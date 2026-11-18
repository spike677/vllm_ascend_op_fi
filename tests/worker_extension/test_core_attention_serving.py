"""Actual CPU Torch calls through the real public Host/Runtime lifecycle.

The Native driver and scheduler are explicit CPU seams. These tests grant no
NPU/layout qualification; real validators have separate negative contracts.
"""
from copy import deepcopy
from types import SimpleNamespace as NS
import threading
import tempfile
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime
from demoncat.runtime.vllm_ascend.worker_extension import attention_execution as execution
from demoncat.runtime.vllm_ascend.worker_extension.attention_native import OwnedLaunchScope, MECHANISMS
from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import (
    install_attention_serving_binding, remove_attention_serving_binding)
from demoncat.runtime.vllm_ascend.worker_extension.providers.attention_serving import (
    FaLegalTilingServingProvider, FiaLegalTilingServingProvider, FiaDenominatorReplayServingProvider,
    PagedDecodeServingProvider)
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from test_extension import _extended_worker
from test_core_host import ExistingExecutor

FACTORIES = dict(zip(MECHANISMS, (FaLegalTilingServingProvider, FiaLegalTilingServingProvider,
                                FiaDenominatorReplayServingProvider, PagedDecodeServingProvider)))


class CpuNative:
    simulation = True
    def __init__(self, mechanism):
        self.mechanism, self.calls, self.stops = mechanism, [], 0
        self.pending, self.fail_stop = False, False
        self.fingerprint = dict(simulation=True, mechanism=mechanism)

    def state(self):
        return dict(native_loaded=True, scope_active=self.pending, scope_owned=self.pending,
                    environment_owned=0, calls=len(self.calls), failures=[], build=self.fingerprint, simulation=True)

    def execute(self, actual, boundary, phase, once):
        self.pending = True
        self.calls.append((actual, boundary['target_key'], phase))
        result = once()
        self.pending = False
        return result, dict(original_calls=1, target_key=boundary['target_key'], phase=phase,
                            selection_applied=phase == 'fault', simulation=True)

    def stop(self):
        self.stops += 1
        if self.fail_stop:raise RuntimeError('CPU_DRAIN_FAILED')
        self.pending = False
        return self.state()


class Attention(torch.nn.Module):
    def __init__(self, api, layer):
        super().__init__()
        self.api, self.layer_name = api, layer
        self.register_buffer('q', torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4))
        self.register_buffer('mask', torch.ones(3, 3, dtype=torch.bool).triu(1))
        self.calls = 0
        self.positional = False

    def forward(self, value):
        self.calls += 1
        args = (self.q, self.q, self.q, 2) if self.positional else (self.q, self.q, self.q)
        return self.api.npu_fused_infer_attention_score(*args, atten_mask=self.mask,
            input_layout='BNSD', scale=.5, num_heads=2, num_key_value_heads=2,
            actual_seq_lengths=[3], actual_seq_lengths_kv=[3], sparse_mode=2)[0] + value


class Logits(torch.nn.Module):
    def forward(self, hidden):return hidden.sum(-1)


class Model(torch.nn.Module):
    def __init__(self, api):
        super().__init__()
        self.a, self.b, self.logits = Attention(api, 'a'), Attention(api, 'b'), Logits()
        self.calls = 0

    def forward(self, input_ids, positions):
        self.calls += 1
        return self.b(self.a(input_ids.float().sum()))


class CoreAttentionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(identity, '_current_device', return_value=0))
        self.worker = _extended_worker(rank=0);self.worker.load_model()
        self.api_calls = []
        def original(*args, **kwargs):
            self.api_calls.append((args, kwargs))
            return (args[0] + 1, torch.empty(0))
        self.original, self.api = original, NS(npu_fused_infer_attention_score=original)
        self.model = Model(self.api).eval()
        self.runner = self.worker.model_runner
        self.runner.model, self.runner.with_prefill = self.model, True
        self.context = NS(model_instance=self.model, no_compile_layers={'a': self.model.a, 'b': self.model.b},
            attn_metadata={'a': NS(attn_mask=self.model.a.mask), 'b': NS(attn_mask=self.model.b.mask)})
        self.executor, self.providers = ExistingExecutor([self.worker]), []
        def resolve(mechanism, version, worker, model, actual_identity):
            provider = FACTORIES[mechanism](worker, model, actual_identity)
            self.providers.append(provider)
            return provider
        self.enterContext(patch.object(runtime, '_resolve_provider', side_effect=resolve))
        self.enterContext(patch.object(runtime, '_provider_capabilities', return_value=[
            dict(mechanism=name, implementation_version='1') for name in MECHANISMS]))
        self.addCleanup(self.dispose)

    def dispose(self):
        for provider in self.providers:
            session = provider.session
            if session and session.handles:
                if isinstance(provider.binding.native, CpuNative):provider.binding.native.fail_stop = False
                session.needs_off = False
                session.close()
        self.assertIsNone(execution._ACTIVE)

    def client(self, mechanism=MECHANISMS[1], native=False):
        self.driver = CpuNative(mechanism) if native else None
        test_context = dict(api_module=self.api, candidates=[('a', self.model.a), ('b', self.model.b)],
            processor=self.model.logits, context=lambda: self.context,
            snapshot=lambda request_id, actual, context: dict(request_id=request_id, num_requests=1,
                status='READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE',
                num_computed_tokens=0, num_prompt_tokens=3, num_scheduled_tokens=3))
        self.binding = install_attention_serving_binding(self.worker, mechanism, prompt_token_ids=[1,2,3],
            provenance={'cpu_only': True}, native=self.driver, _simulation=True, _test_context=test_context)
        fault, variant = mechanism.split('.', 1)
        return WorkerFaultClient(HostAdapter(self.executor), FaultConfig(int(fault), variant))

    def ok(self, reply):
        self.assertEqual(reply['status'], 'ACTION_COMPLETED', repr(reply.get('response', reply))[:1600])
        return reply['response']['provider_state']

    def start(self, client):
        for action in ('prepare', 'register', 'check', 'enable'):self.ok(getattr(client, action)())
        self.assertEqual(self.model.calls, 0)
        return self.providers[-1]

    def request(self, client, index):
        self.ok(client.advance('real-'+str(index)))
        with torch.inference_mode():
            value = self.model(torch.tensor([1,2,3]), torch.tensor([0,1,2]))
            self.model.logits(value)
        return self.ok(client.collect_replay())

    def cleanup(self, client):
        self.ok(client.unregister())
        self.assertEqual(client.detach()['status'], 'DETACHED')
        remove_attention_serving_binding(self.worker, client.config.mechanism)

    def test_four_factories_are_distinct_and_unqualified_before_actual_calls(self):
        for mechanism in MECHANISMS:
            with self.subTest(mechanism=mechanism):
                client = self.client(mechanism)
                state = self.ok(client.prepare())
                self.assertEqual(state['mechanism'], mechanism)
                self.assertFalse(state['native_loaded'] or state['C1'] or state['C2'] or state['C3'])
                self.assertEqual(state['evidence']['requests'], [])
                self.cleanup(client)
        self.assertEqual(self.model.calls, 0)

    def test_real_unsupported_calls_preserve_full_denominator_and_original_results(self):
        client = self.client();provider = self.start(client)
        self.request(client, 0);state = self.request(client, 1)
        rows = state['evidence']['requests']
        self.assertEqual(len(rows), 2)
        self.assertEqual([len(row['forwards'][0]['calls']) for row in rows], [2,2])
        self.assertIsNone(state['evidence']['selection'])
        for request in rows:
            for call in request['forwards'][0]['calls']:
                self.assertFalse(call['input_contract_matches'])
                self.assertTrue(call['original_returned'] and call['inputs_unchanged'])
                self.assertEqual(call['original_calls'], 1)
                self.assertTrue(call['reasons'])
        self.assertEqual((self.model.calls, len(self.api_calls)), (2,4))
        self.assertEqual(client.advance('refused-baseline')['status'], 'PROVIDER_ACTION_FAILED')
        self.cleanup(client)
        self.assertIs(self.api.npu_fused_infer_attention_score, self.original)
        with torch.inference_mode():self.model(torch.tensor([1,2,3]), torch.tensor([0,1,2]))
        self.assertEqual((self.model.calls, len(self.api_calls)), (3,6))

    def test_query_is_read_only_and_no_forward_is_created_by_advance(self):
        client = self.client();provider = self.start(client)
        self.ok(client.advance('not-executed'))
        before = (self.model.calls, deepcopy(provider.records), provider.pending)
        for _ in range(3):self.assertEqual(client.query()['status'], 'QUERY_OK')
        self.assertEqual(before, (self.model.calls, provider.records, provider.pending))
        self.assertEqual(client.collect_replay()['status'], 'PROVIDER_ACTION_FAILED')
        self.cleanup(client)

    def test_unknown_positional_signature_is_retained_without_guessing_or_native(self):
        self.model.a.positional = self.model.b.positional = True
        client = self.client();self.start(client)
        self.request(client, 0);state = self.request(client, 1)
        calls = state['evidence']['requests'][-1]['forwards'][0]['calls']
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(row['target_key'] is None and row['original_calls'] == 1
            and row['original_returned'] and not row['input_contract_matches'] for row in calls))
        self.assertIsNone(state['evidence']['selection'])
        self.cleanup(client)

    def test_matching_native_baseline_is_exactly_once_but_missing_stock_bridge_refuses_f(self):
        client = self.client(native=True);provider = self.start(client)
        # This seam changes only the small CPU geometry validator. All public
        # RPCs, real model calls, original API identity and receipts remain real.
        with patch.object(execution.AttentionExecutionAdapter, '_boundary', return_value={'input_contract_matches': True}):
            for index in range(4):state = self.request(client, index)
        self.assertEqual(len(self.driver.calls), 2)
        self.assertEqual([call[2] for call in self.driver.calls], ['baseline','baseline_repeat'])
        self.assertEqual((self.model.calls, len(self.api_calls)), (4,8))
        reply = client.advance('would-be-fault')
        self.assertEqual(reply['status'], 'PROVIDER_ACTION_FAILED')
        self.assertEqual(reply['response']['provider_code'], 'ATTENTION_STOCK_PRIVATE_BRIDGE_REQUIRED')
        self.assertEqual(len(self.driver.calls), 2)
        self.assertIsNone(provider.pending)
        self.cleanup(client)

    def test_wrong_worker_model_never_releases_owner(self):
        client = self.client();provider = self.start(client)
        self.runner.model = Model(self.api).eval()
        try:
            self.assertNotEqual(client.unregister()['status'], 'ACTION_COMPLETED')
            self.assertTrue(self.binding.leases)
            self.assertTrue(provider.session.handles)
        finally:self.runner.model = self.model
        # Runtime quarantines an observed model identity change permanently;
        # putting back an old pointer is not authority to revive the ticket.
        self.assertNotEqual(client.unregister()['status'], 'ACTION_COMPLETED')
        self.assertTrue(self.binding.leases)

    def test_foreign_api_wrapper_is_not_uninstalled(self):
        client = self.client();provider = self.start(client)
        owned = self.api.npu_fused_infer_attention_score
        foreign = lambda *args, **kwargs: owned(*args, **kwargs)
        self.api.npu_fused_infer_attention_score = foreign
        try:
            self.assertEqual(client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
            self.assertIs(self.api.npu_fused_infer_attention_score, foreign)
            self.assertTrue(self.binding.leases)
        finally:self.api.npu_fused_infer_attention_score = owned
        self.cleanup(client)

    def test_native_drain_failure_retains_hooks_owner_and_retries(self):
        client = self.client(native=True);provider = self.start(client)
        self.driver.fail_stop = True
        self.assertEqual(client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(provider.session.handles and self.binding.leases)
        self.driver.fail_stop = False
        self.cleanup(client)
        self.assertTrue(provider.failures)

    def test_clean_cannot_release_binding_until_unregister(self):
        client = self.client();provider = self.start(client)
        self.ok(client.clean())
        self.assertEqual(self.binding.leases, {provider._ticket})
        with self.assertRaisesRegex(RuntimeError, 'STILL_OWNED'):
            remove_attention_serving_binding(self.worker, client.config.mechanism)
        self.cleanup(client)

    def test_existing_hooks_survive_cleanup_and_later_input_wrapper_is_refused(self):
        seen=[]
        existing=self.model.register_forward_hook(lambda module, args, output:seen.append('existing'))
        client = self.client();provider = self.start(client)
        self.request(client,0)
        self.assertEqual(seen,['existing'])
        late=self.model.register_forward_pre_hook(lambda module, args:None)
        self.ok(client.advance('later'))
        with self.assertRaisesRegex(RuntimeError, 'ROOT_INPUT_NOT_FINAL'):
            self.model(torch.tensor([1,2,3]), torch.tensor([0,1,2]))
        self.cleanup(client)
        self.assertIn(existing.id,self.model._forward_hooks)
        self.assertIn(late.id,self.model._forward_pre_hooks)
        existing.remove();late.remove()

    def test_failed_actual_request_cleans_resources_without_permanent_owner_lock(self):
        client = self.client();provider = self.start(client)
        self.ok(client.advance('bad-request'))
        with self.assertRaisesRegex(RuntimeError, 'CONTEXT_OBJECT_CHANGED'):
            # A request/context error is recorded as an ineligible call by the
            # observer, so force an original API failure to exercise unwinding.
            with patch.object(provider.session, '_invoke', side_effect=RuntimeError('CONTEXT_OBJECT_CHANGED')):
                self.model(torch.tensor([1,2,3]), torch.tensor([0,1,2]))
        self.cleanup(client)

    def test_partial_install_failure_is_owned_and_public_cleanup_retries(self):
        client = self.client()
        for action in ('prepare','register','check'):self.ok(getattr(client, action)())
        original = self.model.a.register_forward_pre_hook
        with patch.object(self.model.a, 'register_forward_pre_hook', side_effect=RuntimeError('partial install')):
            self.assertEqual(client.enable()['status'], 'PROVIDER_ACTION_FAILED')
        self.assertTrue(self.providers[-1].session.handles)
        self.cleanup(client)
        self.assertEqual(len(self.model._forward_hooks), 0)

    def test_missing_deployment_refuses_before_runtime_owner_is_created(self):
        client = WorkerFaultClient(HostAdapter(self.executor), FaultConfig(28, 'fia.legal_tiling'))
        self.assertEqual(client.prepare()['status'], 'PROVIDER_UNAVAILABLE')
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)
        self.assertEqual(self.model.calls, 0)
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_actual_request_id_context_mismatch_is_not_accepted_as_complete(self):
        client = self.client();provider = self.start(client)
        self.ok(client.advance('expected'))
        self.binding.test_context['snapshot'] = lambda *args: (_ for _ in ()).throw(RuntimeError('ACTUAL_REQUEST_SET_MISMATCH'))
        with self.assertRaisesRegex(RuntimeError, 'ACTUAL_REQUEST_SET_MISMATCH'):
            self.model(torch.tensor([1,2,3]), torch.tensor([0,1,2]))
        self.assertFalse(provider.records)
        self.cleanup(client)

    def test_zero_f_probe_uses_real_rpc_and_post_unregister_ordinary_enqueue(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location('attention_zero_f_probe',
            root/'experiments/c2c3_closure_20261117/natural_attention_worker_core_probe.py')
        probe = importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
        client = self.client()
        class Llm:
            def __init__(slf):
                slf.llm_engine=NS(has_unfinished_requests=lambda:False)
                slf.index=0
            def reset_prefix_cache(slf, **kwargs):return True
            def enqueue(slf, *args, **kwargs):
                slf.index+=1;slf.request='natural-'+str(slf.index);return [slf.request]
            def wait_for_completion(slf, **kwargs):
                with torch.inference_mode():
                    hidden=self.model(torch.tensor([1,2,3]), torch.tensor([0,1,2]))
                    self.model.logits(hidden)
                return [NS(request_id=slf.request, outputs=[NS(token_ids=[7])])]
        with tempfile.TemporaryDirectory() as temp, patch.object(probe, 'read_enqueued_identity',
                side_effect=lambda llm, ids, **kwargs: dict(internal_request_id=ids[0], external_request_id=ids[0])):
            report=dict(status='STARTED', healthy_before=[7])
            probe.drive(Llm(), client, prompt={'prompt_token_ids':[1,2,3]}, params=object(), out=Path(temp), report=report)
            self.assertEqual(report['status'], probe.SUCCESS, report.get('error'))
            self.assertFalse(report['worker_core']['ownership_retained'])
            self.assertEqual(len(report['requests']),2)
            self.assertEqual(self.model.calls,3)
            self.assertEqual(report['post_unregister_health']['scope'], 'TOKENS_ONLY_NO_POST_UNREGISTER_LOGITS_OBSERVER')
            self.assertTrue(Path(temp, 'host_receipt.json').is_file())
        remove_attention_serving_binding(self.worker, client.config.mechanism)

    def test_after_removal_exception_retires_only_confirmed_hook_then_public_retry(self):
        client = self.client();provider = self.start(client)
        handle = provider.session.handles[-1][0]
        remove = handle.remove
        def after():remove();raise RuntimeError('after remove')
        handle.remove = after
        self.assertEqual(client.unregister()['status'], 'PROVIDER_ACTION_FAILED')
        self.assertNotIn(handle, [item[0] for item in provider.session.handles])
        self.assertTrue(self.binding.leases)
        self.cleanup(client)


class NativeScopeRecoveryTests(unittest.TestCase):
    def fixture(self):
        from fault_injection.native_adapters.attention28.launch import LaunchObserver
        observer = LaunchObserver.__new__(LaunchObserver)
        state = dict(active=0, mode='ok', ends=0, token=None)
        class Function:
            def __init__(self, fn):self.fn=fn
            def __call__(self, *args):return self.fn(*args)
        def begin(token):state['active']=1;state['token']=token;return 0
        def end():
            state['ends'] += 1
            if state['mode'] == 'before':raise RuntimeError('before end')
            state['active']=0
            if state['mode'] == 'after':raise RuntimeError('after end')
            return 0
        observer.lib = NS(dcat_attention_observer_scope_active_v1=Function(lambda:state['active']),
            dcat_attention_observer_scope_matches_v1=Function(lambda token: 0 if not state['active'] else
                                                              1 if token == state['token'] else 2),
            dcat_attention_observer_begin_v1=Function(begin), dcat_attention_observer_end_v1=Function(end))
        observer.active=False
        return OwnedLaunchScope(observer), state

    def test_before_end_keeps_owned_native_scope_until_actual_retry(self):
        scope, state = self.fixture();state['mode']='before'
        with self.assertRaisesRegex(RuntimeError, 'before end'):
            with scope.scope('test'):pass
        self.assertEqual(scope.state()['active'], True);self.assertTrue(scope.owned)
        state['mode']='ok';scope.stop()
        self.assertFalse(scope.owned);self.assertEqual(state['active'], 0)
        self.assertIn('before end', scope.first_error)

    def test_after_end_readback_retires_resource_but_keeps_first_error(self):
        scope, state = self.fixture();state['mode']='after'
        with self.assertRaisesRegex(RuntimeError, 'after end'):
            with scope.scope('test'):pass
        self.assertFalse(scope.owned)
        state['mode']='ok';scope.stop()
        self.assertEqual(state['ends'], 1)
        self.assertIn('after end', scope.first_error)

    def test_foreign_scope_and_thread_cannot_be_drained(self):
        scope, state = self.fixture();state['active']=1
        with self.assertRaisesRegex(RuntimeError, 'FOREIGN_NATIVE_SCOPE'):scope.stop()
        state['active']=0
        errors=[]
        thread=threading.Thread(target=lambda: self._thread_stop(scope, errors));thread.start();thread.join()
        self.assertEqual(len(errors),1);self.assertIn('OWNER_THREAD', errors[0])
        self.assertEqual(state['ends'],0)

    def test_replaced_live_native_token_never_clears_foreign_scope(self):
        scope, state = self.fixture()
        with self.assertRaisesRegex(RuntimeError, 'TOKEN_OWNER_CHANGED'):
            with scope.scope('ours'):state['token']=b'foreign'
        self.assertTrue(scope.owned)
        self.assertEqual(state['active'], 1)
        self.assertEqual(state['ends'], 0)
        state['token']=b'ours';scope.stop()
        self.assertFalse(scope.owned)

    @staticmethod
    def _thread_stop(scope, errors):
        try:scope.stop()
        except RuntimeError as error:errors.append(str(error))


if __name__ == '__main__':unittest.main()
