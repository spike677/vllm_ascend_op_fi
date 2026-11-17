"""Executed CPU controller contracts; patched host metadata is never Worker evidence."""
from contextlib import ExitStack
import os
import sys
from types import MethodType, ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch
from test_rmsnorm_natural import CanonicalNorm
from fault_injection.runtime_discovery import _process_start_identity
from demoncat.runtime.vllm_ascend.worker_extension import serving_tensor
from test_tensor_statistics_window import GuardWindow


class Processor(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.register_buffer('weight', torch.arange(32.).reshape(8, 4)/32)
    def forward(self, selected):
        return torch.mm(selected, self.weight)


class WorkerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('embedding', torch.arange(64.).reshape(8, 8)/8)
        self.first = CanonicalNorm(8); self.last = CanonicalNorm(8)
        self.processor = Processor()
    def forward(self, input_ids, positions):
        return self.last(self.first(self.embedding[input_ids]) + 1)
    def logits(self, hidden):
        return self.processor(hidden[torch.tensor([hidden.shape[0]-1])])


class ServingTensorGroupTests(unittest.TestCase):
    def setUp(self):
        self.patches = ExitStack()
        fake_logits = ModuleType('vllm.model_executor.layers.logits_processor')
        fake_logits.LogitsProcessor = Processor
        fake_context = ModuleType('vllm.forward_context')
        fake_context.get_forward_context = lambda: SimpleNamespace(cpu_contract=True)
        self.patches.enter_context(patch.dict(sys.modules, {fake_logits.__name__: fake_logits,
                                                           fake_context.__name__: fake_context}))
        self.patches.enter_context(patch.object(serving_tensor, 'require_serial_eager', lambda runner: None))
        self.patches.enter_context(patch.object(serving_tensor, 'discover_logits_boundary',
            lambda model, **kwargs: dict(processor=model.processor, report=dict(cpu_contract=True))))
        def snapshot(runner, model, actual, context, **kwargs):
            self.assertTrue(context.cpu_contract)
            self.assertEqual(actual['input_ids'].tolist(), kwargs['expected_prompt_token_ids'])
            self.assertNotIn(kwargs['expected_internal_request_id'], kwargs['seen_internal_request_ids'])
            return dict(cpu_contract=True, request_id=kwargs['expected_internal_request_id'])
        self.patches.enter_context(patch.object(serving_tensor, 'snapshot_fresh_prefill', snapshot))
        self.model = WorkerModel().eval()
        self.worker = SimpleNamespace(model_runner=SimpleNamespace(model=self.model,
            vllm_config=SimpleNamespace(model_config=SimpleNamespace(hf_config=SimpleNamespace(vocab_size=4)))))
        nonce = uuid.uuid4().hex
        self.worker.demoncat_worker_identity = lambda: dict(status='WORKER_IDENTIFIED', quarantined=False,
            worker=dict(pid=os.getpid(), rank=0, process_start_identity=_process_start_identity(),
                        worker_start_identity=nonce, model_instance_id=nonce))
        self.session = serving_tensor.ServingTensorGroup(self.worker, prompt=[1, 2, 3],
            weight_manifest=dict(cpu_fixture=True), workload_sha256='a'*64, environment_sha256='b'*64,
            _simulation=True, _synchronize=lambda: None)
        self.next_request = 0

    def tearDown(self):
        try:
            if self.session.handles:
                self.session.abort()
        finally:
            self.patches.close()

    def request(self, phase, arm=None):
        request_id = 'cpu-contract-request-' + str(self.next_request); self.next_request += 1
        self.session.expect(request_id, phase, arm)
        hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
        logits = self.model.logits(hidden)
        row = self.session.collect()
        self.assertEqual(row['request_id'], request_id)
        self.assertTrue(row['logits_boundary_returned'])
        return row, logits

    def test_real_forwards_and_post_root_logits_through_both_coverage_groups(self):
        for phase, arm in (('native_baseline', None), ('baseline', 'baseline'),
                           ('baseline_repeat', 'baseline'), ('lossless', 'baseline')):
            self.request(phase, arm)
        discovery = self.session.qualify()
        self.assertEqual(discovery['candidate_set']['eligible'], 2)
        self.session.provider.open_group('once', forwards=1)
        fault, _ = self.request('c2_fault', 'fault')
        self.request('c2_off')
        c2 = self.session.finish_group()
        self.assertTrue(c2['contract_satisfied'], c2['rejection_reasons'])
        self.assertFalse(c2['real_worker_evidence'])
        self.assertFalse(c2['public_qualification_granted'])
        self.assertFalse(self.session.handles)
        last = fault['mechanism']['events'][-1]
        self.assertTrue(any(row['post_root'] for row in last['consumers']))
        proof = self.session.provider.group.c2_proof()
        self.session.provider.open_group('every_match', forwards=2, c2_proof=proof)
        self.request('c3_fault_0', 'fault'); self.request('c3_fault_1', 'fault'); self.request('c3_off')
        c3 = self.session.finish_group()
        self.assertTrue(c3['contract_satisfied'], c3['rejection_reasons'])
        self.assertEqual(c3['observed_eligible_calls'], 4)
        self.assertEqual(c3['hits'], 4)
        self.assertTrue(c3['recovery']['complete'])
        self.assertFalse(self.session.close()['standard_worker'])
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for m in self.model.modules()))

    def test_foreign_serving_callback_is_not_removed(self):
        handle = self.session.handles[0]; hooks = handle.hooks_dict_ref(); original = hooks[handle.id]
        foreign = lambda *args: None
        hooks[handle.id] = foreign
        try:
            with self.assertRaisesRegex(RuntimeError, 'HOOK_OWNERSHIP_CHANGED'):
                self.session.close()
            self.assertIs(hooks[handle.id], foreign)
        finally:
            hooks[handle.id] = original
        self.session.close()

    def test_target_change_refuses_new_request_but_owned_cleanup_still_runs(self):
        for phase, arm in (('native_baseline', None), ('baseline', 'baseline'),
                           ('baseline_repeat', 'baseline'), ('lossless', 'baseline')):
            self.request(phase, arm)
        self.session.qualify()
        original = self.model.last.forward
        self.model.last.forward = MethodType(Processor.forward, self.model.last)
        try:
            with self.assertRaisesRegex(RuntimeError, 'DISPATCH_CALLABLE_CHANGED'):
                self.session.expect('changed-target', 'c2_fault', 'fault')
            self.assertIsNone(self.session.adapter)
            self.session.close()
            self.assertFalse(self.session.handles)
        finally:
            self.model.last.forward = original

    def test_worker_identity_drift_refuses_arm_and_retains_cleanup(self):
        original = self.worker.demoncat_worker_identity
        changed = original(); changed['worker']['worker_start_identity'] = 'new-worker-lifetime'
        self.worker.demoncat_worker_identity = lambda: changed
        try:
            with self.assertRaisesRegex(RuntimeError, 'WORKER_IDENTITY_CHANGED'):
                self.session.expect('wrong-worker', 'c2_fault', 'fault')
            self.session.close()
        finally:
            self.worker.demoncat_worker_identity = original

    def test_rpc_capture_includes_model_logits_and_sampler_but_excludes_owned_snapshots(self):
        from fault_injection.rmsnorm_return import RMSNormReturnSession
        class RpcWindow(GuardWindow):
            def begin(self, model, phase):
                self.rows.append(phase); self.active = True
            def end(self, *, completed):
                self.completed = completed; self.active = False
        for phase, arm in (('native_baseline', None), ('baseline', 'baseline'),
                           ('baseline_repeat', 'baseline'), ('lossless', 'baseline')):
            self.request(phase, arm)
        self.session.qualify(); capture = RpcWindow()
        self.session.configure_statistics(capture, 'fault')
        original_capture = serving_tensor._capture
        original_cpu = RMSNormReturnSession.cpu
        original_stride = torch.Tensor.stride
        snapshots = []
        def guarded_capture(value):
            self.assertFalse(capture.active); snapshots.append('complete_snapshot')
            return original_capture(value)
        def guarded_cpu(adapter, tensor):
            self.assertFalse(capture.active); snapshots.append('local_readback')
            return original_cpu(adapter, tensor)
        def guarded_stride(tensor, *args):
            self.assertFalse(capture.active); snapshots.append('owned_metadata')
            return original_stride(tensor, *args)
        returns = []
        h1 = self.model.first.register_forward_hook(lambda *args: returns.append(('model', capture.active)))
        h2 = self.model.processor.register_forward_pre_hook(lambda *args: returns.append(('logits', capture.active)))
        try:
            with patch.object(serving_tensor, '_capture', guarded_capture), \
                 patch.object(RMSNormReturnSession, 'cpu', guarded_cpu), \
                 patch.object(torch.Tensor, 'stride', guarded_stride):
                self.session.expect('cpu-captured-request', 'statistics_request', 'fault')
                hidden = self.model(input_ids=torch.tensor([1, 2, 3]), positions=torch.arange(3))
                logits = self.model.logits(hidden)
                self.assertTrue(capture.active)
                torch.argmax(logits, dim=-1)  # Real sampling work remains in the capture.
                row = self.session.collect()
        finally:
            h1.remove(); h2.remove()
        self.assertTrue(capture.completed)
        self.assertFalse(capture.active)
        self.assertEqual(returns, [('model', True), ('logits', True)])
        self.assertEqual(row['statistics']['local_hits'], 2)
        self.assertFalse(row['statistics']['grants_coverage'])
        self.assertEqual(set(snapshots), {'complete_snapshot', 'local_readback', 'owned_metadata'})
        self.request('statistics_off')
        self.assertIsNotNone(self.session.provider.off_row)
        self.session.close()


if __name__ == '__main__':
    unittest.main()
