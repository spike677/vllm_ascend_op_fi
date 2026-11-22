"""Actual CPU serving/provider hooks with explicit non-CANN Native doubles."""
import copy
import ctypes
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection import quant_group_native as native
from fault_injection.deployments.quant_group_c1 import verify_completion
from demoncat.runtime.vllm_ascend.worker_extension import serving_quant as serving
from demoncat.runtime.vllm_ascend.worker_extension import serving_quant_group_c1 as c1
from demoncat.runtime.vllm_ascend.worker_extension.providers.quant import QuantProvider
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows, source_bindings
from demoncat.runtime.vllm_ascend.worker_extension.quant_mla_metadata_rows import MlaMetadataRows
from test_quant_group_inventory import CpuJournal, cpu_view
from test_quant_inventory import TwoQuantProjections
from test_serving_session_lifecycle import CpuProcessor


_CPU_DISPATCH = torch.library.Library('cpu_quant_group_c1', 'DEF')
_CPU_DISPATCH.define('quant(Tensor input, *, Tensor? smooth=None) -> (Tensor, Tensor)')
_CPU_DISPATCH.define('unknown(Tensor input) -> (Tensor, Tensor)')
_CPU_DISPATCH.define('mla(Tensor hidden_states, bool need_gather_q_kv, Tensor(a!) output, str layer_name) -> ()')
_CPU_BINDING = threading.local()


def cpu_native_quant(input, *, smooth=None):
    return _CPU_BINDING.current._quant(input)


def cpu_unknown_quant(input):
    return torch.ops.cpu_quant_group_c1.quant(input)


def cpu_mla_quant(hidden_states, need_gather_q_kv, output, layer_name):
    self = _CPU_BINDING.model
    forward_context = NS(no_compile_layers={layer_name: self})
    kv_cache = self.mla_attn.kv_cache[0]
    output.copy_(self.first(hidden_states))


_CPU_DISPATCH.impl('quant', cpu_native_quant, 'CPU')
_CPU_DISPATCH.impl('unknown', cpu_unknown_quant, 'CPU')
_CPU_DISPATCH.impl('mla', cpu_mla_quant, 'CPU')


class CpuGroupBinding:
    """Typed ABI event fixtures, never an oracle for CANN quantization."""
    def __init__(self):
        self.journal = CpuJournal()
        self.journal.binding = self
        self.identity = self.journal.build_identity
        self.config = self.observed = self.finished = None
        self.lease = self.completion = None
        self.lifetime = 0
        self.selected_modes, self.calls, self.seals = [], [], []
        self.drain_result = True
        self.corrupt_role = False
        self.no_effect = False
        self.omit_selected_completion = False
        self.dispatch_quant = self.unknown_dispatch = False
        self.dispatch_smooth = None

    def query(self):
        return copy.deepcopy(self.journal.state_value)

    def raw_range(self, first, last):
        return copy.deepcopy(self.journal.rows[first-1:last])

    def raw(self, sequence):
        return self.raw_range(sequence, sequence)[0]

    def install(self, config, observed, finished):
        self.calls.append('install')
        self.config, self.observed, self.finished = config, observed, finished
        self.journal.state_value['core'].update(installed=1, owner_hi=config.owner_hi, owner_lo=config.owner_lo,
                                              generation=config.generation)

    def pass_call(self, call):
        return 0

    def admit(self, lease):
        self.lease = native.copy_struct(lease)
        self.selected_modes.append(lease.mode)
        core = self.journal.state_value['core']
        core.update(targets=1, host_bytes=512, pending_parent=1)
        core['admitted_model_calls'] += 1
        return 0

    def invoke(self, x, original):
        self.lifetime += 1
        self.journal.state_value['public_active'] = 1
        try:
            result = original()
            return result, native.snapshot(self.completion)
        finally:
            self.journal.state_value['public_active'] = 0

    def _append(self, row):
        self.journal.rows.append(native.snapshot(row))
        census = self.journal.state_value['census']
        census['total'] = self.journal.state_value['last_sequence'] = len(self.journal.rows)

    def quant(self, x):
        if not self.dispatch_quant:
            return self._quant(x)
        _CPU_BINDING.current = self
        try:
            if self.unknown_dispatch:
                return torch.ops.cpu_quant_group_c1.unknown(x)
            return torch.ops.cpu_quant_group_c1.quant(x, smooth=self.dispatch_smooth)
        finally:
            del _CPU_BINDING.current

    def _quant(self, x):
        # Fixed CPU fixtures exercise ownership, publication and byte checks.
        # They do not simulate the Ascend quantization rounding formula.
        q, scale = x.to(torch.int8), torch.ones(x.shape[0], dtype=torch.float32)
        parent = native.Call(token=len(self.journal.rows)+1, thread=threading.get_native_id(), device=0,
            lifetime_token=self.lifetime if self.journal.state_value['public_active'] else 0)
        for ordinal, (name, tensor) in enumerate((('input', x), ('quant', q), ('scale', scale))):
            value = cpu_view(tensor, name, 0)
            value.descriptor = 100 + ordinal
            setattr(parent, name, value)
        self._append(native.RawRecord(kind=1, parent=parent))
        census = self.journal.state_value['census']
        census['model_parents'] += 1
        self.lease = None
        if self.config is not None:
            self.observed(native.copy_struct(parent))
        lease = self.lease
        if lease is None:
            records = self._child(parent, None, 2, parent.input, parent.quant, parent.scale)
            self._prepare(records)
            self._execute(records)
            done = native.Completion(call=parent, prepare_enter=records.prepare_enter, prepare_exit=records.prepare_exit,
                execute_enter=records.execute_enter, execute_exit=records.execute_exit)
            if self.config is not None:
                self.finished(done)
            return q, scale
        done = native.Completion(call=parent, admitted=1)
        receipt = done.core
        receipt.version, receipt.size = 1, ctypes.sizeof(native.Receipt)
        receipt.mode, receipt.owner_hi, receipt.owner_lo = lease.mode, lease.owner_hi, lease.owner_lo
        receipt.generation, receipt.target, receipt.actual_row_proof = lease.generation, lease.target, lease.actual_row_proof
        receipt.parent_token, receipt.lifetime_token = parent.token, parent.lifetime_token
        keep = []
        def owned(view, role):
            value = native.copy_struct(view)
            tensor = torch.empty(parent.quant.rows, parent.quant.columns, dtype=torch.int8) if role == 'quant' else torch.empty(parent.scale.rows)
            keep.append(tensor)
            value.address, value.descriptor = tensor.data_ptr(), 1000 + len(keep)
            return value
        effect_q, effect_s = parent.quant, parent.scale
        if lease.mode in (2, 3):
            effect_q, effect_s = owned(parent.quant, 'quant'), owned(parent.scale, 'scale')
        effect_x = native.copy_struct(parent.input)
        if lease.mode == 3:
            receipt.reference = self._child(parent, lease, 1, parent.input, owned(parent.quant, 'quant'), owned(parent.scale, 'scale'))
            effect_x.rows = effect_q.rows = effect_s.rows = 1
            effect_x.columns = effect_q.columns = parent.input.rows * parent.input.columns
            effect_x.stride0 = effect_q.stride0 = effect_x.columns
            effect_x.descriptor, effect_s.bytes = 900, 4
        receipt.effect = self._child(parent, lease, 2, effect_x, effect_q, effect_s)
        children = [receipt.reference, receipt.effect] if lease.mode == 3 else [receipt.effect]
        for child in children:
            self._prepare(child)
        for child in children:
            self._execute(child)
            child.drained = child.readback = 1
        receipt.input_before_captured = receipt.input_unchanged = receipt.published = receipt.publication_exact = 1
        if lease.mode == 3:
            receipt.reference_effect_compared = 1
            if not self.no_effect:
                q[0, 0] += 1
                receipt.changed_quant_bytes = 1
        if not self.omit_selected_completion:
            self.finished(native.copy_struct(done))
        receipt.released = 1
        self.journal.state_value['core']['pending_parent'] = 0
        self.completion = native.copy_struct(done)
        return q, scale

    def _child(self, parent, lease, role, x, q, scale):
        child = native.ChildReceipt()
        child.plan.role = role
        child.plan.parent_token = parent.token
        child.plan.child_lease = parent.token * 4 + role
        child.plan.input, child.plan.quant, child.plan.scale = x, q, scale
        self._child_lease = lease
        return child

    def _prepare(self, child):
        child.prepare_enter = child.registration = len(self.journal.rows) + 1
        child.prepare_exit = child.prepare_enter + 1
        child.executor = child.registration + 10000
        child.prepared = child.prepare_original_calls = 1
        census = self.journal.state_value['census']
        census['native_prepares'] += 1
        if self.lease is None:
            census['unclassified_prepares'] += 1
        else:
            census['helper_prepares' if child.plan.role == 1 else 'model_prepares'] += 1
        self._event(child, 1)
        self._event(child, 2)

    def _execute(self, child):
        child.execute_enter = len(self.journal.rows) + 1
        child.execute_exit = child.execute_enter + 1
        child.stream = 71
        child.returned = child.execute_original_calls = 1
        self.journal.state_value['census']['native_executes'] += 1
        self._event(child, 3)
        self._event(child, 4)

    def _event(self, child, kind):
        event = native.NativeEvent(kind=kind, sequence=len(self.journal.rows)+1, registration=child.registration,
            thread=threading.get_native_id(), device=0, executor=0 if kind == 1 else child.executor,
            stream=71 if kind >= 3 else 0, original_calls=int(kind in (2, 4)), destination_dtype=2,
            input=child.plan.input, quant=child.plan.quant, scale=child.plan.scale)
        binding = native.ChildBinding()
        if self.lease is not None:
            binding = native.ChildBinding(parent_token=child.plan.parent_token, child_lease=child.plan.child_lease,
                owner_hi=self.lease.owner_hi, owner_lo=self.lease.owner_lo, generation=self.lease.generation,
                role=child.plan.role if not self.corrupt_role else 9)
        self._append(native.RawRecord(kind=2, native=event, binding=binding))

    def seal(self, targets, scope, *, unresolved_candidates):
        self.seals.append((targets, scope, unresolved_candidates))
        self.journal.state_value['core']['sealed'] = 1

    def stop(self, target):
        self.calls.append('stop')
        self.journal.state_value['core']['stopped'] = 1

    def drain(self, target):
        self.calls.append('drain')
        return self.drain_result

    def recover(self):
        self.calls.append('recover')
        self.journal.state_value['core'].update(installed=0, targets=0, host_bytes=0, pending_parent=0)
        return self.query()

    def cancel_empty(self):
        self.calls.append('cancel_empty')
        self.journal.state_value['core']['installed'] = 0
        return True

    def remove(self):
        self.calls.append('remove')
        self.config = self.observed = self.finished = None


class QuantGroupC1Tests(unittest.TestCase):
    def setUp(self):
        from research_fixture import install_research_resolver
        install_research_resolver(self,"quant.batch_shared_scale")
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.binding = CpuGroupBinding()
        self.enterContext(patch.object(native.QuantGroupNative, 'tensor_view', side_effect=cpu_view))
        self.api = ModuleType('torch_npu')
        self.api.get_npu_format = lambda _: 2
        self.api.npu_dynamic_quant = self.binding.quant
        self.api.npu_quant_matmul = lambda x1, x2, scale, pertoken_scale, output_dtype: (x1.float() @ x2.float()).to(output_dtype)
        self.synchronizes = []
        self.api.npu = NS(current_device=lambda: 0, synchronize=lambda: self.synchronizes.append(True))
        class Embedding(torch.nn.Module):
            def forward(self, input_):
                return input_[:, None].expand(-1, 4).to(torch.bfloat16).contiguous()
        class Head(Embedding):
            pass
        class Norm(torch.nn.Module):
            def forward(self, x):
                return x.clone()
        native_fixture = self.binding
        class Model(TwoQuantProjections):
            def __init__(self):
                super().__init__()
                self.embed, self.norm = Embedding(), Norm()
                self.head = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
                self.processor = CpuProcessor()
                self.unknown_first = False
                self.fault_nonfinite = False
                self.before_first = self.after_first = None
                self.first_call = None
            def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
                x = self.norm(self.embed(input_ids))
                if self.unknown_first:
                    x = torch.roll(x, 1, dims=0)
                if self.before_first is not None:
                    self.before_first()
                first = self.first(x) if self.first_call is None else self.first_call(x)
                if self.after_first is not None:
                    self.after_first()
                output = self.second(torch.roll(first, 1, dims=0))
                if self.fault_nonfinite and native_fixture.config is not None and native_fixture.selected_modes[-1:] == [3]:
                    output = output * float('nan')
                return output
        self.model = Model().eval()
        self.runner = NS(model=self.model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))),
            logits_indices=torch.arange(8), with_prefill=True)
        self.worker = NS(model_runner=self.runner, rank=0, demoncat_worker_identity=lambda:
            dict(status='WORKER_IDENTIFIED', quarantined=False, worker={'rank': 0, 'worker_id': 'CPU'}))
        from fault_injection.runtime_discovery import _process_start_identity
        self.identity = dict(pid=os.getpid(), process_start_identity=_process_start_identity(), rank=0,
            configured_logical_device=0, logical_device=0, observed_logical_device=0,
            worker_start_identity='EXPLICIT_CPU_WORKER', model_instance_id='EXPLICIT_CPU_MODEL')
        self.worker.device = NS(type='npu', index=0)
        modules = {name: ModuleType(name) for name in ('vllm', 'vllm.model_executor', 'vllm.model_executor.layers',
            'vllm.model_executor.layers.logits_processor', 'vllm.model_executor.layers.vocab_parallel_embedding',
            'vllm.model_executor.layers.layernorm', 'vllm.forward_context')}
        modules['torch_npu'] = self.api
        modules['vllm.model_executor.layers.logits_processor'].LogitsProcessor = CpuProcessor
        modules['vllm.model_executor.layers.vocab_parallel_embedding'].VocabParallelEmbedding = Embedding
        modules['vllm.model_executor.layers.vocab_parallel_embedding'].ParallelLMHead = Head
        modules['vllm.model_executor.layers.layernorm'].RMSNorm = Norm
        modules['vllm.forward_context'].get_forward_context = lambda: NS()
        self.enterContext(patch.dict(sys.modules, modules))
        self.enterContext(patch.object(serving, 'require_serial_eager'))
        self.enterContext(patch.object(serving, 'discover_logits_boundary', return_value=dict(processor=self.model.processor,
            lm_head=self.model.head, report={'output_contract': [8, 3]})))
        self.enterContext(patch.object(serving, 'snapshot_first_decode', side_effect=lambda runner, model, actual, context, expected:
            dict(request_ids=list(expected), logical_ids=list(range(8)), row_token_positions=[2]*8,
                prefixes=[{'prompt': [1, 2], 'output': [3]} for _ in range(8)], input_ids=[3]*8)))
        self.enterContext(patch.object(native.QuantGroupNative, 'load', return_value=self.binding))
        self.enterContext(patch('fault_injection.deployments.quant_group_inventory.GroupQuantJournal', return_value=self.binding.journal))
        self.enterContext(patch.object(SourceBoundRows, 'from_runtime', side_effect=lambda graph, model:
            SourceBoundRows(graph, model, [], {}, simulation=True)))
        self.enterContext(patch.object(MlaMetadataRows, 'from_runtime', side_effect=lambda runner:
            MlaMetadataRows(runner, [], {}, simulation=True)))
        self.session = c1.QuantGroupC1ServingSession(self.worker, self.model, owner='actual-provider-owner', generation=3,
            worker_identity=self.identity, root='EXPLICIT_CPU_ONLY', build_sha256='b'*64,
            folder=self.folder/'runtime', prompts=[[1, 2] for _ in range(8)])
        self.provider = QuantProvider(self.worker, self.model, self.identity)
        factory = c1.QuantGroupC1ServingBinding(lambda worker, model, *, owner, generation, worker_identity:
            self.session if (worker is self.worker and model is self.model and owner == self.session.owner
                             and generation == self.session.generation and worker_identity == self.identity) else None)
        self.enterContext(patch('fault_injection.providers.resolve_runtime', return_value=factory))
        self.addCleanup(self.cleanup)
        for action in ('prepare', 'register', 'check', 'execute'):
            self.dispatch(action)

    def cleanup(self):
        # An explicit fixture teardown is not production recovery authority.
        # Production retains a failed fault owner until healthy OFF is proven.
        if self.session.installed:
            self.binding.drain_result = True
            self.session.phase_attempts.discard('fault')
            self.session.error = None
            self.session.close(abort=True)

    def dispatch(self, action, settings=None):
        return self.provider.dispatch(action, settings or {}, owner=self.session.owner, generation=self.session.generation)

    def phase(self, *, collect=True):
        index = len(self.session.rows)
        ids = ['cpu-' + str(index) + '-' + str(value) for value in range(8)]
        self.dispatch('advance', {'request_ids': ids})
        self.runner.with_prefill = True
        hidden = self.model(input_ids=torch.tensor([1, 2]*8), positions=torch.arange(16))
        self.model.processor(self.model.head, hidden[-8:])
        self.runner.input_batch = NS(req_ids=ids)
        self.runner.requests = {value: NS(prompt_token_ids=[1, 2], output_token_ids=[3]) for value in ids}
        self.dispatch('advance', {'capture_decode': True})
        self.runner.with_prefill = False
        hidden = self.model(input_ids=torch.tensor([3]*8), positions=torch.tensor([2]*8))
        self.model.processor(self.model.head, hidden)
        if collect:
            self.session.collect()

    def qualified(self):
        for _ in range(5):
            self.phase()

    def test_existing_provider_runs_single_site_and_keeps_full_unknown_denominator(self):
        for _ in range(7):
            self.phase()
        self.assertFalse(self.session.report()['c1'])  # Hooks still belong to this Worker owner.
        result = self.dispatch('finish')['serving']
        self.assertTrue(result['c1'] and result['functional_effect_proven'])
        self.assertFalse(result['c2'] or result['c3'] or result['all_eligible_qualification'])
        self.assertEqual(result['e5'], 'NOT_ASSESSED')
        self.assertEqual(self.binding.selected_modes, [1, 2, 4, 3])
        self.assertEqual(result['native_faults'], 1)
        self.assertEqual(len(self.binding.seals[0][0]), 1)
        self.assertEqual(self.binding.calls[-4:], ['stop', 'drain', 'recover', 'remove'])
        self.assertEqual(len(self.synchronizes), 1)
        discovery = self.session.discovery['report']
        self.assertEqual(discovery['structural_candidate_count'], 3)
        self.assertEqual(discovery['actual_native_calls'], 4)
        self.assertTrue(any(row['status'] == 'NOT_OBSERVED' for row in discovery['structural_candidates']))
        self.assertTrue(all(forward['candidates'][1]['status'] == 'UNVERIFIED' for forward in discovery['native_forwards']))
        packet = next(row for row in result['receipts'] if row['phase'] == 'fault')
        self.assertEqual((len(packet['raw_events']), packet['binding']['original_model_calls'],
                          packet['binding']['original_execute_calls'], packet['binding']['helper_calls']), (9, 1, 2, 1))
        self.assertFalse(result['owned_hooks'] or result['owned_nested_dispatch_modes'] or result['observer_cleanup_pending'])
        self.assertIsNone(sys.getprofile())
        self.dispatch('unregister')

    def test_unknown_all_rows_refuse_before_native_install_without_deleting_candidates(self):
        self.model.unknown_first = True
        self.phase()
        with self.assertRaisesRegex(RuntimeError, 'NO_SINGLE_NATURAL_SITE'):
            self.phase()
        self.assertEqual(self.binding.calls, [])
        self.assertEqual(self.session.discovery['report']['actual_native_calls'], 4)
        self.assertTrue(all(row['status'] == 'UNVERIFIED' for forward in
            self.session.discovery['report']['native_forwards'] for row in forward['candidates']))

    def test_frozen_site_new_unknown_row_cannot_become_passthrough_or_a_new_target(self):
        self.phase()
        self.phase()
        self.model.unknown_first = True
        with self.assertRaisesRegex(RuntimeError, 'CURRENT_ACTUAL_REQUEST_ROW_PROOF_REQUIRED'):
            self.phase()
        self.assertFalse(self.binding.selected_modes)
        self.session.close(abort=True)
        self.assertEqual(self.binding.calls, ['install', 'cancel_empty', 'remove'])

    def test_drain_failure_keeps_owner_and_hooks_retry_does_not_repeat_fault(self):
        self.qualified()
        self.binding.drain_result = False
        self.phase(collect=False)
        with self.assertRaisesRegex(RuntimeError, 'DRAIN_PENDING'):
            self.session.collect()
        self.assertIsNotNone(self.binding.config)
        self.assertNotIn('recover', self.binding.calls)
        self.assertNotIn('remove', self.binding.calls)
        self.assertTrue(self.session.installed and self.session.handles)
        self.binding.drain_result = True
        self.session.collect()
        self.assertEqual(self.binding.selected_modes, [1, 2, 4, 3])
        self.phase()
        self.dispatch('finish')

    def test_successful_fault_cannot_release_worker_owner_without_same_worker_off(self):
        self.qualified()
        self.phase()
        with self.assertRaisesRegex(RuntimeError, 'SAME_WORKER_HEALTHY_OFF_REQUIRED'):
            self.dispatch('disable')
        self.assertTrue(self.session.installed and self.session.handles)
        self.assertFalse(self.session.report()['c1'])
        self.phase()
        self.dispatch('finish')

    def test_native_receipt_tampering_cannot_hide_helper_or_swap_owner_role(self):
        self.qualified()
        self.phase()
        packet = self.session.receipts[-1]
        lease = self.session.current_lease
        for mutation in ('drop_helper', 'wrong_owner', 'wrong_role', 'old_registration', 'not_drained', 'no_effect', 'input_changed'):
            with self.subTest(mutation=mutation):
                done, rows = copy.deepcopy(packet['completion']), copy.deepcopy(packet['raw_events'])
                if mutation == 'drop_helper':
                    rows.pop(1)
                elif mutation == 'wrong_owner':
                    rows[1]['binding']['owner_hi'] += 1
                elif mutation == 'wrong_role':
                    rows[1]['binding']['role'] = 2
                elif mutation == 'old_registration':
                    rows[1]['native']['registration'] -= 9
                elif mutation == 'not_drained':
                    done['core']['reference']['drained'] = 0
                elif mutation == 'no_effect':
                    done['core']['changed_quant_bytes'] = done['core']['changed_scale_bytes'] = 0
                else:
                    done['core']['input_unchanged'] = 0
                with self.assertRaises(RuntimeError):
                    verify_completion(done, rows, lease, released=True)
        self.phase()
        self.dispatch('finish')

    def host_client(self):
        """Load the real Core API; a sibling checkout is a local test dependency."""
        package = 'demoncat.runtime.vllm_ascend.worker_extension'
        try:
            host = importlib.import_module(package + '.host')
        except ModuleNotFoundError:
            source = os.environ.get('DCAT_QUANT_C1_CORE_SOURCE')
            if not source:
                self.skipTest('Integrated Worker Core host API required (or explicit CPU source overlay)')
            loaded = {}
            for name in ('config', 'host'):
                path = Path(source) / 'demoncat/runtime/vllm_ascend/worker_extension' / (name + '.py')
                spec = importlib.util.spec_from_file_location(package + '.' + name, path)
                module = importlib.util.module_from_spec(spec)
                loaded[spec.name] = module
                self.enterContext(patch.dict(sys.modules, {spec.name: module}))
                spec.loader.exec_module(module)
            host = loaded[package + '.host']
        from demoncat.runtime.vllm_ascend.worker_extension.runtime import WorkerFaultRuntime
        from demoncat.runtime.vllm_ascend.worker_extension.identity import WorkerContext
        self.session.close(abort=True)
        self.core = WorkerFaultRuntime(os.getpid())
        self.core._identity = NS(read=lambda worker: WorkerContext(self.identity, self.runner.model, True))
        self.worker.demoncat_worker_identity = lambda: self.core.invoke(self.worker, 'worker_identity')
        def make(worker, model, *, owner, generation, worker_identity):
            self.session = c1.QuantGroupC1ServingSession(worker, model, owner=owner, generation=generation,
                worker_identity=worker_identity, root='EXPLICIT_CPU_ONLY', build_sha256='b'*64,
                folder=self.folder/'host-runtime', prompts=[[1, 2] for _ in range(8)])
            return self.session
        self.enterContext(patch('fault_injection.providers.resolve_runtime', return_value=c1.QuantGroupC1ServingBinding(make)))
        fixture = self
        class Executor:
            drop_action = None
            def collective_rpc(self, method, *, timeout, args=()):
                actions = {'demoncat_worker_identity': 'worker_identity', 'demoncat_capabilities': 'capabilities',
                    'demoncat_attach': 'attach', 'demoncat_detach': 'detach', 'demoncat_fault_query': 'query'}
                action, payload = (args[0], args[1]) if method == 'demoncat_fault_rpc' else (actions[method], args[0] if args else None)
                response = fixture.core.invoke(fixture.worker, action, payload)
                if self.drop_action == action:
                    self.drop_action = None
                    raise ConnectionError('EXPLICIT_CPU_LOST_RPC_AFTER_PROVIDER_ACTION')
                return [response]
        self.executor = Executor()
        self.client = host.WorkerFaultClient(host.HostAdapter(self.executor), host.FaultConfig('quant', 'batch_shared_scale'))
        def dispatch(action, settings=None):
            if action == 'advance':
                result = (self.client.advance_batch(settings['request_ids']) if 'request_ids' in settings
                          else self.client.collect_replay() if 'collect' in settings
                          else self.client.begin_decode_capture())
            else:
                result = getattr(self.client, action)()
            self.assertEqual(result['status'], 'QUERY_OK' if action == 'query' else 'ACTION_COMPLETED', result['response'])
            return result['response'].get('provider_state', {})
        self.dispatch = dispatch
        for action in ('prepare', 'register', 'check', 'execute'):
            self.dispatch(action)
        return self.client

    def test_actual_host_query_does_not_collect_completed_observations(self):
        client = self.host_client()
        self.phase(collect=False)
        before = self.session.report()
        expected = self.session.expected
        native_calls = list(self.binding.calls)
        with patch.object(self.session, 'collect', side_effect=AssertionError('QUERY_MUST_NOT_COLLECT')):
            for _ in range(2):
                state = self.dispatch('query')
                self.assertEqual(state['serving'], before)
        self.assertEqual(self.session.rows, [])
        self.assertIs(self.session.expected, expected)
        self.assertEqual(self.binding.calls, native_calls)
        self.assertTrue(client.snapshot()['ownership_retained'])
        self.dispatch('advance', {'collect': True})
        self.assertEqual(len(self.session.rows), 1)
        self.assertIsNone(self.session.expected)

    def test_actual_host_explicit_collect_reports_selection_failure_and_retains_owner(self):
        client = self.host_client()
        self.phase()
        self.phase(collect=False)
        with patch.object(self.session, '_select_target', side_effect=RuntimeError('ACTUAL_NATIVE_FORMALS_REFUSED')):
            self.assertEqual(client.query()['status'], 'QUERY_OK')
            self.assertEqual(len(self.session.rows), 1)
            result = client.collect_replay()
        self.assertEqual(result['response']['status'], 'PROVIDER_ACTION_FAILED')
        self.assertIn('ACTUAL_NATIVE_FORMALS_REFUSED', str(self.session.collection_errors))
        self.assertNotIn('NATIVE_STATE_UNAVAILABLE', str(result['response']))
        self.assertTrue(result['ownership_retained'])
        self.assertTrue(self.session.installed and self.session.handles)
        self.assertFalse(self.binding.selected_modes)
        self.assertEqual(len(self.session.collection_errors), 1)
        self.dispatch('disable')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_collection_advance_requires_exclusive_explicit_true(self):
        for settings in ({'collect': False}, {'collect': 1}, {'collect': True, 'request_ids': ['x']},
                         {'collect': True, 'capture_decode': True}):
            with self.subTest(settings=settings), self.assertRaisesRegex(RuntimeError, 'EXPLICIT_REPLAY_COLLECTION'):
                self.dispatch('advance', settings)

    def test_actual_runtime_and_public_host_complete_all_phases_without_recursive_identity_rpc(self):
        client = self.host_client()
        for _ in range(7):
            self.phase(collect=False)
            self.dispatch('advance', {'collect': True})
            state = self.dispatch('query')
            if self.session.phase_name() in ('lossless', 'qualification_off', 'fault'):
                self.assertTrue(state['native_registered'])
                self.assertGreater(state['native']['raw']['core']['host_bytes'], 0)
                self.assertEqual(state['leases'], 0)
                self.assertFalse(state['enabled'])
        state = self.dispatch('finish')
        self.assertTrue(state['serving']['c1'])
        self.assertFalse(state['native_registered'])
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(client.snapshot()['ownership_retained'])
        self.assertFalse(self.core._leases)

    def test_actual_dispatcher_native_frame_keeps_fresh_proof_through_runtime_and_host(self):
        self.binding.dispatch_quant = True
        self.enterContext(patch.object(c1, 'dynamic_quant_operator', return_value=torch.ops.cpu_quant_group_c1.quant.default))
        client = self.host_client()
        for _ in range(7):
            self.phase(collect=False)
            self.dispatch('advance', {'collect': True})
        result = self.dispatch('finish')['serving']
        self.assertTrue(result['c1'])
        for packet in result['receipts']:
            fresh, public = packet['admission_row_proof'], packet['row_proof']
            self.assertEqual(fresh['native_input_dispatch']['operator'], 'cpu_quant_group_c1.quant.default')
            self.assertGreater(fresh['native_input_dispatch']['sequence'], public['observation_prefix']['actual_calls'])
            self.assertGreaterEqual(fresh['observation_prefix']['actual_calls'], public['observation_prefix']['actual_calls'])
            expected = native.key(c1.digest(dict(proof=fresh, public_proof=public,
                actual_call=packet['completion']['call'], target=result['selected_site']['sha256'],
                owner=result['owner'], generation=result['generation'])))
            self.assertEqual(native.snapshot(expected), packet['lease']['actual_row_proof'])
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_reviewed_outer_opaque_source_and_current_native_frame_both_remain_visible(self):
        self.model.mla_attn = NS(kv_cache=[object()])
        def first_call(x):
            _CPU_BINDING.model = self.model
            try:
                output = torch.empty_like(x)
                torch.ops.cpu_quant_group_c1.mla(x, False, output, 'cpu-layer')
                return output
            finally:
                del _CPU_BINDING.model
        self.model.first_call = first_call
        pin = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        bindings, sources = source_bindings({__name__: (pin, {'cpu_mla_quant': 'mla_opaque'})})
        self.enterContext(patch.object(SourceBoundRows, 'from_runtime', side_effect=lambda graph, model:
            SourceBoundRows(graph, model, bindings, sources, simulation=True,
                operator_aliases={torch.ops.cpu_quant_group_c1.mla.default: 'vllm.mla_forward.default'})))
        self.binding.dispatch_quant = True
        self.enterContext(patch.object(c1, 'dynamic_quant_operator', return_value=torch.ops.cpu_quant_group_c1.quant.default))
        client = self.host_client()
        for _ in range(7):
            self.phase(collect=False)
            self.dispatch('advance', {'collect': True})
        result = self.dispatch('finish')['serving']
        self.assertTrue(result['c1'])
        self.assertTrue(all(row['admission_row_proof']['observation_prefix']['parent_source_sequences']
                            for row in result['receipts']))
        self.assertTrue(all(row['admission_row_proof']['native_input_dispatch'] is not None
                            for row in result['receipts']))
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def assert_native_admission_refused(self, mutation, *, dispatcher=False, unknown=False, smooth=None, restore=None):
        """Invalidate after public query; exercise the actual Core recovery route."""
        client = self.host_client()
        for _ in range(5):
            self.phase(collect=False)
            self.dispatch('advance', {'collect': True})
        if dispatcher:
            self.binding.dispatch_quant = True
            self.binding.unknown_dispatch = unknown
            self.binding.dispatch_smooth = smooth
            self.enterContext(patch.object(c1, 'dynamic_quant_operator', return_value=torch.ops.cpu_quant_group_c1.quant.default))
        observed = self.binding.observed
        def changed(call):
            selected = self.session.current_call is not None
            if selected:
                mutation()
            try:
                return observed(call)
            finally:
                if selected and restore is not None:
                    restore()  # Only undo the explicit fixture perturbation, never production cleanup.
        self.binding.observed = changed
        with self.assertRaisesRegex(RuntimeError, 'NATIVE_ROW_|C1_ROW_PROOF_CHANGED'):
            self.phase(collect=False)
        self.assertEqual(self.binding.selected_modes, [1, 2, 4])
        self.assertNotIn('fault', self.session.phase_attempts)
        self.assertEqual(self.binding.query()['core']['admitted_model_calls'], 3)
        self.assertFalse(any(row['phase'] == 'fault' for row in self.session.receipts))
        raw = self.session.callbacks[-1]
        self.assertTrue(raw['selected'])
        self.assertFalse(raw['completed'])
        self.assertNotIn('lease', raw)
        self.assertEqual(self.binding.raw(raw['call']['token'])['parent'], raw['call'])
        result = client.query()
        self.assertTrue(result['response']['recovery_required'] and result['ownership_retained'])
        self.assertTrue(self.session.installed and self.session.handles)
        self.assertIsNotNone(self.binding.config)
        self.assertFalse(self.session.report()['c1'])
        self.assertEqual(client.begin_decode_capture()['status'], 'HOST_REJECTED')
        # A failed device drain still retains the Native owner and every hook.
        self.binding.drain_result = False
        failed = client.disable()
        self.assertTrue(failed['ownership_retained'])
        self.assertIsNotNone(self.binding.config)
        self.assertTrue(self.session.installed and self.session.handles)
        self.binding.drain_result = True
        self.dispatch('disable')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertIsNone(self.binding.config)
        self.assertFalse(self.session.installed or self.session.row_observer.active)
        self.assertIsNone(sys.getprofile())

    def test_current_graph_error_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: self.session.row_observer.errors.append('EXPLICIT_CPU_AFTER_PUBLIC_QUERY'))

    def test_current_source_error_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: self.session.source_rows.errors.append('EXPLICIT_CPU_AFTER_PUBLIC_QUERY'))

    def test_current_overflow_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: setattr(self.session.row_observer, 'overflow', 1))

    def test_current_ending_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: setattr(self.session.row_observer, 'ending', True))

    def test_current_inactive_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: setattr(self.session.row_observer, 'active', False),
            restore=lambda: setattr(self.session.row_observer, 'active', True))

    def test_current_request_changed_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: self.session.active['decode']['request_ids'].reverse())

    def test_current_forward_changed_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: setattr(self.session.row_observer, 'forward_id', 'foreign-forward'))

    def test_current_observer_changed_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: self.session.current_call['row_proof']['observation_prefix'].update(observer_id=0))

    def test_current_source_owner_changed_between_public_query_and_native_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: sys.setprofile(None),
            restore=lambda: sys.setprofile(self.session.source_rows.callback))

    def test_unknown_actual_opaque_dispatch_refuses_before_fault(self):
        # Installed after qualification; the raw Native parent still records this refused call.
        self.assert_native_admission_refused(lambda: None, dispatcher=True, unknown=True)

    def test_nondefault_actual_native_dispatch_input_refuses_before_fault(self):
        self.assert_native_admission_refused(lambda: None, dispatcher=True, smooth=torch.ones(8))

    def test_actual_runtime_rejects_busy_native_call_and_retains_owner_until_cleanup(self):
        client = self.host_client()
        for _ in range(2):
            self.phase(collect=False)
            self.dispatch('advance', {'collect': True})
        self.dispatch('advance', {'request_ids': ['busy-' + str(i) for i in range(8)]})
        self.runner.with_prefill = True
        hidden = self.model(input_ids=torch.tensor([1, 2]*8), positions=torch.arange(16))
        self.model.processor(self.model.head, hidden[-8:])
        self.runner.input_batch = NS(req_ids=list(self.session.expected))
        self.runner.requests = {value: NS(prompt_token_ids=[1, 2], output_token_ids=[3]) for value in self.session.expected}
        self.dispatch('advance', {'capture_decode': True})
        self.binding.journal.state_value['core']['pending_parent'] = 1
        result = client.query()
        self.assertTrue(result['response']['recovery_required'])
        self.assertGreater(result['response']['provider_state']['leases'], 0)
        self.assertTrue(result['ownership_retained'])
        self.assertEqual(client.begin_decode_capture()['status'], 'HOST_REJECTED')
        self.binding.journal.state_value['core']['pending_parent'] = 0
        self.dispatch('disable')
        self.assertIn('cancel_empty', self.binding.calls)
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')

    def test_lost_rpc_after_actual_advance_retains_host_owner_and_uses_existing_cleanup_actions(self):
        client = self.host_client()
        self.executor.drop_action = 'advance'
        result = client.advance_batch(['lost-' + str(i) for i in range(8)])
        self.assertEqual(result['status'], 'RPC_UNCERTAIN')
        self.assertTrue(result['ownership_retained'] and self.session.installed)
        self.assertIsNotNone(self.session.expected)
        self.assertEqual(client.begin_decode_capture()['status'], 'HOST_REJECTED')
        self.dispatch('disable')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(self.session.installed or self.session.handles)

    def test_actual_runtime_retries_partial_method_cleanup_after_foreign_owner_is_returned(self):
        client = self.host_client()
        x, positions = torch.tensor([3]*8), torch.tensor([2]*8)
        self.phase(collect=False)
        self.dispatch('advance', {'collect': True})
        baseline = self.session.rows[-1]['decode']['hidden_sha256']
        first, last = self.session.inventory.patches
        foreign = lambda *args: None
        first[0].apply = foreign
        try:
            failed = client.disable()
            self.assertTrue(failed['ownership_retained'])
            self.assertIn('QUANT_METHOD_OBSERVER_OWNER_CHANGED', str(failed['response']))
            self.assertTrue(self.session.installed and self.session.handles)
            self.assertEqual(self.session.inventory.patches, [first])
            self.assertIs(first[0].apply, foreign)
            self.assertNotIn('apply', vars(last[0]))
        finally:
            first[0].apply = first[2]
        self.dispatch('disable')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(self.session.installed or self.session.handles or self.session.inventory.patches)
        self.assertEqual(serving.complete_tensor(self.model(input_ids=x, positions=positions))[1], baseline)

    def test_actual_runtime_retries_partial_api_cleanup_without_releasing_worker_early(self):
        client = self.host_client()
        ids = ['cleanup-' + str(i) for i in range(8)]
        self.dispatch('advance', {'request_ids': ids})
        self.runner.with_prefill = True
        hidden = self.model(input_ids=torch.tensor([1, 2]*8), positions=torch.arange(16))
        self.model.processor(self.model.head, hidden[-8:])
        self.runner.input_batch = NS(req_ids=ids)
        self.runner.requests = {value: NS(prompt_token_ids=[1, 2], output_token_ids=[3]) for value in ids}
        self.dispatch('advance', {'capture_decode': True})
        self.runner.with_prefill = False
        x, positions = torch.tensor([3]*8), torch.tensor([2]*8)
        baseline = self.model(input_ids=x, positions=positions)
        first, last = self.session.inventory.api_patches
        foreign = lambda *args: None
        setattr(first[0], first[1], foreign)
        try:
            failed = client.disable()
            self.assertTrue(failed['ownership_retained'])
            self.assertIn('PUBLIC_QUANT_OBSERVER_OWNER_CHANGED', str(failed['response']))
            self.assertTrue(self.session.installed and self.session.handles)
            self.assertEqual(self.session.inventory.api_patches, [first])
            self.assertIs(getattr(first[0], first[1]), foreign)
            self.assertIs(getattr(last[0], last[1]), last[2])
        finally:
            setattr(first[0], first[1], first[3])
        self.dispatch('disable')
        self.dispatch('unregister')
        self.assertEqual(client.detach()['status'], 'DETACHED')
        self.assertFalse(self.session.installed or self.session.handles or self.session.inventory.api_patches)
        self.assertFalse(self.session.row_observer.active)
        self.assertIsNone(sys.getprofile())
        self.assertTrue(torch.equal(self.model(input_ids=x, positions=positions), baseline))

    def test_wrong_current_device_refuses_before_native_owner_install(self):
        self.phase()
        self.phase()
        self.worker.device.index = 1
        try:
            with self.assertRaisesRegex(RuntimeError, 'SAME_ACTUAL_TP1_WORKER_REQUIRED'):
                self.phase()
            self.assertEqual(self.binding.calls, [])
        finally:
            self.worker.device.index = 0

    def test_missing_selected_completion_keeps_worker_owner_after_safe_native_release(self):
        self.qualified()
        self.binding.omit_selected_completion = True
        with self.assertRaisesRegex(RuntimeError, 'ONE_REAL_SELECTED_COMPLETION_REQUIRED'):
            self.phase()
        self.assertTrue(self.session.installed and self.session.handles)
        self.assertFalse(self.session.report()['c1'])
        with self.assertRaisesRegex(RuntimeError, 'SAME_WORKER_HEALTHY_OFF_REQUIRED'):
            self.session.close(abort=True)
        self.assertIsNone(self.binding.config)
        self.assertTrue(self.session.installed and self.session.handles)
        self.assertFalse(self.session.recovered)

    def test_fault_nonfinite_output_is_retained_and_exact_healthy_off_still_required(self):
        self.qualified()
        self.model.fault_nonfinite = True
        self.phase()
        row = self.session.rows[-1]['decode']
        self.assertFalse(row['hidden_finite'] or row['logits_finite'])
        self.assertTrue(self.session.report()['functional_effect_proven'])
        self.assertFalse(self.session.report()['c1'])
        self.phase()
        result = self.dispatch('finish')['serving']
        self.assertTrue(result['c1'])
        self.assertEqual(result['e5'], 'NOT_ASSESSED')

    def test_report_cannot_rewrite_frozen_target_rows_or_native_qualification_receipts(self):
        self.qualified()
        report = self.session.report()
        report['selected_site']['identity']['rows']['input_ids'][0] = 999
        report['receipts'][0]['completion']['core']['owner_hi'] = 0
        report['receipts'][1]['raw_events'][1]['binding']['role'] = 99
        report['selected_discovery_prefixes'][0]['consumer']['actual_rows']['request_ids'].reverse()
        self.assertEqual(self.session.target['identity']['rows']['input_ids'][0], 3)
        self.assertNotEqual(self.session.receipts[0]['completion']['core']['owner_hi'], 0)
        self.phase()
        self.phase()
        self.assertTrue(self.dispatch('finish')['serving']['c1'])

    def test_later_independent_unadapted_path_keeps_closed_c1_prefix_and_full_refusals(self):
        def later():
            if self.session.row_observer.active:
                self.session.row_observer.errors.append('EXPLICIT_CPU_UNADAPTED_INDEPENDENT_LATER_PATH')
        self.model.after_first = later
        for _ in range(7):
            self.phase()
        result = self.dispatch('finish')['serving']
        self.assertTrue(result['c1'])
        self.assertFalse(result['row_operator_observation_complete'])
        first = self.session.discovery['report']['native_forwards'][0]
        self.assertEqual(first['candidates'][0]['status'], 'UNVERIFIED')
        self.assertIn('COMPLETE_REQUEST_ROW_OPERATOR_OBSERVATION_UNPROVEN', first['candidates'][0]['reasons'])
        self.assertFalse(first['row_provenance']['complete'])
        self.assertTrue(result['selected_discovery_prefixes'][0]['complete_at_consumer'])
        self.assertFalse(result['c2'] or result['c3'])

    def test_error_before_target_cannot_be_relabelled_as_later_unknown(self):
        def before():
            if self.session.row_observer.active:
                self.session.row_observer.errors.append('EXPLICIT_CPU_ERROR_BEFORE_TARGET')
        self.model.before_first = before
        self.phase()
        with self.assertRaisesRegex(RuntimeError, 'NO_SINGLE_NATURAL_SITE'):
            self.phase()
        self.assertEqual(self.binding.calls, [])

    def test_error_before_real_qmm_consumer_prevents_prefix_qualification(self):
        original = self.api.npu_quant_matmul
        def consumer(*args, **kwargs):
            if self.session.row_observer.active and self.session.inventory.frame['module'] is self.model.first:
                self.session.row_observer.errors.append('EXPLICIT_CPU_ERROR_BEFORE_CONSUMER')
            return original(*args, **kwargs)
        self.api.npu_quant_matmul = consumer
        self.phase()
        with self.assertRaisesRegex(RuntimeError, 'NO_SINGLE_NATURAL_SITE'):
            self.phase()
        public = self.session.discovery['report']['public_calls'][0]
        self.assertTrue(public['request_row_lineage']['verified'])
        self.assertEqual(self.binding.calls, [])

    def test_closed_prefix_snapshot_wrong_request_or_call_order_is_refused(self):
        self.phase()
        self.phase()
        report = self.session.discovery['report']
        for change in ('request', 'sequence', 'earlier_error', 'parent_not_closed'):
            with self.subTest(change=change):
                public = copy.deepcopy(report['public_calls'][0])
                projection = copy.deepcopy(report['forwards'][0]['projections'][0])
                forward = copy.deepcopy(report['native_forwards'][0])
                consumer = projection['output_row_lineage']
                if change == 'request':
                    consumer['actual_rows']['request_ids'][0] = 'foreign-request'
                elif change == 'sequence':
                    consumer['observation_prefix']['actual_calls'] = forward['row_provenance']['actual_calls'] + 1
                elif change == 'earlier_error':
                    public['request_row_lineage']['observation_prefix']['errors'] = 1
                else:
                    consumer['observation_prefix']['parent_source_sequences'] = [999]
                with self.assertRaises(RuntimeError):
                    c1.closed_projection_prefix(public, projection, forward)


if __name__ == '__main__':
    unittest.main()
