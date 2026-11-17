"""CPU-only real projection execution, explicit native V2 inventory mock."""
import ctypes as C
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.dynamic_quant_inventory import (KINDS, NativeDynamicQuantInventory,
    QuantEvent, QuantState, verify_public_native_call)
from fault_injection.quantization import _discover
from demoncat.runtime.vllm_ascend.worker_extension.shared_scale_inventory import SharedScaleInventory
from test_quant_inventory import TwoQuantProjections
from test_serving_session_lifecycle import CpuProcessor


class CpuNativeQuant(NativeDynamicQuantInventory):
    def __init__(self):
        self.build_identity = {'evidence_kind': 'EXPLICIT_CPU_NATIVE_MOCK_NOT_NPU'}
        self.device = 0
        self.raw = QuantState(version=1, size=C.sizeof(QuantState), capacity=32768)
        self.events = []
        self.state_fn, self.read_fn = self._state, self._read
        self.missing_execute = self.wrong_formals = self.native_only = False

    def _state(self, output):
        C.memmove(output, C.byref(self.raw), C.sizeof(self.raw))
        return 0

    def _read(self, index, output):
        C.memmove(output, C.byref(self.events[index]), C.sizeof(QuantEvent))
        return 0

    def quant(self, x):
        q, scale = x.round().to(torch.int8), torch.ones(x.shape[0], dtype=torch.float32)
        self.emit(x, q, scale)
        return q, scale

    def emit(self, x, q, scale):
        call = len(self.events) + 1
        for kind in ((1, 2, 3) if self.missing_execute else (1, 2, 3, 4)):
            event = QuantEvent(version=1, size=C.sizeof(QuantEvent), kind=kind, flags=31,
                device=0, sequence=len(self.events) + 1, call=call, thread=threading.get_native_id(),
                executor=0 if kind == 1 else 900, input=x.data_ptr() + int(self.wrong_formals),
                quant=q.data_ptr(), scale=scale.data_ptr(), rows=x.shape[0], columns=x.shape[1],
                input_dtype=27, quant_dtype=2, scale_dtype=0, input_format=2, quant_format=2, scale_format=2,
                original_calls=int(kind in (2, 4)))
            self.events.append(event)
            self.raw.total_events += 1
            self.raw.stored_events += 1
            name = KINDS[kind]
            setattr(self.raw, name, getattr(self.raw, name) + 1)
            if kind == 1:
                self.raw.metadata_complete += 1
            if kind == 2:
                self.raw.registrations += 1


class SharedScaleInventoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.native = CpuNativeQuant()
        self.api = ModuleType('torch_npu')
        self.api.get_npu_format = lambda x: 2
        self.api.npu = NS(current_device=lambda: 0)
        self.api.npu_dynamic_quant = self.native.quant
        self.qmm_calls = 0
        def qmm(x1, x2, scale, *, pertoken_scale, output_dtype):
            self.qmm_calls += 1
            return (x1.float() @ x2.float()).to(output_dtype)
        self.api.npu_quant_matmul = qmm
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.api}))
        self.model = TwoQuantProjections().eval()
        self.input = torch.ones(8, 4, dtype=torch.bfloat16)
        candidates, _ = _discover(self.model, (1,))
        self.observer = SharedScaleInventory(self.model, candidates,
            worker_identity={'rank': 0, 'worker_id': 'CPU-only'}, native=self.native, folder=self.folder / 'inventory',
            row_lineage=lambda x: dict(verified=x is self.input, evidence_kind='EXPLICIT_CPU_ROW_FIXTURE_NOT_NPU'))
        self.observer.install()
        self.addCleanup(self.observer.close)

    def phase(self, request):
        self.observer.begin(request, 'a' * 64, request_ids=tuple(request + '-' + str(i) for i in range(8)),
            row_mapping={'evidence_kind': 'EXPLICIT_CPU_SCHEDULER_NOT_NPU'})
        with torch.inference_mode():
            self.model(self.input)
        self.observer.end('b' * 64, model_output_finite=True)

    def test_all_native_calls_observed_without_promoting_equal_shape_to_request_rows(self):
        for request in ('first', 'second'):
            self.phase(request)
        report = self.observer.report()
        self.assertEqual((report['actual_projection_calls'], report['actual_native_calls'], self.qmm_calls), (4, 4, 4))
        self.assertEqual(report['request_row_proven_calls'], 2)
        self.assertEqual(report['qualified_native_targets'], 0)
        self.assertFalse(report['c2'] or report['c3'])
        for row in report['native_forwards']:
            first, second = row['candidates']
            self.assertEqual(first['reasons'], [])
            self.assertIn('ACTUAL_PER_REQUEST_PROJECTION_ROW_LINEAGE_UNPROVEN', second['reasons'])
        self.assertTrue(report['complete_healthy_replays'])

    def test_missing_execute_is_preserved_in_denominator(self):
        self.native.missing_execute = True
        self.phase('missing')
        report = self.observer.report()
        self.assertEqual(report['actual_native_calls'], 2)
        self.assertEqual(report['request_row_proven_calls'], 0)
        self.assertTrue(all('ONE_ACTUAL_NATIVE_QUANT_CALL_REQUIRED' in c['reasons']
                            for c in report['native_forwards'][0]['candidates']))

    def test_exact_formal_pointer_mismatch_is_not_shape_qualified(self):
        self.native.wrong_formals = True
        self.phase('wrong-formal')
        candidates = self.observer.native_forwards[0]['candidates']
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all('ACTUAL_PUBLIC_NATIVE_QUANT_FORMALS_DIFFER' in c['reasons'] for c in candidates))

    def test_native_call_without_any_public_observer_is_not_dropped(self):
        self.observer.begin('unowned', 'a' * 64, request_ids=('unowned',))
        self.native.quant(self.input)
        self.model(self.input)
        self.observer.end('b' * 64, model_output_finite=True)
        candidates = self.observer.native_forwards[0]['candidates']
        self.assertEqual(len(candidates), 3)
        self.assertIn('NATIVE_QUANT_HAS_NO_OWNED_PUBLIC_CHAIN', candidates[0]['reasons'])

    def test_public_call_outside_projection_still_keeps_native_record(self):
        self.observer.begin('public-unowned', 'a' * 64, request_ids=('public-unowned',))
        self.api.npu_dynamic_quant(self.input)
        self.model(self.input)
        self.observer.end('b' * 64, model_output_finite=True)
        candidates = self.observer.native_forwards[0]['candidates']
        self.assertEqual(len(candidates), 3)
        self.assertIn('PUBLIC_CALL_OUTSIDE_DISCOVERED_DYNAMIC_PROJECTION', candidates[0]['reasons'])
        self.assertNotIn('NATIVE_QUANT_HAS_NO_OWNED_PUBLIC_CHAIN', candidates[0]['reasons'])

    def test_orphan_overflow_and_details_gap_refuse_complete_inventory(self):
        before = self.native.state()
        self.native.quant(self.input)
        self.native.raw.execute_exit += 1
        with self.assertRaisesRegex(RuntimeError, 'DETAILS_DO_NOT_MATCH_COUNTERS'):
            self.native.delta(before)
        self.native.raw.execute_exit -= 1
        self.native.raw.overflow_events = 1
        with self.assertRaisesRegex(RuntimeError, 'OVERFLOW_OR_ERROR'):
            self.native.state()
        self.native.raw.overflow_events = 0
        self.native.raw.orphan_executions = 1
        with self.assertRaisesRegex(RuntimeError, 'WITHOUT_ACTUAL_PREPARE'):
            self.native.state()

    def test_executor_address_reuse_has_distinct_actual_registration_lifetimes(self):
        x = self.input
        q, scale = self.native.quant(x)
        first = [self.native.read(i) for i in range(4)]
        a = verify_public_native_call(first, x, q, scale, threading.get_native_id(), expected_device=0)
        q, scale = self.native.quant(x)
        second = [self.native.read(i) for i in range(4, 8)]
        b = verify_public_native_call(second, x, q, scale, threading.get_native_id(), expected_device=0)
        self.assertEqual(a['executor'], b['executor'])
        self.assertNotEqual(a['call'], b['call'])
        self.assertEqual(C.sizeof(QuantEvent), 128)
        self.assertEqual(C.sizeof(QuantState), 128)

    def run_complete_serving(self, *, unknown_boundary=False):
        from demoncat.runtime.vllm_ascend.worker_extension import serving_quant as serving
        from demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_inventory import SharedScaleInventorySession
        self.observer.close()
        class Embedding(torch.nn.Module):
            def forward(self, input_):
                return input_[:, None].expand(-1, 4).to(torch.bfloat16).contiguous()
        class Head(Embedding):
            pass
        class Norm(torch.nn.Module):
            def forward(self, x):
                return x.clone()
        class Model(TwoQuantProjections):
            def __init__(self):
                super().__init__()
                del self.unused
                self.embed = Embedding()
                self.norm = Norm()
                self.head = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
                self.processor = CpuProcessor()
            def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
                hidden = self.first(self.norm(self.embed(input_ids)))
                if unknown_boundary:
                    hidden = torch.roll(hidden, 1, dims=0)
                return self.second(hidden)
        model = Model().eval()
        runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))),
            logits_indices=torch.arange(8), with_prefill=True)
        worker = NS(model_runner=runner, rank=0,
            demoncat_worker_identity=lambda: dict(status='WORKER_IDENTIFIED', worker={'rank': 0, 'worker_id': 'CPU'}))
        modules = {name: ModuleType(name) for name in ('vllm', 'vllm.model_executor', 'vllm.model_executor.layers',
            'vllm.model_executor.layers.logits_processor', 'vllm.model_executor.layers.vocab_parallel_embedding',
            'vllm.model_executor.layers.layernorm', 'vllm.forward_context')}
        modules['vllm.model_executor.layers.logits_processor'].LogitsProcessor = CpuProcessor
        modules['vllm.model_executor.layers.vocab_parallel_embedding'].VocabParallelEmbedding = Embedding
        modules['vllm.model_executor.layers.vocab_parallel_embedding'].ParallelLMHead = Head
        modules['vllm.model_executor.layers.layernorm'].RMSNorm = Norm
        modules['vllm.forward_context'].get_forward_context = lambda: NS()
        native = self.native
        runtime = NS(build_identity=native.build_identity,
            backend=NS(lib=NS(dcat_dq_inventory_state_v1=lambda out: native._state(out),
                dcat_dq_inventory_read_v1=lambda index, out: native._read(index, out))),
            native_state=lambda: dict(armed=0, pending=0, executing=0, owned=0))
        def snapshot(runner, actual_model, actual, context, expected):
            return dict(request_ids=list(expected), logical_ids=list(range(8)), row_token_positions=[2]*8,
                prefixes=[{'prompt': [1, 2], 'output': [3]} for _ in range(8)], input_ids=[3]*8)
        with patch.dict(sys.modules, modules), patch.object(serving, 'require_serial_eager'), \
                patch.object(serving, 'discover_logits_boundary', return_value=dict(processor=model.processor,
                    lm_head=model.head, report={'output_contract': [8, 3]})), \
                patch.object(serving, 'snapshot_first_decode', side_effect=snapshot), \
                patch('fault_injection.quantization.PinnedQuantRuntime.from_portable_v2_backend', return_value=runtime):
            session = SharedScaleInventorySession(worker, model, root='CPU', build_sha256='c'*64,
                folder=self.folder / 'serving', prompts=[[1, 2] for _ in range(8)])
            session.install()
            try:
                for step in range(2):
                    ids = [str(step) + '-' + str(i) for i in range(8)]
                    session.expect(ids)
                    runner.with_prefill = True
                    hidden = model(input_ids=torch.tensor([1, 2]*8), positions=torch.arange(16))
                    model.processor(model.head, hidden[-8:])
                    self.assertFalse(session.inventory.active)
                    runner.input_batch = NS(req_ids=ids)
                    runner.requests = {key: NS(prompt_token_ids=[1, 2], output_token_ids=[3]) for key in ids}
                    session.begin_decode_capture()
                    runner.with_prefill = False
                    hidden = model(input_ids=torch.tensor([3]*8), positions=torch.tensor([2]*8))
                    model.processor(model.head, hidden)
                    session.collect()
                session.close()
                report = session.report()
                self.assertEqual(report['actual_native_calls'], 4)
                self.assertTrue(report['complete_healthy_replays'])
                self.assertTrue(report['recovered'])
                self.assertTrue(report['row_operator_observation_complete'])
                self.assertFalse(report['c2'] or report['c3'])
                self.assertEqual((report['owned_hooks'], report['owned_method_patches'], report['owned_api_patches']), (0, 0, 0))
                return report, session.inventory.report()
            finally:
                session.close(abort=True)

    def test_complete_actual_cpu_prefill_decode_hook_order_and_row_bridge_then_remove(self):
        report, inventory = self.run_complete_serving()
        self.assertEqual(report['request_row_proven_calls'], 4)
        self.assertEqual(inventory['qualified_native_targets'], 0)
        self.assertTrue(all(not candidate['reasons'] for row in inventory['native_forwards'] for candidate in row['candidates']))

    def test_unknown_row_permutation_retains_all_native_calls_without_inheriting_rows(self):
        report, inventory = self.run_complete_serving(unknown_boundary=True)
        self.assertEqual((report['actual_native_calls'], report['request_row_proven_calls']), (4, 2))
        for row in inventory['native_forwards']:
            self.assertEqual(len(row['candidates']), 2)
            self.assertIn('ACTUAL_PER_REQUEST_PROJECTION_ROW_LINEAGE_UNPROVEN', row['candidates'][1]['reasons'])
            self.assertTrue(any(event['operator'] == 'aten.roll.default' and not event['row_preserving']
                                for event in row['row_provenance']['events']))


if __name__ == '__main__':
    unittest.main()
