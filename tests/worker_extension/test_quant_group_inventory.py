"""CPU fixture boundary tests; no CANN, actual scheduler or numerical oracle."""
import copy
from pathlib import Path
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection import quant_group_native as native
from fault_injection.deployments.quant_group_inventory import GroupQuantJournal, verify_public_group_call
from fault_injection.quantization import _discover
from demoncat.runtime.vllm_ascend.worker_extension.shared_scale_group_inventory import SharedScaleGroupInventory
from test_quant_inventory import TwoQuantProjections


def cpu_view(tensor, role, device):
    value = native.View(address=tensor.data_ptr(), bytes=tensor.numel() * tensor.element_size(),
        rows=tensor.shape[0], columns=tensor.shape[1] if tensor.ndim == 2 else 0,
        stride0=tensor.stride(0), stride1=tensor.stride(1) if tensor.ndim == 2 else 0,
        dtype={torch.float32: 0, torch.bfloat16: 27, torch.int8: 2}[tensor.dtype], format=2, dimensions=tensor.ndim)
    return value


class CpuJournal(GroupQuantJournal):
    def __init__(self):
        self.device = 0
        self.build_identity = dict(evidence_kind='EXPLICIT_CPU_GROUP_JOURNAL_FIXTURE')
        self.state_value = native.snapshot(native.RuntimeState())
        self.state_value['census']['capacity'] = 32768
        self.rows = []
        self.binding = NS(query=lambda: copy.deepcopy(self.state_value),
                          raw_range=lambda first, last: copy.deepcopy(self.rows[first-1:last]))
        self.wrong_pointer = self.missing_execute = False

    def quant(self, x):
        q, scale = x.round().to(torch.int8), torch.ones(x.shape[0], dtype=torch.float32)
        self.emit(x, q, scale)
        return q, scale

    def emit(self, x, q, scale):
        parent = native.Call(token=len(self.rows)+1, thread=threading.get_native_id(), device=0)
        for index, (name, tensor, role) in enumerate((('input', x, 'input'), ('quant', q, 'quant'), ('scale', scale, 'scale'))):
            value = cpu_view(tensor, role, 0)
            value.descriptor = 100 + index
            if name == 'input':
                value.address += int(self.wrong_pointer)
            setattr(parent, name, value)
        raw = native.RawRecord(kind=1, parent=parent)
        self.rows.append(native.snapshot(raw))
        registration = len(self.rows) + 1
        for kind in ((1, 2, 3) if self.missing_execute else (1, 2, 3, 4)):
            event = native.NativeEvent(kind=kind, sequence=len(self.rows)+1, registration=registration,
                thread=parent.thread, device=0, executor=0 if kind == 1 else 700,
                stream=71 if kind in (3, 4) else 0, original_calls=int(kind in (2, 4)),
                destination_dtype=2, input=parent.input, quant=parent.quant, scale=parent.scale)
            self.rows.append(native.snapshot(native.RawRecord(kind=2, native=event)))
        census = self.state_value['census']
        census['model_parents'] += 1
        census['native_prepares'] += 1
        census['unclassified_prepares'] += 1
        census['native_executes'] += 1
        census['total'] = self.state_value['last_sequence'] = len(self.rows)
        census['pending'] = int(self.missing_execute)


class GroupQuantInventoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.native = CpuJournal()
        self.enterContext(patch.object(native.QuantGroupNative, 'tensor_view', side_effect=cpu_view))
        self.api = ModuleType('torch_npu')
        self.api.get_npu_format = lambda value: 2
        self.api.npu_dynamic_quant = self.native.quant
        self.api.npu_quant_matmul = lambda x1, x2, scale, pertoken_scale, output_dtype: (x1.float() @ x2.float()).to(output_dtype)
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.api}))
        self.model = TwoQuantProjections().eval()
        self.input = torch.ones(8, 4, dtype=torch.bfloat16)
        candidates, _ = _discover(self.model, (1,))
        self.observer = SharedScaleGroupInventory(self.model, candidates, worker_identity={'rank': 0, 'worker_id': 'CPU'},
            native=self.native, folder=self.folder / 'inventory',
            row_lineage=lambda value: dict(verified=value is self.input, evidence_kind='EXPLICIT_CPU_ROW_FIXTURE'))
        self.observer.install()
        self.addCleanup(self.observer.close)

    def phase(self, request='first'):
        self.observer.begin(request, 'a'*64, request_ids=tuple(request+'-'+str(i) for i in range(8)))
        with torch.inference_mode():
            self.model(self.input)
        self.observer.row_trace = dict(complete=True, evidence_kind='CPU_FIXTURE')
        self.observer.end('b'*64, model_output_finite=True)

    def test_all_parents_and_native_events_are_retained_with_unknown_rows(self):
        self.phase('first')
        self.phase('second')
        report = self.observer.report()
        self.assertEqual(report['actual_native_calls'], 4)
        self.assertEqual(report['actual_native_prepare_calls'], 4)
        self.assertEqual(report['actual_native_execute_calls'], 4)
        self.assertEqual(report['request_row_proven_calls'], 2)
        self.assertTrue(report['native_public_binding_complete'])
        self.assertEqual(report['qualified_native_targets'], 0)
        self.assertFalse(report['c2'] or report['c3'])
        for forward in report['native_forwards']:
            self.assertEqual(len(forward['native_events']), 10)
            self.assertIn('ACTUAL_PER_REQUEST_PROJECTION_ROW_LINEAGE_UNPROVEN', forward['candidates'][1]['reasons'])

    def test_unowned_native_parent_and_children_remain_in_whole_model_denominator(self):
        self.observer.begin('unowned', 'a'*64, request_ids=('actual-cpu-fixture',))
        self.native.quant(self.input)
        self.model(self.input)
        self.observer.row_trace = dict(complete=True)
        self.observer.end('b'*64, model_output_finite=True)
        report = self.observer.report()
        self.assertEqual(report['actual_native_calls'], 3)
        self.assertFalse(report['native_public_binding_complete'])
        self.assertEqual(len(report['native_forwards'][0]['unowned_native_events']), 5)

    def test_wrong_formal_pointer_does_not_borrow_geometry(self):
        self.native.wrong_pointer = True
        self.phase()
        self.assertEqual(self.observer.report()['request_row_proven_calls'], 0)
        self.assertTrue(all(any('FORMALS_DIFFER' in reason for reason in row['reasons'])
                            for row in self.observer.native_forwards[0]['candidates']))

    def test_missing_execute_keeps_candidates_and_raw_rows_before_failure(self):
        self.native.missing_execute = True
        with self.assertRaisesRegex(RuntimeError, 'EXECUTION_STILL_ACTIVE'):
            self.phase()
        report = self.observer.report()
        self.assertEqual(report['actual_native_calls'], 2)
        self.assertEqual(len(report['native_forwards'][0]['native_events']), 8)
        self.observer.fail('fixture incomplete actual execution')

    def test_reused_executor_has_fresh_parent_and_registration(self):
        q, scale = self.native.quant(self.input)
        first = verify_public_group_call(self.native.rows, self.input, q, scale, threading.get_native_id(), expected_device=0)
        q, scale = self.native.quant(self.input)
        second = verify_public_group_call(self.native.rows[-5:], self.input, q, scale, threading.get_native_id(), expected_device=0)
        self.assertEqual(first['executor'], second['executor'])
        self.assertNotEqual(first['registration'], second['registration'])
        self.assertNotEqual(first['call'], second['call'])

    def test_classified_helper_cannot_be_disguised_as_readonly_model_call(self):
        q, scale = self.native.quant(self.input)
        rows = copy.deepcopy(self.native.rows)
        rows[1]['binding']['role'] = 1
        with self.assertRaisesRegex(RuntimeError, 'HIDE_CLASSIFIED_CHILDREN'):
            verify_public_group_call(rows, self.input, q, scale, threading.get_native_id(), expected_device=0)

    def test_census_counter_or_raw_sequence_gap_refuses_completeness(self):
        before = self.native.state()
        self.native.quant(self.input)
        self.native.state_value['census']['native_executes'] += 1
        with self.assertRaisesRegex(RuntimeError, 'DO_NOT_RECONCILE'):
            self.native.delta(before)
        self.native.state_value['census']['native_executes'] -= 1
        self.native.rows[2]['native']['sequence'] += 99
        with self.assertRaisesRegex(RuntimeError, 'SEQUENCE_GAP'):
            self.native.delta(before)

    def test_complete_serving_path_uses_group_loader_and_preserves_unknown_row_edge(self):
        self.complete_serving_path(enable_sources=False)

    def test_actual_serving_hooks_handoff_early_profile_and_release_both_observers(self):
        self.complete_serving_path(enable_sources=True)

    def complete_serving_path(self, *, enable_sources):
        from demoncat.runtime.vllm_ascend.worker_extension import serving_quant as serving
        from demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_group_inventory import SharedScaleGroupInventorySession
        from test_serving_session_lifecycle import CpuProcessor
        if enable_sources:
            from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import SourceBoundRows
            from demoncat.runtime.vllm_ascend.worker_extension.quant_mla_metadata_rows import MlaMetadataRows
            self.enterContext(patch.object(SourceBoundRows, 'from_runtime', side_effect=lambda graph, model:
                SourceBoundRows(graph, model, [], {}, simulation=True)))
            self.enterContext(patch.object(MlaMetadataRows, 'from_runtime', side_effect=lambda runner:
                MlaMetadataRows(runner, [], {}, simulation=True)))
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
                self.embed, self.norm = Embedding(), Norm()
                self.head = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
                self.processor = CpuProcessor()
            def forward(self, input_ids, positions, inputs_embeds=None, intermediate_tensors=None):
                hidden = self.first(self.norm(self.embed(input_ids)))
                return self.second(torch.roll(hidden, 1, dims=0))
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
        self.api.npu = NS(current_device=lambda: 0)
        actual_binding = NS(identity=self.native.build_identity, query=lambda: copy.deepcopy(self.native.state_value))
        def scheduler(runner, actual_model, actual, context, expected):
            return dict(request_ids=list(expected), logical_ids=list(range(8)), row_token_positions=[2]*8,
                prefixes=[{'prompt': [1, 2], 'output': [3]} for _ in range(8)], input_ids=[3]*8)
        with patch.dict(sys.modules, modules), patch.object(serving, 'require_serial_eager'), \
                patch.object(serving, 'discover_logits_boundary', return_value=dict(processor=model.processor,
                    lm_head=model.head, report={'output_contract': [8, 3]})), \
                patch.object(serving, 'snapshot_first_decode', side_effect=scheduler), \
                patch.object(native.QuantGroupNative, 'load', return_value=actual_binding) as group_loader, \
                patch('fault_injection.deployments.quant_group_inventory.GroupQuantJournal', return_value=self.native), \
                patch('fault_injection.quantization.PinnedQuantRuntime.from_portable_v2_backend') as legacy_loader:
            session = SharedScaleGroupInventorySession(worker, model, root='CPU', build_sha256='c'*64,
                folder=self.folder/'serving', prompts=[[1, 2] for _ in range(8)])
            session.install()
            try:
                for step in range(2):
                    ids = [str(step)+'-'+str(index) for index in range(8)]
                    session.expect(ids)
                    runner.with_prefill = True
                    hidden = model(input_ids=torch.tensor([1, 2]*8), positions=torch.arange(16))
                    model.processor(model.head, hidden[-8:])
                    runner.input_batch = NS(req_ids=ids)
                    runner.requests = {key: NS(prompt_token_ids=[1, 2], output_token_ids=[3]) for key in ids}
                    session.begin_decode_capture()
                    runner.with_prefill = False
                    hidden = model(input_ids=torch.tensor([3]*8), positions=torch.tensor([2]*8))
                    model.processor(model.head, hidden)
                    session.collect()
                session.close()
                report = session.report()
                self.assertTrue(report['recovered'] and report['native_public_binding_complete'])
                self.assertEqual(report['actual_native_calls'], 4)
                self.assertEqual(report['request_row_proven_calls'], 2)
                self.assertFalse(report['c2'] or report['c3'] or report['owned_hooks'] or report['owned_row_dispatch_modes'])
                self.assertEqual(report['source_row_adapter']['enabled'], enable_sources)
                if enable_sources:
                    self.assertFalse(session.source_rows.active or session.metadata_rows.active)
                    self.assertTrue(report['metadata_producers']['complete'])
                    self.assertIsNone(sys.getprofile())
                group_loader.assert_called_once()
                legacy_loader.assert_not_called()
            finally:
                if session.installed:
                    session.close(abort=True)


if __name__ == '__main__':
    unittest.main()
