"""CPU public-lineage observer tests; no native qualification or mocked C2."""
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection.quantization import _discover
from demoncat.runtime.vllm_ascend.worker_extension.quant_inventory import PublicQuantInventory
try:
    from .test_serving20_qmm import CpuProjection, CpuQmmModel
    from . import test_serving_session_lifecycle as life
except ImportError:
    from test_serving20_qmm import CpuProjection, CpuQmmModel
    import test_serving_session_lifecycle as life


class TwoQuantProjections(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = CpuProjection()
        self.second = CpuProjection()
        self.unused = CpuProjection()
        self.second.quant_method = self.first.quant_method

    def forward(self, x):
        return self.second(self.first(x))


class QuantInventoryTests(unittest.TestCase):
    def setUp(self):
        self.api = ModuleType('torch_npu')
        self.api.get_npu_format = lambda value: 2  # Explicit CPU fixture only.
        self.q_calls = self.mm_calls = 0
        def quant(x):
            self.q_calls += 1
            return x.round().to(torch.int8), torch.ones(x.shape[0], dtype=torch.float32)
        def qmm(x1, x2, scale, *, pertoken_scale, output_dtype):
            self.mm_calls += 1
            return torch.mm(x1.float(), x2.float()).to(output_dtype)
        self.api.npu_dynamic_quant = self.original_quant = quant
        self.api.npu_quant_matmul = self.original_qmm = qmm
        self.enterContext(patch.dict(sys.modules, {'torch_npu': self.api}))
        self.model = TwoQuantProjections().eval()
        self.observers = []

    def tearDown(self):
        for observer in self.observers:
            observer.close()

    def observer(self):
        candidates, rejected = _discover(self.model, (1,))
        self.assertFalse(rejected)
        observer = PublicQuantInventory(self.model, candidates,
            worker_identity=dict(rank=0, model_instance_id='CPU', evidence_kind='CPU_CONTRACT_ONLY'))
        self.observers.append(observer)
        observer.install()
        return observer

    def phase(self, observer, number):
        observer.begin('actual-forward-' + str(number), 'a' * 64,
                       request_ids=('actual-request-' + str(number),),
                       row_mapping={'kind': 'CPU_FIXTURE_PREFILL_TOKEN_ROWS_NOT_REQUEST_BATCH'})
        result = self.model(torch.ones(2, 4, dtype=torch.bfloat16))
        observer.end('b' * 64, model_output_finite=bool(torch.isfinite(result).all()))
        return result

    def test_all_actual_projections_observed_unused_is_not_qualified(self):
        observer = self.observer()
        first, second = self.phase(observer, 0), self.phase(observer, 1)
        self.assertTrue(torch.equal(first, second))
        report = observer.report()
        self.assertEqual(report['structural_candidate_count'], 3)
        self.assertEqual(report['actual_projection_calls'], 4)
        self.assertEqual(report['actual_public_api_calls'], 8)
        self.assertEqual(report['qualified_native_targets'], 0)
        self.assertEqual(self.q_calls, 4)
        self.assertEqual(self.mm_calls, 4)
        self.assertTrue(report['repeated_public_inventory_stable'])
        states = {row['path']: row for row in report['structural_candidates']}
        self.assertEqual(states['first']['public_verified_calls'], 2)
        self.assertEqual(states['second']['public_verified_calls'], 2)
        self.assertEqual(states['unused']['status'], 'NOT_OBSERVED')
        self.assertFalse(report['c2'])
        observer.close()
        self.assertIs(self.api.npu_dynamic_quant, self.original_quant)
        self.assertIs(self.api.npu_quant_matmul, self.original_qmm)

    def test_clone_lineage_rejection_preserves_original_once_and_full_denominator(self):
        def cloned_apply(layer, x):
            import torch_npu
            q, scale = torch_npu.npu_dynamic_quant(x)
            return torch_npu.npu_quant_matmul(q.clone(), layer.weight, layer.weight_scale,
                                             pertoken_scale=scale, output_dtype=x.dtype)
        self.model.first.quant_method.apply = cloned_apply
        observer = self.observer()
        self.phase(observer, 0)
        report = observer.report()
        self.assertEqual(report['actual_projection_calls'], 2)
        self.assertEqual(report['actual_public_api_calls'], 4)
        self.assertEqual(self.mm_calls, 2)
        rows = report['forwards'][0]['projections']
        self.assertTrue(all(not row['public_chain_verified'] for row in rows))
        self.assertTrue(all('QMM_INPUT_NOT_ACTUAL_DYNAMIC_QUANT_OUTPUT_STORAGE' in row['errors'] for row in rows))
        self.assertTrue(all(row['original_calls'] == 1 for row in report['public_calls']))

    def test_public_call_outside_candidate_not_discarded(self):
        observer = self.observer()
        observer.begin('actual-forward', 'a' * 64, request_ids=('actual-request',))
        self.api.npu_dynamic_quant(torch.ones(2, 4, dtype=torch.bfloat16))
        self.model(torch.ones(2, 4, dtype=torch.bfloat16))
        observer.end('b' * 64, model_output_finite=True)
        report = observer.report()
        self.assertEqual(report['unowned_public_calls'], 1)
        self.assertEqual(report['actual_public_api_calls'], 5)
        self.assertEqual(report['public_calls'][0]['status'], 'PUBLIC_LINEAGE_REJECTED')

    def test_changed_healthy_output_refuses_repeatability_and_restores_api(self):
        observer = self.observer()
        self.phase(observer, 0)
        observer.begin('actual-forward-1', 'a' * 64, request_ids=('actual-request-1',))
        self.model(torch.ones(2, 4, dtype=torch.bfloat16))
        with self.assertRaisesRegex(RuntimeError, 'HEALTHY_WORKLOAD_NOT_REPEATABLE'):
            observer.end('c' * 64, model_output_finite=True)
        self.assertFalse(observer.report()['complete_healthy_replays'])
        self.assertIs(self.api.npu_dynamic_quant, self.original_quant)
        self.assertIs(self.api.npu_quant_matmul, self.original_qmm)

    def test_capacity_failure_keeps_total_and_original_execution(self):
        candidates, _ = _discover(self.model, (1,))
        observer = PublicQuantInventory(self.model, candidates, worker_identity={'rank': 0}, max_calls=1)
        self.observers.append(observer)
        observer.install()
        self.phase(observer, 0)
        report = observer.report()
        self.assertEqual(report['actual_public_api_calls'], 4)
        self.assertEqual(report['retained_public_api_calls'], 1)
        self.assertEqual(report['missing_public_call_details'], 3)
        self.assertFalse(report['public_observer_complete'])
        self.assertFalse(report['complete_healthy_replays'])
        self.assertEqual((self.q_calls, self.mm_calls), (2, 2))
        self.assertFalse(report['c2'])


class WholeWorkerQuantInventoryTests(unittest.TestCase):
    setUp = life.ServingSessionLifecycleTests.setUp
    tearDown = life.ServingSessionLifecycleTests.tearDown
    snapshot = life.ServingSessionLifecycleTests.snapshot
    root = life.ServingSessionLifecycleTests.root

    def test_real_cpu_root_and_logits_replay_all_projection_observation_then_remove(self):
        from demoncat.runtime.vllm_ascend.worker_extension.serving_quant_inventory import ServingQuantInventorySession
        model = CpuQmmModel().eval()
        candidates, _ = _discover(model, (1,))
        backend = PublicQuantInventory(model, candidates, worker_identity={'rank': 0})
        runner = NS(model=model, vllm_config=NS(model_config=NS(hf_config=NS(vocab_size=3))))
        worker = NS(model_runner=runner, rank=0)
        session = ServingQuantInventorySession(worker, model, backend, None, 'cpu-observer',
            prompt_token_ids=[1, 2, 3, 4], evidence_dir=self.evidence_root / 'inventory')
        self.sessions.append(session)
        api = sys.modules['torch_npu']
        api.npu_dynamic_quant = lambda x: (x.round().to(torch.int8), torch.ones(x.shape[0]))
        api.npu_quant_matmul = lambda x1, x2, scale, pertoken_scale, output_dtype: torch.mm(x1.float(), x2.float()).to(output_dtype)
        original = model.forward
        session.install()
        for index in range(2):
            self.actual_request_id = 'actual-worker-request-' + str(index)
            session.expect(self.actual_request_id)
            output = life.ServingSessionLifecycleTests.forward(self, session)
            self.assertEqual(list(output.shape), [1, 3])
        session.close()
        report = session.report()
        self.assertTrue(report['complete_healthy_replays'])
        self.assertEqual(report['actual_projection_calls'], 2)
        self.assertEqual(report['structural_candidate_count'], 2)
        self.assertFalse(report['c2'])
        self.assertEqual(report['qualified_native_targets'], 0)
        self.assertEqual(model.forward, original)
        self.assertEqual(session.handles, [])
        self.assertEqual(backend.patches, [])
        self.assertEqual(backend.api_patches, [])



if __name__ == '__main__':
    unittest.main()
