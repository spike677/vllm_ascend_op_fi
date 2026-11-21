from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension.readonly_inspect import inspect_worker,aggregate_inspection


class InspectTests(unittest.TestCase):
    def test_tp8_missing_runner_fields_does_not_use_fault_gate(self):
        model=NS(named_modules=lambda:iter((str(i),NS()) for i in range(300)))
        worker=NS(rank=7,model_runner=NS(model=model))
        with patch('demoncat.runtime.vllm_ascend.worker_extension.readonly_inspect.observe_deployment',return_value=dict(tp=8)):
            result=inspect_worker(worker)
        self.assertEqual(len(result['modules']),128);self.assertEqual(result['next_offset'],128)
        self.assertFalse(hasattr(worker,'_demoncat_worker_fault_runtime'))
        self.assertFalse(result['qualification_granted'])
    def test_missing_and_failed_ranks_are_explicit(self):
        rows=[dict(rank=i,status='READ_ONLY_INSPECTION') for i in range(7)]
        result=aggregate_inspection(range(8),rows)
        self.assertEqual(result['missing_ranks'],[7]);self.assertEqual(result['status'],'PARTIAL')
        rows.append(dict(rank=7,status='INSPECTION_UNAVAILABLE'))
        self.assertEqual(aggregate_inspection(range(8),rows)['failed_ranks'],[7])
