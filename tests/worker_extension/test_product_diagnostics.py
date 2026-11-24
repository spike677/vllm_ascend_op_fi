import unittest
from copy import deepcopy
from unittest.mock import Mock
from demoncat.runtime.vllm_ascend.worker_extension.product_diagnostics import DiagnosticProductClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig


class DiagnosticTests(unittest.TestCase):
    def client(self,rows):
        ex=Mock();ex.collective_rpc.return_value=rows
        c=DiagnosticProductClient(ex,FaultConfig(18,'api_half'),profile='norm',operator='trusted',run_canary=lambda:None,drain=lambda:None)
        c.members=[dict(rank=0),dict(rank=1)];c.enabled=True
        return c,ex
    def test_mismatch_preserves_exact_both_rank_values_and_error(self):
        rows=[dict(rank=1,requests=[],cursor=0,more=False,active_request_id='q',failure=None),
              dict(rank=0,requests=[dict(request_id='q',coverage={'applied':7})],cursor=1,more=False,active_request_id=None,failure=None)]
        original=deepcopy(rows);c,ex=self.client(rows)
        with self.assertRaisesRegex(RuntimeError,'COMPLETION_MISMATCH') as caught:c.status()
        d=caught.exception.status_diagnostic
        self.assertEqual([r['cursor'] for r in d['ranks']],[0,1]);self.assertEqual([r['active'] for r in d['ranks']],[True,False])
        self.assertEqual(d['ranks'][1]['requests'][0]['coverage'],{'applied':7})
        self.assertEqual(c.cursor,0);self.assertFalse(c.requests);self.assertEqual(rows,original)
        self.assertEqual(ex.collective_rpc.call_count,1)
    def test_transport_failure_keeps_unknown_ranks_null(self):
        c,ex=self.client([]);ex.collective_rpc.side_effect=TimeoutError('timeout')
        with self.assertRaises(TimeoutError):c.status()
        self.assertIsNone(c.diagnostics()[0]['ranks'])
    def test_success_does_not_create_failure_or_change_cursor(self):
        c,ex=self.client([dict(rank=i,requests=[],cursor=0,more=False,failure=None) for i in (0,1)])
        self.assertEqual(c.status()['requests'],[]);self.assertEqual(c.diagnostics(),[])
    def test_missing_active_field_is_not_false(self):
        c,ex=self.client([dict(rank=0,requests=[],cursor=0,more=False,failure='x'),dict(rank=1,requests=[],cursor=0,more=False,failure=None)])
        with self.assertRaises(RuntimeError):c.status()
        self.assertIsNone(c.diagnostics()[0]['ranks'][0]['active'])
    def test_malformed_request_cannot_mask_original_failure(self):
        c,ex=self.client([dict(rank=0,requests=[None],cursor=1,more=False,failure='original'),
                          dict(rank=1,requests=[],cursor=0,more=False,failure=None)])
        with self.assertRaises(RuntimeError) as caught:c.status()
        self.assertIn('original',str(caught.exception))
        self.assertEqual(c.diagnostics()[0]['ranks'][0]['requests'],[{'malformed_type':'NoneType'}])
        self.assertEqual(ex.collective_rpc.call_count,1)
