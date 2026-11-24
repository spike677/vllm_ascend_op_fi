import unittest
from types import SimpleNamespace as NS
from demoncat.runtime.vllm_ascend.worker_extension.external_control import control_reply,validate_control_replies


class ControlTests(unittest.TestCase):
    def worker(self,rank):return NS(demoncat_worker_identity=lambda:dict(worker=dict(rank=rank)))
    def test_all_rank_structured_errors_remain_correlated(self):
        def reject():raise ValueError('expected')
        rows=[control_reply(self.worker(0),'call-1',reject),control_reply(self.worker(1),'call-1',lambda:{'ok':True})]
        got=validate_control_replies(rows,request_token='call-1',world_size=2)
        self.assertEqual([r['exit_code'] for r in got],[4,0])
    def test_internal_error_does_not_escape(self):
        def fail():raise KeyError('implementation')
        self.assertEqual(control_reply(self.worker(0),'id',fail)['exit_code'],5)
    def test_stale_missing_duplicate_ranks_fail_closed(self):
        a=control_reply(self.worker(0),'new',lambda:None);b=control_reply(self.worker(1),'old',lambda:None)
        for rows in ([a,b],[a],[a,a]):
            with self.assertRaisesRegex(RuntimeError,'CORRELATION_UNPROVEN'):
                validate_control_replies(rows,request_token='new',world_size=2)
    def test_single_rank_uses_same_validation(self):
        r=control_reply(self.worker(0),'single',lambda:1)
        self.assertEqual(validate_control_replies([r],request_token='single',world_size=1),[r])
    def test_non_serializable_result_is_error_data(self):
        self.assertEqual(control_reply(self.worker(0),'id',lambda:object())['exit_code'],5)


if __name__=='__main__':unittest.main()
