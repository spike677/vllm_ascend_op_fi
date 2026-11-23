import unittest
from demoncat.runtime.vllm_ascend.worker_extension.product_batch import actual_forward_coverage


class ActualBatchCallTests(unittest.TestCase):
    def test_mla_absorbed_candidate_is_not_an_actual_call(self):
        proof=actual_forward_coverage([{'site':'a'}],['a'],['a','absorbed'])
        self.assertEqual(proof['actual_apply_calls'],{'a':1})
        self.assertEqual(proof['not_called'],[dict(site='absorbed',reason='NOT_CALLED_IN_ACTUAL_FORWARD')])

    def test_repeat_real_call_is_counted_not_silently_deduplicated(self):
        proof=actual_forward_coverage([{'site':'a'},{'site':'a'}],['a','a'],['a'])
        self.assertEqual(proof['actual_apply_calls'],{'a':2})

    def test_missing_extra_and_wrong_call_evidence_refused(self):
        for events,calls in [([],['a']),([{'site':'a'}],['a','a']),
                             ([{'site':'a'},{'site':'a'}],['a']),([{'site':'b'}],['a'])]:
            with self.subTest(events=events,calls=calls),self.assertRaisesRegex(RuntimeError,'FORWARD_COVERAGE'):
                actual_forward_coverage(events,calls,['a','b'])
        for calls in ([],['unknown']):
            with self.assertRaisesRegex(RuntimeError,'CENSUS_INVALID'):
                actual_forward_coverage([],calls,['a'])


if __name__=='__main__':unittest.main()
