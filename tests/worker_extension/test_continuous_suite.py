import unittest
from tools.run_continuous_suite import run_cases
class SuiteTests(unittest.TestCase):
    def test_recovered_failure_does_not_drop_next_case(self):
        ran=[]
        def execute(case):
            ran.append(case);return dict(passed=case!='bad',cleanup_verified=True)
        r=run_cases(['bad','good'],execute)
        self.assertEqual(ran,['bad','good']);self.assertFalse(r['passed'])
        self.assertTrue(r['cases'][1]['passed'])
    def test_quarantine_is_not_cleared_by_next_case(self):
        ran=[]
        def execute(case):ran.append(case);return dict(passed=False,cleanup_verified=False)
        r=run_cases(['bad','next'],execute)
        self.assertEqual(ran,['bad']);self.assertEqual(r['cases'][1]['status'],'NOT_RUN_UNRESOLVED_RECOVERY')
