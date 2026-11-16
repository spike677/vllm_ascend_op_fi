import copy
import unittest
from demoncat_fault39.target import select_stable_target, TargetError


class TargetPolicyTests(unittest.TestCase):
    def setUp(self):
        self.before = [dict(signature=str(i), candidate_count=1, observations=3, M=m, K=k, N=n, rank=0, device=0)
                       for i, (m, k, n) in enumerate([(21, 1024, 4096), (21, 3072, 1024), (1, 3072, 1024)])]
        self.after = copy.deepcopy(self.before)
        for c in self.after: c['observations'] += 1
    def test_strict_default_keeps_ambiguity(self):
        with self.assertRaisesRegex(TargetError, 'AMBIGUOUS'):
            select_stable_target(self.after, self.before)
    def test_explicit_policy_uses_geometry_not_list_order(self):
        self.assertEqual(select_stable_target(self.after[::-1], self.before, 'max_k_min_m_unique'), '2')
    def test_tie_is_not_broken_by_name_or_order(self):
        self.after.append(dict(self.after[-1], signature='other', N=2048))
        with self.assertRaisesRegex(TargetError, 'AMBIGUOUS'):
            select_stable_target(self.after, self.before, 'max_k_min_m_unique')
    def test_new_stronger_candidate_refuses_stability(self):
        self.after.append(dict(self.after[-1], signature='new', K=8192))
        with self.assertRaisesRegex(TargetError, 'UNSTABLE'):
            select_stable_target(self.after, self.before, 'max_k_min_m_unique')
    def test_same_geometry_multiple_cache_identities_refuses(self):
        self.after[-1]['candidate_count'] = 2
        with self.assertRaisesRegex(TargetError, 'AMBIGUOUS'):
            select_stable_target(self.after, self.before, 'max_k_min_m_unique')
