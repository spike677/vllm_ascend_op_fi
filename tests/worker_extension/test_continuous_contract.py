from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from fault_injection.continuous_contract import continuous_request_plan, review_rank_coverage, MECHANISMS
from fault_injection.delivery_scope import describe_delivery
from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report
from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider


def receipt(hits=3):
    return dict(counts=dict(observed=hits, eligible=hits, attempted=hits, applied=hits,
        changed=min(hits, 1), consumed=hits, restored=hits, failed=0),
        observation_complete=True, same_stream_verified=True, bytes_restored=True,
        off_exact=True, health_exact=True, uncovered_sites=[],
        resources=dict(leases=0, hooks=0, inflight=0), inapplicable_reasons={})


class ContinuousContractTests(unittest.TestCase):
    def test_scope_and_plans(self):
        self.assertEqual(len(MECHANISMS),12)
        for mechanism in MECHANISMS:
            plan=continuous_request_plan(mechanism, batch_size=3 if mechanism.startswith('quant.') else 1)
            self.assertEqual([p['phase'] for p in plan['phases']], ['normal','fault','off','health'])
            self.assertTrue(all(p['max_tokens']==4 for p in plan['phases']))
            self.assertFalse(plan['qualification_granted'])
        with self.assertRaises(ValueError): continuous_request_plan('quant.batch_shared_scale')

    def test_world_sizes_and_ep_missing_local_call(self):
        for n in (1,2,8):
            rows={i:receipt(0 if i == n-1 and n > 1 else i+1) for i in range(n)}
            before=deepcopy(rows)
            self.assertTrue(review_rank_coverage(range(n),rows)['passed'])
            self.assertEqual(rows,before)
        self.assertFalse(review_rank_coverage([0],{0:receipt(0)})['passed'])

    def test_missing_failed_or_uncovered_cannot_be_skipped(self):
        for mutate in (lambda r:r['counts'].update(failed=1),
                       lambda r:r['counts'].update(attempted=2),
                       lambda r:r.update(uncovered_sites=['unknown_impl']),
                       lambda r:r.update(bytes_restored=False),
                       lambda r:r['resources'].update(leases=1),
                       lambda r:r.update(same_stream_verified=False)):
            row=receipt(); mutate(row)
            self.assertFalse(review_rank_coverage([0],{0:row})['passed'])
        self.assertFalse(review_rank_coverage([0,1],{0:receipt()})['passed'])

    def test_residual_query_and_creation_excluded_before_import(self):
        key='26.residual_stomp'
        delivery=describe_delivery(key)
        self.assertFalse(delivery['selected_for_current_delivery'])
        self.assertEqual(delivery['next_step'],'本轮排除')
        report=ordinary_route_report(key)
        self.assertFalse(report['can_prepare'])
        with patch('importlib.import_module',side_effect=AssertionError('provider must not load')):
            with self.assertRaisesRegex(RuntimeError,'本轮排除'):
                create_worker_provider(key,'1',SimpleNamespace(),object(),{})

if __name__=='__main__': unittest.main()
