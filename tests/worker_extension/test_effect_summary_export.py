from copy import deepcopy
import unittest
from demoncat.runtime.vllm_ascend.worker_extension.deployment_summary import summarize_scope,_FIELDS


def receipt():
    actual={k:'observed' for k in _FIELDS};actual['hardware_family']='A3'
    state=dict(deployment_acceptance_use=dict(actual=actual,scope_valid=True,registration_source='YELLOW_TEST_DEPLOYMENT'),
        inflight=0,leases=0,registered=False,enabled=False,native_registered=False,
        evidence=dict(actual_requests=[dict(phase='fault',request_id='internal-1')],
            recovered=None,matched_off_and_health_confirmed=True,
            last_fault_effect=dict(positions=[dict(semantic_position=dict(path='norm',ordinal=1),
                actual_consumer_observed=True,values=dict(changed_elements=8,elements=16))])))
    return dict(mechanism='18.api_half',owner='scope',worker=dict(rank=0),history=[dict(response=dict(provider_state=state))],
                ownership_retained=False,status='DETACHED')


class ExportTests(unittest.TestCase):
    def test_element_fields_and_derived_recovery_survive(self):
        r=receipt();result=summarize_scope(r)
        self.assertEqual(result['effects']['last_fault_effect']['positions'][0]['values']['changed_elements'],8)
        self.assertTrue(result['effects']['recovered']);self.assertIsNone(result['effects']['reported_recovered'])
        self.assertTrue(result['recovery_confirmed']);self.assertTrue(result['evidence_complete'])
        self.assertIsNone(r['history'][0]['response']['provider_state']['evidence']['recovered'])
    def test_contradiction_is_not_normalized_into_success(self):
        r=receipt();r['history'][0]['response']['provider_state']['evidence']['recovered']=False
        self.assertIn('RECOVERY_FIELDS_CONTRADICT',summarize_scope(r)['missing_or_failed'])
    def test_rope_row_effect_and_rank_coverage_survive(self):
        r=receipt();r['mechanism']='22.wrong_row';ev=r['history'][0]['response']['provider_state']['evidence']
        ev['last_fault_effect']=dict(complete_qk_changed=True,non_target_unchanged=True,correct_source_row=3,replacement_source_row=0)
        ev['coverage']=dict(observed=4,eligible=4,attempted=4,applied=4,changed=3,consumed=4,restored=4,failed=0)
        result=summarize_scope(r)
        self.assertEqual(result['effects']['coverage'],ev['coverage'])
        self.assertEqual(result['effects']['last_fault_effect'],ev['last_fault_effect'])
