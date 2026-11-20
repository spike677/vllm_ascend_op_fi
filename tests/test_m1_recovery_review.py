from copy import deepcopy
import unittest
from experiments.worker_core_v11_20261118.review_m1_recovery import recovery_protocol


def event(action, phase, complete=False):
    local = dict(next_phase=phase, complete=complete, recovered=complete,
                 actual_requests=[dict(phase=p) for p in ('normal', 'fault')])
    state = dict(next_phase=phase, complete=complete,
                 rank_states={str(rank): dict(serving=deepcopy(local)) for rank in (0, 1)})
    return dict(action=action, status='QUERY_OK' if action == 'query' else 'ACTION_COMPLETED',
                response=dict(provider_state=state))


class RecoveryReviewTests(unittest.TestCase):
    def host(self):
        return dict(history=[event('recover', 'off'), event('advance', 'off'),
                    event('query', 'health'), event('advance', 'health'),
                    event('query', 'complete', True), event('finish', 'complete', True)])
    def test_aligned_terminal(self):
        self.assertEqual(recovery_protocol(self.host())['all_rank_terminal_phase'], 'complete')
    def test_output39_evidence_is_separate_from_serving_resources(self):
        host=self.host()
        for e in host['history']:
            for state in e['response']['provider_state']['rank_states'].values():
                local=state.pop('serving');state['next_phase']=local.pop('next_phase')
                local['matched_off_and_health_confirmed']=local.pop('recovered');local['mechanism']='39.output_proxy'
                state['evidence']=local;state['serving']={'owned_hooks':4}
        self.assertEqual(recovery_protocol(host)['all_rank_terminal_phase'],'complete')
    def test_old_terminal_off_bug(self):
        host=self.host();host['history'][-2]=event('query', 'off', True)
        with self.assertRaisesRegex(ValueError, 'GROUP_PLAN_NOT_COMPLETE'): recovery_protocol(host)
    def test_missing_recovery_rank(self):
        host=self.host();del host['history'][0]['response']['provider_state']['rank_states']['1']
        with self.assertRaisesRegex(ValueError, 'RECOVER_MISSING_RANK'): recovery_protocol(host)
    def test_unfinished_peer(self):
        host=self.host();host['history'][-2]['response']['provider_state']['rank_states']['1']['serving']['complete']=False
        with self.assertRaisesRegex(ValueError, 'RANK_PLAN_NOT_COMPLETE'): recovery_protocol(host)
    def test_missing_recovery_confirmation(self):
        host=self.host();host['history'][-2]['response']['provider_state']['rank_states']['1']['serving']['recovered']=False
        with self.assertRaisesRegex(ValueError, 'RANK_RECOVERY_UNCONFIRMED'): recovery_protocol(host)
    def test_repeated_fault_after_recover(self):
        host=self.host();host['history'].insert(1,event('advance','fault'))
        with self.assertRaisesRegex(ValueError, 'RECOVERY_REQUEST_SEQUENCE'): recovery_protocol(host)

if __name__ == '__main__': unittest.main()
