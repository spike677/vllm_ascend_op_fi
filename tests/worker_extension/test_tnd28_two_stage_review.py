from copy import deepcopy
import unittest
from experiments.worker_core_v11_20261118 import review_tnd28_two_stage as R
class ReviewTests(unittest.TestCase):
    def fixture(self):
        rows=[]
        for i,phase in enumerate(R.PHASES):
            native={} if phase in ('baseline','baseline_repeat','fault','off') else None
            calls=[] if phase=='health' else [dict(native=native if j==0 else None,target_key='chosen' if j==0 else str(j),
                input='same-qkv',output='different' if phase=='fault' and j==0 else 'same',original_calls=1,original_returned=True) for j in range(28)]
            rows.append(dict(phase=phase,request_id=str(i),forwards=[dict(input='same-root',hidden='F' if phase=='fault' else 'B',logits='F' if phase=='fault' else 'B',calls=calls)]))
        state=dict(serving={'next_phase':'complete'},registered=False,binding_reserved=False,owned_hooks=0,owned_api_wrappers=0,inflight=0,leases=0,
            native_state=dict(scope_active=False,scope_owned=False,environment_owned=0,pending_drains=0,pending_references=0),
            evidence=dict(requests=rows,actual_requests_sha256=R.E.digest(rows),selection={'target_key':'chosen'}))
        h=[dict(action=a) for a in ('prepare','register','check','enable')]
        for i,row in enumerate(rows):
            h.extend([dict(action='advance',request_id=row['request_id']),dict(action='query'),dict(action='collect',response={'provider_state':{'evidence':{'requests':deepcopy(rows[:i+1])}}})])
        h.extend([dict(action='finish'),dict(action='unregister'),dict(action='detach',response={'provider_state':state})])
        return dict(protocol='EXPLICIT_TWO_STAGE_RESEARCH_NOT_PUBLIC_RUNTIME',ownership_retained=False,history=h)
    def test_complete_research_history(self):self.assertEqual(len(R.verify_history(self.fixture())[0]),7)
    def test_residual_resources_refused(self):
        h=self.fixture();h['history'][-1]['response']['provider_state']['leases']=1
        with self.assertRaisesRegex(ValueError,'RESOURCES'):R.verify_history(h)
    def test_exact_recovery_and_normal_bridge_required(self):
        for index in (1,2,3,5,6):
            h=self.fixture();state=h['history'][-1]['response']['provider_state'];rows=state['evidence']['requests']
            rows[index]['forwards'][0]['hidden']='wrong';state['evidence']['actual_requests_sha256']=R.E.digest(rows)
            for i in range(7):h['history'][6+3*i]['response']['provider_state']['evidence']['requests']=deepcopy(rows[:i+1])
            with self.assertRaisesRegex(ValueError,'VALUES_DIFFER'):R.verify_history(h)
    def test_request_mapping_not_just_stage_labels(self):
        h=self.fixture();h['history'][4]['request_id']='other'
        with self.assertRaisesRegex(ValueError,'HISTORY_DISAGREES'):R.verify_history(h)
    def test_historical_snapshot_cannot_be_silently_replaced(self):
        h=self.fixture();h['history'][6]['response']['provider_state']['evidence']['requests'][0]['phase']='fault'
        with self.assertRaisesRegex(ValueError,'HISTORY_DISAGREES'):R.verify_history(h)
