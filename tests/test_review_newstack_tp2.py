import copy,unittest
from experiments.worker_core_v11_20261118.review_newstack_tp2 import review

def fixture():
    out={'hidden':'h','logits':'l'};controls=[{'rank_rows':[{'row':{'output':out},'fault_calls':0} for _ in range(2)],'cleanup':[dict(owned_hooks=0,owned_wrappers=0,leases=0,inflight=0) for _ in range(2)]} for _ in range(3)]
    report=dict(mechanism='18.api_half',validation_scope='WARMED_WORKERS_ONLY_COLD_FIRST_REQUEST_NOT_QUALIFIED',preflight_controls=controls,scopes=[])
    hosts=[]
    for scope in range(2):
        group=dict(members=[{'rank':0},{'rank':1}],owner='o'+str(scope));states={};requests=[]
        for i,phase in enumerate(['normal','fault','off','health']):requests.append(dict(phase=phase,mapping={'internal_request_id':f'{scope}-{i}'},output={'tokens':[3]}))
        for rank in ('0','1'):
            rows=[]
            for i,phase in enumerate(['normal','fault','off','health']):
                effect=phase=='fault' and rank=='0'
                event=dict(path='actual.norm',applied=effect,output_changed=effect,return_readback_exact=True,consumers=[{'returned':True}],actual={'input':'fixed'},effect={'changed_elements':1} if effect else None)
                rows.append(dict(phase=phase,request_id=f'{scope}-{i}',input='same',output={'hidden':'f' if phase=='fault' else 'h','logits':'f' if phase=='fault' else 'l'},
                    request_trace=dict(status='OBSERVED',visual_calls=0,visual_execution_excluded=True,model_input_association_verified=True),
                    device_input_bytes_equality='COMPLETE_DEVICE_READBACK_VERIFIED',mechanism={'events':[event]},guard_boundary_unchanged=True))
            states[rank]={'serving':dict(actual_requests=rows,complete=True,recovered=True,failure=None)}
        report['scopes'].append(dict(index=scope,group=group,requests=requests,final_states=states))
        hosts.append(dict(group=group,ownership_retained=False,failure=None,role_decision={'selected_rank':0},ranks={r:{'history':[dict(action='finish',response={'provider_state':dict(owned_hooks=0,owned_wrappers=0,leases=0,inflight=0,replay_pending=False)})]} for r in ('0','1')}))
    import json
    return json.loads(json.dumps(report)),json.loads(json.dumps(hosts))
class ReviewTests(unittest.TestCase):
    def test_recomputes_without_status_or_probe_checks(self):
        r,h=fixture();self.assertEqual(review(r,h)['status'],'TWO_WARMED_TP2_SCOPES_INDEPENDENTLY_REVIEWED')
    def test_semantic_negative_cases(self):
        cases=[('warmup',lambda r,h:r['preflight_controls'][2]['rank_rows'][0]['row'].update(output={'hidden':'wrong'}),'WARMED_BASELINE'),
            ('off',lambda r,h:r['scopes'][0]['final_states']['0']['serving']['actual_requests'][2]['output'].update(hidden='wrong'),'OFF_HEALTH'),
            ('guard',lambda r,h:r['scopes'][0]['final_states']['1']['serving']['actual_requests'][1]['mechanism']['events'][0].update(applied=True),'TARGET_OR_CONSUMER'),
            ('owner',lambda r,h:h[1].update(ownership_retained=True),'OWNERSHIP'),
            ('resource',lambda r,h:h[0]['ranks']['0']['history'][0]['response']['provider_state'].update(owned_wrappers=1),'RESOURCES'),
            ('chunk',lambda r,h:r['scopes'][0]['final_states']['0']['serving']['actual_requests'][0]['request_trace'].update(status='INCOMPLETE'),'ACTUAL_TEXT'),
            ('input',lambda r,h:r['scopes'][0]['final_states']['0']['serving']['actual_requests'][1].update(input='changed'),'INPUTS')]
        for name,change,why in cases:
            with self.subTest(name=name):
                r,h=fixture();change(r,h)
                with self.assertRaisesRegex(ValueError,why):review(r,h)
if __name__=='__main__':unittest.main()
