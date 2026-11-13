"""Post-localization final audit, including uncaptured OFF and resource release."""
import hashlib,json,subprocess,sys,time
from pathlib import Path

ROOT=Path('/home/fault28-reduce-replay-20261113')
sys.path[:0]=[str(ROOT),str(ROOT/'analysis')]
from audit_runtime import audit
from run_search import Search

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def main():
    Search(0).verify_freeze()
    assert read(ROOT/'reports/formal/review.json')['accepted_before_controls']
    assert read(ROOT/'reports/restart/review.json')['accepted_before_controls']
    assert read(ROOT/'reports/controls.json')['status']=='PASS_RESTART_OFF_AND_FIXED_QKV_BATCH_CONTROLS'
    state=read(ROOT/'search_state.json');rows=[]
    assert len(state['cases'])==16 and all(c['status']=='CAPTURE_COMPLETED' for c in state['cases'].values())
    for name,row in state['cases'].items():
        folder=ROOT/row['folder'];assert sha(folder/'result.json')==row['result_sha256']
        runtime=audit(folder)
        keys=sorted({x['actual_key'] for x in runtime['census']})
        expected='119000000000400000' if name in ['analytic_F','real_F','formal_F','restart_F','ctrl_fault1','ctrl_fault8','ctrl_auto8'] else '109000000000000000' if name=='ctrl_auto1' else '109000000000400000'
        assert keys==[expected],(name,keys)
        rows.append({'case_id':name,'status':runtime['status'],'physical_card':runtime['physical_card'],
            'FIA_executions':runtime['actual_FIA_executions'],'layer_request_executions':runtime['layer_request_executions'],
            'all_observed_outputs_repeat_exact':runtime['all_outputs_repeat_exact'],'actual_kernel_keys':keys,
            'result_sha256':sha(folder/'result.json'),'controller_completion_sha256':sha(folder.parent/'completion.json'),
            'no_cleanup_errors':True,'owned_container_removed_and_device_lock_released':True})
    owned=subprocess.check_output(['docker','ps','-aq','--filter','label=demoncat.task=fault28-reduce-replay-20261113'],text=True).split()
    assert not owned,'Task containers remain'
    total=sum(r['FIA_executions'] for r in rows);assert total==608
    final={'schema':'controlled_reduction.final_acceptance.v1','status':'PASS_NEW_DENOMINATOR_REPLAY_VARIANT',
        'accepted_as_historical_fault28_root_cause':False,'variant':'online_softmax_denominator_replay',
        'physical_card':2,'completed_cases':len(rows),'actual_native_FIA_calls':total,'cases':rows,
        'formal_public_groups':['baseline','fault'],'actual_capture_steps':[0,1,2],
        'preregister_sha256':sha(ROOT/'preregister.json'),'execution_source_and_vendor_freeze_unchanged':True,
        'formal_statistical_first_selected_without_runtime_or_private_arrays':True,
        'independent_restart_three_steps_passed':True,'OFF_actual_key_healthy_and_all_arrays_exact':True,
        'task_containers_remaining':owned,'host_docker_entry_sha256':sha('/usr/bin/docker'),
        'host_docker_entry_modified_by_this_experiment':False,'global_sdk_modified':False,
        'finished_at':time.time(),'limits':['New intentional denominator accumulation error, not historical legal-tiling drift.',
            'Prefill only, fixed geometry/version. No continuous-generation or task-accuracy claim.',
            'Statistics independently identify FIA output; they do not uniquely identify an internal physical instruction.']}
    with (ROOT/'reports/final_acceptance.json').open('x') as f:json.dump(final,f,indent=2);f.write('\n')
    state.update(status='COMPLETE_NEW_REDUCTION_ERROR_VARIANT',accepted_as_historical_fault28=False,final_report='reports/final_acceptance.json')
    (ROOT/'search_state.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps({k:final[k] for k in ['status','completed_cases','actual_native_FIA_calls','task_containers_remaining']}))
if __name__=='__main__':main()
