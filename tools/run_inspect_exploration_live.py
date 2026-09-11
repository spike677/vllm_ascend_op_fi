"""Serial, frozen MT-Bench configuration matrix using the EXISTING lifecycle owner.

Needs an already leased isolated service. Does not launch, acquire a second lock,
change a fault algorithm, or change detector thresholds.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from ascend_fi.inspect.client import InspectClient
from ascend_fi.inspect.qwen_r1_adapter import create_owner
from ascend_fi.inspect.validation import ExistingOwnerIdentityValidator
from ascend_fi.inspect.plans import validate_plan
from ascend_fi.inspect.exploration import ROLE_GROUPS, make_experiment, evaluate_exploration, window_gate
from ascend_fi.inspect.dataset import load_samples
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate
from ascend_fi.evaluation.distortion.report import write_report
from run_interactive_inspect_live import save, selected, paged, owned_lease

PRIOR = Path('/home/haoran/inspect_20260910_mtbench_v1/dataset_acceptance')


def run(session, *, schedule='matrix'):
    if schedule not in {'matrix', 'c3-fast', 'c3-fault-only'}:
        raise ValueError('UNSUPPORTED_EXPLORATION_SCHEDULE')
    out = session/'exploration'
    out.mkdir(exist_ok=False)
    suite = json.loads((PRIOR/'suite.json').read_text())
    order = json.loads((PRIOR/'pre_run_contract.json').read_text())['sample_order']
    data_manifest, original = load_samples(session/'dataset/question.jsonl', session/'dataset/dataset_manifest.json')
    assert data_manifest['sha256'] == suite['dataset']['sha256']
    assert {p['prompt'] for p in suite['probes']} == {s['prompt'] for s in original}
    assert suite['hash'] == content_hash(suite)
    probes = [next(p for p in suite['probes'] if p['dataset_sample']['question_id']==qid) for qid in order]
    assert all(p['generation_config']['max_tokens']==1024 for p in probes)
    owner = create_owner(session,suite)
    client = InspectClient('http://127.0.0.1:18031')
    final, completed = {}, []
    preflight = False
    try:
        save(out/'preflight.json',owner.preflight()); preflight=True
        save(out/'frozen_r1_source_manifest.json',owner.manifest)
        save(out/'suite.json',suite)
        scorer = Path(__file__).resolve().parents[1]/'src/ascend_fi/evaluation/distortion'
        save(out/'pre_run_contract.json',{
            'suite_sha256':suite['hash'],'sample_order':order,'thresholds':suite['thresholds'],
            'original_r1_arm':owner.manifest['r1_arm_config'],
            'scorer_files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                [scorer/n for n in ('evaluator.py','detectors.py','contracts.py','mechanism.py')]},
            'matrix': ({'A0':'immutable prior eight-sample R1 evidence, not rerun',
                        'A1':'down .25 complete window','B':[1,2,5,10,20],
                        'C':ROLE_GROUPS,'final_saturation_control':.25}
                       if schedule == 'matrix' else
                       {'C3_ALL_FAMILIES_M20_FULL_WINDOW_FAST': {
                           'roles': ROLE_GROUPS[-1], 'multiplier': 20,
                           'identity_probe': 'mtbench_82_turn0',
                           'formal_probe_order': [p['probe_id'] for p in probes],
                           'repeat_rule': 'one independent B/F/R only when an existing '
                                          'baseline-relative rule confirms a sample'},
                        'fault_only': schedule == 'c3-fault-only'}),
            'schedule': schedule,
            'selection_rule':'Max contextual confirmed sample count, tie: smallest multiplier; if no confirmed B candidate, C uses predeclared 20 as stress exploration, NOT an effective pathology claim.',
            'repeat_rule':'After finite matrix choose max confirmed sample count, tie: first config in execution order. Two extra full eight-sample B/F/R cycles; search + repeats = N3. No confirmed candidate => no padding repeats.',
            'confirmed_pattern_rule':'Only baseline-relative repetition/lexical/punctuation/garbage rules; exclude Markdown separators. Length-only is not pathology. No semantic judge.',
            'recovery':'EXACT_FULL_TEXT_AND_TOKEN_PIECES; any mismatch stops escalation',
            'request_timeout_seconds':3600,'observer_history_limit':32,'dose_sample_interval_per_module':64,
            'phase_token_mapping':'UNKNOWN','original_artifacts':'READONLY',
            'request_isolation':'new private /v1 API key, project lease, serial owner, running/waiting zero and exactly-one engine completion'} )
        import urllib.request, urllib.error
        try:
            urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:18031/v1/chat/completions',b'{}',{'Content-Type':'application/json'}),timeout=10)
            raise RuntimeError('UNAUTHENTICATED_GENERATION_NOT_BLOCKED')
        except urllib.error.HTTPError as e:
            assert e.code==401
            save(out/'generation_isolation.json',{'unauthenticated_request':401,'token_persisted_in_evidence':False})
        status=selected(client.call('status'))
        assert len(status)==2 and {s['ranks']['global']['value'] for s in status}=={0,1}
        epochs=[s['service_epoch'] for s in status]
        rank0=next(s for s in status if s['ranks']['global']['value']==0)
        modules={e:paged(client,'list_modules',e) for e in epochs}
        save(out/'live_modules.json',modules)
        r1=copy.deepcopy(owner.manifest['r1_arm_config'])
        max_prompt=max(json.loads((PRIOR/'request_protocol.json').read_text())['prompt_tokens'].values())
        plans, declarations = {}, {}

        def run_config(config_id,roles,multiplier,cycle_index=1,*,probe_subset=None,identity_probe=None,
                       fault_only=False):
            target=out/config_id/f'cycle_{cycle_index:02d}'
            target.mkdir(parents=True,exist_ok=False)
            active_probes=probes if probe_subset is None else probe_subset
            if not active_probes:
                raise ValueError('EMPTY_PROBE_SUBSET')
            identity_probe=identity_probe or active_probes[0]
            exp=make_experiment(config_id,r1,modules[rank0['service_epoch']],roles,multiplier,1024,max_prompt)
            owner.configure_experiment(exp)
            owner.rawroot=target/'raw'
            save(target/'experiment.json',exp)
            names=exp['selected_modules']
            role_key=tuple(roles)
            if role_key not in plans:
                active=selected(client.call('observe_start',target_service_epochs=epochs,module_names=names,
                    request_window=config_id+'_binding',exclusive=True,ttl_seconds=300,
                    max_events=1024,max_bytes=2097152,samples_per_module=2))
                sessions={s['service_epoch']:s['session'] for s in active}
                try:
                    observed=owner.request(identity_probe,'observation',0)
                    events={e:paged(client,'observe_poll',e,session=sessions[e]) for e in epochs}
                finally:
                    stopped=selected(client.call('observe_stop',target_service_epochs=epochs,session_by_epoch=sessions))
                save(target/'binding_observations.json',events);save(target/'observer_stop.json',stopped)
                assert all(not s['active'] and not s['installed_handles'] for s in stopped)
                assert all({e['module_name'] for e in ev}==set(names) for ev in events.values())
                candidates=paged(client,'list_candidates',rank0['service_epoch'])
                candidates=[c for c in candidates if c['module_name'] in names]
                assert len(candidates)==len(names)
                plan=selected(client.call('export_plan',target_service_epochs=[rank0['service_epoch']],
                    candidate_ids=[c['candidate_id'] for c in candidates],runtime_fingerprint=rank0['runtime_fingerprint'],frozen_arm=owner.arm))[0]
                save(target/'draft_plan.json',plan)
                try:
                    config=owner.install()
                finally:
                    owner.disarm_and_remove()
                def identity():
                    current=selected(client.call('status'))
                    assert all(not s['active'] and not s['installed_handles'] for s in current)
                    w=next(s for s in current if s['service_epoch']==rank0['service_epoch'])
                    return w['service_epoch'],w['runtime_fingerprint']
                v=validate_plan(plan,epoch=plan['service_epoch'],fingerprint=plan['runtime_fingerprint'],
                    executor=ExistingOwnerIdentityValidator(owner,config,identity_probe,identity))
                save(target/'identity_validation.json',v.to_dict())
                assert v.lossless_passed and v.restore_exact
                raw=v.to_dict()['raw'];base=raw['baseline']
                assert all(raw[k].get('token_pieces')==base.get('token_pieces') for k in ('disabled','identity','recovery'))
                assert observed['response_text']==base['response_text'] and observed['token_pieces']==base['token_pieces']
                plans[role_key]=plan
            else:
                # Re-export against the SAME epoch and observed exact target set
                # when strength changes; do not silently reuse a stale frozen arm.
                previous=plans[role_key]
                plan=selected(client.call('export_plan',target_service_epochs=[rank0['service_epoch']],
                    candidate_ids=previous['candidate_ids'],runtime_fingerprint=rank0['runtime_fingerprint'],frozen_arm=owner.arm))[0]
                assert set(plan['selected_modules'])==set(names)
                save(target/'draft_plan.json',plan)
            before=owner.clean_query()
            cycle={'cycle_id':f'cycle_{cycle_index:02d}','before_baseline':before,'service_healthy_before':True,'pairs':{}}
            config=None
            for p in active_probes:
                current=selected(client.call('status'))
                assert all(not s['active'] and not s['installed_handles'] for s in current)
                assert any(s['service_epoch']==plan['service_epoch'] and s['runtime_fingerprint']==plan['runtime_fingerprint'] for s in current)
                assert cleanup_gate(owner.clean_query(),owner.service_healthy(),2)['status']=='PASS'
                if fault_only:
                    try:
                        config=owner.install()
                        fault=owner.fault_request(p,config,cycle_index)
                    finally:
                        owner.disarm_and_remove()
                    after=owner.clean_query()
                    cleanup=cleanup_gate(after,owner.service_healthy(),2)
                    window=window_gate(fault,config)
                    record={'scope':'FAST_FAULT_ONLY_NOT_A_COMPARATIVE_EFFECT_EVALUATION',
                            'fault':fault,'request_window_gate':window,'cleanup':cleanup,
                            'identity_probe_id':identity_probe['probe_id'],
                            'identity_validation':v.to_dict(),
                            'config':config.to_dict(),'experiment':exp}
                    save(target/(p['probe_id']+'_fault_only.json'),record)
                    assert window['status']=='PASS', 'WINDOW_MECHANISM_GATE_FAILED; no escalation'
                    assert cleanup['status']=='PASS', 'CLEANUP_GATE_FAILED; no escalation'
                    cycle['pairs'][p['probe_id']]={'fault':fault}
                    cycle.update(after_remove=after,service_healthy=owner.service_healthy())
                    print(json.dumps({'config':config_id,'cycle':cycle_index,'sample':p['probe_id'],
                        'scope':'FAST_FAULT_ONLY','window':window['status'],
                        'hit':fault.get('mechanism_summary',{}).get('measurement_hit_count'),
                        'tokens':fault.get('generation_length')}),flush=True)
                    continue
                baseline=owner.request(p,'baseline',cycle_index)
                try:
                    actual=owner.install()
                    if config is not None: assert actual.to_dict()==config.to_dict()
                    config=actual
                    fault=owner.fault_request(p,config,cycle_index)
                finally:
                    owner.disarm_and_remove()
                after=owner.clean_query()
                assert cleanup_gate(after,owner.service_healthy(),2)['status']=='PASS'
                recovery=owner.request(p,'recovery',cycle_index)
                cycle['pairs'][p['probe_id']]={'baseline':baseline,'fault':fault,'recovery':recovery}
                cycle.update(after_remove=owner.clean_query(),service_healthy=owner.service_healthy())
                save(target/(p['probe_id']+'.json'),cycle['pairs'][p['probe_id']])
                subset={**suite,'probes':[p for p in suite['probes'] if p['probe_id'] in cycle['pairs']], 'probe_count':len(cycle['pairs'])}
                subset['hash']=content_hash(subset)
                result=evaluate_exploration(subset,config,cycle,evaluation_id=config_id+f'-{cycle_index}-{len(cycle["pairs"])}')
                save(target/('checkpoint_'+str(len(cycle['pairs']))+'.json'),result.to_dict())
                x=result.to_dict()['exploration']
                print(json.dumps({'config':config_id,'cycle':cycle_index,'sample':p['probe_id'],'integration':result.status,
                    'window':x['REQUEST_WINDOW_COVERAGE'],'hit':fault.get('mechanism_summary',{}).get('measurement_hit_count'),
                    'BFR_tokens':[r['generation_length'] for r in (baseline,fault,recovery)],'confirmed':x['confirmed_sample_ids']}),flush=True)
                assert result.status=='PASS', 'WINDOW_MECHANISM_OR_EXACT_RECOVERY_GATE_FAILED; no escalation'
            if fault_only:
                save(target/'live_raw.json',{'suite':suite,'config':config.to_dict(),'cycles':[cycle],
                    'scope':'FAST_FAULT_ONLY_NOT_A_COMPARATIVE_EFFECT_EVALUATION'})
                entry={'config_id':config_id,'cycle':cycle_index,'roles':roles,'multiplier':multiplier,
                    'fault_only':True,'output':str(target),'status':'PASS'}
                completed.append(entry);save(out/f'progress_{len(completed):02d}.json',completed)
                declarations[config_id]=exp
                return entry
            save(target/'live_raw.json',{'suite':suite,'config':config.to_dict(),'cycles':[cycle]})
            write_report(target/'effect',result,{'scope':'CONFIG_EXPLORATION','complete_cycles':1,'same_official_subset':True})
            entry={'config_id':config_id,'cycle':cycle_index,'roles':roles,'multiplier':multiplier,
                   'confirmed_sample_ids':x['confirmed_sample_ids'],'output':str(target),'status':result.status}
            completed.append(entry);save(out/f'progress_{len(completed):02d}.json',completed)
            declarations[config_id]=exp
            return entry

        if schedule == 'c3-fault-only':
            sample_82=next(p for p in probes if p['probe_id']=='mtbench_82_turn0')
            sample_149=next(p for p in probes if p['probe_id']=='mtbench_149_turn0')
            run_config('C3_ALL_FAMILIES_M20_FULL_WINDOW_FAST', ROLE_GROUPS[-1], 20,
                       probe_subset=[sample_149], identity_probe=sample_82, fault_only=True)
        elif schedule == 'c3-fast':
            sample_82=next(p for p in probes if p['probe_id']=='mtbench_82_turn0')
            c3=run_config('C3_ALL_FAMILIES_M20_FULL_WINDOW_FAST', ROLE_GROUPS[-1], 20,
                          identity_probe=sample_82)
            if c3['confirmed_sample_ids']:
                repeat_id=c3['confirmed_sample_ids'][0]
                repeat_probe=next(p for p in probes if p['probe_id']==repeat_id)
                save(out/'repeat_selection.json', {
                    'selected_config': c3, 'selected_probe_id': repeat_id,
                    'rule': 'one independent B/F/R after an existing confirmed rule; N=2 only'})
                run_config('C3_ALL_FAMILIES_M20_FULL_WINDOW_FAST', ROLE_GROUPS[-1], 20, 2,
                           probe_subset=[repeat_probe], identity_probe=sample_82)
        else:
            run_config('A1_down_025_full',ROLE_GROUPS[0],.25)
            b=[run_config('B_down_'+str(m)+'_full',ROLE_GROUPS[0],m) for m in (1,2,5,10,20)]
            candidates=[c for c in b if c['confirmed_sample_ids']]
            chosen=sorted(candidates,key=lambda c:(-len(c['confirmed_sample_ids']),c['multiplier']))[0] if candidates else b[-1]
            save(out/'coverage_multiplier_selection.json',{'selected':chosen,'pathology_candidate':bool(candidates),'fallback_predeclared':20})
            for n,roles in enumerate(ROLE_GROUPS[1:],1):
                run_config(f'C{n}_m{chosen["multiplier"]}_full',roles,chosen['multiplier'])
            run_config('C3_all_025_full',ROLE_GROUPS[-1],.25)
            all_candidates=[c for c in completed if c['confirmed_sample_ids']]
            if all_candidates:
                selected_candidate=sorted(enumerate(all_candidates),key=lambda t:(-len(t[1]['confirmed_sample_ids']),t[0]))[0][1]
                save(out/'repeat_selection.json',selected_candidate)
                for repeat in (2,3):
                    run_config(selected_candidate['config_id'],selected_candidate['roles'],selected_candidate['multiplier'],repeat)
        final.update(matrix_complete=True,completed=completed)
    except BaseException as e:
        final.update(error=repr(e),traceback=traceback.format_exc(),completed=completed)
        print(final['traceback'],flush=True)
        raise
    finally:
        if preflight and owned_lease(session):
            try:
                client.call('observe_stop');owner.disarm_and_remove()
                final['cleanup']=cleanup_gate(owner.clean_query(),owner.service_healthy(),2)
            except BaseException as e:
                final['cleanup_error']=repr(e)
        save(out/'exploration_final.json',final)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--session',type=Path,required=True)
    ap.add_argument('--schedule', choices=('matrix','c3-fast','c3-fault-only'), default='matrix')
    args=ap.parse_args()
    run(args.session, schedule=args.schedule)
