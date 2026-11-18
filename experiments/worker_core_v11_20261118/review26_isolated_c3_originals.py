"""Independent delta originals for isolated C2 followed by cumulative C3.

This reader imports only pure audit helpers. Saved proof objects document a
historical live transition; they cannot recreate one or qualify a new scope.
"""
from pathlib import Path

STAGE='c2_isolated_c3'
SCOPE='ALL_STABLE_ELIGIBLE_TARGETS_ISOLATED_C2_THEN_SAME_WORKER_BOUNDED_C3_EVERY_MATCH'
C3_SCOPE='SAME_REQUEST_CUMULATIVE_EVERY_MATCH_TWO_FRESH_REQUESTS'


def helpers(r):
    return tuple(r.C.pure_module(Path(__file__).with_name(name),'_transition_'+name[:-3]) for name in
        ('review26_isolated_originals.py','review26_c3_originals.py'))


def scope(r,a,value,ref,context,run,*,natural):
    generation=context['ticket']['generation']
    a.check(value['schema']=='demoncat.writer26.isolated_core_observations.v1' and
        value['action']==ref['action'] and value['observation_kind']==ref['observation_kind'] and
        value['worker']==context['worker'] and value['native_owner']==context['provider_owner'] and
        value['run']==run and value['simulation'] is (not natural) and value['qualification_granted'] is False and
        all(r.C.positive(value[k]) for k in ('generation','rpc_generation','native_generation')) and
        value['generation']==value['rpc_generation']==generation and
        value['native_generation']==generation+int(run=='c3'), 'TRANSITION_DELTA_EXACT_SCOPE_AND_GENERATION',run)


def rpc(r,i,a,inv,value,evidence,context,run,*,state=None):
    compact={k:v for k,v in evidence.items() if k not in ('public_core','original_observations','current_run','prior_c2')}
    public=evidence['public_core'];generation=context['ticket']['generation']
    a.check(r.D(compact)==r.D(i.projection(r,inv)), 'TRANSITION_RECONSTRUCTED_FULL_RPC_PROJECTION',run)
    a.check(evidence['current_run']==run and public['stage']==STAGE and public['owner']==value['native_owner'] and
        public['worker_identity']==value['worker'] and public['generation']==value['rpc_generation']==generation and
        public['native_generation']==value['native_generation']==generation+int(run=='c3') and
        public['simulation'] is value['simulation'] and public['qualification_granted'] is False,
        'TRANSITION_PUBLIC_RPC_EXACT_SCOPE_NOT_AUTHORITY',run)
    if state is not None:
        a.check(r.D(value['serving'])==r.D(state['serving']) and r.D(inv['native_state'])==r.D(state['native']),
            'TRANSITION_ORIGINAL_NATIVE_AND_SERVING_READBACK',run)


def add_delta(r,i,a,originals,ref,item,context,run,*,natural,rid):
    value=originals.load(ref,mounted=True);scope(r,a,value,ref,context,run,natural=natural)
    inv=value['inventory'];previous=item['refs'];index=len(item['rows'])
    a.check(value['observation_kind']=='COMPLETE_REQUEST_DELTA' and ref['action']=='collect' and
        value['previous_original_sha256']==(previous[-1]['sha256'] if previous else None) and
        all(type(inv['complete_request_delta'][k]) is int for k in ('start','stop')) and type(ref['complete_requests']) is int and
        inv['complete_request_delta']==dict(start=index,stop=index+1,complete_rows=True,complete_native_outcomes=True) and
        len(inv['rows'])==len(value['serving_requests'])==1 and ref['complete_requests']==index+1 and
        inv['rows'][0]['request_id']==value['serving_requests'][0]['actual_internal_request_id']==rid and
        all(x['request_id']==rid for x in inv['outcomes']), 'PER_GENERATION_EXACT_REQUEST_DELTA_SHA_CHAIN',run+':'+rid)
    cov=inv['coverage'];calls=[] if cov is None else cov['calls']
    a.check(cov is None or cov['calls_are_complete_request_delta'] is True and all(x['forward_id']==rid for x in calls),
        'PER_GENERATION_LEDGER_DELTA_NO_FOREIGN_REQUEST',run+':'+rid)
    for name,values in (('rows',inv['rows']),('outcomes',inv['outcomes']),('calls',calls),('serving',value['serving_requests'])):
        item[name].extend(values)
    rebuilt=dict(inv,rows=list(item['rows']),outcomes=list(item['outcomes']));rebuilt.pop('complete_request_delta')
    if cov is not None:
        rebuilt['coverage']=dict(cov,calls=list(item['calls']));rebuilt['coverage'].pop('calls_are_complete_request_delta')
    item['refs'].append(ref);item['latest']=dict(value,inventory=rebuilt,serving_requests=list(item['serving']))
    return value,rebuilt


def close(r,i,a,originals,ref,item,context,run,evidence,*,natural,state=None):
    final=originals.load(ref,mounted=True);scope(r,a,final,ref,context,run,natural=natural)
    inv=final['inventory'];previous=item['refs'];last=item['latest']
    a.check(ref['action']=='cleanup' and final['observation_kind']=='FINAL_COMPLETE_REPORT' and
        final['previous_original_sha256']==(previous[-1]['sha256'] if previous else None) and
        type(ref['complete_requests']) is int and ref['complete_requests']==len(item['rows']) and
        'complete_request_delta' not in inv and
        all(r.D(inv[k])==r.D(item[k]) for k in ('rows','outcomes')) and
        r.D(final['serving_requests'])==r.D(item['serving']) and
        (inv['coverage'] is None and not item['calls'] or inv['coverage'] is not None and
         r.D(inv['coverage']['calls'])==r.D(item['calls']) and 'calls_are_complete_request_delta' not in inv['coverage']),
        'PER_GENERATION_ALL_DELTAS_EXACTLY_RECONSTRUCT_FINAL',run)
    if last is not None:
        before=last['inventory']
        for name in ('candidates','alias_inventory','lifetime','build','public_output_observer'):
            a.check(r.D(inv[name])==r.D(before[name]),'TRANSITION_FINAL_PRESERVES_COMPLETE_OBSERVATION',run+':'+name)
        bridge=before['fia_typed_stock_bridge']
        a.check(r.D(inv['fia_typed_stock_bridge'])==r.D(dict(bridge,stock_target_count=0,retained_host_bytes=0) if bridge is not None else None),
            'TRANSITION_FINAL_PRESERVES_STOCK_BYTES_RELEASES_STORAGE',run)
    rpc(r,i,a,inv,final,evidence,context,run,state=state)
    item['refs'].append(ref);item.update(final=final,final_ref=ref,summary=evidence)
    return final


def new_runs():
    return {run:dict(refs=[],rows=[],outcomes=[],calls=[],serving=[],latest=None,
        final=None,final_ref=None,summary=None,incomplete_requests=[],join_consistent=True) for run in ('c2','c3')}


def join(r,i,a,originals,d,context,runs,*,natural):
    states=context['states']
    seen=set();transition=None
    for driver in d['requests']:
        run=driver['run'];rid=driver['mapping']['internal_request_id'];item=runs[run];index=len(item['rows'])
        before=len(a.issues);was_valid=item['join_consistent'];item['join_consistent']=False
        a.check(rid not in seen and (run=='c2' and transition is None or run=='c3' and transition is not None),
            'UNIQUE_REQUESTS_ISOLATED_C2_THEN_CUMULATIVE_C3',rid);seen.add(rid)
        if 'original_observations' not in driver:
            item['incomplete_requests'].append(rid);a.check(bool(d.get('error')),'MISSING_TRANSITION_DELTA_WITHOUT_FAILURE',rid)
            item['join_consistent']=was_valid and len(a.issues)==before
            continue
        queries=[s for s in states if s['raw']['_reference']==driver['pre_collect_query']]
        arms=[s for s in states if s['action']=='advance' and r.C.rpc_action(s['raw'])[1].get('settings')=={'request_id':rid}]
        a.check(len(queries)==len(arms)==1,'EXACT_TRANSITION_ARM_AND_QUERY',rid);q=queries[0];arm=arms[0]
        following=[s for s in states if s['index']>q['index'] and s['action']=='advance'];collect=following[0]
        a.check(r.C.rpc_action(collect['raw'])[1].get('settings')=={'collect':True},'EXPLICIT_TRANSITION_DELTA_COLLECT',rid)
        state=collect['state'];evidence=state['evidence'];changing=evidence['current_run']!=run
        if changing:
            a.check(run=='c2' and evidence['current_run']=='c3' and driver['phase']=='health' and transition is None and
                index==5+2*len(evidence['prior_c2']['isolated_plan']['target_order']),
                'ONLY_COMPLETE_ISOLATED_C2_HEALTH_CAN_START_C3',rid)
            evidence=evidence['prior_c2'];transition=collect['index']
        refs=evidence['original_observations']['originals'];previous=list(item['refs'])
        a.check(refs[:len(previous)]==previous and len(refs)==len(previous)+(2 if changing else 1) and
            driver['original_observations']==refs[-1] and
            q['state']['evidence']['original_observations']['originals']==previous and
            arm['state']['evidence']['current_run']==q['state']['evidence']['current_run']==run and
            arm['index']<q['index']<collect['index'] and arm['state']['serving']['expected']==rid and
            q['state']['serving']['needs_collect'] is True and state['serving']['needs_collect'] is False and
            len(arm['state']['evidence']['request_index'])==index and len(q['state']['evidence']['request_index'])==index+1,
            'TRANSITION_REAL_ENQUEUE_QUERY_COLLECT_NO_REPLACEMENT',rid)
        raw,rebuilt=add_delta(r,i,a,originals,refs[len(previous)],item,context,run,natural=natural,rid=rid)
        # The query observes the same complete request before collect/cleanup.
        rpc(r,i,a,rebuilt,raw,q['state']['evidence'],context,run)
        a.check(r.D(q['state']['native'])==r.D(rebuilt['native_state']), 'TRANSITION_QUERY_NATIVE_READBACK',rid)
        if changing:
            close(r,i,a,originals,refs[-1],item,context,run,evidence,natural=natural)
            a.check(d.get('c2_attempt')==evidence,'DRIVER_C2_SUMMARY_JOINS_EXACT_ORIGINAL_TRANSITION')
        else:rpc(r,i,a,rebuilt,raw,evidence,context,run,state=state)
        item['join_consistent']=was_valid and len(a.issues)==before
    finals=[s for s in states if s['action']=='finish' and s['reply']['status']=='ACTION_COMPLETED']
    a.check(len(finals)<=1,'ONE_FINAL_TRANSITION_FINISH')
    if finals:
        state=finals[0]['state'];evidence=state['evidence'];item=runs['c3'];refs=evidence['original_observations']['originals']
        before=len(a.issues);was_valid=item['join_consistent'];item['join_consistent']=False
        a.check(evidence['current_run']=='c3' and refs[:-1]==item['refs'],'FINAL_C3_DELTA_CHAIN_NOT_REPLACED')
        close(r,i,a,originals,refs[-1],item,context,'c3',evidence,natural=natural,state=state)
        a.check(d.get('c3_attempt')==evidence,'DRIVER_C3_SUMMARY_JOINS_EXACT_ORIGINAL_FINISH')
        item['join_consistent']=was_valid and len(a.issues)==before
    if transition is not None:
        for state in states:
            evidence=state['state'].get('evidence')
            if state['index']>=transition and evidence is not None:
                valid=evidence.get('prior_c2')==runs['c2']['summary']
                a.check(valid, 'ALL_LATER_RPCS_PRESERVE_PRIOR_C2',state['index'])
                if not valid:
                    for item in runs.values():item['join_consistent']=False
    return runs


def partial_cleanup(r,i,a,originals,d,context,runs,*,natural):
    """Read a failed attempt's actual cleanup without declaring it complete."""
    details={}
    for run,item in runs.items():
        if item['final'] is not None:continue
        candidates=[s for s in context['states'] if s['action'] in ('recover','unregister') and
            s['reply']['status']=='ACTION_COMPLETED' and (s['state'].get('evidence') or {}).get('current_run')==run]
        if not candidates:continue
        state=candidates[-1]['state'];evidence=state['evidence'];refs=evidence['original_observations']['originals']
        if not refs:continue
        # Missing request deltas are not silently substituted by a final report.
        # A forced collect during unwind is retained separately, with its real
        # request association; it is not a successful completed request.
        a.check(refs[:len(item['refs'])]==item['refs'],'PARTIAL_CLEANUP_PRESERVES_COLLECTED_PREFIX',run)
        previous=item['refs'][-1]['sha256'] if item['refs'] else None;raw=None;uncollected=[];failed_originals={}
        for ref in refs[len(item['refs']):]:
            raw=originals.load(ref,mounted=True);scope(r,a,raw,ref,context,run,natural=natural)
            a.check(raw['previous_original_sha256']==previous,'PARTIAL_CLEANUP_ORIGINAL_SHA_CHAIN',run)
            previous=ref['sha256']
            failed=raw.get('failed_request_original');failed_request=None
            if failed is not None:
                failed_request=originals.load(failed,mounted=True);rid=failed_request['request_id']
                drivers=[v for v in d['requests'] if v['run']==run and v['mapping']['internal_request_id']==rid]
                a.check(rid in item['incomplete_requests'] and rid in d.get('owned_request_abort',[]) and
                    len(drivers)==1 and drivers[0]['phase']==failed_request['phase'] and
                    isinstance(failed_request['failure'],str) and bool(failed_request['failure']),
                    'FAILED_REQUEST_ORIGINAL_JOINS_REAL_ARM_ABORT_AND_PHASE',run)
                failed_originals[failed['sha256']]=dict(request_id=rid,phase=failed_request['phase'],
                    failure=failed_request['failure'],recorded_native_events=len(failed_request['native_events']),
                    recorded_candidates=len(failed_request['candidates']),completed_request=False)
            if raw['observation_kind']=='COMPLETE_REQUEST_DELTA':
                ids=[v['request_id'] for v in raw['inventory']['rows']]
                empty_failed=(not ids and failed_request is not None and raw['inventory']['complete_request_delta']==dict(
                    start=len(item['rows']),stop=len(item['rows']),complete_rows=True,complete_native_outcomes=True) and
                    raw['serving_requests']==[] and raw['inventory']['outcomes']==[])
                a.check(ref['action']=='collect' and (bool(ids) and set(ids).issubset(item['incomplete_requests']) or empty_failed),
                    'UNWIND_DELTA_IS_RECORDED_INCOMPLETE_REQUEST',run)
                uncollected.extend(ids)
            else:a.check(raw['observation_kind']=='FINAL_COMPLETE_REPORT' and ref['action']=='cleanup',
                'PARTIAL_CLEANUP_EXPLICIT_FINAL_ORIGINAL',run)
        if raw is None:continue
        inv=raw['inventory'];n=len(item['rows'])
        a.check(raw['observation_kind']=='FINAL_COMPLETE_REPORT' and ref['complete_requests']==len(inv['rows']) and
            r.D(inv['rows'][:n])==r.D(item['rows']) and r.D(raw['serving_requests'][:n])==r.D(item['serving']) and
            r.D(inv['outcomes'][:len(item['outcomes'])])==r.D(item['outcomes']),
            'PARTIAL_FINAL_PRESERVES_COMPLETED_ORIGINAL_REQUESTS',run)
        rpc(r,i,a,inv,raw,evidence,context,run,state=state)
        a.check(inv['installed'] is False and inv['native_removed'] is True and
            all(inv[k]==0 for k in ('owned_hooks','owned_public_api_wrappers','pending_output_tensors','pending_residual_tensors','pending_public_storage_owners')) and
            not inv['pending_outcomes'] and not inv['pending_per_call_leases'] and
            all(inv['native_state']['core'][k]==0 for k in ('installed','pending','executing','device_bytes','host_bytes','descriptor_pending','error')),
            'PARTIAL_FINAL_ORIGINAL_OWNED_RESOURCES_ZERO',run)
        details[run]=dict(final_original_sha256=ref['sha256'],original_complete_rows=len(inv['rows']),
            collected_before_failure=n,unwind_collected_incomplete_requests=uncollected,failed_request_originals=failed_originals,
            additional_owned_outcomes_not_completed_requests=len(inv['outcomes'])-len(item['outcomes']),group_completion_reconciled=False)
    return details


def isolated_complete(r,i,a,b,d,final,ref,originals,context,*,mechanism,expected_count,natural,native_build_binding,partial):
    inv=final['inventory'];details={};artifacts=r.GroupArtifacts(originals,ref)
    a.check(inv['schema']=='demoncat.writer26.complete_set.v2' and inv['stage']=='coverage' and inv['mechanism']==mechanism and
        inv['phase']=='complete' and inv['finalized'] is True and inv['failure'] is inv['first_failure'] is None and
        not inv['failure_events'] and not inv['blockers'],'ISOLATED_C2_COMPLETE_BEFORE_TRANSITION_NO_ERASED_FAILURE')
    details['request_outputs']=a.section('c2:requests',i.requests,r,a,b,d,final,context['worker'],natural=natural,require_post_health=not partial)
    values=a.section('c2:isolated_native',i.native,r,a,b,inv,context,artifacts,natural=natural,expected_count=expected_count,transition=True)
    if values:
        frozen,eligible,outcomes,boundaries=values;details['outside_request_native_boundaries']=boundaries
        details['coverage']=a.section('c2:isolated_ledger',i.ledger,r,a,b,inv,frozen,eligible,outcomes,artifacts,context,
            natural=natural,native_build_binding=native_build_binding,execution_scope=SCOPE)
        a.check(d['isolated_plan']==inv['isolated_plan'],'TRANSITION_DRIVER_PLAN_EQUALS_COMPLETE_C2_CENSUS')
    if natural and mechanism=='26.fia_destination':
        typed=r.OLD.helper('review_native26_fia_single_source62.py')
        details['typed_fia']=a.section('c2:typed',r.N.typed_group_review,b,typed,inv,typed.FiaArtifacts(b,originals.root),
            expected_count=expected_count,isolated=True,isolated_scope=SCOPE)
        a.check(details['typed_fia'] is not None and details['typed_fia']['consistent'],'TRANSITION_C2_ALL_TYPED_AND_STOCK_BYTES')
    a.check(inv['installed'] is False and inv['native_removed'] is True and inv['isolated_logical_inflight']==0 and
        all(inv[k]==0 for k in ('owned_hooks','owned_public_api_wrappers','pending_output_tensors','pending_residual_tensors','pending_public_storage_owners')) and
        not inv['pending_outcomes'] and not inv['pending_per_call_leases'] and all(inv['native_state']['core'][k]==0 for k in
        ('installed','pending','executing','device_bytes','host_bytes','descriptor_pending','error')), 'TRANSITION_C2_RESOURCES_ZERO_BEFORE_NEW_GENERATION')
    return details


def cumulative_scope(r,a,inv,expected_count):
    keys=[v['target_key'] for v in inv['candidates'] if v['status']=='OBSERVED_NOT_QUALIFIED'];count=len(keys)
    a.check(inv['execution_scope']==inv['scope']==SCOPE and inv['independent_per_target_replay'] is False and
        inv['coverage']['trigger']=='every_match' and inv['isolated_pairs']==[] and inv['isolated_logical_inflight']==0 and
        inv['isolated_plan']==dict(target_order=keys,group_requests=9,ordinary_post_unregister_requests=1,
            full_native_writer_count=expected_count,native_owned_execute_records=6*count,max_targets_is_selector=False,c3_requested=True,
            c2_group_requests=6+2*count,c3_group_requests=9,combined_group_requests=15+2*count,current_phase_scope=C3_SCOPE),
        'EXACT_C3_CUMULATIVE_NOT_ISOLATED_REPLAY_SCOPE_AND_PLAN')


def review(r,d,host,records,originals,*,mechanism,expected_count,natural,native_build_binding=None):
    i,t=helpers(r);C=r.C;a=C.Audit();b=r.OLD.helper('review_native26_single_source54.py');details={}
    a.check(d['stage']==STAGE,'EXPLICIT_ISOLATED_C2_CUMULATIVE_C3_STAGE')
    context=a.section('protocol',C.protocol,a,d,host,records,mechanism,natural=natural)
    if not context:return a.report()
    protocol_consistent=not a.issues;runs=new_runs()
    joined=a.section('two_generation_deltas',join,r,i,a,originals,d,context,runs,natural=natural)
    C.final_cleanup(a,d,host,context)
    partial=bool(d.get('error') or d.get('cleanup_error') or d.get('abort_error') or d.get('blocked_run'))
    if joined is None:partial=True
    if partial:
        details['partial_cleanup']=a.section('partial_cleanup_originals',partial_cleanup,r,i,a,originals,d,context,runs,natural=natural)
    for run,item in runs.items():
        drivers=[v for v in d['requests'] if v['run']==run];final=item['final']
        details[run]=dict(collected_requests=len(item['rows']),incomplete_requests=item['incomplete_requests'],complete_originals_reconciled=False)
        if final is not None:
            before=len(a.issues);local=dict(d,requests=drivers)
            if run=='c2':
                checked=a.section('complete_isolated_c2',isolated_complete,r,i,a,b,local,final,item['final_ref'],originals,context,
                    mechanism=mechanism,expected_count=expected_count,natural=natural,native_build_binding=native_build_binding,partial=partial)
            else:
                a.section('c3:cumulative_scope',cumulative_scope,r,a,final['inventory'],expected_count)
                checked=a.section('complete_cumulative_c3',r.complete_group,a,b,local,final,item['final_ref'],originals,dict(context,execution_scope=SCOPE),
                    mechanism=mechanism,expected_count=expected_count,natural=natural,run=run,native_build_binding=native_build_binding,transition_isolated=True)
            details[run].update(checked or {})
            details[run]['complete_originals_reconciled']=checked is not None and len(a.issues)==before and item['join_consistent'] and protocol_consistent
        else:
            partial=True;latest=item['latest'];counts=[]
            if latest is not None:
                inv=latest['inventory']
                for index,row in enumerate(inv['rows']):
                    actual=a.section(run+':partial_trace',b.trace,a,row,inv['lifetime'],str(index),mechanism=1 if mechanism=='26.residual_stomp' else 2)
                    if actual is not None:
                        counts.append(len(actual));a.check(len(actual)==len(row['candidates'])==expected_count,'PARTIAL_COMPLETE_NATIVE_DENOMINATOR',run+':'+str(index))
                details[run].update(observed_complete_request_native_counts=counts,completed_phases=[v['phase'] for v in inv['rows']],
                    recorded_owned_outcomes=len(inv['outcomes']),recorded_fault_outcomes=sum(v['mode']=='fault' for v in inv['outcomes']))
    if all(details.get(x,{}).get('coverage') for x in ('c2','c3')):
        # Different request plans, but the same exact live object, scope, owner,
        # denominator and target identities must bridge the two generations.
        first=dict(details['c2']['coverage']);first['projection_calls_per_request']=first['full_calls_per_request']
        details['transition']=a.section('same_original_live_transition',t.transition,r,a,runs['c2']['final'],runs['c3']['final'],first,details['c3']['coverage'])
    elif not partial:a.check(False,'BOTH_GENERATIONS_REQUIRE_COMPLETE_INDEPENDENT_LEDGER_AUDITS')
    if not partial:
        keys=runs['c2']['final']['inventory']['isolated_plan']['target_order'];phases=i.plan(keys)
        a.check([(x['run'],x['phase']) for x in d['requests']]==[('c2',p['phase']) for p in phases]+[('c3',p) for p in r.C3_PHASES],
            'ALL_ISOLATED_AND_CUMULATIVE_REQUESTS_IN_ORDER')
        a.check(context['actions']==['attach','prepare','register','check','enable']+['advance','query','advance']*(len(phases)+9)+
            ['finish','disable','unregister','detach'],'EXACT_ISOLATED_TRANSITION_PUBLIC_PROTOCOL')
    return dict(**a.report(),details=details,classification='FAILED_OR_PARTIAL_NOT_QUALIFIED' if partial else
        'SAVED_STANDARD_WORKER_ISOLATED_C2_CUMULATIVE_C3_ORIGINALS' if natural else 'CPU_ISOLATED_C2_CUMULATIVE_C3_CONTRACT_ONLY',
        group_completion_reconciled=not partial and not a.issues,originals_rehashed=len(originals.checked),
        completed_original_requests=sum(len(v['rows']) for v in runs.values()),original_error=d.get('error'),
        original_cleanup_error=d.get('cleanup_error'),aborted_request=d.get('owned_request_abort'),blocked_run=d.get('blocked_run'),
        limits=['Saved originals document a historical transition; they never create live C2/C3 authority or qualify another scope.',
            'C2 isolates one target per fresh F/OFF pair; C3 admits all eligible targets within each of two fresh requests.',
            'A failed later stage retains completed-stage evidence without becoming complete C3.',
            'FIA full stock/writer bytes are rehashed in natural mode; full residual/logits equality remains a runtime digest observation.',
            'E5 requires independent raw statistics and is not inferred from these Native or coverage records.'])
