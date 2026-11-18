"""Offline original-receipt audit for complete-set Worker Native26 C2/C3.

No saved successful flag grants authority. Native/model code is never imported.
The two mechanisms share the full census and public RPC contract, but FIA also
requires typed descriptors and every original stock/owned output byte artifact.
"""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import sys
import json


def support():
    import importlib.util
    path=Path(__file__).with_name('group_review_common.py')
    spec=importlib.util.spec_from_file_location('_group_common26',path)
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    old=common.pure_module(Path(__file__).parents[1]/'c2c3_closure_20261117/review26_worker_core_result.py','_old26group')
    native=common.pure_module(Path(__file__).with_name('review26_group_native.py'),'_native26group')
    return common,old,native


C,OLD,N=support()
def D(value):
    """Writer26's original JSON digest uses default separators, unlike Core."""
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()
PHASES=('inventory_1','inventory_2','qualification_reference_all','qualification_lossless_all',
        'qualification_off_all','group_fault','group_off','health')
MODES=(None,None,'reference','lossless','off','fault','off',None)
C3_PHASES=PHASES[:6]+('group_fault_repeat',)+PHASES[6:]
C3_MODES=MODES[:6]+('fault',)+MODES[6:]


def profile(inv):
    trigger=inv['coverage']['trigger']
    if trigger not in ('once','every_match'):raise ValueError('EXPLICIT_GROUP_TRIGGER_REQUIRED')
    return (C3_PHASES,C3_MODES) if trigger=='every_match' else (PHASES,MODES)


def native_generation(context):
    return context.get('native_generation',context['ticket']['generation'])


def projection(inv):
    result=OLD.project_inventory(inv)
    if inv.get('coverage') is not None:
        full=inv['coverage']; compact={k:v for k,v in full.items() if k not in ('calls','discovery','observation')}
        compact.update({k+'_original_sha256':D(full[k]) for k in ('calls','discovery','observation')})
        compact.update(compact_rpc_projection=True,original_report_sha256=D(full))
        result['coverage']=compact
    return result


def load_original(a,originals,ref,state,context,*,natural,evidence=None,run=None):
    value=originals.load(ref,mounted=True);inv=value['inventory']
    a.check(value['schema']=='demoncat.writer26.core_observations.v1' and value['worker']==context['worker'] and
        value['native_owner']==context['provider_owner'] and C.positive(value['generation']) and value['generation']==context['ticket']['generation'] and
        value['action']==ref['action'] and value['simulation'] is (not natural) and value['qualification_granted'] is False,
        'ORIGINAL_COMPLETE_GROUP_SCOPE')
    if run is not None or 'run' in value:
        run=run or 'c2'
        a.check(value['run']==run and C.positive(value['rpc_generation']) and C.positive(value['native_generation']) and
            value['rpc_generation']==value['generation']==context['ticket']['generation'] and
            value['native_generation']==context['ticket']['generation']+int(run=='c3'), 'ORIGINAL_TWO_GENERATIONS_EXACT_JOIN')
    if evidence is None and state is not None:evidence=state['evidence']
    if evidence is not None:
        compact={k:v for k,v in evidence.items() if k not in ('public_core','original_observations','current_run','prior_c2')}
        a.check(compact==projection(inv),'ORIGINAL_TO_COMPACT_RPC_EXACT_JOIN')
        if 'current_run' in evidence:a.check(evidence['current_run']==(run or 'c2'),'ORIGINAL_RPC_RUN')
        public=evidence['public_core']
        a.check(public['worker_identity']==value['worker'] and public['owner']==value['native_owner'] and
            public['generation']==value['generation'] and public['native_generation']==value.get('native_generation',value['generation']) and
            public['simulation'] is (not natural) and public['qualification_granted'] is False,'ORIGINAL_PUBLIC_NATIVE_GENERATION')
    if state is not None:
        a.check(value['serving']==state['serving'] and state['native']==inv['native_state'],'ORIGINAL_NATIVE_SERVING_READBACK')
    a.check(ref['complete_requests']==len(inv['rows'])==len(value['serving_requests']),'ORIGINAL_FULL_REQUEST_COUNT')
    return value


def originals_join(a,originals,d,context,*,natural):
    previous=[];snapshots=[];states=context['states']; incomplete=[]
    for index,driver in enumerate(d['requests']):
        rid=driver['mapping']['internal_request_id']
        if 'original_observations' not in driver:
            incomplete.append(rid);a.check(bool(d.get('error')),'MISSING_REQUEST_ORIGINAL_WITHOUT_FAILURE',rid);continue
        query=[x for x in states if x['raw']['_reference']==driver['pre_collect_query']]
        arm=[x for x in states if x['action']=='advance' and C.rpc_action(x['raw'])[1].get('settings')=={'request_id':rid}]
        a.check(len(query)==len(arm)==1,'EXACT_REQUEST_ARM_AND_QUERY',rid)
        q=query[0]; following=[x for x in states if x['index']>q['index'] and x['action']=='advance']
        a.check(bool(following) and C.rpc_action(following[0]['raw'])[1].get('settings')=={'collect':True},
                'EXPLICIT_COLLECT_FOLLOWS_QUERY',rid)
        state=following[0]['state'];refs=state['evidence']['original_observations']['originals']
        a.check(q['state']['evidence']['original_observations']['originals']==previous and refs[:-1]==previous and
            refs[-1]==driver['original_observations'] and len(refs)==index+1 and
            arm[0]['index']<q['index'] and arm[0]['state']['serving']['expected']==rid and
            len(arm[0]['state']['evidence']['request_index'])==index and
            len(q['state']['evidence']['request_index'])==index+1 and
            q['state']['serving']['needs_collect'] is True and state['serving']['needs_collect'] is False,
            'RPC_CANNOT_REPLACE_ACTUAL_ENQUEUED_FORWARD',rid)
        value=load_original(a,originals,refs[-1],state,context,natural=natural)
        if snapshots:
            a.check(value['inventory']['rows'][:-1]==snapshots[-1]['inventory']['rows'] and
                value['serving_requests'][:-1]==snapshots[-1]['serving_requests'],'IMMUTABLE_REQUEST_PREFIX',rid)
        previous=refs;snapshots.append(value)
    finals=[x for x in states if x['action']=='finish' and x['reply']['status']=='ACTION_COMPLETED']
    final=None
    if finals:
        state=finals[-1]['state'];refs=state['evidence']['original_observations']['originals']
        a.check(refs[:-1]==previous and refs[-1]['action']=='cleanup','FINAL_CLEANUP_ORIGINAL_NOT_REPLACED')
        final=load_original(a,originals,refs[-1],state,context,natural=natural)
        if snapshots:
            a.check(final['inventory']['rows']==snapshots[-1]['inventory']['rows'] and
                final['serving_requests']==snapshots[-1]['serving_requests'],'CLEANUP_PRESERVES_ACTUAL_REQUESTS')
        a.check(d.get('c2_attempt')==state['evidence'],'DRIVER_SUMMARY_JOINS_ORIGINAL_FINISH')
    return dict(final=final,snapshots=snapshots,incomplete_requests=incomplete,
                final_ref=finals[-1]['state']['evidence']['original_observations']['originals'][-1] if final else None)


def current_absence(b,a,candidate,row,lifetime):
    status,reasons=b.classify(candidate);proof=candidate.get('current_forward_residual_absence')
    if proof is None:return status,reasons
    allowed={'ACTUAL_PUBLIC_PRODUCER_LIFETIME_UNPROVEN','CURRENT_PUBLIC_RESIDUAL_LIFETIME_UNPROVEN',
             'NEXT_ACTUAL_NATIVE_CONSUMER_UNPROVEN','PUBLIC_CONSUMER_LIFETIME_NOT_PROVEN'}
    events=row['native_events'];before,after=row['native_before'],row['native_after'];native=candidate['native'];cb=candidate['callback']
    earlier=[x for x in events if x['kind_name'].startswith('add_rms_') and x['sequence']<native['sequence']]
    later=[x for x in events if x['kind_name']=='add_rms_exit' and x['sequence']>native['sequence']]
    valid=(status=='UNVERIFIED' and bool(reasons) and set(reasons).issubset(allowed) and
        not any(x[k] for x in (before,after) for k in ('overflow_events','errors')) and
        all(x['stored_events']==x['total_events'] for x in (before,after)) and
        [x['sequence'] for x in events]==list(range(before['stored_events']+1,after['stored_events']+1)) and
        all(x['thread']==lifetime['native_thread'] for x in events) and bool(cb) and
        cb['actual_request_id']==row['request_id'] and cb['original_continuations']==1 and cb['claimed']==0 and
        not earlier and bool(later) and later[0]['status']==0 and later[0]['producer']==native['producer']+1 and
        len(candidate['execution'])==1 and candidate['execution'][0]['status']==0)
    a.check(valid,'INDEPENDENT_CURRENT_FORWARD_ABSENCE',row['request_id'])
    expected=dict(schema='demoncat.writer26.current_forward_residual_absence.v1',request_id=row['request_id'],
        complete_native_events_sha256=D(events),native_before=before,native_after=after,writer_sequence=native['sequence'],
        inherited_producer=native['producer'],next_actual_producer=later[0] if later else None,original_unresolved_reasons=reasons,
        earlier_current_forward_producer_count=0,qualification_granted=False)
    a.check(proof==expected,'ABSENCE_PROOF_BINDS_COMPLETE_CURRENT_NATIVE_TRACE')
    return ('UNSUPPORTED',['CURRENT_FRESH_FORWARD_HAS_NO_RESIDUAL_PRODUCER']) if valid else (status,reasons)


class GroupArtifacts:
    def __init__(self,originals,final_ref,*,run='c2'):
        path=final_ref['path']
        if path.startswith('/evidence/'):path=path[len('/evidence/'):]
        elif not originals.natural:path=Path(path).resolve().relative_to(originals.root).as_posix()
        else:raise ValueError('EXPLICIT_ORIGINAL_MOUNT_REQUIRED')
        if run not in ('c2','c3'):raise ValueError('EXPLICIT_ORIGINAL_GROUP_RUN_REQUIRED')
        self.folder=(Path(path).parents[2]/('candidate-experiment-c3' if run=='c3' else 'candidate-experiment')).as_posix();self.originals=originals
        self.checked={}

    def load(self,name,expected_sha=None,expected=None):
        path=C.under(self.originals.root,self.folder+'/'+name+'.json');sha=C.file_sha(path)
        if expected_sha is not None and expected_sha!=sha:raise ValueError('GROUP_ARTIFACT_SHA_MISMATCH:'+name)
        value=C.read(path)
        if expected is not None and value!=expected:raise ValueError('GROUP_ARTIFACT_CANONICAL_OBJECT_MISMATCH:'+name)
        self.checked[name]=sha;return value

    def health(self,sha):
        folder=C.under(self.originals.root,self.folder+'/complete-candidate-set.json').parent
        files=list(folder.glob('worker-health-*.json'))
        if len(files)>64:raise ValueError('HEALTH_ARTIFACT_SEARCH_BUDGET_EXCEEDED')
        matches=[p for p in files if C.file_sha(p)==sha]
        if len(matches)!=1:raise ValueError('EXACT_UNIQUE_HEALTH_SHA_REQUIRED')
        return self.load(matches[0].stem,sha)


def rebuild_identity(inv,frozen,worker):
    native,cb=frozen['second_native'],frozen['callback'];unowned=cb is None
    module=inv['lifetime']['model_object_identity'] if unowned else cb['module_object_identity']
    signature=D({k:native[k] for k in ('dtype','format','dimensions','output_bytes','residual_bytes','shape','residual_shape')})
    method='UNOWNED_NATIVE_BOUNDARY_IN_ACTUAL_ROOT_FORWARD' if unowned else D(cb['method'])
    common=dict(process_id=inv['lifetime']['pid'],process_start_identity=inv['lifetime']['process_start_identity'],
        worker_id=worker['worker_start_identity'],rank=worker['rank'],logical_device=worker['logical_device'],
        model_instance_id=worker['model_instance_id'],model_object_identity=inv['lifetime']['model_object_identity'],
        module_object_identity=module,module_roles=[],tensor_signature=signature,provider_identity='native26.group.v2',
        provider_proof=[['method_identity',method],['native_build',inv['build']]])
    if inv['mechanism']=='26.residual_stomp':
        return dict(common,operator_api='aclnnQuantMatmulWeightNz',execution_boundary='op_exec_cache_run',
            executor_identity='opcache:'+str(native['cache']),cache_lifecycle='actual_registration_epoch:'+str(native['registration']),
            native_build_fingerprint=inv['build'])
    operator='aclnnFusedInferAttentionScoreV3'
    common['module_roles']=['native_unowned_in_root'] if unowned else []
    return dict(common,operator_api=operator,scope_kind='INITIALIZED_WORKER',device_type='npu',
        boundary=dict(kind='operator_api',implementation_kind='public_aclnn',schema_sha256=inv['build'],
            implementation_sha256=inv['lifetime']['native_writer_sha256'],call_token='root-unowned:'+D([operator,native['cache'],signature])
                if unowned else 'module-writer:'+str(module)+':'+method))


def native_group(b,a,inv,worker,context,artifacts,*,natural,expected_count):
    rows=inv['rows']; lifetime=inv['lifetime'];mechanism=1 if inv['mechanism']=='26.residual_stomp' else 2
    phases,modes=profile(inv);off_index=phases.index('group_off');generation=native_generation(context)
    a.check(len(rows)==len(phases),'FULL_GENERATION_PHASE_DENOMINATOR')
    aliases={(x['path'],x['object_identity']) for x in inv['alias_inventory']}
    a.check(len(aliases)==len(inv['alias_inventory']) and ('<root>',lifetime['model_object_identity']) in aliases,
            'COMPLETE_MODULE_ALIASES')
    if natural:
        a.check(lifetime['worker']==worker and lifetime['pid']==worker['pid'] and
            lifetime['process_start_identity']==worker['process_start_identity'],'ACTUAL_WORKER_NATIVE_LIFETIME')
    rebuilt=[];seen=set()
    for index,row in enumerate(rows):
        label=row['phase']; native=b.trace(a,row,lifetime,label,mechanism=mechanism)
        if index:
            a.check(row['native_before']==rows[index-1]['native_after'],'NO_UNACCOUNTED_NATIVE_CALLS_BETWEEN_REQUESTS',label)
        a.check(len(native)==len(row['candidates'])==expected_count and not seen.intersection(x['call'] for x in native),
                'FULL_NATIVE_WRITER_DENOMINATOR',label);seen.update(x['call'] for x in native)
        correlated=b.correlate(row,mechanism=mechanism)
        a.check([{k:v for k,v in x.items() if k!='current_forward_residual_absence'} for x in row['candidates']]==correlated,
                'INDEPENDENT_NATIVE_PUBLIC_CONSUMER_CORRELATION',label)
        a.check(row['complete_inventory']==dict(native_writer_count=expected_count,native_events_sha256=D(row['native_events']),
            candidates_sha256=D(row['candidates']),actual_aliases_sha256=D(inv['alias_inventory'])),
            'COMPLETE_INVENTORY_HASHES',label)
        if natural:
            for producer in row['producers']:b.public_producer(a,producer,row,aliases,label)
            for candidate in row['candidates']:b.callback_proof(a,candidate,row,lifetime,aliases,b.Artifacts(artifacts.originals.root),label,mechanism=mechanism)
        for candidate in row['candidates']:current_absence(b,a,candidate,row,lifetime)
        rebuilt.append(row['candidates'])
    frozen=[]
    for index,(left,right) in enumerate(zip(rebuilt[0],rebuilt[1])):
        ls,_=current_absence(b,a,left,rows[0],lifetime);status,reasons=current_absence(b,a,right,rows[1],lifetime)
        cb=right['callback'];key=cb.get('target_key') if cb else None
        a.check(ls==status and all(left['native'][k]==right['native'][k] for k in b.STABLE_NATIVE),
                'HEALTHY_COMPLETE_TARGET_SIGNATURE',index)
        if status=='OBSERVED_NOT_QUALIFIED':
            a.check(left['callback'] and left['callback']['target_key']==key and all(
                sum(c['callback'] is not None and c['callback'].get('target_key')==key for c in group)==1 for group in rebuilt[:2]),
                'ONE_STABLE_TARGET_OCCURRENCE',index)
        a.check(status in ('OBSERVED_NOT_QUALIFIED','UNSUPPORTED'),'UNRESOLVED_TARGET_BLOCKS_WHOLE_GROUP',index)
        candidate=dict(index=index,target_key=key,status=status,reasons=reasons,first_native=left['native'],
                       second_native=right['native'],callback=cb)
        candidate['identity']=rebuild_identity(inv,candidate,worker);frozen.append(candidate)
    a.check(inv['candidates']==frozen,'FROZEN_TARGET_SET_REBUILT_FROM_TWO_HEALTHY_TRACES')
    eligible={x['target_key']:x for x in frozen if x['status']=='OBSERVED_NOT_QUALIFIED'}
    a.check(bool(eligible) and inv['resolved_candidate_count']==inv['qualified_candidate_count']==len(eligible),
            'ALL_RESOLVED_TARGETS_QUALIFIED')
    outcomes={};native_owner=b.words(hashlib.sha256(context['provider_owner'].encode()).hexdigest())[:2]
    for outcome in inv['outcomes']:
        key=(outcome['request_id'],outcome['target']);a.check(key not in outcomes,'ONE_OUTCOME_PER_TARGET_PER_REQUEST');outcomes[key]=outcome
    for index,row in enumerate(rows):
        mode=modes[index];count=len(eligible) if mode else 0
        a.check(row['plan']==dict(phase=phases[index],mode=mode,target='*' if mode else None) and
            row['admissions']==row['completed']==count and sum(k[0]==row['request_id'] for k in outcomes)==count,
            'EVERY_TARGET_REFERENCE_LOSSLESS_FAULT_OFF',row['phase'])
        for candidate in row['candidates']:
            cb=candidate['callback'];target=cb.get('target_key') if cb else None
            selected=bool(mode and target in eligible)
            a.check(not cb or cb['claimed']==int(selected),'CURRENT_NATIVE_ADMISSION_NOT_HIT_LOG',row['phase'])
            if not selected:continue
            effect=outcomes[(row['request_id'],target)]
            if natural:
                N.group_outcome(b,a,effect,candidate,row,dict(target_key=target,native_owner=native_owner,
                    generation=generation),row['phase']+':'+target,mechanism=mechanism)
            a.check(effect['mode']==mode and effect['native_call']==cb['native_packet'] and
                effect['request_id']==row['request_id'] and effect['public_consumer_lifetime'] is True and
                effect['residual_before_sha256']==effect['residual_after_sha256'], 'TARGET_OUTCOME_CURRENT_CALL_AND_RECOVERY')
    for target in eligible:
        q=[outcomes[(rows[i]['request_id'],target)] for i in (2,3,4,off_index)]
        a.check(len({x['writer_output_sha256'] for x in q})==len({x['residual_before_sha256'] for x in q})==1,
                'COMPLETE_PER_TARGET_REFERENCE_LOSSLESS_OFF_EQUAL',target)
        if 'group_fault_repeat' in phases:
            first=outcomes[(rows[5]['request_id'],target)];second=outcomes[(rows[6]['request_id'],target)]
            a.check(first['writer_output_sha256']==second['writer_output_sha256'] and
                first['residual_before_sha256']==second['residual_before_sha256'],'EVERY_TARGET_FAULT_REPEAT_EXACT',target)
    if natural:
        state=inv['native_state'];core=state['core']
        a.check([core['owner_hi'],core['owner_lo']]==native_owner and core['generation']==generation and
            core['thread']==lifetime['native_thread'] and core['mechanism']==mechanism and core['calls']==sum(x is not None for x in modes)*len(eligible) and
            state['admitted_calls']==state['completion_calls']==core['calls'],'FINAL_REAL_NATIVE_OWNER_AND_CALL_COUNTS')
        b.runtime_state(a,state,dict(native_owner=native_owner,generation=generation),
            lifetime['native_thread'],sum(len(x['callbacks']) for x in rows),len(outcomes),
            removed=True,sealed=True,where='complete_group_final',mechanism=mechanism)
        b.metadata(a,[('initial',inv['metadata_arrays_initial'])]+[(x['phase'],x['metadata_arrays']) for x in rows]+
            [('final',inv['metadata_arrays_latest'])],lifetime,inv['build'])
    return frozen,eligible,outcomes


def ledger(a,b,inv,frozen,eligible,outcomes,artifacts,context,*,natural):
    cov=inv['coverage'];rows=inv['rows'];owner=context['provider_owner'];generation=native_generation(context)
    phases,modes=profile(inv);off_index=phases.index('group_off');faults=modes.count('fault')
    a.check(cov['scope']['mechanism']==inv['mechanism'] and cov['scope']['evidence_kind']==
        ('natural_worker' if natural else 'cpu_contract') and cov['owner']==owner and cov['generation']==generation and
        cov['trigger']==('every_match' if context.get('run')=='c3' else 'once') and cov['selection']=='all_eligible' and cov['budget']['max_concurrency']==1 and
        inv['execution_scope']=='SAME_REQUEST_CUMULATIVE_ALL_STABLE_ELIGIBLE_TARGETS' and
        inv['independent_per_target_replay'] is False,'EXACT_COMPLETE_GROUP_SCOPE')
    candidates=[dict(candidate_id=x['target_key'] or 'unsupported-native-'+str(i),identity=x['identity'],
        eligibility=dict(status='SUPPORTED' if x['target_key'] in eligible else 'UNSUPPORTED',reasons=x['reasons']))
        for i,x in enumerate(frozen)]
    expected=dict(candidates=candidates,discovery_complete=True,eligible=len(eligible),discovered=len(candidates),
        eligibility_counts={s:sum(x['eligibility']['status']==s for x in candidates) for s in
            ('SUPPORTED','UNSUPPORTED','UNVERIFIED','AMBIGUOUS','NOT_OBSERVED','RESOURCE_LIMITED')},execution_authorized=False)
    artifacts.load('complete-candidate-set',expected=expected)
    artifacts.load('full-healthy-discovery',cov['discovery']['artifact_sha256'],rows[:2])
    artifacts.load('full-native-public-observation',cov['observation']['artifact_sha256'],rows)
    a.check(cov['discovery']['candidates_sha256']==C.digest(expected) and cov['discovery']['observed_target_ids']==
        sorted(C.digest(x['identity']) for x in frozen),'FULL_INDEPENDENT_CANDIDATE_ARTIFACTS')
    recovery=cov['recovery'];recovered={C.digest(x['lease']['target']):x for x in recovery['targets']}
    target_ids={x['target_key']:C.digest(x['identity']) for x in frozen if x['target_key'] in eligible}
    a.check(len(recovered)==len(recovery['targets'])==len(eligible) and set(recovered)==set(target_ids.values()),
            'ALL_ELIGIBLE_TARGET_RECOVERIES')
    for key,target in eligible.items():
        reference=outcomes[(rows[2]['request_id'],key)];off=outcomes[(rows[off_index]['request_id'],key)]
        recovered_row=recovered[target_ids[key]];lease=recovered_row['lease'];call=reference['native_call']
        expected_lease=dict(target=target['identity'],owner=owner,generation=generation,
            lease_id=D([owner,generation,key]),device_bytes=call['output_bytes'],
            host_bytes=2*call['residual_bytes']+3*call['output_bytes']+8192)
        a.check(lease==expected_lease and recovered_row['drained'] is recovered_row['released'] is recovered_row['off_exact'] is True and
            recovered_row['pending_reason'] is None and recovered_row['off_output_sha256']==reference['writer_output_sha256'],
            'EXACT_TARGET_LEASE_AND_OFF',key)
        artifacts.load('lossless-'+key,expected=outcomes[(rows[3]['request_id'],key)])
        saved=artifacts.load('recovery-'+key,recovered_row['proof_sha256'])
        a.check(saved['lease']==lease and saved['off']==off and all(saved['native_state']['core'][k]==0 for k in
            ('installed','pending','executing','device_bytes','host_bytes','descriptor_pending','error')),
            'ORIGINAL_NATIVE_OFF_RELEASE_ARTIFACT',key)
    expected_calls=[]
    for row in rows[5:off_index+1]:
        # Callbacks are observed at entry. Truly unowned Native calls are joined
        # at request completion; their original trace position is retained.
        ordered=[x for x in row['candidates'] if x['callback'] is not None]+[x for x in row['candidates'] if x['callback'] is None]
        expected_calls.extend((row,x) for x in ordered)
    calls=cov['calls'];a.check(len(calls)==len(expected_calls),'COMPLETE_LEDGER_INCLUDES_UNSUPPORTED_CALLS')
    tickets=[];hits=0;anomaly=False
    for index,(call,(row,candidate)) in enumerate(zip(calls,expected_calls)):
        ordinal=row['candidates'].index(candidate);frozen_row=frozen[ordinal];identity=frozen_row['identity']
        cb=candidate['callback'];key=cb.get('target_key') if cb else None;chosen=key in eligible;off=row['phase']=='group_off'
        hit=chosen and not off;phase='OFF_RECOVERY' if off else 'POST_FIRST_ANOMALY' if anomaly else 'PRE_FAULT_PROPAGATION'
        tid=C.digest(identity);call_id=str(candidate['native']['call']);worker=[identity[k] for k in
            ('process_id','process_start_identity','worker_id','rank')]
        if 'boundary' in identity:worker.append(identity['device_type'])
        worker.append(identity['logical_device'])
        ticket=dict(observer_id=cov['discovery']['observer_id'],sequence=index,eligible=chosen,propagation_added=False,
            call=dict(target=identity,call_id=call_id,forward_context=row['request_id'],request_ids=[row['request_id']],phase=phase))
        tickets.append(ticket);decision=call['decision'];receipt=call['receipt']
        a.check(call['call_id']==call_id and call['target_id']==tid and call['forward_id']==row['request_id'] and
            call['request_ids']==[row['request_id']] and call['phase']==phase and call['eligible'] is chosen and
            call['worker']==worker and call['model_instance']==identity['model_instance_id'] and
            call['operator']==identity['operator_api'] and call['rank']==identity['rank'] and
            call['device']==identity['logical_device'] and call['owner']==owner and call['generation']==generation and
            call['propagation_added'] is False and decision['ticket']==ticket and decision['session_id']==cov['session_id'] and
            decision['admitted'] is hit and decision['reason']==('OFF_RECOVERY' if off else 'ADMITTED' if chosen else 'NON_TARGET') and
            decision['lease']==(recovered[tid]['lease'] if chosen else None) and receipt['decision']==decision and
            receipt['native_applied'] is hit and receipt['original_calls']==1 and
            receipt['outcome']==('HIT' if hit else 'PASSTHROUGH'),'INDEPENDENT_ACTUAL_CALL_ADMISSION',call_id)
        value=outcomes.get((row['request_id'],key)) if chosen else candidate
        artifacts.load(('execution-' if chosen else 'passthrough-')+call_id,receipt['execution_sha256'],value)
        a.check(receipt['effect_sha256']==(receipt['execution_sha256'] if hit else None),'ORIGINAL_EFFECT_REFERENCE',call_id)
        if hit:hits+=1;anomaly=True
    a.check(cov['hits']==hits==faults*len(eligible) and cov['planned_targets']==cov['planned_targets_hit']==len(eligible) and
        cov['observed_calls']==len(calls) and cov['observed_eligible_calls']==faults*len(eligible) and
        not cov['missing_outcomes'] and not cov['propagation_added_calls'] and
        cov['observation']['calls_sha256']==C.digest(tickets) and cov['observation']['complete'] is True and
        cov['observation']['dropped_calls']==0,'RECOMPUTED_COMPLETE_GROUP_TOTALS')
    a.check(recovery['state']=='DISABLED_CONFIRMED' and recovery['inflight']==0 and
        not any(recovery[k] for k in ('errors','held_leases','pending_drains','pending_leases')) and len(recovery['workers'])==1,
        'ALL_OWNED_GROUP_RESERVATIONS_CLOSED')
    health=recovery['workers'][0];raw=artifacts.health(health['proof_sha256'])
    a.check(raw['baseline']==raw['off']==rows[0]['output_sha256']==rows[-1]['output_sha256'] and
        all(raw[k]==0 for k in ('owned_hooks','residual_refs','output_refs','fia_stock_host_bytes')) and
        raw['native']==inv['native_state'] and health['baseline_output_sha256']==health['off_output_sha256']==raw['baseline'] and
        all(health[k]==0 for k in ('active_leases','device_bytes','host_bytes','inflight_calls','owned_hooks')),
        'ACTUAL_POST_RELEASE_HEALTH_ZERO_RESOURCES')
    if natural:
        b.metadata(a,[('initial',inv['metadata_arrays_initial']),('health',raw['metadata_arrays'])],inv['lifetime'],inv['build'])
        environment=artifacts.load('actual-environment')
        helper=C.pure_module(Path(__file__).with_name('review26_environment_originals.py'),'_env_originals26')
        helper.review(C,a,environment,cov['scope']['environment_sha256'],inv['lifetime'],
            native_images={x['name']:x['sha256'] for x in inv['metadata_arrays_initial']['images']})
    if cov['trigger']=='once':check_c2_proof(a,inv,target_ids)
    else:
        live=cov['live_c2_proof'];a.check(live['issued']==[] and isinstance(live['consumed'],dict),'C3_REQUIRES_ORIGINAL_LIVE_C2_CONSUMPTION')
    return dict(projection_calls_per_request=len(frozen),eligible_targets=len(eligible),fault_requests=faults,
        per_target_native_outcomes=sum(x is not None for x in modes),ledger_calls=len(calls),recomputed_hits=hits,artifact_files_rehashed=len(artifacts.checked))


def check_c2_proof(a,inv,target_ids):
    cov=inv['coverage'];owner=cov['owner'];generation=cov['generation'];recovery=cov['recovery']
    live=cov['live_c2_proof'];a.check(len(live['issued'])==1 and live['consumed'] is None,'ONE_ORIGINAL_C2_OBJECT_ISSUANCE')
    issued=live['issued'][0];proof=issued['proof'];stripped={k:v for k,v in cov.items() if k!='live_c2_proof'}
    a.check(issued['proof_sha256']==C.digest(proof) and proof==dict(scope=cov['scope'],owner=owner,generation=generation,
        session_id=cov['session_id'],target_ids=sorted(target_ids.values()),candidates_sha256=cov['discovery']['candidates_sha256'],
        discovery_sha256=cov['discovery']['artifact_sha256'],observation_sha256=cov['observation']['artifact_sha256'],
        recovery_sha256=C.digest(recovery),report_sha256=C.digest(stripped)) and
        issued['exact_issued_object_verified'] is True and issued['object_binding']['pid']==inv['lifetime']['pid'] and
        C.positive(issued['object_binding']['python_object_id']) and issued['qualification_granted'] is False and
        issued['serializable_authority'] is False,'SAVED_LIVE_C2_PROOF_EXACT_SCOPE_NOT_AUTHORITY')


def requests(a,b,d,final,worker,*,natural):
    rows=final['inventory']['rows'];observed=final['serving_requests'];seen=set()
    phases,modes=profile(final['inventory'])
    a.check(len(rows)==len(observed)==len(d['requests'])==len(phases),'EXACT_COMPLETE_GROUP_REQUEST_DENOMINATOR')
    for index,(row,obs,driver) in enumerate(zip(rows,observed,d['requests'])):
        rid=row['request_id'];snap=obs['snapshot'];mapping=driver['mapping']
        a.check(row['phase']==obs['phase']==driver['phase']==phases[index] and rid not in seen and
            rid==obs['actual_internal_request_id']==snap['internal_request_id']==mapping['internal_request_id'] and
            mapping['source']=='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR' and C.positive(mapping['request_state_object_id']) and
            obs['root_returned'] is True and obs['model_output_sha256']==row['output_sha256'] and
            obs['model_output_finite'] is row['output_finite'] and driver['output']['tokens']==[obs['top1']],
            'REAL_ENQUEUED_ROOT_LOGITS_REQUEST',rid)
        seen.add(rid)
        if natural:
            tokens=d['actual_prompt_token_ids']
            a.check(snap['num_requests']==1 and snap['num_computed_tokens']==0 and
                snap['num_prompt_tokens']==snap['num_scheduled_tokens']==len(tokens) and snap['token_ids']==tokens and
                snap['positions']==list(range(len(tokens))) and snap['device_input_bytes_equality']=='COMPLETE_DEVICE_READBACK_VERIFIED' and
                row['input_sha256']==obs['input_sha256']==rows[0]['input_sha256']==
                    b.input_hash(tokens,snap['positions'],snap['input_tensor_metadata']), 'FULL_FIXED_FRESH_PREFILL_INPUT',rid)
        if modes[index]!='fault':
            a.check(row['output_sha256']==rows[0]['output_sha256'] and row['output_finite'] is True and
                driver['output']==d['healthy_before'],'REFERENCE_LOSSLESS_OFF_COMPLETE_LOGITS',rid)
    if 'group_fault_repeat' in phases:
        a.check(rows[5]['output_sha256']==rows[6]['output_sha256'] and
            d['requests'][5]['output']==d['requests'][6]['output'],'FULL_FAULT_REPEAT_LOGITS_EXACT')
    health=d.get('post_unregister_health')
    if health is None:
        a.check(bool(d.get('error')),'MISSING_POST_UNREGISTER_HEALTH_WITHOUT_FAILURE');return
    a.check(health['worker']==worker and health['output']==d['healthy_before'] and
        health['mapping']['internal_request_id'] not in seen and
        health['mapping']['source']=='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR','POST_UNREGISTER_REAL_HEALTH')


def complete_group(a,b,d,final,final_ref,originals,context,*,mechanism,expected_count,natural,run='c2'):
    details={};inv=final['inventory'];context=dict(context,run=run,
        native_generation=context['ticket']['generation']+int(run=='c3'))
    a.check(inv['schema']=='demoncat.writer26.complete_set.v2' and inv['mechanism']==mechanism and
        inv['stage']=='coverage' and d['stage'] in ('c2_once','c2_c3') and inv['phase']=='complete' and inv['finalized'] is True and
        inv['failure'] is inv['first_failure'] is None and not inv['failure_events'] and not inv['blockers'],
        'COMPLETE_GROUP_NO_ERASED_FAILURE')
    artifacts=GroupArtifacts(originals,final_ref,run=run)
    a.section(run+':requests',requests,a,b,d,final,context['worker'],natural=natural)
    native=a.section(run+':native',native_group,b,a,inv,context['worker'],context,artifacts,natural=natural,expected_count=expected_count)
    if native:
        details['coverage']=a.section(run+':coverage',ledger,a,b,inv,*native,artifacts,context,natural=natural)
    if natural and mechanism=='26.fia_destination':
        typed=OLD.helper('review_native26_fia_single_source62.py')
        details['typed_fia']=a.section(run+':typed_fia',N.typed_group_review,b,typed,inv,typed.FiaArtifacts(b,originals.root),expected_count=expected_count)
        a.check(details['typed_fia'] is not None and details['typed_fia']['consistent'],'EVERY_FIA_TYPED_AND_STOCK_BRIDGE',run)
    a.check(inv['installed'] is False and inv['native_removed'] is True and all(inv[k]==0 for k in
        ('owned_hooks','owned_public_api_wrappers','pending_output_tensors','pending_residual_tensors','pending_public_storage_owners')) and
        not inv['pending_outcomes'] and not inv['pending_per_call_leases'] and all(inv['native_state']['core'][k]==0 for k in
        ('installed','pending','executing','device_bytes','host_bytes','descriptor_pending','error')),
        'FINAL_NATIVE_TENSOR_HOOK_RESOURCE_ZERO',run)
    return details


def review(d,host,records,originals,*,mechanism,expected_count,natural=True):
    if d.get('stage')=='c2_c3':
        helper=C.pure_module(Path(__file__).with_name('review26_c3_originals.py'),'_c3_originals26')
        from types import SimpleNamespace
        return helper.review(SimpleNamespace(**globals()),d,host,records,originals,
                             mechanism=mechanism,expected_count=expected_count,natural=natural)
    a=C.Audit();b=OLD.helper('review_native26_single_source54.py');details={}
    context=a.section('protocol',C.protocol,a,d,host,records,mechanism,natural=natural)
    if not context:return a.report()
    observed=a.section('originals',originals_join,a,originals,d,context,natural=natural)
    C.final_cleanup(a,d,host,context)
    failed=bool(d.get('error') or d.get('cleanup_error') or d.get('abort_error'))
    if failed or observed is None or observed['final'] is None:
        return dict(**a.report(),classification='FAILED_OR_PARTIAL_NOT_QUALIFIED',group_completion_reconciled=False,
            original_error=d.get('error'),original_cleanup_error=d.get('cleanup_error'),
            completed_original_requests=len(observed['snapshots']) if observed else None,
            aborted_request=d.get('owned_request_abort'))
    a.check(context['actions']==['attach','prepare','register','check','enable']+['advance','query','advance']*8+
        ['finish','disable','unregister','detach'],'EXACT_COMPLETE_SET_PUBLIC_PROTOCOL')
    a.check(d['stage']=='c2_once','EXACT_C2_ONLY_DRIVER_STAGE')
    details=complete_group(a,b,d,observed['final'],observed['final_ref'],originals,context,
        mechanism=mechanism,expected_count=expected_count,natural=natural)
    return dict(**a.report(),details=details,classification='SAVED_STANDARD_WORKER_GROUP_RECEIPTS' if natural else 'CPU_TOPOLOGY_CONTRACT_ONLY',
        group_completion_reconciled=not a.issues,originals_rehashed=len(originals.checked),
        limits=['Saved receipt reconciliation never issues a live C2/C3 capability.',
            'Full residual/logits equality is a frozen runtime receipt; only FIA stock/writer artifacts are rehashed as full tensor bytes.',
            'CPU mode checks public protocol/census/topology only and deliberately does not validate Native ABI double effects.',
            'E5 raw statistics are separate and are not inferred from execution outcomes or hit counts.'])


def provenance(d,final,args):
    loaded=OLD.LOADED;required=OLD.REQUIRED_SOURCES
    extra={'fault_injection/deployments/writer26_coverage.py','fault_injection/coverage.py','fault_injection/coverage_session.py'}
    producer=C.under(args.source_root,'experiments/c2c3_closure_20261117/providers26_worker_core_probe.py').read_text(encoding='utf-8')
    environment_declared="'fault_injection.environment'" in producer
    if environment_declared:extra.add('fault_injection/environment.py')
    try:
        OLD.LOADED=loaded|extra;OLD.REQUIRED_SOURCES=required|extra
        result=OLD.provenance(d,final,args.source_root,args.native_root,source_commit=args.source_commit,
            source_sha=args.source_manifest_sha256,build_sha=args.native_build_sha256,
            native_source_root=args.native_source_root,native_source_commit=args.native_source_commit,
            native_source_sha=args.native_source_manifest_sha256)
    finally:OLD.LOADED,OLD.REQUIRED_SOURCES=loaded,required
    hashes=d['sources'];prefix='fault_injection/deployments/'
    components=[hashes[prefix+'writer26_coverage.py'],hashes[prefix+'writer26_experiment.py'],final['inventory']['build'],
        hashes[prefix+'writer26_public.py'],hashes[prefix+'writer26_fia_typed.py']]
    coverage_source=C.under(args.source_root,prefix+'writer26_coverage.py').read_text(encoding='utf-8')
    if "joinpath('environment.py')" in coverage_source:components.append(hashes['fault_injection/environment.py'])
    components.append(hashes['fault_injection/metadata_arrays.py']);shared=D(components)
    implementation=D([shared,hashes[OLD.PREFIX+'serving26_core.py'],hashes[OLD.PREFIX+'providers/fault26_core.py']])
    a=C.Audit();a.check(final['inventory']['coverage']['scope']['implementation_sha256']==implementation,
        'GROUP_SCOPE_BINDS_ACTUAL_CORE_COVERAGE_NATIVE_AND_METADATA_SOURCE')
    result['group_scope_source']=a.report();result['consistent']=result['consistent'] and not a.issues
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('evidence-root','source-root','native-root'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('result-sha256','source-commit','source-manifest-sha256','native-build-sha256'):p.add_argument('--'+name,required=True)
    p.add_argument('--native-source-root',type=Path);p.add_argument('--native-source-commit');p.add_argument('--native-source-manifest-sha256')
    p.add_argument('--mechanism',choices=tuple(OLD.SCOPES),required=True);p.add_argument('--expected-writers',type=int,required=True)
    args=p.parse_args(argv);sys.dont_write_bytecode=True;sys.addaudithook(C.deny_runtime)
    a=C.Audit();d=None
    try:
        originals=C.Originals(args.evidence_root);d=originals.load(dict(path='probe/result.json',sha256=args.result_sha256))
        host,records=C.load_rpc(originals,d);result=review(d,host,records,originals,mechanism=args.mechanism,expected_count=args.expected_writers)
        proofs={}
        for run in ('c2','c3') if d.get('stage')=='c2_c3' else ('c2',):
            refs=d.get(run+'_attempt',{}).get('original_observations',{}).get('originals',[])
            if not refs:continue
            final=originals.load(refs[-1],mounted=True)
            if final['inventory'].get('coverage') is None:continue
            proof=provenance(d,final,args);proofs[run]=proof
            result['consistent']=bool(result['consistent'] and proof['consistent'])
        result['provenance_by_run']=proofs
        if proofs:result['provenance']=proofs.get('c2')
        else:
            result.update(consistent=False,provenance=dict(consistent=False,reason='NO_FINAL_COMPLETE_GROUP_ORIGINAL'))
        result['group_completion_reconciled']=bool(result.get('group_completion_reconciled') and result['consistent'])
    except (ValueError,KeyError,TypeError,OSError,IndexError,AttributeError) as error:
        a.check(False,'ORIGINAL_REVIEW_REFUSED',repr(error));result=a.report()
    if d is not None:
        result.update(original_driver_error=d.get('error'),original_cleanup_error=d.get('cleanup_error'),
            original_request_phases=[x.get('phase') for x in d.get('requests',[])])
        if d.get('error'):
            result.update(classification='FAILED_OR_PARTIAL_NOT_QUALIFIED',group_completion_reconciled=False)
    result.update(schema='demoncat.writer26.worker_group.independent_originals.v1',reviewer_sha256=C.file_sha(__file__),
        atomic_helpers=OLD.HELPERS,independent_helpers={name:C.file_sha(Path(__file__).with_name(name)) for name in
            ('group_review_common.py','review26_group_native.py','review26_c3_originals.py','review26_environment_originals.py')},
        read_only=True,runtime_executed=False)
    print(json.dumps(result,indent=2,allow_nan=False));return 0 if result['consistent'] else 2


if __name__=='__main__':raise SystemExit(main())
