"""Offline isolated-target original audit; saved JSON never grants live C2."""
import hashlib
from pathlib import Path

STAGE='c2_isolated_once'
SCOPE='ALL_STABLE_ELIGIBLE_TARGETS_ISOLATED_F_OFF_FRESH_REQUEST_PAIRS'


def plan(keys):
    rows=[dict(phase=p,mode=m,target='*' if m else None) for p,m in zip(
        ('inventory_1','inventory_2','qualification_reference_all','qualification_lossless_all','qualification_off_all'),
        (None,None,'reference','lossless','off'))]
    for i in range(len(keys)):
        rows.extend(dict(phase=p,mode=m,target='*',isolated_pair=i) for p,m in (('group_fault','fault'),('group_off','off')))
    return rows+[dict(phase='health',mode=None,target=None)]


def projection(r,inv):
    value=r.projection(inv)
    value['alias_inventory']=dict(count=len(inv['alias_inventory']),original_sha256=r.D(inv['alias_inventory']),compact_rpc_projection=True)
    bridge=value.get('fia_typed_stock_bridge')
    if bridge is not None:
        bridge=dict(bridge);artifacts=bridge.pop('artifacts')
        bridge['artifacts_original']=dict(count=len(artifacts),sha256=r.D(artifacts));value['fia_typed_stock_bridge']=bridge
    value['compact_rpc_projection']=True
    return value


def original_scope(r,a,value,ref,context,*,natural):
    a.check(value['schema']=='demoncat.writer26.isolated_core_observations.v1' and
        value['action']==ref['action'] and value['observation_kind']==ref['observation_kind'] and
        value['worker']==context['worker'] and value['native_owner']==context['provider_owner'] and
        value['run']=='c2' and value['simulation'] is (not natural) and value['qualification_granted'] is False and
        all(r.C.positive(value[k]) and value[k]==context['ticket']['generation'] for k in
            ('generation','rpc_generation','native_generation')),'ISOLATED_ORIGINAL_EXACT_WORKER_OWNER_GENERATION')


def original_rpc(r,a,inv,value,state,context):
    evidence=state['evidence'];public=evidence['public_core']
    compact={k:v for k,v in evidence.items() if k not in ('public_core','original_observations','current_run','prior_c2')}
    a.check(r.D(compact)==r.D(projection(r,inv)) and r.D(value['serving'])==r.D(state['serving']) and
        r.D(state['native'])==r.D(inv['native_state']),
        'DELTA_REBUILT_FULL_ORIGINAL_TO_COMPACT_RPC')
    a.check(evidence['current_run']=='c2' and 'prior_c2' not in evidence and public['stage']==STAGE and
        public['worker_identity']==value['worker'] and public['owner']==value['native_owner'] and
        public['generation']==public['native_generation']==value['generation'] and
        public['simulation'] is value['simulation'] and public['qualification_granted'] is False,
        'ISOLATED_PUBLIC_RPC_EXACT_SCOPE_NOT_AUTHORITY')


def join(r,a,originals,d,context,*,natural):
    states=context['states'];previous=[];rows=[];outcomes=[];calls=[];serving=[];last=None;incomplete=[]
    for index,driver in enumerate(d['requests']):
        rid=driver['mapping']['internal_request_id'];a.check(driver['run']=='c2','ISOLATED_ONLY_C2_REQUEST',rid)
        if 'original_observations' not in driver:
            incomplete.append(rid);a.check(bool(d.get('error')),'MISSING_DELTA_WITHOUT_FAILURE',rid);continue
        queries=[s for s in states if s['raw']['_reference']==driver['pre_collect_query']]
        arms=[s for s in states if s['action']=='advance' and r.C.rpc_action(s['raw'])[1].get('settings')=={'request_id':rid}]
        a.check(len(queries)==len(arms)==1,'EXACT_ISOLATED_ARM_AND_QUERY',rid);q=queries[0];arm=arms[0]
        following=[s for s in states if s['index']>q['index'] and s['action']=='advance'];collect=following[0]
        a.check(r.C.rpc_action(collect['raw'])[1].get('settings')=={'collect':True},'EXPLICIT_DELTA_COLLECT',rid)
        evidence=collect['state']['evidence'];refs=evidence['original_observations']['originals'];ref=driver['original_observations']
        a.check(refs==previous+[ref] and ref['action']=='collect' and
            q['state']['evidence']['original_observations']['originals']==previous and arm['index']<q['index']<collect['index'] and
            arm['state']['serving']['expected']==rid and q['state']['serving']['needs_collect'] is True and
            collect['state']['serving']['needs_collect'] is False and
            len(arm['state']['evidence']['request_index'])==index and len(q['state']['evidence']['request_index'])==index+1,
            'REAL_ENQUEUE_FORWARD_QUERY_COLLECT_NO_DELTA_REPLACEMENT',rid)
        value=originals.load(ref,mounted=True);original_scope(r,a,value,ref,context,natural=natural);inv=value['inventory']
        a.check(value['observation_kind']=='COMPLETE_REQUEST_DELTA' and
            value['previous_original_sha256']==(previous[-1]['sha256'] if previous else None) and
            all(type(inv['complete_request_delta'][k]) is int for k in ('start','stop')) and type(ref['complete_requests']) is int and
            inv['complete_request_delta']==dict(start=index,stop=index+1,complete_rows=True,complete_native_outcomes=True) and
            len(inv['rows'])==len(value['serving_requests'])==1 and ref['complete_requests']==index+1 and
            inv['rows'][0]['request_id']==value['serving_requests'][0]['actual_internal_request_id']==rid and
            all(x['request_id']==rid for x in inv['outcomes']),'EXACT_ONE_REQUEST_DELTA_SHA_CHAIN',rid)
        cov=inv['coverage'];new_calls=[] if cov is None else cov['calls']
        a.check(cov is None or cov['calls_are_complete_request_delta'] is True and
            all(x['forward_id']==rid for x in new_calls),'COVERAGE_DELTA_IS_CURRENT_REQUEST_ONLY',rid)
        rows.extend(inv['rows']);outcomes.extend(inv['outcomes']);calls.extend(new_calls);serving.extend(value['serving_requests'])
        rebuilt=dict(inv,rows=list(rows),outcomes=list(outcomes));rebuilt.pop('complete_request_delta')
        if cov is not None:
            rebuilt['coverage']=dict(cov,calls=list(calls));rebuilt['coverage'].pop('calls_are_complete_request_delta')
        query_compact={k:v for k,v in q['state']['evidence'].items() if k not in
            ('public_core','original_observations','current_run','prior_c2')}
        a.check(r.D(query_compact)==r.D(projection(r,rebuilt)) and
            r.D(q['state']['native'])==r.D(inv['native_state']),'PRE_COLLECT_QUERY_MATCHES_COMPLETE_ACTUAL_REQUEST',rid)
        original_rpc(r,a,rebuilt,value,collect['state'],context);previous=refs;last=rebuilt
    finals=[s for s in states if s['action']=='finish' and s['reply']['status']=='ACTION_COMPLETED'];a.check(len(finals)<=1,'ONE_FINAL_ISOLATED_REPORT')
    final=ref=None
    if finals:
        state=finals[0]['state'];refs=state['evidence']['original_observations']['originals'];ref=refs[-1]
        a.check(refs[:-1]==previous and ref['action']=='cleanup','ISOLATED_FINAL_EXTENDS_DELTA_CHAIN')
        final=originals.load(ref,mounted=True);original_scope(r,a,final,ref,context,natural=natural);inv=final['inventory']
        a.check(final['observation_kind']=='FINAL_COMPLETE_REPORT' and final['previous_original_sha256']==previous[-1]['sha256'] and
            ref['complete_requests']==len(rows) and 'complete_request_delta' not in inv and
            r.D(inv['rows'])==r.D(rows) and r.D(inv['outcomes'])==r.D(outcomes) and r.D(final['serving_requests'])==r.D(serving) and
            r.D(inv['coverage']['calls'])==r.D(calls) and 'calls_are_complete_request_delta' not in inv['coverage'],
            'ALL_DELTAS_EXACTLY_RECONSTRUCT_FINAL_COMPLETE_COLLECTIONS')
        for name in ('candidates','alias_inventory','lifetime','build','public_output_observer'):
            a.check(r.D(inv[name])==r.D(last[name]),'FINAL_PRESERVES_LAST_COMPLETE_OBSERVATION',name)
        bridge=last['fia_typed_stock_bridge']
        a.check(r.D(inv['fia_typed_stock_bridge'])==r.D(dict(bridge,stock_target_count=0,retained_host_bytes=0) if bridge is not None else None),
            'FINAL_STOCK_ARTIFACTS_UNCHANGED_ONLY_OWNED_HOST_STORAGE_RELEASED')
        original_rpc(r,a,inv,final,state,context)
        a.check(d['c2_attempt']==state['evidence'],'DRIVER_ISOLATED_SUMMARY_EXACT_FINAL_RPC')
    return dict(final=final,final_ref=ref,completed_requests=len(rows),incomplete_requests=incomplete,
        latest_inventory=last)


def native(r,a,b,inv,context,artifacts,*,natural,expected_count):
    rows=inv['rows'];life=inv['lifetime'];mechanism=1 if inv['mechanism']=='26.residual_stomp' else 2
    aliases={(x['path'],x['object_identity']) for x in inv['alias_inventory']}
    a.check(len(aliases)==len(inv['alias_inventory']) and ('<root>',life['model_object_identity']) in aliases,'COMPLETE_ISOLATED_ALIAS_SET')
    if natural:a.check(life['worker']==context['worker'] and life['pid']==context['worker']['pid'] and
        life['process_start_identity']==context['worker']['process_start_identity'],'ACTUAL_WORKER_NATIVE_LIFETIME')
    seen=set();boundaries=[]
    for i,row in enumerate(rows):
        label=str(i)+':'+row['phase'];events=b.trace(a,row,life,label,mechanism=mechanism)
        if i:boundaries.append(r.N.between_requests(a,rows[i-1]['native_after'],row['native_before'],label))
        a.check(len(events)==len(row['candidates'])==expected_count and not seen.intersection(x['call'] for x in events),
            'ISOLATED_FULL_NATIVE_DENOMINATOR_EVERY_REQUEST',label);seen.update(x['call'] for x in events)
        correlated=b.correlate(row,mechanism=mechanism)
        a.check(r.D([{k:v for k,v in x.items() if k!='current_forward_residual_absence'} for x in row['candidates']])==r.D(correlated),
            'INDEPENDENT_ISOLATED_WRITER_CONSUMER_JOIN',label)
        a.check(row['complete_inventory']==dict(native_writer_count=expected_count,native_events_sha256=r.D(row['native_events']),
            candidates_sha256=r.D(row['candidates']),actual_aliases_sha256=r.D(inv['alias_inventory'])),'FULL_REQUEST_INVENTORY_HASHES',label)
        if natural:
            for producer in row['producers']:b.public_producer(a,producer,row,aliases,label)
            for candidate in row['candidates']:b.callback_proof(a,candidate,row,life,aliases,b.Artifacts(artifacts.originals.root),label,mechanism=mechanism)
        for candidate in row['candidates']:r.current_absence(b,a,candidate,row,life)
    frozen=[]
    for i,(left,right) in enumerate(zip(rows[0]['candidates'],rows[1]['candidates'])):
        ls,_=r.current_absence(b,a,left,rows[0],life);status,reasons=r.current_absence(b,a,right,rows[1],life)
        cb=right['callback'];key=cb.get('target_key') if cb else None
        a.check(ls==status and all(left['native'][k]==right['native'][k] for k in b.STABLE_NATIVE),'TWO_HEALTHY_FULL_NATIVE_IDENTITIES',i)
        if status=='OBSERVED_NOT_QUALIFIED':
            a.check(left['callback'] and left['callback']['target_key']==key and all(sum(
                c['callback'] is not None and c['callback'].get('target_key')==key for c in row['candidates'])==1 for row in rows[:2]),
                'UNIQUE_ACTUAL_HEALTHY_TARGET',i)
        a.check(status in ('OBSERVED_NOT_QUALIFIED','UNSUPPORTED'),'NO_UNRESOLVED_ISOLATED_TARGET',i)
        target=dict(index=i,target_key=key,status=status,reasons=reasons,first_native=left['native'],second_native=right['native'],callback=cb)
        target['identity']=r.rebuild_identity(inv,target,context['worker']);frozen.append(target)
    a.check(inv['candidates']==frozen,'REBUILT_COMPLETE_FROZEN_ISOLATED_SET')
    eligible={x['target_key']:x for x in frozen if x['status']=='OBSERVED_NOT_QUALIFIED'};keys=list(eligible);phases=plan(keys)
    a.check(bool(keys) and inv['resolved_candidate_count']==inv['qualified_candidate_count']==len(keys) and len(rows)==len(phases),
        'COMPLETE_ISOLATED_TARGET_AND_REQUEST_COUNTS')
    a.check(inv['isolated_plan']==dict(target_order=keys,group_requests=len(phases),ordinary_post_unregister_requests=1,
        full_native_writer_count=expected_count,native_owned_execute_records=5*len(keys),max_targets_is_selector=False,c3_requested=False),
        'ISOLATED_PLAN_REBUILT_FROM_ACTUAL_CENSUS')
    outcomes={};owner=b.words(hashlib.sha256(context['provider_owner'].encode()).hexdigest())[:2];generation=context['ticket']['generation']
    for outcome in inv['outcomes']:
        key=(outcome['request_id'],outcome['target']);a.check(key not in outcomes,'UNIQUE_ISOLATED_NATIVE_OUTCOME');outcomes[key]=outcome
    pairs=[]
    for i,(row,expected) in enumerate(zip(rows,phases)):
        mode=expected['mode'];targets=set(keys) if 2<=i<=4 else {keys[(i-5)//2]} if 5<=i<len(phases)-1 else set()
        a.check(row['plan']==expected and all(type(row[k]) is int for k in ('admissions','completed')) and
            ('isolated_pair' not in row['plan'] or type(row['plan']['isolated_pair']) is int) and
            row['admissions']==row['completed']==len(targets) and
            {k for rid,k in outcomes if rid==row['request_id']}==targets,'ONLY_PLANNED_TARGETS_HAVE_NATIVE_EXECUTION',i)
        for candidate in row['candidates']:
            cb=candidate['callback'];key=cb.get('target_key') if cb else None;selected=key in targets
            a.check(not cb or type(cb['claimed']) is int and cb['claimed']==int(selected),'ACTUAL_CLAIMED_SET_ISOLATED_NOT_HIT_LOG',i)
            if not selected:continue
            effect=outcomes[(row['request_id'],key)]
            if natural:r.N.group_outcome(b,a,effect,candidate,row,dict(target_key=key,native_owner=owner,generation=generation),str(i)+':'+key,mechanism=mechanism)
            a.check(effect['mode']==mode and effect['native_call']==cb['native_packet'] and effect['request_id']==row['request_id'] and
                effect['public_consumer_lifetime'] is True and effect['residual_before_sha256']==effect['residual_after_sha256'],
                'ISOLATED_CURRENT_WRITER_CONSUMER_AND_RESTORATION',i)
    a.check(len(outcomes)==5*len(keys),'EXACT_FIVE_OWNED_EXECUTIONS_PER_TARGET')
    for i,key in enumerate(keys):
        fault,off=rows[5+2*i:7+2*i];reference=outcomes[(rows[2]['request_id'],key)];f=outcomes[(fault['request_id'],key)];o=outcomes[(off['request_id'],key)]
        controls=[outcomes[(rows[j]['request_id'],key)] for j in (2,3,4)]+[o]
        a.check(len({v['writer_output_sha256'] for v in controls})==len({v['residual_before_sha256'] for v in controls})==1 and
            f['residual_before_sha256']==reference['residual_before_sha256'],'EACH_F_STARTS_WITH_INDEPENDENT_HEALTHY_RESIDUAL',key)
        cb=next(c['callback'] for c in fault['candidates'] if c['callback'] and c['callback'].get('target_key')==key)
        proof=dict(actual_residual_before_sha256=f['residual_before_sha256'],qualified_reference_residual_sha256=reference['residual_before_sha256'],
            earlier_faults_this_request=0,uses_existing_digest_no_extra_tensor_copy=True)
        a.check(cb['isolated_pre_fault_baseline']==proof and type(cb['isolated_pre_fault_baseline']['earlier_faults_this_request']) is int,
            'PRE_FAULT_PROOF_REBUILT_FROM_ORIGINAL_REFERENCE',key)
        artifact='execution-'+str(f['native_call']['independent_call']);artifacts.load(artifact,expected=f)
        pairs.append(dict(index=i,target_key=key,target_id=r.C.digest(eligible[key]['identity']),fault_request_id=fault['request_id'],
            fault_independent_call=f['native_call']['independent_call'],fault_execution_sha256=artifacts.checked[artifact],
            actual_request_returned=True,pre_fault_baseline=proof,native_recovered_before_logical_completion=True,
            off_request_id=off['request_id'],off_independent_call=o['native_call']['independent_call'],off_outcome_sha256=r.D(o)))
    a.check(inv['isolated_pairs']==pairs and inv['isolated_logical_inflight']==0,'SAVED_ISOLATED_PAIRS_MATCH_INDEPENDENT_REQUEST_OUTCOMES')
    if natural:
        state=inv['native_state'];core=state['core'];count=5*len(keys)
        a.check([core['owner_hi'],core['owner_lo']]==owner and core['generation']==generation and core['thread']==life['native_thread'] and
            core['mechanism']==mechanism and core['calls']==state['admitted_calls']==state['completion_calls']==count,'ACTUAL_ISOLATED_NATIVE_OWNER_AND_EXECUTE_TOTAL')
        b.runtime_state(a,state,dict(native_owner=owner,generation=generation),life['native_thread'],sum(len(x['callbacks']) for x in rows),count,
            removed=True,sealed=True,where='isolated_final',mechanism=mechanism)
        b.metadata(a,[('initial',inv['metadata_arrays_initial'])]+[(str(i),x['metadata_arrays']) for i,x in enumerate(rows)]+
            [('final',inv['metadata_arrays_latest'])],life,inv['build'])
    return frozen,eligible,outcomes,boundaries


def ledger(r,a,b,inv,frozen,eligible,outcomes,artifacts,context,*,natural,native_build_binding):
    C=r.C;cov=inv['coverage'];rows=inv['rows'];keys=list(eligible);owner=context['provider_owner'];generation=context['ticket']['generation']
    a.check(inv['execution_scope']==inv['scope']==SCOPE and inv['independent_per_target_replay'] is True and
        cov['scope']['mechanism']==inv['mechanism'] and cov['scope']['evidence_kind']==('natural_worker' if natural else 'cpu_contract') and
        cov['owner']==owner and cov['generation']==generation and cov['trigger']=='once' and cov['selection']=='all_eligible' and
        cov['budget']['max_concurrency']==1,'EXACT_ISOLATED_C2_SCOPE')
    candidates=[dict(candidate_id=x['target_key'] or 'unsupported-native-'+str(i),identity=x['identity'],
        eligibility=dict(status='SUPPORTED' if x['target_key'] in eligible else 'UNSUPPORTED',reasons=x['reasons'])) for i,x in enumerate(frozen)]
    expected=dict(candidates=candidates,discovery_complete=True,eligible=len(keys),discovered=len(candidates),
        eligibility_counts={s:sum(x['eligibility']['status']==s for x in candidates) for s in
            ('SUPPORTED','UNSUPPORTED','UNVERIFIED','AMBIGUOUS','NOT_OBSERVED','RESOURCE_LIMITED')},execution_authorized=False)
    artifacts.load('complete-candidate-set',expected=expected);artifacts.load('full-healthy-discovery',cov['discovery']['artifact_sha256'],rows[:2])
    artifacts.load('full-native-public-observation',cov['observation']['artifact_sha256'],rows)
    a.check(cov['discovery']['candidates_sha256']==C.digest(expected) and
        cov['discovery']['observed_target_ids']==sorted(C.digest(x['identity']) for x in frozen),'ISOLATED_FULL_DISCOVERY_ARTIFACT')
    recovery=cov['recovery'];recovered={C.digest(x['lease']['target']):x for x in recovery['targets']}
    target_ids={k:C.digest(x['identity']) for k,x in eligible.items()}
    a.check(len(recovered)==len(recovery['targets'])==len(keys) and set(recovered)==set(target_ids.values()),'ALL_ISOLATED_TARGETS_RECOVERED')
    for i,key in enumerate(keys):
        target=eligible[key];reference=outcomes[(rows[2]['request_id'],key)];off=outcomes[(rows[6+2*i]['request_id'],key)]
        rr=recovered[target_ids[key]];lease=rr['lease'];call=reference['native_call']
        expected_lease=dict(target=target['identity'],owner=owner,generation=generation,lease_id=r.D([owner,generation,key]),
            device_bytes=call['output_bytes'],host_bytes=2*call['residual_bytes']+3*call['output_bytes']+8192)
        a.check(lease==expected_lease and rr['drained'] is rr['released'] is rr['off_exact'] is True and
            rr['pending_reason'] is None and rr['off_output_sha256']==reference['writer_output_sha256'],'ISOLATED_LEASE_EXACT_OFF',key)
        artifacts.load('lossless-'+key,expected=outcomes[(rows[3]['request_id'],key)])
        saved=artifacts.load('recovery-'+key,rr['proof_sha256'])
        a.check(saved['lease']==lease and saved['off']==off and all(saved['native_state']['core'][k]==0 for k in
            ('installed','pending','executing','device_bytes','host_bytes','descriptor_pending','error')),'ORIGINAL_ISOLATED_OFF_RELEASE',key)
    expected_calls=[]
    for i,row in enumerate(rows[5:-1]):
        ordered=[x for x in row['candidates'] if x['callback'] is not None]+[x for x in row['candidates'] if x['callback'] is None]
        expected_calls.extend((i,row,c) for c in ordered)
    calls=cov['calls'];a.check(len(calls)==len(expected_calls)==2*len(keys)*len(frozen),'FULL_ISOLATED_LEDGER_DENOMINATOR')
    tickets=[];attempted=set();hits=0;previous_request=None;anomaly=False
    for index,(call,(i,row,candidate)) in enumerate(zip(calls,expected_calls)):
        if previous_request!=row['request_id']:anomaly=False;previous_request=row['request_id']
        cb=candidate['callback'];key=cb.get('target_key') if cb else None;chosen=key in eligible
        selected=chosen and key==keys[i//2];off=row['phase']=='group_off';hit=selected and not off
        identity=frozen[row['candidates'].index(candidate)]['identity'];tid=C.digest(identity);call_id=str(candidate['native']['call'])
        phase='OFF_RECOVERY' if off else 'POST_FIRST_ANOMALY' if anomaly else 'PRE_FAULT_PROPAGATION'
        reason=('OFF_RECOVERY' if off else 'NON_TARGET' if not chosen else 'TARGET_ALREADY_ATTEMPTED' if key in attempted
            else 'ADMITTED' if hit else 'CONCURRENCY_BUDGET_EXHAUSTED')
        worker=[identity[k] for k in ('process_id','process_start_identity','worker_id','rank')]
        if 'boundary' in identity:worker.append(identity['device_type'])
        worker.append(identity['logical_device'])
        ticket=dict(observer_id=cov['discovery']['observer_id'],sequence=index,eligible=chosen,propagation_added=False,
            call=dict(target=identity,call_id=call_id,forward_context=row['request_id'],request_ids=[row['request_id']],phase=phase))
        tickets.append(ticket);decision=call['decision'];receipt=call['receipt']
        a.check(call['call_id']==call_id and call['target_id']==tid and call['forward_id']==row['request_id'] and
            call['request_ids']==[row['request_id']] and call['phase']==phase and call['eligible'] is chosen and
            call['worker']==worker and call['model_instance']==identity['model_instance_id'] and call['operator']==identity['operator_api'] and
            call['rank']==identity['rank'] and call['device']==identity['logical_device'] and call['owner']==owner and
            call['generation']==generation and call['propagation_added'] is False and decision['ticket']==ticket and
            decision['session_id']==cov['session_id'] and decision['admitted'] is hit and decision['reason']==reason and
            decision['lease']==(recovered[tid]['lease'] if chosen else None) and receipt['decision']==decision and
            receipt['native_applied'] is hit and receipt['original_calls']==1 and receipt['outcome']==('HIT' if hit else 'PASSTHROUGH'),
            'INDEPENDENT_ISOLATED_ADMISSION_AND_UNSELECTED_PASSTHROUGH',call_id)
        value=outcomes[(row['request_id'],key)] if selected else candidate
        artifacts.load(('execution-' if selected else 'passthrough-')+call_id,receipt['execution_sha256'],value)
        a.check(receipt['effect_sha256']==(receipt['execution_sha256'] if hit else None),'ISOLATED_EFFECT_ORIGINAL',call_id)
        if hit:attempted.add(key);hits+=1;anomaly=True
    a.check(cov['hits']==hits==len(keys) and attempted==set(keys) and cov['planned_targets']==cov['planned_targets_hit']==len(keys) and
        cov['observed_calls']==len(calls) and cov['observed_eligible_calls']==len(keys)**2 and
        not cov['missing_outcomes'] and not cov['propagation_added_calls'] and cov['observation']['calls_sha256']==C.digest(tickets) and
        cov['observation']['complete'] is True and cov['observation']['dropped_calls']==0,'REBUILT_ISOLATED_ONCE_ONLY_TOTALS')
    a.check(recovery['state']=='DISABLED_CONFIRMED' and recovery['inflight']==0 and
        not any(recovery[k] for k in ('errors','held_leases','pending_drains','pending_leases')) and len(recovery['workers'])==1,
        'ALL_ISOLATED_OWNERS_RELEASED')
    health=recovery['workers'][0];raw=artifacts.health(health['proof_sha256'])
    a.check(raw['baseline']==raw['off']==rows[0]['output_sha256']==rows[-1]['output_sha256'] and
        all(raw[k]==0 for k in ('owned_hooks','residual_refs','output_refs','fia_stock_host_bytes')) and raw['native']==inv['native_state'] and
        health['baseline_output_sha256']==health['off_output_sha256']==raw['baseline'] and
        all(health[k]==0 for k in ('active_leases','device_bytes','host_bytes','inflight_calls','owned_hooks')),'ACTUAL_ISOLATED_HEALTH_RESOURCE_ZERO')
    if natural:
        b.metadata(a,[('initial',inv['metadata_arrays_initial']),('health',raw['metadata_arrays'])],inv['lifetime'],inv['build'])
        helper=C.pure_module(Path(__file__).with_name('review26_environment_originals.py'),'_env_isolated26')
        helper.review(C,a,artifacts.load('actual-environment'),cov['scope']['environment_sha256'],inv['lifetime'],
            native_images={x['name']:x['sha256'] for x in inv['metadata_arrays_initial']['images']},
            build_binding=native_build_binding,expected_build_sha=inv['build'])
    r.check_c2_proof(a,inv,target_ids)
    return dict(full_calls_per_request=len(frozen),eligible_targets=len(keys),group_requests=len(rows),
        fault_requests=len(keys),owned_native_executes=len(outcomes),ledger_calls=len(calls),recomputed_hits=hits,
        artifact_files_rehashed=len(artifacts.checked),execution_scope=SCOPE)


def requests(r,a,b,d,final,worker,*,natural):
    rows=final['inventory']['rows'];observed=final['serving_requests'];seen=set()
    a.check(len(rows)==len(observed)==len(d['requests']),'ISOLATED_REAL_REQUEST_DENOMINATOR')
    for row,obs,driver in zip(rows,observed,d['requests']):
        rid=row['request_id'];snap=obs['snapshot'];mapping=driver['mapping']
        a.check(row['phase']==obs['phase']==driver['phase'] and rid not in seen and driver['run']=='c2' and
            rid==obs['actual_internal_request_id']==snap['internal_request_id']==mapping['internal_request_id'] and
            mapping['source']=='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR' and r.C.positive(mapping['request_state_object_id']) and
            obs['root_returned'] is True and obs['model_output_sha256']==row['output_sha256'] and
            obs['model_output_finite'] is row['output_finite'] and driver['output']['tokens']==[obs['top1']],
            'ACTUAL_ISOLATED_ENQUEUE_ROOT_LOGITS_AND_OUTPUT_PROCESSOR',rid);seen.add(rid)
        if natural:
            tokens=d['actual_prompt_token_ids']
            a.check(snap['num_requests']==1 and snap['num_computed_tokens']==0 and snap['num_prompt_tokens']==snap['num_scheduled_tokens']==len(tokens) and
                snap['token_ids']==tokens and snap['positions']==list(range(len(tokens))) and
                snap['device_input_bytes_equality']=='COMPLETE_DEVICE_READBACK_VERIFIED' and row['input_sha256']==obs['input_sha256']==
                rows[0]['input_sha256']==b.input_hash(tokens,snap['positions'],snap['input_tensor_metadata']),'FRESH_FIXED_INPUT_EACH_ISOLATED_REQUEST',rid)
        if row['phase']!='group_fault':a.check(row['output_sha256']==rows[0]['output_sha256'] and row['output_finite'] is True and
            driver['output']==d['healthy_before'],'REFERENCE_LOSSLESS_EACH_OFF_AND_FINAL_HEALTH_EXACT',rid)
    health=d.get('post_unregister_health')
    a.check(health is not None and health['worker']==worker and health['output']==d['healthy_before'] and
        health['mapping']['internal_request_id'] not in seen and health['mapping']['source']=='ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR',
        'ORDINARY_POST_UNREGISTER_HEALTH_REQUEST')
    faults=[(row,driver) for row,driver in zip(rows,d['requests']) if row['phase']=='group_fault']
    return dict(group_requests=len(rows),ordinary_post_unregister_health_requests=int(health is not None),
        isolated_fault_requests=len(faults),fault_complete_model_output_hash_changed=sum(row['output_sha256']!=rows[0]['output_sha256'] for row,_ in faults),
        fault_returned_tokens_changed=sum(driver['output']['tokens']!=d['healthy_before']['tokens'] for _,driver in faults),
        complete_residual_and_logits_arrays_rehashed=False,hashes_are_original_runtime_observations=True)


def review(r,d,host,records,originals,*,mechanism,expected_count,natural,native_build_binding=None):
    C=r.C;a=C.Audit();b=r.OLD.helper('review_native26_single_source54.py');details={}
    a.check(d['stage']==STAGE,'EXPLICIT_ISOLATED_DRIVER_STAGE')
    context=a.section('protocol',C.protocol,a,d,host,records,mechanism,natural=natural)
    if not context:return a.report()
    found=a.section('isolated_delta_originals',join,r,a,originals,d,context,natural=natural)
    C.final_cleanup(a,d,host,context)
    partial=bool(d.get('error') or d.get('cleanup_error') or d.get('abort_error') or d.get('blocked_run') or
        found is None or found['final'] is None or found['incomplete_requests'])
    if found is not None and found['final'] is not None:
        final=found['final'];inv=final['inventory'];artifacts=r.GroupArtifacts(originals,found['final_ref'])
        a.check(inv['schema']=='demoncat.writer26.complete_set.v2' and inv['stage']=='coverage' and inv['mechanism']==mechanism and
            inv['phase']=='complete' and inv['finalized'] is True and inv['failure'] is inv['first_failure'] is None and
            not inv['failure_events'] and not inv['blockers'],'COMPLETE_ISOLATED_RUN_HAS_NO_ERASED_FAILURE')
        details['request_outputs']=a.section('isolated_requests',requests,r,a,b,d,final,context['worker'],natural=natural)
        values=a.section('isolated_native',native,r,a,b,inv,context,artifacts,natural=natural,expected_count=expected_count)
        if values:
            frozen,eligible,outcomes,boundaries=values;details['outside_request_native_boundaries']=boundaries
            details['coverage']=a.section('isolated_ledger',ledger,r,a,b,inv,frozen,eligible,outcomes,artifacts,context,
                natural=natural,native_build_binding=native_build_binding)
            a.check(d['isolated_plan']==inv['isolated_plan'],'DRIVER_FROZEN_PLAN_EQUALS_ORIGINAL_COMPLETE_CENSUS')
            if not partial:a.check(context['actions']==['attach','prepare','register','check','enable']+
                ['advance','query','advance']*len(plan(eligible))+['finish','disable','unregister','detach'],'EXACT_ISOLATED_PUBLIC_PROTOCOL')
        if natural and mechanism=='26.fia_destination':
            typed=r.OLD.helper('review_native26_fia_single_source62.py')
            details['typed_fia']=a.section('isolated_typed_fia',r.N.typed_group_review,b,typed,inv,typed.FiaArtifacts(b,originals.root),
                expected_count=expected_count,isolated=True)
            a.check(details['typed_fia'] is not None and details['typed_fia']['consistent'],'ALL_ISOLATED_TYPED_AND_STOCK_OUTPUT_BYTES')
        a.check(inv['installed'] is False and inv['native_removed'] is True and inv['isolated_logical_inflight']==0 and
            all(inv[k]==0 for k in ('owned_hooks','owned_public_api_wrappers','pending_output_tensors','pending_residual_tensors','pending_public_storage_owners')) and
            not inv['pending_outcomes'] and not inv['pending_per_call_leases'] and all(inv['native_state']['core'][k]==0 for k in
            ('installed','pending','executing','device_bytes','host_bytes','descriptor_pending','error')),'ISOLATED_FINAL_ALL_OWNED_RESOURCES_ZERO')
    elif found is not None and found['latest_inventory'] is not None:
        inv=found['latest_inventory'];counts=[]
        for index,row in enumerate(inv['rows']):
            actual=a.section('partial_trace',b.trace,a,row,inv['lifetime'],str(index),mechanism=1 if mechanism=='26.residual_stomp' else 2)
            if actual is not None:
                counts.append(len(actual));a.check(len(actual)==len(row['candidates'])==expected_count,'PARTIAL_COMPLETE_NATIVE_DENOMINATOR',index)
        details['partial']=dict(complete_request_native_counts=counts,completed_phases=[row['phase'] for row in inv['rows']],
            recorded_owned_outcomes=len(inv['outcomes']),recorded_fault_outcomes=sum(v['mode']=='fault' for v in inv['outcomes']),
            group_completion_reconciled=False)
    return dict(**a.report(),details=details,classification='FAILED_OR_PARTIAL_NOT_QUALIFIED' if partial else
        'SAVED_STANDARD_WORKER_ISOLATED_C2_ORIGINALS' if natural else 'CPU_ISOLATED_TOPOLOGY_CONTRACT_ONLY',
        group_completion_reconciled=not partial and not a.issues,originals_rehashed=len(originals.checked),
        completed_original_requests=found['completed_requests'] if found else 0,original_error=d.get('error'),
        original_cleanup_error=d.get('cleanup_error'),aborted_request=d.get('owned_request_abort'),blocked_run=d.get('blocked_run'),
        limits=['Saved JSON cannot issue or recreate live C2 authority; C3 is not requested.',
            'This is isolated single-target F/OFF per fresh request, not cumulative all-target injection.',
            'FIA stock/writer bytes are rehashed; residual/logits equality remains recorded full runtime digest evidence.',
            'No E5 conclusion is inferred from outcomes, Native execution or coverage.'])
