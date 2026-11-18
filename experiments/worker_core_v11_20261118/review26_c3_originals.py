"""Pure saved-original joins for two generations; never a live-proof authority."""


def join(r,a,originals,d,context,*,natural):
    C=r.C; states=context['states']; runs={}
    for name in ('c2','c3'):
        runs[name]=dict(final=None,final_ref=None,snapshots=[],incomplete_requests=[],refs=[],summary=None)
    seen=set(); transition=None
    for driver in d['requests']:
        run=driver['run']; item=runs[run]; rid=driver['mapping']['internal_request_id']
        a.check(rid not in seen,'UNIQUE_REQUEST_ACROSS_NATIVE_GENERATIONS',rid);seen.add(rid)
        a.check(run=='c2' and transition is None or run=='c3' and transition is not None,'C2_THEN_C3_ORDER',rid)
        if 'original_observations' not in driver:
            item['incomplete_requests'].append(rid)
            a.check(bool(d.get('error')),'MISSING_REQUEST_ORIGINAL_WITHOUT_FAILURE',rid)
            continue
        queries=[x for x in states if x['raw']['_reference']==driver['pre_collect_query']]
        arms=[x for x in states if x['action']=='advance' and C.rpc_action(x['raw'])[1].get('settings')=={'request_id':rid}]
        a.check(len(queries)==len(arms)==1,'EXACT_REQUEST_ARM_AND_QUERY',rid)
        q=queries[0]; arm=arms[0]; index=len(item['snapshots'])
        following=[x for x in states if x['index']>q['index'] and x['action']=='advance']
        a.check(bool(following) and C.rpc_action(following[0]['raw'])[1].get('settings')=={'collect':True},
                'EXPLICIT_COLLECT_FOLLOWS_QUERY',rid)
        current=following[0]; state=current['state']; evidence=state['evidence']
        changing=evidence['current_run']!=run
        if changing:
            a.check(run=='c2' and evidence['current_run']=='c3' and driver['phase']=='health' and index==7 and
                transition is None,'ONLY_COMPLETED_C2_HEALTH_CAN_START_C3',rid)
            evidence=evidence['prior_c2'];transition=current['index']
        refs=evidence['original_observations']['originals']; previous=item['refs']
        a.check(q['state']['evidence']['current_run']==arm['state']['evidence']['current_run']==run and
            q['state']['evidence']['original_observations']['originals']==previous and
            refs[:len(previous)]==previous and len(refs)==len(previous)+(2 if changing else 1) and
            refs[-1]==driver['original_observations'] and arm['index']<q['index'] and
            arm['state']['serving']['expected']==rid and len(arm['state']['evidence']['request_index'])==index and
            len(q['state']['evidence']['request_index'])==index+1 and q['state']['serving']['needs_collect'] is True and
            state['serving']['needs_collect'] is False,'RPC_CANNOT_REPLACE_ACTUAL_ENQUEUED_FORWARD',rid)
        added=[]
        for ref in refs[len(previous):]:
            is_last=ref==refs[-1]
            value=r.load_original(a,originals,ref,state if is_last and not changing else None,context,
                natural=natural,evidence=evidence if is_last else None,run=run)
            a.check(ref['action']==('cleanup' if changing and is_last else 'collect'),'ORIGINAL_ACTION_SEQUENCE',rid)
            if added:
                a.check(value['inventory']['rows']==added[-1]['inventory']['rows'] and
                    value['serving_requests']==added[-1]['serving_requests'],'C2_CLOSE_PRESERVES_LAST_REQUEST')
            added.append(value)
        value=added[-1]
        if item['snapshots']:
            a.check(value['inventory']['rows'][:-1]==item['snapshots'][-1]['inventory']['rows'] and
                value['serving_requests'][:-1]==item['snapshots'][-1]['serving_requests'],'IMMUTABLE_REQUEST_PREFIX',rid)
        item['snapshots'].append(value);item['refs']=refs
        if changing:
            item.update(final=value,final_ref=refs[-1],summary=evidence)
            a.check(d.get('c2_attempt')==evidence,'DRIVER_C2_SUMMARY_JOINS_ORIGINAL_TRANSITION')
    finals=[x for x in states if x['action']=='finish' and x['reply']['status']=='ACTION_COMPLETED']
    a.check(len(finals)<=1,'ONLY_ONE_PUBLIC_FINISH')
    if finals:
        state=finals[0]['state'];evidence=state['evidence'];item=runs['c3'];refs=evidence['original_observations']['originals']
        a.check(evidence['current_run']=='c3' and refs[:-1]==item['refs'] and refs[-1]['action']=='cleanup',
                'FINAL_C3_CLEANUP_ORIGINAL_NOT_REPLACED')
        value=r.load_original(a,originals,refs[-1],state,context,natural=natural,run='c3')
        a.check(bool(item['snapshots']) and value['inventory']['rows']==item['snapshots'][-1]['inventory']['rows'] and
            value['serving_requests']==item['snapshots'][-1]['serving_requests'],'CLEANUP_PRESERVES_ACTUAL_REQUESTS')
        item.update(final=value,final_ref=refs[-1],summary=evidence,refs=refs)
        a.check(d.get('c3_attempt')==evidence,'DRIVER_C3_SUMMARY_JOINS_ORIGINAL_FINISH')
    if transition is not None:
        for state in states:
            if state['index']>=transition and state['state'].get('evidence') is not None:
                a.check(state['state']['evidence'].get('prior_c2')==runs['c2']['summary'],
                        'IMMUTABLE_PRIOR_C2_IN_ALL_LATER_RPC',state['index'])
    return runs


def transition(r,a,first,second,first_details,second_details):
    C=r.C; left,right=first['inventory'],second['inventory']; one,two=left['coverage'],right['coverage']
    issued,=one['live_c2_proof']['issued']; consumed=two['live_c2_proof']['consumed']
    a.check(two['live_c2_proof']['issued']==[] and set(consumed)==set(issued)|{'consumer'} and
        {k:v for k,v in consumed.items() if k!='consumer'}==issued and
        consumed['consumer']==dict(session_id=two['session_id'],scope=two['scope'],owner=two['owner'],
            generation=two['generation'],candidates_sha256=two['discovery']['candidates_sha256'],
            target_ids=issued['proof']['target_ids']) and
        consumed['exact_issued_object_verified'] is True and consumed['serializable_authority'] is False and
        consumed['qualification_granted'] is False,'EXACT_SAVED_LIVE_OBJECT_ISSUANCE_CONSUMPTION')
    a.check(one['scope']==two['scope'] and one['owner']==two['owner'] and
        two['generation']==one['generation']+1 and two['session_id']!=one['session_id'] and
        first['worker']==second['worker'] and first['rpc_generation']==second['rpc_generation']==one['generation'] and
        first['native_generation']==one['generation'] and second['native_generation']==two['generation'],
        'SAME_WORKER_SCOPE_OWNER_NEXT_NATIVE_GENERATION')
    def targets(inv):
        return [(x['target_key'],x['status'],x['identity']) for x in inv['candidates']]
    a.check(targets(left)==targets(right) and left['alias_inventory']==right['alias_inventory'] and
        one['discovery']['candidates_sha256']==two['discovery']['candidates_sha256'],
        'FULL_NATIVE_DENOMINATOR_REQUALIFIED_WITHOUT_TARGET_LOSS')
    a.check(left['build']==right['build'] and left['lifetime']==right['lifetime'] and
        left['rows'][0]['output_sha256']==right['rows'][0]['output_sha256'] and
        left['rows'][0]['input_sha256']==right['rows'][0]['input_sha256'], 'SAME_BUILD_LIFETIME_FIXED_INPUT_AND_HEALTHY_OUTPUT')
    a.check(left['rows'][-1]['native_after']==right['rows'][0]['native_before'],
            'NO_UNACCOUNTED_NATIVE_CALLS_AT_GENERATION_BOUNDARY')
    if left['mechanism']=='26.fia_destination':
        a.check(left['rows'][-1]['fia_typed_after']==right['rows'][0]['fia_typed_before'],
                'NO_UNACCOUNTED_TYPED_CALLS_AT_GENERATION_BOUNDARY')
    a.check(first_details['projection_calls_per_request']==second_details['projection_calls_per_request'] and
        first_details['eligible_targets']==second_details['eligible_targets'] and
        second_details['recomputed_hits']==2*first_details['recomputed_hits'], 'INDEPENDENT_PER_GENERATION_ACTUAL_COUNTS')
    return dict(rpc_generation=first['rpc_generation'],native_generations=[one['generation'],two['generation']],
        same_saved_object_binding=issued['object_binding']==consumed['object_binding'],
        proof_sha256=C.digest(issued['proof']), serializable_authority=False, live_authority_recreated=False)


def review(r,d,host,records,originals,*,mechanism,expected_count,natural):
    C=r.C;a=C.Audit();b=r.OLD.helper('review_native26_single_source54.py');details={}
    context=a.section('protocol',C.protocol,a,d,host,records,mechanism,natural=natural)
    if not context:return a.report()
    runs=a.section('two_generation_originals',join,r,a,originals,d,context,natural=natural)
    C.final_cleanup(a,d,host,context)
    partial=bool(d.get('error') or d.get('cleanup_error') or d.get('abort_error') or d.get('blocked_run'))
    if runs is None:partial=True;runs={}
    for run,item in runs.items():
        final=item['final'];drivers=[v for v in d['requests'] if v['run']==run]
        details[run]=dict(collected_requests=len(item['snapshots']),incomplete_requests=item['incomplete_requests'])
        if final is not None:
            # A partial C3 attempt cannot erase the completed, independently
            # retained C2 run. The final public cleanup/health is still checked.
            before=len(a.issues)
            checked=a.section(run+':complete_group',r.complete_group,a,b,dict(d,requests=drivers),final,
                item['final_ref'],originals,context,mechanism=mechanism,expected_count=expected_count,natural=natural,run=run)
            details[run].update(checked or {})
            details[run]['complete_originals_reconciled']=checked is not None and len(a.issues)==before
        else:
            partial=True
            latest=item['snapshots'][-1] if item['snapshots'] else None
            if latest is not None:
                inv=latest['inventory'];counts=[]
                for row in inv['rows']:
                    native=a.section(run+':partial_native',b.trace,a,row,inv['lifetime'],row['phase'],
                                     mechanism=1 if mechanism=='26.residual_stomp' else 2)
                    if native is not None:
                        counts.append(len(native))
                        a.check(len(native)==len(row['candidates'])==expected_count,'PARTIAL_FULL_NATIVE_DENOMINATOR',run+':'+row['phase'])
                details[run]['observed_complete_request_native_counts']=counts
    if not partial:
        a.check(context['actions']==['attach','prepare','register','check','enable']+['advance','query','advance']*17+
            ['finish','disable','unregister','detach'],'EXACT_C2_THEN_C3_PUBLIC_PROTOCOL')
        a.check([(x['run'],x['phase']) for x in d['requests']]==
                [('c2',p) for p in r.PHASES]+[('c3',p) for p in r.C3_PHASES],'ALL_SEVENTEEN_REQUESTS_IN_ORDER')
        if all(details[x].get('coverage') for x in ('c2','c3')):
            details['transition']=a.section('live_transition',transition,r,a,runs['c2']['final'],runs['c3']['final'],
                details['c2']['coverage'],details['c3']['coverage'])
        else:a.check(False,'BOTH_COMPLETE_GENERATION_LEDGER_AUDITS_REQUIRED')
    return dict(**a.report(),details=details,group_completion_reconciled=not partial and not a.issues,
        classification='FAILED_OR_PARTIAL_NOT_QUALIFIED' if partial else
            'SAVED_STANDARD_WORKER_C2_C3_RECEIPTS' if natural else 'CPU_TOPOLOGY_CONTRACT_ONLY',
        original_error=d.get('error'),original_cleanup_error=d.get('cleanup_error'),blocked_run=d.get('blocked_run'),
        aborted_request=d.get('owned_request_abort'),originals_rehashed=len(originals.checked),
        completed_original_requests=sum(len(x['snapshots']) for x in runs.values()),
        limits=['Saved original reconciliation never recreates or grants live C2/C3 authority.',
            'CPU topology mode does not validate Native ABI double effects or qualify an NPU Worker.',
            'FIA stock/owned bytes are separately rehashed in natural mode; full residual/logits equality is a runtime receipt.',
            'No E5 inference is made from runtime outcomes, proof records, or hit counts.'])
