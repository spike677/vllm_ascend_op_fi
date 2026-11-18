"""Independent metadata fixtures, never Native/observer assessment or device tests."""
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import review_quant_gmm2_observations as audit


def fixture(count=1):
    process = dict(pid=9, thread=11, process_start_identity='proc:01234567-0123-0123-0123-0123456789ab:9:123')
    worker = dict(pid=9, process_start_identity=process['process_start_identity'], rank=0,
        logical_device=0, configured_logical_device=0, observed_logical_device=0,
        worker_start_identity='worker-live-fixture', model_instance_id='model-live-fixture')
    identity = dict(status='WORKER_IDENTIFIED', worker=worker, quarantined=False)
    profile = dict(images={name:dict(path='/pinned/'+name+'.so', file_bytes=50000000,
        file_sha256=sha, segments=[dict(vaddr=0, size=50000000)]) for name, sha in audit.PINS.items()})
    owners = {name: dict(symbol=symbol, address=100000000+offset, base=100000000, offset=offset,
        canonical_path=profile['images'][image]['path'], file_bytes=50000000, file_sha256=audit.PINS[image])
        for name, (image, symbol, offset) in audit.FUNCTIONS.items()}
    raw, typed, scopes, graph_events = [], [], [], []
    clock = 0; payload_bytes = 0
    def tick():
        nonlocal clock
        clock += 10; return clock
    def tensor(number, shape, dtype):
        stride = [math.prod(shape[n+1:]) for n in range(len(shape))]
        width = {'torch.int8':1, 'torch.bfloat16':2, 'torch.float32':4, 'torch.int64':8}[dtype]
        return dict(tensor=dict(storage_address=number*100000, storage_bytes=math.prod(shape)*width,
            shape=shape, stride=stride, offset=0, dtype=dtype, device='npu:0'))
    def descriptor(value, number):
        t = value['tensor']; shape, stride = t['shape'], t['stride']
        return dict(descriptor=number, address=t['storage_address'], counts=[len(shape), len(stride), 1],
            called_mask=127, null_mask=0, truncated_mask=0, offset=0,
            shape=shape+[0]*(8-len(shape)), strides=stride+[0]*(8-len(stride)),
            storage=[math.prod(shape)]+[0]*7,
            dtype={'torch.int8':2, 'torch.bfloat16':27, 'torch.float32':0, 'torch.int64':9}[t['dtype']],
            format=2, complete=1, statuses=[0]*7)
    def original(op, scope, handle, *, content=(), registration=0):
        nonlocal payload_bytes
        call = len(raw)//2+1; creator = op in (1, 3)
        base = dict(operation=op, call=call, scope_token=scope['token'],
            **{k:scope[k] for k in ('forward', 'operator_sequence', 'boundary_sequence', 'stage', 'thread')},
            pid=9, depth=1, error=0, status=0, reserved=0, count=len(content) if creator else 0,
            function=100000000+op*16, function_base=100000000, function_offset=op*16,
            caller=200000000+100, caller_base=200000000, caller_offset=100, caller_kind=1)
        for phase in (1, 2):
            row = dict(base, phase=phase, sequence=len(raw)+1, monotonic_ns=tick(), original_calls=phase-1,
                handle=0 if creator and phase==1 else handle,
                registration=(0 if phase==1 else call) if creator else registration, payload_bytes=0, payload_offset=0)
            if creator and phase==1:
                row.update(payload_offset=payload_bytes, payload_bytes=len(content)*(304 if op==1 else 8))
                row['tensor_descriptors' if op==1 else 'integer_values'] = copy.deepcopy(list(content))
                payload_bytes += row['payload_bytes']
            raw.append(row)
        return raw[-2], raw[-1]
    def event(scope, operation, call, registration):
        name = 'prepare_function' if operation==1 else 'execute_function'
        row = dict.fromkeys(audit.G_FIELDS, 0)
        row.update(version=1, size=audit.EVENT_BYTES, operation=operation, call=call,
            prepare_registration=registration, owner_hi=7, owner_lo=8, generation=1,
            scope_token=scope['token'], **{k:scope[k] for k in ('forward', 'operator_sequence', 'boundary_sequence', 'stage', 'thread')},
            pid=9, function=owners[name]['address'], function_base=100000000, function_offset=owners[name]['offset'],
            caller=200000000+audit.CALLERS[operation], caller_base=200000000, caller_offset=audit.CALLERS[operation],
            device=0, device_status=0, status=0, reserved=0, formals=[0]*23, scalars=[0]*5,
            containers=[], group_list={}, workspace=0, workspace_bytes=0, executor=0, stream=0)
        return row
    def append(row, phase):
        row.update(sequence=len(typed)+1, phase=phase, monotonic_ns=tick(), original_calls=phase-1, container_events=len(raw))
        typed.append(row)
    for n in range(count):
        scope = dict(version=1, size=56, token=n+1, forward=1, operator_sequence=n+1,
                     boundary_sequence=n+1, stage=3, thread=11)
        formals = {name:None for name in ('bias', 'offset', 'antiquant_scale', 'antiquant_offset', 'activation_input',
            'activation_quant_scale', 'activation_quant_offset', 'tuning_config')}
        formals.update(x=[tensor(1,[48,1408],'torch.int8')], weight=[tensor(2,[64,1408,2048],'torch.int8')],
            scale=[tensor(3,[64,2048],'torch.bfloat16')], per_token_scale=[tensor(4,[48],'torch.float32')],
            group_list=tensor(6,[64],'torch.int64'), split_item=2, group_type=0, group_list_type=1, act_type=0)
        output = tensor(5,[48,2048],'torch.bfloat16')
        scopes.append(dict(native_scope=scope, evidence=dict(formals=formals)))
        graph_events.append(dict(sequence=n+1, outputs=[output['tensor']]))
        content = {0:[descriptor(formals['x'][0],901)], 1:[descriptor(formals['weight'][0],902)],
            3:[descriptor(formals['scale'][0],903)], 7:[descriptor(formals['per_token_scale'][0],904)],
            18:[descriptor(output,905)]}
        slots=[]
        for index in audit.SLOTS:
            kind=3 if index==16 else 1; handle=500+index
            first,last=original(kind,scope,handle,content=content.get(index,[]))
            slots.append(dict(formal_index=index,kind=kind,handle=handle,registration=last['registration'],
                creator_entry=first['sequence'],creator_return=last['sequence'],count=first['count'],
                payload_offset=first['payload_offset'],payload_bytes=first['payload_bytes'],
                actual_count=first['count'],joined=1,live_at_return=0,size_status=0,reserved=0))
        call=len(typed)//2+1; entry=event(scope,1,call,call)
        entry.update(containers=slots,scalars=[2,0,1,0,-11],group_list=descriptor(formals['group_list'],906))
        for slot in slots:entry['formals'][slot['formal_index']]=slot['handle']
        entry['formals'][8]=906;entry['formals'][21:]=[888,889]
        for i,index in enumerate(audit.SCALARS):entry['formals'][index]=entry['scalars'][i]&((1<<64)-1)
        append(copy.deepcopy(entry),1)
        done=copy.deepcopy(entry);done.update(workspace_bytes=16,executor=777)
        for slot in done['containers']:slot['live_at_return']=1
        append(done,2)
        execute=event(scope,2,call+1,call);execute.update(workspace=999,workspace_bytes=32,executor=777,stream=111)
        append(copy.deepcopy(execute),1);append(copy.deepcopy(execute),2)
        for slot in slots:original(slot['kind']+1,scope,slot['handle'],registration=slot['registration'])
    entries=[sum(r['phase']==1 and r['operation']==n for r in raw) for n in (1,2,3,4)]
    state=dict(version=1,size=400,installed=1,stopped=1,retired=0,resolver_ready=1,owner_hi=7,owner_lo=8,
        generation=1,owner_thread=11,total_calls=len(raw)//2,total_events=len(raw),retained_events=len(raw),
        entries=entries,returns=entries[:],original_calls=entries[:],metadata_bytes=payload_bytes,
        record_capacity_bytes=300000,payload_capacity_bytes=8388608,elapsed_ns=clock)
    state.update(dict.fromkeys(('active_calls','scope_token','live_registrations','registration_count','first_error',
        'abi_exceptions','getter_capture_failures','overflow','foreign_scope_thread_calls','outside_scope_calls'),0))
    state['arrays']=dict(capture_calls=count*6,getter_calls=count*18,acquired_arrays=count*18,released_arrays=count*18,
        **dict.fromkeys(('active_captures','active_getters','live_arrays','getter_failures','getter_exceptions',
            'capture_exceptions','ownership_errors'),0))
    gstate=dict(version=1,size=344,installed=1,stopped=1,retired=0,resolver_ready=1,owner_hi=7,owner_lo=8,
        generation=1,thread=11,total_calls=len(typed)//2,total_events=len(typed),retained_events=len(typed),
        record_capacity_bytes=256*audit.EVENT_BYTES,elapsed_ns=clock)
    gstate.update(dict.fromkeys(('active_calls','pending_executors','untracked_executors','overflow','first_error',
        'abi_exceptions','outside_scope_calls','foreign_thread_calls','getter_failures','failed_with_executor'),0))
    gstate.update({k:count for k in ('prepares','prepare_returns','executes','execute_returns','original_prepares',
        'original_executes','paired_execute_returns')});gstate.update({k:r['address'] for k,r in owners.items()})
    fresh={j:dict.fromkeys(('installed','retired','total_calls','first_error','abi_exceptions','active_calls','record_capacity_bytes'),0)
           for j in ('containers','gmm2')}
    after=copy.deepcopy(fresh);after['gmm2'].update(resolver_ready=1,**{k:r['address'] for k,r in owners.items()})
    startup=dict(schema='demoncat.quant.gmm2_startup.v1',status='RESOLVED_BEFORE_LLM',identity=process,
        native_build_sha256='a'*64,native_profile_sha256='b'*64,native_compile_source_commit='c'*40,
        original_functions=owners,state_before=fresh,state_after=after,operator_calls=0,explicit_device_query_calls=0,
        install_calls=0,loader_constructor_driver_calls='NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION')
    o=dict(schema='demoncat.quant.container_trace.v1',events=raw,state=state,public_scopes=scopes,
        mapping=dict(base=300000000,path='/container-trace/fixture.so',sha256='d'*64),build_sha256='a'*64,
        profile_sha256='b'*64,gmm2=dict(schema='demoncat.quant.gmm2_trace.v1',events=typed,state=gstate,
        formal_names=list(audit.FORMALS),failures=[],startup=startup,worker_identity=identity,
        original_execute_return_is_not_device_drain=True))
    return o,profile,dict(events=graph_events)


class Gmm2OriginalTests(unittest.TestCase):
    def good(self,n=1):
        o,p,g=fixture(n);summary,groups=audit.review_typed(o,p,expected_pairs=n)
        audit.review_containers(o,p,groups);audit.review_named_outputs(o,g)
        return o,p,g,summary

    def test_import_does_not_load_runtime_native_ctypes_or_torch(self):
        script=('import sys; from experiments.worker_core_v11_20261118 import review_quant_gmm2_observations; '
            'assert all(n not in sys.modules for n in ("torch","ctypes","fault_injection.quant_gmm2_native",'
            '"fault_injection.quant_container_native"))')
        subprocess.run([sys.executable,'-B','-c',script],cwd=audit.ROOT,check=True,capture_output=True)

    def test_actual_review_dependency_closure_is_separate_from_runtime(self):
        sources=audit.reviewer_sources()
        self.assertEqual(set(sources),{'review_quant_gmm2_observations.py','review_quant_container_observations.py',
            'review_quant_routed_frames.py','audit_quant_source87_routes.py'})
        for name,sha in sources.items():
            self.assertEqual(sha,audit.digest((Path(audit.__file__).parent/name).read_bytes()))

    def test_full_26_pairs_recount_current_slots_and_no_authority(self):
        o,p,g,s=self.good(26)
        self.assertEqual(s['actual_pairs'],26);self.assertEqual(s['joined_nonnull_slots'],390)
        self.assertEqual(s['signed_quant_group_sizes'],{'-11':26});self.assertFalse(s['row_authority'])
        self.assertEqual(audit.review_named_outputs(o,g)['actual_named_output_returns'],26)

    def test_group_metadata_arrays_add_to_creators_not_subtracted_from_raw(self):
        o,p,g,s=self.good();before=copy.deepcopy(o)
        _,groups=audit.review_typed(o,p,1);summary=audit.review_containers(o,p,groups)
        self.assertEqual(summary['creator_descriptor_members'],5)
        self.assertEqual(summary['metadata_array_captures'],6);self.assertEqual(o,before)
        o['state']['arrays'].update(capture_calls=5,getter_calls=15,acquired_arrays=15,released_arrays=15)
        with self.assertRaisesRegex(ValueError,'ARRAY_ACCOUNTING'):audit.review_containers(o,p,groups)

    def test_assessment_pass_and_refused_strings_are_never_oracles(self):
        for reported in ('PASS','OBSERVABILITY_REFUSED'):
            o,p,g=fixture();o['assessment']=dict(status=reported);o['gmm2']['assessment']=dict(status=reported)
            self.assertEqual(audit.review_typed(o,p,1)[0]['actual_pairs'],1)
        o['gmm2']['events'][0]['formals'][0]=999
        with self.assertRaises(ValueError):audit.review_typed(o,p,1)

    def test_missing_extra_or_duplicate_events_refuse(self):
        for mode in ('missing','extra','duplicate_call'):
            o,p,g=fixture()
            if mode=='missing':o['gmm2']['events'].pop()
            elif mode=='extra':o['gmm2']['events'].append(copy.deepcopy(o['gmm2']['events'][0]))
            else:o['gmm2']['events'][2]['call']=1
            with self.assertRaises(ValueError):audit.review_typed(o,p,1)

    def test_typed_signed_range_and_bool_are_not_uint64(self):
        for value in (None, True,1<<64,-1):
            o,p,g=fixture();o['gmm2']['events'][0]['formals'][0]=value
            o['gmm2']['events'][1]['formals'][0]=value
            with self.assertRaises(ValueError):audit.review_typed(o,p,1)
        o,p,g=fixture()
        for e in o['gmm2']['events'][:2]:e['scalars'][4]=1<<63;e['formals'][17]=1<<63
        with self.assertRaisesRegex(ValueError,'FORMALS'):audit.review_typed(o,p,1)

    def test_formals_order_signed_scalar_and_public_named_inputs(self):
        for mutate in (
            lambda e:e['formals'].__setitem__(0,999),
            lambda e:e['scalars'].__setitem__(4,11),
            lambda e:e['containers'].reverse(),
            lambda e:e['group_list'].__setitem__('address',999)):
            o,p,g=fixture()
            for e in o['gmm2']['events'][:2]:mutate(e)
            with self.assertRaises(ValueError):audit.review_typed(o,p,1)
        o,p,g=fixture();o['public_scopes'][0]['evidence']['formals']['x'][0]['tensor']['storage_address']=999
        with self.assertRaisesRegex(ValueError,'NAMED_GWS_PUBLIC_TENSOR'):audit.review_typed(o,p,1)

    def test_null_slot_cannot_borrow_empty_live_registration(self):
        o,p,g=fixture()
        for e in o['gmm2']['events'][:2]:
            e['formals'][2]=0;e['containers'][2]['handle']=0
        with self.assertRaisesRegex(ValueError,'NULL_SLOT'):audit.review_typed(o,p,1)

    def test_reused_address_requires_new_current_creator(self):
        o,p,g=fixture(2)
        for i in (4,5):o['gmm2']['events'][i]['containers'][0]=copy.deepcopy(o['gmm2']['events'][i-4]['containers'][0])
        with self.assertRaisesRegex(ValueError,'CONTAINER_NOT_LIVE'):audit.review_typed(o,p,2)

    def test_late_destroy_or_missing_destroy_cannot_pass(self):
        o,p,g=fixture();death=next(r for r in o['events'] if r['operation']==2 and r['phase']==1)
        death['registration']=999
        with self.assertRaisesRegex(ValueError,'MISSING_CREATOR_DESTROY'):audit.review_typed(o,p,1)

    def test_executor_reuse_cross_scope_short_workspace_or_stream_drift(self):
        for field,value in (('executor',9999),('prepare_registration',99),('workspace_bytes',15),('scope_token',99),('thread',12)):
            o,p,g=fixture()
            for e in o['gmm2']['events'][2:]:e[field]=value
            with self.assertRaises(ValueError):audit.review_typed(o,p,1)
        o,p,g=fixture();o['gmm2']['events'][3]['stream']=222
        with self.assertRaisesRegex(ValueError,'EXECUTE_RETURN_ARGUMENTS'):audit.review_typed(o,p,1)

    def test_unknown_owner_caller_device_pending_and_errors_refuse(self):
        for field,value in (('caller_base',300000000),('function',0),('pid',10),('device',1),('error',1),('original_calls',2)):
            o,p,g=fixture();o['gmm2']['events'][1][field]=value
            with self.assertRaises(ValueError):audit.review_typed(o,p,1)
        for name in ('pending_executors','untracked_executors','overflow','abi_exceptions','first_error'):
            o,p,g=fixture();o['gmm2']['state'][name]=1
            with self.assertRaisesRegex(ValueError,'NOT_COMPLETE_IDLE'):audit.review_typed(o,p,1)

    def test_cross_journal_clock_boundary_is_checked(self):
        o,p,g=fixture();o['gmm2']['events'][0]['container_events']=0
        with self.assertRaisesRegex(ValueError,'CROSS_JOURNAL_TIME_JOIN'):audit.review_typed(o,p,1)

    def test_named_output_is_actual_public_return_not_same_shape(self):
        o,p,g,s=self.good();g['events'][0]['outputs'][0]['storage_address']+=16
        with self.assertRaisesRegex(ValueError,'OUTPUT_ACTUAL_PUBLIC_RETURN'):audit.review_named_outputs(o,g)

    def startup_fixture(self):
        o,p,g=fixture();startup=o['gmm2']['startup'];identity=o['gmm2']['worker_identity']
        result=dict(identity=identity,worker_before_requests=copy.deepcopy(identity),pre_llm_startup=copy.deepcopy(startup),
            setup=dict(worker_binding=copy.deepcopy(identity),pre_llm_startup=copy.deepcopy(startup)))
        inv=dict(worker_identity=identity['worker'],lifetime=dict(worker=copy.deepcopy(identity['worker'])))
        build=dict(source_commit='c'*40,profile_sha256='b'*64)
        return o,p,startup,result,inv,build

    def test_same_process_startup_function_metadata_is_independently_joined(self):
        o,p,s,r,i,b=self.startup_fixture()
        self.assertTrue(audit.review_startup(s,o,r,i,b,p,'a'*64)['same_worker'])
        for mutation in ('pid','symbol','canonical','sha','preinstalled','process_start'):
            o,p,s,r,i,b=self.startup_fixture()
            if mutation=='pid':s['identity']['pid']=10
            elif mutation=='symbol':s['original_functions']['prepare_function']['symbol']='other'
            elif mutation=='canonical':s['original_functions']['prepare_function']['canonical_path']='/other.so'
            elif mutation=='sha':s['original_functions']['prepare_function']['file_sha256']='e'*64
            elif mutation=='preinstalled':s['state_before']['gmm2']['installed']=1
            else:s['identity']['process_start_identity']='process_local:9:nonce'
            r['pre_llm_startup']=copy.deepcopy(s);r['setup']['pre_llm_startup']=copy.deepcopy(s)
            with self.assertRaises(ValueError):audit.review_startup(s,o,r,i,b,p,'a'*64)

    def cleanup_fixture(self):
        o,p,g=fixture();cleanup=copy.deepcopy(o)
        released=dict(containers=copy.deepcopy(o['state']),gmm2=copy.deepcopy(o['gmm2']['state']),
            resource_scope='OBSERVER_HOST_JOURNALS_ONLY_NO_DEVICE_BUFFERS',device_drain_proven=False,
            preloaded_library_remains_resident=True)
        for state,names in ((released['containers'],('retained_events','metadata_bytes','live_registrations',
            'active_calls','scope_token','record_capacity_bytes','payload_capacity_bytes','registration_count')),
            (released['gmm2'],('retained_events','active_calls','pending_executors','untracked_executors','record_capacity_bytes'))):
            state.update(installed=0,retired=1,stopped=1)
            state.update(dict.fromkeys(names,0))
        serving=dict(installed=False,observer_cleanup_pending=False,error=None,owned_hooks=0)
        tokens=[[1,2] for _ in range(8)]
        result=dict(cleanup=dict(errors=[],failures=[],first_failure=None,released=released,inventory=serving),
            healthy_before=copy.deepcopy(tokens),healthy_after=copy.deepcopy(tokens),phases=[dict(output_tokens=copy.deepcopy(tokens))])
        return result,o,cleanup,serving

    def test_cleanup_reads_original_events_and_owner_counters_not_success_bool(self):
        result,o,c,s=self.cleanup_fixture();before=copy.deepcopy((result,o,c,s))
        summary=audit.review_cleanup(result,o,c,s)
        self.assertTrue(summary['captured_tokens_equal_healthy_before_and_after'])
        self.assertFalse(summary['text_or_logits_comparison']);self.assertEqual(before,(result,o,c,s))
        for kind in ('event','owner','counter','pending','capacity','hooks','failed_stop','late_error','device_drain','tokens_after','tokens_capture'):
            result,o,c,s=self.cleanup_fixture()
            if kind=='event':c['gmm2']['events'][0]['formals'][0]=12345
            elif kind=='owner':result['cleanup']['released']['gmm2']['owner_hi']=123
            elif kind=='counter':result['cleanup']['released']['gmm2']['original_executes']=0
            elif kind=='pending':result['cleanup']['released']['gmm2']['pending_executors']=1
            elif kind=='capacity':result['cleanup']['released']['containers']['record_capacity_bytes']=1
            elif kind=='hooks':s['owned_hooks']=1
            elif kind=='failed_stop':result['cleanup']['errors']=['STOP_FAILED']
            elif kind=='late_error':result['cleanup']['released']['gmm2']['first_error']=1
            elif kind=='device_drain':result['cleanup']['released']['device_drain_proven']=True
            elif kind=='tokens_after':result['healthy_after'][0]=[1,3]
            else:result['phases'][0]['output_tokens'][0]=[1,3]
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.review_cleanup(result,o,c,s)

    def test_raw_numeric_false_and_null_do_not_equal_integer_zero(self):
        for value in (False,None):
            o,p,g=fixture();o['gmm2']['events'][0]['error']=value
            with self.assertRaisesRegex(ValueError,'NOT_NATIVE_INTEGER_FIELDS'):audit.review_typed(o,p,1)

    def test_missing_loaded_source_or_unhashed_support_cannot_pass(self):
        result=dict(setup=dict(loaded_sources=[]))
        with self.assertRaisesRegex(ValueError,'EXACT_LOADED_RUNTIME_SOURCE_SET'):
            audit.review_runtime_sources(result,{}, {},audit.ROOT)

    def test_partial_result_and_success_boolean_never_grant_saved_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            report=audit.review({audit.PREFIX+'result.json':dict(status='PASS',c2=True)}, {}, audit.ROOT,
                                directory,'a'*64,audit.ROOT)
        self.assertFalse(report['observation_consistent']);self.assertTrue(report['issues'])
        for name in audit.NONE_GRANTED:self.assertFalse(report[name])

    def test_strict_json_keeps_null_but_refuses_duplicates_or_nan(self):
        self.assertIsNone(audit.parsed(b'{"value":null}')['value'])
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":Infinity}'):
            with self.assertRaises(ValueError):audit.parsed(raw)

    def test_archive_exact_original_bytes_and_no_path_or_duplicate_injection(self):
        def archive(path, names):
            with tarfile.open(path,'w:gz') as t:
                for name in names:
                    raw=b'{}';m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
            return audit.digest(path.read_bytes())
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'raw.tar.gz';sha=archive(p,audit.MEMBERS)
            values,hashes=audit.load_archive(p,sha);self.assertEqual(len(values),9)
            self.assertEqual(hashes['controller.json']['sha256'],audit.digest(b'{}'))
            for names in (['../x.json'],['controller.json']*2):
                sha=archive(p,names)
                with self.assertRaises(ValueError):audit.load_archive(p,sha)
            with self.assertRaisesRegex(ValueError,'PACKAGE_CHANGED'):audit.load_archive(p,'a'*64)


if __name__=='__main__':unittest.main()
