import copy
import ctypes
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from fault_injection import quant_container_native as native
from demoncat.runtime.vllm_ascend.worker_extension.quant_container_scopes import ContainerScopes
from demoncat.runtime.vllm_ascend.worker_extension.quant_token_routes import SCHEMAS
from experiments.worker_core_v11_20261118.prepare_quant_container_case import adapt
from experiments.worker_core_v11_20261118 import probe_quant_containers as probe


def observation():
    state = dict(installed=1, stopped=1, active_calls=0, scope_token=0, live_registrations=0, first_error=0,
        abi_exceptions=0, overflow=0, getter_capture_failures=0, foreign_scope_thread_calls=0,
        total_events=8, retained_events=8, total_calls=4, entries=[1]*4, returns=[1]*4, original_calls=[1]*4,
        arrays=dict(acquired_arrays=3, released_arrays=3, active_captures=0, active_getters=0, live_arrays=0,
            getter_failures=0, getter_exceptions=0, capture_exceptions=0, ownership_errors=0))
    scopes = [dict(native_scope=dict(token=i, forward=1, operator_sequence=i*10, boundary_sequence=i*9,
        stage=i, thread=1), status='RETURNED', original_calls=1, original_error=None) for i in range(1,5)]
    events = []
    for call, operation, stage, registration, handle in ((1,1,3,1,100),(2,2,3,1,100),(3,3,1,3,200),(4,4,1,3,200)):
        for phase in (1,2):
            events.append(dict(sequence=len(events)+1, phase=phase, operation=operation, call=call,
                registration=registration if phase==2 or operation%2==0 else 0, handle=handle if phase==2 or operation%2==0 else 0,
                pid=9, thread=1, scope_token=stage, forward=1, operator_sequence=stage*10, boundary_sequence=stage*9,
                stage=stage, caller_kind=2 if stage==1 else 1, function=1000+operation, caller=2000+stage,
                error=0, status=0, original_calls=phase-1))
    return dict(schema='demoncat.quant.container_trace.v1', state=state, events=events, public_scopes=scopes)


class AssessmentTests(unittest.TestCase):
    def test_complete_fixture_only_establishes_observability(self):
        value = native.assess_observation(observation())
        self.assertEqual(value['status'], 'FOUR_CONTAINER_OBSERVABILITY_COMPLETE')
        for key in ('qualified', 'request_rows_proven', 'c1', 'c2', 'c3'):
            self.assertFalse(value[key])

    def test_missing_raw_entry_cannot_hide_behind_total(self):
        value = observation(); value['events'].pop(0)
        reasons = native.assess_observation(value)['reasons']
        self.assertIn('INDEPENDENT_EVENT_DENOMINATOR_INCOMPLETE', reasons)
        self.assertIn('ORIGINAL_CALL_PAIR_INCOMPLETE', reasons)

    def test_library_name_without_actual_caller_classification_refuses(self):
        value = observation(); value['events'][0]['caller_kind'] = 0
        self.assertIn('ACTUAL_CALLER_IMAGE_UNPROVEN', native.assess_observation(value)['reasons'])

    def test_cross_scope_or_thread_relabel_refuses(self):
        for key in ('scope_token', 'thread', 'operator_sequence', 'boundary_sequence'):
            with self.subTest(key=key):
                value = observation(); value['events'][0][key] = 77
                self.assertNotEqual(native.assess_observation(value)['status'], 'FOUR_CONTAINER_OBSERVABILITY_COMPLETE')

    def test_destroy_must_join_current_registration_not_address(self):
        value = observation(); value['events'][3]['registration'] = 99
        reasons = native.assess_observation(value)['reasons']
        self.assertIn('UNOBSERVED_OR_DUPLICATE_DESTROY', reasons)
        self.assertIn('RETAINED_CONTAINER_LIFETIMES', reasons)

    def test_failed_destroy_or_overflow_keeps_refusal(self):
        for key in ('first_error','overflow','live_registrations','abi_exceptions','foreign_scope_thread_calls'):
            with self.subTest(key=key):
                value=observation();value['state'][key]=1
                self.assertIn('NATIVE_OBSERVATION_NOT_COMPLETE_AND_IDLE', native.assess_observation(value)['reasons'])

    def test_no_synthetic_integer_entry_to_complete_empty_optional(self):
        value=observation();value['state']['entries'][2]=0
        self.assertIn('FOUR_REAL_FUNCTIONS_NOT_OBSERVED',native.assess_observation(value)['reasons'])

    def test_sdk_arrays_must_all_be_released(self):
        value=observation();value['state']['arrays']['live_arrays']=1
        self.assertIn('GETTER_ARRAY_ACCOUNTING_FAILED',native.assess_observation(value)['reasons'])

    def test_failed_public_original_is_not_recovered_by_native_pairs(self):
        value=observation();value['public_scopes'][0]['original_error']='RuntimeError'
        self.assertIn('PUBLIC_SCOPE_INCOMPLETE',native.assess_observation(value)['reasons'])

    def test_record_is_a_deep_snapshot(self):
        value=native.instance(native.State);value.entries[0]=7
        first=native.record(value);first['entries'][0]=99
        self.assertEqual(native.record(value)['entries'][0],7)

    def test_all_struct_sizes_are_bounded_and_abi_initialized(self):
        for cls in (native.Config,native.Scope,native.Event,native.State,native.Sizes):
            value=native.instance(cls)
            self.assertEqual(value.version,1);self.assertEqual(value.size,ctypes.sizeof(cls))
        self.assertLess(ctypes.sizeof(native.Event)*8192,4*1024*1024)

    def test_bundle_sha_refuses_before_any_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory,'build.json').write_text('{}')
            with patch.object(native.C,'CDLL') as loader:
                with self.assertRaisesRegex(RuntimeError,'BUILD_SHA_CHANGED'):
                    native.validate_bundle(directory,'0'*64)
                loader.assert_not_called()

    def test_scope_end_retry_preserves_first_original_failure_and_receipts(self):
        value=native.NativeContainers.__new__(native.NativeContainers)
        value.config=native.instance(native.Config)
        row=dict(native_scope=dict(token=1));value.active_scope=row
        codes=iter([1,0]);value.functions={'scope_end':lambda *args:next(codes)}
        with self.assertRaisesRegex(RuntimeError,'SCOPE_END_FAILED:1'):
            value.leave(original_calls=1,error='ACTUAL_ORIGINAL_FAILURE')
        self.assertIs(value.active_scope,row)
        value.leave(original_calls=1,error='CLEANUP_RETRY_AFTER_SCOPE_FAILURE')
        self.assertIsNone(value.active_scope)
        self.assertEqual(row['original_error'],'ACTUAL_ORIGINAL_FAILURE')
        self.assertEqual(row['status'],'ORIGINAL_FAILED')
        self.assertEqual([x['native_status'] for x in row['leave_attempts']],[1,0])

    def test_cleanup_retry_saves_new_original_and_retains_owner_until_release(self):
        class Trace:
            started=True
            attempts=0
            def stop(self):pass
            def snapshot(self):return dict(original_calls=[1,1,1,1])
            def query(self):return dict(installed=1)
            def release(self):
                self.attempts+=1
                if self.attempts==1:raise RuntimeError('REAL_DESTROY_STILL_PENDING')
                return dict(installed=0)
        worker=SimpleNamespace(_demoncat_container_trace=Trace())
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'trace'
            with patch.object(probe,'actual',return_value=worker),patch.object(probe,'Path',return_value=folder):
                first=probe.cleanup(worker)
                self.assertTrue(first['errors']);self.assertTrue(hasattr(worker,'_demoncat_container_trace'))
                first_bytes=(folder/'cleanup-observation-001.json').read_bytes()
                second=probe.cleanup(worker)
            self.assertFalse(second['errors']);self.assertFalse(hasattr(worker,'_demoncat_container_trace'))
            self.assertEqual((folder/'cleanup-observation-001.json').read_bytes(),first_bytes)
            self.assertTrue((folder/'cleanup-observation-002.json').is_file())

    def test_case_keeps_quant_first_no_fault_no_integer_helpers(self):
        case=dict(id='old',mounts=[],review=dict(resource_budget={}))
        value=adapt(case,PurePosixPath('/root/trace'),'a'*64,'b'*64)
        self.assertIn('ASCEND_CUSTOM_OPP_PATH=/quant/runtime/vendor:/container-trace/runtime/vendor:',value['command'])
        self.assertNotIn('capture-integer-routes',value['command'])
        self.assertNotIn('arm',value['command'])
        self.assertEqual(value['review']['expected_hits'],0)
        self.assertEqual(value['review']['resource_budget']['container_calls'],4096)
        self.assertEqual(value['review']['resource_budget']['timeout_seconds'],1800)
        self.assertEqual(value['mounts'],[['/root/trace','/container-trace','ro']])


class FakeNative:
    def __init__(self):
        self.active_scope=None;self.scopes=[];self.fail_leave=False;self.fail_enter=False
    def enter(self,**values):
        if self.fail_enter:raise RuntimeError('BEGIN_REFUSED')
        self.active_scope=copy.deepcopy(values);self.scopes.append(self.active_scope)
    def leave(self,**values):
        if self.fail_leave:raise RuntimeError('TRANSIENT_NATIVE_BUSY')
        self.active_scope.update(values);self.active_scope=None


class Function:
    def __init__(self):self._schema=torch._C.parse_schema(SCHEMAS['gmm2'])
    def __str__(self):return 'npu.npu_grouped_matmul.default'


class ScopeTests(unittest.TestCase):
    def setUp(self):
        self.previous=sys.getprofile();self.callback=lambda *a:None;sys.setprofile(self.callback)
        self.addCleanup(sys.setprofile,self.previous)
        graph=SimpleNamespace(active=True,ending=False,errors=[],overflow=0,epochs={},forward_id='request-A',
            storage_key=lambda meta:(meta['storage_address'],meta['storage_bytes']))
        self.boundary=SimpleNamespace(graph=graph,active=True,ending=False,errors=[],callback=self.callback,
            thread=threading.get_ident(),model=torch.nn.Linear(2,2).eval(),operator_aliases={},
            scope=dict(request_ids=['request-A']),_verify_bindings=lambda:None,
            stack=[dict(role='rchain_gmm2',sequence=17,function='actual_cpu_fixture',source_sha256='a'*64)])
        self.native=FakeNative();self.observer=ContainerScopes(self.boundary,self.native)
        self.state=dict(func=Function(),args=([torch.ones(8,4,dtype=torch.int8)],
            [torch.ones(2,4,4,dtype=torch.int8)]),kwargs=dict(group_list=torch.tensor([4,4])),
            event=dict(sequence=21,original_calls=0))
        self.inference=torch.inference_mode();self.inference.__enter__();self.addCleanup(self.inference.__exit__,None,None,None)

    def test_actual_args_and_current_source_enter_then_return_once(self):
        self.observer.before_operator(self.state)
        self.assertIs(self.observer.active,self.state)
        row=self.native.scopes[0]
        self.assertEqual(row['boundary'],17);self.assertEqual(row['sequence'],21)
        self.assertEqual(row['evidence']['argument_origins']['group_list'],'KEYWORD')
        self.assertTrue(row['evidence']['input_objects'])
        self.state['event']['original_calls']=1;self.observer.after_operator(self.state)
        self.assertIsNone(self.native.active_scope);self.assertIsNone(self.observer.active)
        self.assertEqual(row['original_calls'],1)
        self.assertFalse(self.observer.report()['request_rows_proven'])

    def test_prior_error_ending_or_training_refuses_scope(self):
        for target,key,value in ((self.boundary,'ending',True),(self.boundary.graph,'errors',['old']),
                                 (self.boundary.model,'training',True)):
            old=getattr(target,key);setattr(target,key,value)
            self.observer.before_operator(self.state)
            self.assertFalse(self.native.scopes)
            setattr(target,key,old)

    def test_unknown_schema_or_missing_frame_never_guesses_scope(self):
        self.state['func']._schema=torch._C.parse_schema('npu::npu_grouped_matmul(Tensor x)->Tensor')
        self.observer.before_operator(self.state);self.assertFalse(self.native.scopes)
        self.state['func']=Function();self.boundary.stack=[]
        self.observer.before_operator(self.state);self.assertFalse(self.native.scopes)

    def test_failed_native_begin_does_not_leave_phantom_scope(self):
        self.native.fail_enter=True;self.observer.before_operator(self.state)
        self.assertIsNone(self.observer.active);self.assertNotIn('container_trace_scope',self.state)
        self.assertTrue(self.observer.report()['errors'])

    def test_native_leave_failure_retains_handle_and_cleanup_retries(self):
        self.observer.before_operator(self.state);self.state['event']['original_calls']=1
        self.native.fail_leave=True;self.observer.after_operator(self.state)
        self.assertIs(self.observer.active,self.state);self.assertIsNotNone(self.native.active_scope)
        with self.assertRaisesRegex(RuntimeError,'TRANSIENT_NATIVE_BUSY'):self.observer.close()
        self.native.fail_leave=False;self.observer.close()
        self.assertIsNone(self.observer.active);self.assertIsNone(self.native.active_scope)
        self.assertTrue(self.observer.report()['errors'])

    def test_foreign_profile_not_removed_or_used(self):
        foreign=lambda *a:None;sys.setprofile(foreign)
        self.observer.before_operator(self.state)
        self.assertIs(sys.getprofile(),foreign);self.assertFalse(self.native.scopes)


if __name__=='__main__':unittest.main()
