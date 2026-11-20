"""Real CPU Attention objects with an explicit fake compiled callback ABI.

This does not execute CANN, establish a prepared workspace, or authorize F.
"""
from copy import deepcopy
import ctypes as C
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import attention28_typed_plan as T
import test_attention28_replay_binding as fixtures


def typed_packet(plan, path):
    def tensor(name):
        p = plan.public_metadata[name]
        return dict(descriptor=id(p), storage_address=p['storage_address'], data_address=p['data_address'],
            offset=p['storage_offset'], native_storage_offset=0, dtype=2 if name == 'atten_mask' else 1,
            format=2 if name == 'atten_mask' else 47, shape=p['shape'], stride=p['stride'],
            storage_shape=[p['storage_bytes'] // p['element_bytes']])
    return dict(function=dict(address=17, path=str(path), mapped_file_identity_matches=True),
        query=tensor('query'), key=dict(object=21, members=[tensor('key')]),
        value=dict(object=22, members=[tensor('value')]), mask=tensor('atten_mask'),
        output=dict(descriptor=23, storage_address=0x40000000, data_address=0x40000000,
            offset=0, native_storage_offset=0, dtype=1, format=47, shape=[2048,16,128],
            stride=[2048,128,1], storage_shape=[4194304]),
        lse=dict(descriptor=24, storage_address=0x50000000, data_address=0x50000000,
            offset=0, native_storage_offset=0, dtype=0, format=2, shape=[0], stride=[0], storage_shape=[0]),
        seq_q=dict(object=25, values=[2048]), seq_kv=dict(object=26, values=[2048]),
        optionals=[0]*25, scalars=[16,2147483647,2147483647,8,3,0,128,0,0,0,0,0,0],
        layout='TND', scale=128**-.5, softmax_lse=False)


class Export:
    def __init__(self, call):self.call=call
    def __call__(self, *args):return self.call(*args)


class FakePlanLibrary:
    """The fixture's only Native ABI seam. No fake here claims device work."""
    def __init__(self):
        self.value=dict(schema='demoncat.tnd28.live_typed_plan.v2', bound=0, prepare_entered=0,
            prepare_accepted=0, prepare_admitted=0, execute_entered=0, safe_release=0, terminal_uncertain=0,
            first_error=0, fault_authorized=False)
        self.callback=None;self.token=None;self.clear_error=None;self.query_error=0
        setattr(self,T.SYMBOLS[0],Export(self.bind))
        setattr(self,T.SYMBOLS[1],Export(self.clear))
        setattr(self,T.SYMBOLS[2],Export(self.query))
    def bind(self, token, callback):
        assert not self.value['bound'];self.token=token;self.callback=callback;self.value['bound']=1;return 0
    def clear(self, token, callback):
        assert token==self.token and C.cast(callback,C.c_void_p).value==C.cast(self.callback,C.c_void_p).value
        if self.clear_error=='before':return -1
        self.value['bound']=0;self.callback=None
        return -1 if self.clear_error=='after' else 0
    def query(self, token, output, capacity):
        if self.query_error:return self.query_error
        assert token==self.token
        raw=json.dumps(self.value).encode()+b'\0';assert len(raw)<=capacity
        C.memmove(output,raw,len(raw));return 0
    def emit(self, stage, value):
        raw=json.dumps(value,separators=(',',':')).encode();buffer=C.create_string_buffer(raw)
        if stage==1:self.value['prepare_entered']=1
        else:self.value['execute_entered']=1
        status=self.callback(self.token,stage,C.cast(buffer,C.c_void_p),len(raw))
        if stage==1:self.value['prepare_accepted']=self.value['prepare_admitted']=int(status==1)
        self.value['safe_release']=int(stage==2 or status!=1)  # explicit successful cancellation seam
        return status


class TypedPlanContracts(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.ReplayInputContracts();self.f.setUp();self.addCleanup(self.f.doCleanups)
        # Source123 actual Q/K are contiguous; only V retains the projection view.
        self.f.model.attn.q=self.f.model.attn.q.contiguous();self.f.model.attn.k=self.f.model.attn.k.contiguous()
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.path=self.root/'libcust_opapi.so';self.path.write_bytes(b'EXPLICIT_CPU_FAKE_NOT_ELF')
        self.native=FakePlanLibrary()
        self.query=NS(library=self.native,path=self.path.resolve(),sha256='b'*64)
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.paged33_loading.symbol_owner',
            return_value=dict(path=str(self.query.path),sha256='b'*64,address=17)))
        self.live=None

    def call(self, action):
        def callback(row, once):
            plan=self.f.binding.begin(row)
            self.live=T.LiveTypedPlan(self.query,self.f.binding,plan,'b'*32);self.live.install()
            self.f.during_original=lambda: action(plan)
            output=once()
            self.live.clear();self.f.binding.close_public(plan)
            return output
        self.f.run_call(callback)

    def test_two_actual_callback_stages_hold_current_objects_without_device_read(self):
        def action(plan):
            value=typed_packet(plan,self.path)
            with patch.object(fixtures.B,'capture_inputs',side_effect=AssertionError('no inside-scope value read')):
                self.assertEqual(self.native.emit(1,value),1)
                self.assertEqual(self.native.emit(2,value),1)
            self.assertFalse(self.live.snapshot()['current_value_proof'])
            self.assertFalse(self.live.snapshot()['fault_authorized'])
        self.call(action)
        self.assertEqual(len(self.live.events),2);self.assertFalse(self.live.bound)

    def test_actual_shapes_offsets_flat_storage_format_and_optionals_are_strict(self):
        def action(plan):
            value=typed_packet(plan,self.path);T.validate_prepare(plan,value)
            changes=(('query','shape',[2047,16,128]),('query','format',2),
                ('value','offset',0),('value','stride',[1024,128,1]),
                ('key','storage_shape',[2048,8,128]),('lse','stride',[1]),
                ('output','shape',[1,16,128]))
            for name,key,changed in changes:
                with self.subTest(name=name,key=key):
                    item=deepcopy(value);target=item[name]
                    if name in ('key','value'):target=target['members'][0]
                    target[key]=changed
                    with self.assertRaises(RuntimeError):T.validate_prepare(plan,item)
            for key,changed in (('optionals',[0]*24+[1]),('scalars',[16]*13),('scale',.125)):
                item=deepcopy(value);item[key]=changed
                with self.assertRaises(RuntimeError):T.validate_prepare(plan,item)
            self.assertEqual(self.native.emit(1,value),1);self.assertEqual(self.native.emit(2,value),1)
        self.call(action)

    def test_same_value_clone_during_original_refuses_before_parent_and_preserves_error(self):
        def action(plan):
            value=typed_packet(plan,self.path);self.assertEqual(self.native.emit(1,value),1)
            frame=self.f.session.live_api_calls[-1]
            self.f.session.live_api_calls[-1]=(frame[0],frame[1],(frame[2][0].clone(),*frame[2][1:]),frame[3])
            self.assertEqual(self.native.emit(2,value),0)
            self.assertIn('CURRENT_CALL_IDENTITY_CHANGED',self.live.first_error)
            self.f.session.live_api_calls[-1]=frame
        self.call(action)

    def test_foreign_thread_callback_and_reordered_or_repeated_callback_refuse(self):
        def action(plan):
            value=typed_packet(plan,self.path);result=[]
            thread=threading.Thread(target=lambda:result.append(self.native.emit(1,value)))
            thread.start();thread.join();self.assertEqual(result,[0])
            self.assertIn('OWNER_THREAD_CHANGED',self.live.first_error)
            self.assertEqual(self.native.emit(2,value),0)
        self.call(action)

    def test_preallocation_typed_rejection_can_remove_callback_without_executor_retry(self):
        def action(plan):
            value=typed_packet(plan,self.path);value['value']['members'][0]['data_address']+=2
            self.assertEqual(self.native.emit(1,value),0)
            self.assertEqual(self.native.value['prepare_accepted'],0)
            self.assertEqual(self.native.value['execute_entered'],0)
        self.call(action)

    def test_pending_native_state_keeps_callback_until_true_completion(self):
        def action(plan):
            value=typed_packet(plan,self.path);self.assertEqual(self.native.emit(1,value),1)
            with self.assertRaisesRegex(RuntimeError,'NATIVE_PENDING_RETAINED'):self.live.clear()
            self.assertIsNotNone(self.native.callback);self.assertTrue(self.live.bound)
            self.assertEqual(self.native.emit(2,value),1)
        self.call(action)

    def test_clear_after_effect_error_is_read_back_and_retry_does_not_reclear(self):
        def action(plan):
            value=typed_packet(plan,self.path);self.native.emit(1,value);self.native.emit(2,value)
            self.native.clear_error='after'
            with self.assertRaisesRegex(RuntimeError,'CLEAR_FAILED'):self.live.clear()
            self.assertFalse(self.live.bound);self.assertIsNone(self.native.callback)
            self.native.clear_error='before'  # A second native clear would fail.
            self.live.clear()
        self.call(action)

    def test_callback_invalid_or_oversize_payload_is_bounded_and_returns_zero(self):
        def action(plan):
            raw=C.create_string_buffer(b'{}')
            self.assertEqual(self.live.callback(b'b'*32,1,C.cast(raw,C.c_void_p),32769),0)
            self.assertIn('ORDER_OR_BOUND',self.live.first_error)
        self.call(action)


if __name__=='__main__':unittest.main()
