"""CPU fixtures for same-child readback; no device or F authority."""
from copy import deepcopy
import json
import unittest
from demoncat.runtime.vllm_ascend.worker_extension.attention28_value_plan import verify_copy_launch,COPY_ELF_SHA256

class CopyLaunchTests(unittest.TestCase):
    def packet(self):
        return dict(schema='demoncat.tnd28.current_copy_launch.v1',before=0,after=1,begin_ns=10,end_ns=20,
            stream=37,prepared_v_address=4096,launch=dict(schema='demoncat.live_launch_readback.v1',sequence=1,
            event='launch',call_id='a'*32,pid=51,tid=61,launch_begin_ns=11,launch_end_ns=19,stream=37,
            result=0,snapshot_valid=True,observer_mutates_arguments=False,generation=1,tiling_key='524000000',
            block_dim=40,binary_file='/actual/registration.o'))
    def verify(self,p,sha=COPY_ELF_SHA256):
        return verify_copy_launch(json.dumps(p).encode(),'a'*32,pid=51,tid=61,file_digest=lambda path:sha)
    def test_same_child_readback_does_not_authorize_fault(self):
        out=self.verify(self.packet());self.assertTrue(out['current_child_launch'])
        self.assertFalse(out['fault_authorized']);self.assertFalse(out['prepared_parent_argument_join'])
    def test_foreign_owner_token_and_stale_sequence_refused(self):
        for key,value in [('call_id','b'*32),('pid',52),('tid',62),('sequence',0)]:
            with self.subTest(key=key):
                p=self.packet();p['launch'][key]=value
                with self.assertRaises(RuntimeError):self.verify(p)
    def test_failed_unknown_mutating_or_wrong_kernel_refused(self):
        for key,value in [('result',1),('result',False),('snapshot_valid',False),('observer_mutates_arguments',True),
                          ('generation',0),('tiling_key','5000000000000200103'),('block_dim',20)]:
            with self.subTest(key=key,value=value):
                p=self.packet();p['launch'][key]=value
                with self.assertRaises(RuntimeError):self.verify(p)
    def test_extra_launch_and_outside_actual_window_refused(self):
        for key,value in [('before',-1),('before',True),('after',2),('begin_ns',12),('end_ns',18),('stream',38),('prepared_v_address',0)]:
            with self.subTest(key=key):
                p=self.packet();p[key]=value
                with self.assertRaises(RuntimeError):self.verify(p)
    def test_same_key_other_binary_refused(self):
        with self.assertRaisesRegex(RuntimeError,'REVIEWED_SLICE'):self.verify(self.packet(),'f'*64)
    def test_duplicate_nonfinite_or_oversized_packet_refused(self):
        for raw in (b'{}'*65536,b'{"before":0,"before":1}',b'{"before":NaN}'):
            with self.subTest(raw=raw[:40]):
                with self.assertRaises((RuntimeError,ValueError,KeyError)):
                    verify_copy_launch(raw,'a'*32,pid=51,tid=61,file_digest=lambda _:COPY_ELF_SHA256)

if __name__=='__main__':unittest.main()
