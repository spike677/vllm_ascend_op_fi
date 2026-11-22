import ctypes
import threading
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
from fault_injection.engines.native_regions import NativeRegions
from fault_injection.engines.region_lease import identity
from fault_injection.deployments.postwrite import PostwriteRefused


class CpuACL:
    def __init__(self):self.rows={};self.next=0
    def dcat_pw_open_stream_v2(self,address,size,device,owner,generation,stream,out):
        self.next+=1;self.rows[self.next]=dict(address=address,size=size,original=ctypes.string_at(address,size))
        out._obj.value=self.next;return 0
    def dcat_pw_copy_region_v2(self,dst,src,owner,generation,changed):
        d=self.rows[dst];s=self.rows[src]
        if d['size']!=s['size']:return 2
        ctypes.memmove(d['address'],s['address'],s['size']);changed._obj.value=d['size']//2;return 0
    def dcat_pw_apply(self,h,owner,generation,mode,indices,count,changed):
        if mode!=5:return 2
        row=self.rows[h];changed._obj.value=0
        for i in indices:
            cell=ctypes.c_uint16.from_address(row['address']+2*i)
            changed._obj.value+=int(cell.value!=0);cell.value=0
        return 0
    def dcat_pw_restore(self,h,owner,generation):
        d=self.rows[h];ctypes.memmove(d['address'],d['original'],d['size']);return 0
    def dcat_pw_release(self,h,owner,generation):del self.rows[h];return 0


class NativeRegionsTests(unittest.TestCase):
    def manager(self):
        # Explicit CPU-only transport; production constructor never accepts this.
        x=NativeRegions.__new__(NativeRegions);x.transport=NS(lib=CpuACL());x.owner=1;x.generation=2
        x.thread=threading.get_ident();x.active={};x.serial=0;x._validate=identity
        stream=NS(npu_stream=5);x._stream=lambda:stream
        return x

    def test_live_subregions_copy_consume_restore_and_repeated_hits(self):
        x=self.manager();a=torch.arange(10,dtype=torch.bfloat16);b=a+10;original=a.clone()
        class Event:
            def record(self,stream):pass
            def synchronize(self):pass
        with patch.object(torch,'npu',NS(Event=Event),create=True):
            for _ in range(3):
                dst=x.acquire(a[2:6]);src=x.acquire(b[2:6]);x.copy(dst,src)
                self.assertTrue(torch.equal(a[2:6],b[2:6]));self.assertTrue(torch.equal(a[:2],original[:2]))
                with self.assertRaisesRegex(PostwriteRefused,'CONSUMER'):x.restore(dst)
                observed=a[2:6]+1;self.assertTrue(torch.equal(observed,b[2:6]+1))
                x.consumed(dst);x.restore(dst);x.restore(src)
                self.assertTrue(torch.equal(a,original));self.assertEqual(x.count(),0)

    def test_overlap_storage_reuse_and_legal_update_refused(self):
        a=torch.ones(8,dtype=torch.bfloat16);x=self.manager();token=x.acquire(a[1:4])
        with self.assertRaisesRegex(PostwriteRefused,'OVERLAP'):x.acquire(a[2:5])
        held=x.active[token]['tensor'];held.set_(torch.zeros(3,dtype=torch.bfloat16))
        with self.assertRaisesRegex(PostwriteRefused,'IDENTITY'):x.restore(token)
        self.assertEqual(x.count(),1)
        x=self.manager();token=x.acquire(a);a.add_(1)
        with self.assertRaisesRegex(PostwriteRefused,'LEGAL_INPLACE'):x.restore(token)
        self.assertTrue(torch.equal(a,torch.full_like(a,2)))

    def test_stream_thread_and_raw_address_refused(self):
        x=self.manager();a=torch.ones(3,dtype=torch.bfloat16);token=x.acquire(a)
        x._stream=lambda:NS(npu_stream=6)
        with self.assertRaisesRegex(PostwriteRefused,'STREAM'):x.restore(token)
        with self.assertRaisesRegex(ValueError,'TENSOR'):x.acquire(a.data_ptr())
        x.thread=-1
        with self.assertRaisesRegex(PostwriteRefused,'THREAD'):x.acquire(a)
