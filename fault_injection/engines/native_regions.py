"""Trusted live-tensor object interface to additive postwrite stream exports.

No user-supplied addresses; retained tensors, storage identities and streams
outlive each lease. Does not grant a deployment or drive task lifecycle.
"""
import ctypes
import hashlib
import threading
import torch
from .region_lease import identity,version
from ..deployments.postwrite import NativePostwriteLibrary,PostwriteRefused


class NativeRegions:
    def __init__(self,library,sha256,owner,generation):
        if not owner or type(generation)!=int or generation<1:raise ValueError('NATIVE_REGION_SCOPE_REQUIRED')
        self.transport=NativePostwriteLibrary(library,sha256)
        self.owner_text=owner;self.owner=int.from_bytes(hashlib.sha256(owner.encode()).digest()[:8],'little') or 1
        self.generation=generation;self.thread=threading.get_ident();self.active={};self.serial=0
        lib=self.transport.lib;u64=ctypes.c_uint64;ptr=ctypes.c_void_p
        signatures={'dcat_pw_open_stream_v2':([ptr,u64,ctypes.c_int32,u64,u64,u64,ctypes.POINTER(u64)],ctypes.c_int),
                    'dcat_pw_copy_region_v2':([u64,u64,u64,u64,ctypes.POINTER(u64)],ctypes.c_int)}
        for name,(args,result) in signatures.items():
            if not hasattr(lib,name):raise PostwriteRefused('FRESH_STREAM_REGION_EXPORT_REQUIRED:'+name)
            f=getattr(lib,name);f.argtypes=args;f.restype=result

    def _validate(self,t):
        import torch_npu
        if (not isinstance(t,torch.Tensor) or t.device.type!='npu' or t.dtype!=torch.bfloat16
                or t.requires_grad or not t.is_contiguous() or t.numel()*2>1024*1024
                or int(torch_npu.get_npu_format(t)) not in (0,2)):
            raise PostwriteRefused('NATIVE_LIVE_BASE_BF16_REGION_REQUIRED')
        return identity(t)

    def _stream(self):return torch.npu.current_stream()

    def _get(self,token):
        if threading.get_ident()!=self.thread:raise PostwriteRefused('NATIVE_REGION_THREAD_CHANGED')
        row=self.active[token]
        if self._validate(row['tensor'])!=row['identity']:raise PostwriteRefused('NATIVE_OBJECT_IDENTITY_CHANGED_NO_WRITEBACK')
        if self._stream()!=row['stream']:raise PostwriteRefused('NATIVE_REGION_STREAM_CHANGED')
        return row

    def _call(self,name,*args):
        code=getattr(self.transport.lib,name)(*args)
        if code:raise PostwriteRefused(name+':'+str(code))

    def acquire(self,tensor):
        if threading.get_ident()!=self.thread:raise PostwriteRefused('NATIVE_REGION_THREAD_CHANGED')
        key=self._validate(tensor);stream=self._stream()
        for row in self.active.values():
            other=row['identity']
            if key[:2]==other[:2] and max(key[2],other[2])<min(key[3],other[3]):
                raise PostwriteRefused('NATIVE_OVERLAPPING_LIVE_OBJECTS')
        handle=ctypes.c_uint64()
        self._call('dcat_pw_open_stream_v2',tensor.data_ptr(),tensor.numel()*2,tensor.device.index,
                   self.owner,self.generation,int(stream.npu_stream),ctypes.byref(handle))
        self.serial+=1
        self.active[self.serial]=dict(tensor=tensor,storage=tensor.untyped_storage(),identity=key,stream=stream,
            handle=handle.value,version=version(tensor),applied=False,consumed=False)
        return self.serial

    def _unmodified(self,row):
        if version(row['tensor'])!=row['version']:raise PostwriteRefused('NATIVE_LEGAL_INPLACE_UPDATE_NO_WRITEBACK')

    def apply(self,token,mode,indices):
        row=self._get(token);self._unmodified(row)
        if row['applied']:raise PostwriteRefused('NATIVE_ONE_WRITE_PER_LEASE')
        if type(mode)!=int or mode not in (0,1,2,3,4,5) or not indices or any(type(i)!=int or i<0 for i in indices):
            raise PostwriteRefused('NATIVE_TYPED_MODE_AND_INDICES_REQUIRED')
        selected=(ctypes.c_uint64*len(indices))(*indices);changed=ctypes.c_uint64()
        # Mark before the C call: a failed transfer may have partially written.
        row['applied']=True
        self._call('dcat_pw_apply',row['handle'],self.owner,self.generation,mode,selected,len(indices),ctypes.byref(changed))
        return changed.value

    def copy(self,target,source):
        dst=self._get(target);src=self._get(source)
        self._unmodified(dst);self._unmodified(src)
        if dst['applied'] or src['applied'] or dst['stream']!=src['stream']:
            raise PostwriteRefused('NATIVE_COPY_REQUIRES_FRESH_SAME_STREAM_OBJECTS')
        changed=ctypes.c_uint64();dst['applied']=True
        self._call('dcat_pw_copy_region_v2',dst['handle'],src['handle'],self.owner,self.generation,ctypes.byref(changed))
        return changed.value

    def consumed(self,token):
        row=self._get(token);self._unmodified(row)
        # Called by the observed value consumer hook, after enqueue on this stream.
        event=torch.npu.Event();event.record(row['stream']);event.synchronize();row['consumed']=True

    def restore(self,token):
        row=self._get(token);self._unmodified(row)
        if row['applied'] and not row['consumed']:raise PostwriteRefused('NATIVE_CONSUMER_COMPLETION_REQUIRED')
        self._call('dcat_pw_restore',row['handle'],self.owner,self.generation)
        self._call('dcat_pw_release',row['handle'],self.owner,self.generation)
        del self.active[token]

    def count(self):return len(self.active)
