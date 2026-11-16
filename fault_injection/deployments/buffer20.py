"""v17 MatmulWeightNz bridge: native callback, automatic live module ownership.

This adapter preserves the original lossless/mantissa transform and its bounds.
It does not claim QMM INT4/INT2 support or coverage for unobserved native paths.
"""
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import threading

from ..core import Decision, UnsupportedCall
from ..native import NativeCall, NativeProof, REQUIREMENTS, _STACK


class Call(ctypes.Structure):
    _fields_=[('version',ctypes.c_uint32),('size',ctypes.c_uint32),('token',ctypes.c_uint64),
              ('device',ctypes.c_int32),('dtype',ctypes.c_int32),('m',ctypes.c_int64),
              ('k',ctypes.c_int64),('bytes',ctypes.c_uint64),('input',ctypes.c_uint64),
              ('weight',ctypes.c_uint64),('output',ctypes.c_uint64),
              ('metadata_abi',ctypes.c_uint32),('slot_proof',ctypes.c_uint32)]

CALLBACK=ctypes.CFUNCTYPE(ctypes.c_uint32,ctypes.POINTER(Call))


class Buffer20Runtime:
    validation_scope='Native v17 MatmulWeightNz; FP16/BF16 ND input <=65536 bytes'

    def __init__(self, library, *, sha256, evidence_dir, keep_bits=0):
        self.path=Path(library).resolve(strict=True)
        if hashlib.sha256(self.path.read_bytes()).hexdigest()!=sha256:
            raise UnsupportedCall('Native20 build identity mismatch')
        if type(keep_bits) is not int or not 0<=keep_bits<=7:
            raise ValueError('keep_bits must be 0..7')
        if os.environ.get('TASK_QUEUE_ENABLE')!='0':
            raise UnsupportedCall('Synchronous dispatch is required for native model ownership')
        self.lib=ctypes.CDLL(str(self.path),mode=os.RTLD_LAZY|os.RTLD_NOLOAD)
        self.lib.fault20_metadata_runtime_abi_v17.restype=ctypes.c_uint32
        if self.lib.fault20_metadata_runtime_abi_v17()!=17:
            raise UnsupportedCall('Native20 metadata ABI differs')
        self.lib.dcat20_install_v1.argtypes=[CALLBACK]
        self.lib.dcat20_remove_v1.argtypes=[CALLBACK]
        self.lib.dcat20_continue_v1.argtypes=[ctypes.c_uint64,ctypes.c_uint32,ctypes.c_uint32]
        self.lib.dcat20_continue_v1.restype=ctypes.c_uint32
        self.lib.dcat20_snapshot_v1.restype=ctypes.c_char_p
        self.sha256,self.keep_bits=sha256,keep_bits
        self.directory=Path(evidence_dir)
        self.directory.mkdir(parents=True,exist_ok=True)
        self.active=False;self.error=None;self.callback=None;self.inputs={};self.handles=[]
        self.mode=0;self.last=None

    def install(self, callback, candidates, mechanism, arm):
        if self.active or mechanism!='20.buffer_substitute':
            raise UnsupportedCall('Wrong or already active native20 provider')
        self.thread=threading.get_ident()
        objects={id(module):module for _,module,_ in candidates}
        def enter(module,args):
            self.inputs[id(module)]=args[0] if args else None
        def leave(module,args,out):
            self.inputs.pop(id(module),None)
        for module in objects.values():
            self.handles.extend([module.register_forward_pre_hook(enter),
                                 module.register_forward_hook(leave,always_call=True)])
        def dispatch(pointer):
            packet=pointer.contents
            try:
                if self.error: return 1
                stack=_STACK.get()
                if not stack or id(stack[-1][1]) not in objects:
                    return self.lib.dcat20_continue_v1(packet.token,0,0)
                owner=stack[-1][1]
                x=self.inputs.get(id(owner))
                if (threading.get_ident()!=self.thread or packet.version!=1 or packet.size!=ctypes.sizeof(Call)
                    or packet.metadata_abi!=17 or packet.slot_proof!=1 or x is None
                    or str(x.device).split(':')[0]!='npu' or x.data_ptr()!=packet.input
                    or x.storage_offset()!=0 or not x.is_contiguous() or x.ndim<2
                    or x.shape[-1]!=packet.k or math.prod(x.shape[:-1])!=packet.m
                    or x.untyped_storage().nbytes()!=packet.bytes
                    or getattr(owner,'weight',None) is None or owner.weight.data_ptr()!=packet.weight):
                    raise UnsupportedCall('Native slot does not bind the live candidate input/weight')
                call_id=str(packet.token)
                details={name:getattr(packet,name) for name,_ in Call._fields_}
                details.update(library_sha256=self.sha256,source='ACTUAL_OP_EXEC_CACHE_RUN',
                               original_transform='v17_mantissa_truncate',keep_bits=self.keep_bits)
                proofs=[]
                for kind in REQUIREMENTS[mechanism]:
                    receipt=dict(schema='demoncat.native.proof.v1',pid=os.getpid(),owner_id=id(owner),
                        call_id=call_id,operator='aclnnMatmulWeightNz',mechanism=mechanism,
                        kind=kind,status='verified',details=details)
                    raw=(json.dumps(receipt,sort_keys=True)+'\n').encode()
                    path=self.directory/f'{os.getpid()}-{call_id}-{kind}.json'
                    with path.open('xb') as f:f.write(raw)
                    proofs.append(NativeProof(kind,str(path),hashlib.sha256(raw).hexdigest()))
                call=NativeCall(owner,call_id,'aclnnMatmulWeightNz',proofs=tuple(proofs))
                self.mode=0
                return callback(call,lambda:self.lib.dcat20_continue_v1(packet.token,self.mode,self.keep_bits))
            except BaseException as exc:
                # Never propagate a Python exception across the C ABI or retry
                # the native call. Its original caller receives a failure.
                self.error=repr(exc)
                return 1
        self.callback=CALLBACK(dispatch)
        if self.lib.dcat20_install_v1(self.callback)!=0:
            for h in reversed(self.handles):h.remove()
            self.handles=[]
            raise UnsupportedCall('Native20 installation refused')
        self.active=True
        def remove():
            if not self.active:return
            if self.lib.dcat20_remove_v1(self.callback)!=0:
                raise RuntimeError('Native20 removal not acknowledged')
            for h in reversed(self.handles):h.remove()
            self.handles=[];self.active=False
        return remove

    def check(self,call,mechanism,arm):
        return Decision(self.active and self.error is None,'Owned, live native20 invocation')

    def execute(self,call,invoke_once,mechanism,arm):
        self.mode=1 if arm=='baseline' else 2
        result=invoke_once()
        self.last=json.loads(self.lib.dcat20_snapshot_v1())
        if (result!=0 or self.last.get('call_id')!=call.call_id or self.last.get('hits')!=1
                or self.last.get('native_status')!=1 or self.last.get('enabled') is not False):
            raise RuntimeError('Native20 execution/recovery failed: '+repr(self.last))
        return result

    def finish(self,call,mechanism,arm):
        return dict(self.last,source='NATIVE_V17_CONTROL_AFTER_RUN',library_sha256=self.sha256)

    def abort(self,call,mechanism,arm):
        record=json.loads(self.lib.dcat20_snapshot_v1())
        if record.get('enabled') is not False:
            raise RuntimeError('Native20 cleanup is unconfirmed; restart process')
