"""Actual native writer bindings; retained producer and deferred consumer proof.

No layer names or caller-supplied addresses. Unproven allocation adjacency and
unmatched consumers refuse acceptance. Recovery is checked after the forward.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from ..core import Decision,UnsupportedCall
from ..native import NativeCall,NativeProof,REQUIREMENTS,_STACK

class QmmContext(C.Structure):
    _fields_=[('abi_version',C.c_uint32),('device',C.c_int32),('run_index',C.c_uint64),
        ('producer_sequence',C.c_uint64),('consumer_sequence',C.c_uint64),('cache',C.c_uint64),
        ('original_output',C.c_uint64),('output_bytes',C.c_uint64),('output_m',C.c_int64),('output_n',C.c_int64),
        ('residual',C.c_uint64),('residual_bytes',C.c_uint64),('residual_m',C.c_int64),('residual_k',C.c_int64)]
class FiaContext(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32),('token',C.c_uint64),('device',C.c_int32),
        ('dtype',C.c_int32),('output',C.c_uint64),('output_bytes',C.c_uint64),('residual',C.c_uint64),
        ('residual_bytes',C.c_uint64),('producer_sequence',C.c_uint64),('metadata_ok',C.c_uint32),
        ('dimensions',C.c_uint32),('shape',C.c_int64*4)]
QCALL=C.CFUNCTYPE(C.c_int,C.POINTER(QmmContext),C.c_uint64)
FCALL=C.CFUNCTYPE(C.c_int,C.POINTER(FiaContext))

class FiaState(C.Structure):
    _fields_=[('version',C.c_uint32),('size',C.c_uint32)]+[(name,C.c_uint64) for name in
        ('runtime_ready','allowed','registered','owner_thread','callback_active','prepared',
         'queued_executors','pending_consumers','active_consumer','environment_enabled')]

def allocation_pair(snapshot,residual,output,residual_bytes,output_bytes):
    """Require two distinct active blocks in one actual allocator segment."""
    for segment in snapshot:
        start=segment['address'];found={}
        for block in segment['blocks']:
            address=block.get('address',start)
            if block['state']=='active_allocated':
                if address<=residual<address+block['size']:found['residual']=(address,address+block['size'])
                if address<=output<address+block['size']:found['output']=(address,address+block['size'])
            start=address+block['size']
        if len(found)==2:
            r,o=found['residual'],found['output']
            if r!=o and r[1]==o[0]==output and residual+residual_bytes<=r[1] and output+output_bytes<=o[1]:
                padding=r[1]-(residual+residual_bytes)
                if padding+32<=4096:return dict(residual_block=r,output_block=o,padding=padding,shift=padding+32)
    raise UnsupportedCall('Actual live writer/residual allocator blocks are not adjacent')

class Residual26Runtime:
    validation_scope='Native CANN8.5.1 typed writer binding; deferred actual AddRmsNormBias consumer/recovery'
    def __init__(self,model,mechanism,*,library,sha256,evidence_dir,log_path,max_effect_calls=None):
        if mechanism not in ('26.residual_stomp','26.fia_destination'):raise ValueError('Unknown26 mechanism')
        self.model,self.mechanism=model,mechanism;self.fia=mechanism=='26.fia_destination'
        p=Path(library).resolve(strict=True)
        if hashlib.sha256(p.read_bytes()).hexdigest()!=sha256:raise UnsupportedCall('Native26 build differs')
        if os.environ.get('TASK_QUEUE_ENABLE')!='0':raise UnsupportedCall('Synchronous model ownership required')
        self.lib=C.CDLL(str(p),mode=os.RTLD_LAZY|os.RTLD_NOLOAD)
        self.prefix='dcat26f' if self.fia else 'dcat26q';self.cbtype=FCALL if self.fia else QCALL
        self.cont=getattr(self.lib,self.prefix+'_continue_v1');self.cont.argtypes=[C.c_uint64];self.cont.restype=C.c_int
        self.pending=getattr(self.lib,self.prefix+'_pending_v1');self.pending.restype=C.c_uint64
        self.install_fn=getattr(self.lib,self.prefix+'_install_v1')
        self.install_fn.argtypes=[FCALL,C.c_uint64] if self.fia else [QCALL]
        self.remove_fn=getattr(self.lib,self.prefix+'_remove_v1');self.remove_fn.argtypes=[self.cbtype]
        self.sha256=sha256;self.out=Path(evidence_dir);self.out.mkdir(parents=True,exist_ok=False)
        self.log=Path(log_path);self.session=time.monotonic_ns();self.events=[];self.retained={};self.error=None
        self.completed=[];self.supported=0;self.claims=0;self.current=None
        self.forward_active=False;self.forward_index=-1
        if max_effect_calls is not None and (type(max_effect_calls) is not int or max_effect_calls!=1):
            raise ValueError('Only an explicit single-call budget is currently supported')
        self.max_effect_calls=max_effect_calls;self.selected_target=None

    def begin_forward(self,module,args):
        if self.forward_active or self.pending():
            raise RuntimeError('Unclosed residual forward or pending recovery')
        # A retained tensor from the preceding prefill is not a producer for
        # the next decode. Never extend its eligibility across root calls.
        self.retained.clear();self.forward_index+=1;self.forward_active=True

    def native_state(self):
        """Actual additive writer ABI; absence is not inferred idle state."""
        fn=getattr(self.lib,('dcat26f' if self.fia else 'dcat26q')+'_state_v1',None)
        if fn is None:raise UnsupportedCall('Native26 common Worker state ABI missing')
        fn.argtypes=[C.POINTER(FiaState)];fn.restype=C.c_int
        packet=FiaState(version=1,size=C.sizeof(FiaState))
        if fn(C.byref(packet)) or packet.version!=1 or packet.size!=C.sizeof(FiaState):
            raise UnsupportedCall('Native26 state ABI differs')
        result={name:int(getattr(packet,name)) for name,_ in packet._fields_}
        if not result['runtime_ready'] or not result['allowed']:
            raise UnsupportedCall('Native26 deployment is not ready/allowed')
        if result['registered'] and not result['owner_thread']:
            raise UnsupportedCall('Foreign native26 owner thread')
        return result

    def finish_forward(self,module,args,result):
        pending=bool(self.pending())
        try:
            if self.error:raise RuntimeError('Native26 callback failed: '+self.error)
            if pending:raise RuntimeError('Actual residual consumer did not finish; fresh process required')
            self.verify_completion()
        finally:
            if not pending:self.retained.clear()
            self.forward_active=False

    def install(self,callback,candidates,mechanism,arm):
        import torch
        if mechanism!=self.mechanism:raise UnsupportedCall('Wrong native26 binding')
        self.owner_thread=threading.get_ident();self.arm=arm
        self.owners={id(m):m for _,m,_ in candidates}
        self.owner_paths={id(m):path for path,m,_ in candidates}
        original=torch.ops._C_ascend.npu_add_rms_norm_bias
        def producer(*args,**kwargs):
            result=original(*args,**kwargs)
            if threading.get_ident()==self.owner_thread and self.forward_active:
                tensor=result[2]
                if tensor.storage_offset()!=0 or not tensor.is_contiguous():raise UnsupportedCall('Residual is not a full contiguous value')
                self.retained[tensor.data_ptr()]=tensor
            return result
        self.original_producer,self.producer=original,producer
        torch.ops._C_ascend.npu_add_rms_norm_bias=producer
        def dispatch(pointer,token=None):
            owner=None
            try:
                p=pointer.contents;token=p.token if self.fia else token
                stack=_STACK.get()
                if not stack or id(stack[-1][1]) not in self.owners:return self.cont(token) and 0
                if threading.get_ident()!=self.owner_thread:raise UnsupportedCall('Foreign native thread')
                owner=stack[-1][1];residual=self.retained.get(p.residual)
                if residual is None or residual.untyped_storage().nbytes()<p.residual_bytes:
                    raise UnsupportedCall('Actual producer residual is not retained by the model observer')
                if self.fia and (p.version!=1 or p.size!=C.sizeof(FiaContext) or not p.metadata_ok):
                    raise UnsupportedCall('FIA native metadata ABI differs')
                if not self.fia and p.abi_version!=1:raise UnsupportedCall('QMM native metadata ABI differs')
                details={n:(list(getattr(p,n)) if n=='shape' else getattr(p,n)) for n,_ in p._fields_}
                details.update(library_sha256=self.sha256,producer_tensor_id=id(residual),
                    consumer_completion='DEFERRED_UNTIL_ACTUAL_NATIVE_CONSUMER',pre_execution=True,
                    actual_model_forward=self.forward_index)
                if self.fia:
                    if p.output_bytes>p.residual_bytes or p.output==p.residual:raise UnsupportedCall('FIA output does not fit independent residual')
                else:
                    import torch_npu
                    details['allocation']=allocation_pair(torch_npu.npu.memory_snapshot(),p.residual,p.original_output,p.residual_bytes,p.output_bytes)
                cid=str(token);operator='aclnnFusedInferAttentionScoreV3' if self.fia else 'aclnnQuantMatmulWeightNz'
                proofs=[]
                for kind in REQUIREMENTS[mechanism]:
                    row=dict(schema='demoncat.native.proof.v1',pid=os.getpid(),owner_id=id(owner),call_id=cid,
                        operator=operator,mechanism=mechanism,kind=kind,status='verified',details=details)
                    raw=(json.dumps(row,sort_keys=True)+'\n').encode();path=self.out/f'{cid}-{kind}.json';path.write_bytes(raw)
                    proofs.append(NativeProof(kind,str(path),hashlib.sha256(raw).hexdigest()))
                self.current=dict(packet=p,token=token,details=details)
                callback(NativeCall(owner,cid,operator,proofs=tuple(proofs)),lambda:self.cont(token))
                self.current=None;return 0
            except UnsupportedCall as error:
                # Missing live/adjacent storage is a non-match, never fabricated
                # proof. Let the unified session record explicit passthrough.
                if owner is not None:
                    self.current=dict(unsupported=str(error))
                    operator='aclnnFusedInferAttentionScoreV3' if self.fia else 'aclnnQuantMatmulWeightNz'
                    try:
                        callback(NativeCall(owner,str(token),operator),lambda:self.cont(token))
                        return 0
                    except BaseException as failed:self.error=repr(failed);return 1
                    finally:self.current=None
                self.error=repr(error);return 1
            except BaseException as error:self.error=repr(error);return 1
        self.callback=self.cbtype(dispatch)
        status=self.install_fn(self.callback,self.session) if self.fia else self.install_fn(self.callback)
        if status:
            torch.ops._C_ascend.npu_add_rms_norm_bias=original
            raise UnsupportedCall('Native26 installation refused')
        self.initial_log_lines=len(self.log.read_text().splitlines()) if self.log.exists() else 0
        self.pre_handle=self.model.register_forward_pre_hook(self.begin_forward)
        self.handle=self.model.register_forward_hook(self.finish_forward,always_call=True)
        def remove():
            if self.pending() or self.remove_fn(self.callback):raise RuntimeError('Native26 recovery unresolved; restart process')
            self.handle.remove();self.pre_handle.remove()
            if torch.ops._C_ascend.npu_add_rms_norm_bias is not producer:raise RuntimeError('Producer observer ownership changed')
            torch.ops._C_ascend.npu_add_rms_norm_bias=original;self.retained.clear()
            self.verify_completion()
        return remove

    def check(self,call,mechanism,arm):
        if self.current and 'unsupported' in self.current:return Decision(False,self.current['unsupported'])
        if self.max_effect_calls is not None and self.supported>=self.max_effect_calls:
            return Decision(False,'Single qualified writer budget already consumed')
        return Decision(self.current is not None,'Actual retained producer and typed writer')
    def execute(self,call,invoke_once,mechanism,arm):
        p=self.current['packet']
        if self.max_effect_calls is not None:
            self.selected_target=dict(module_path=self.owner_paths[id(call.owner)],operator=call.operator,
                module_object_id=id(call.owner),cache=int(p.cache) if not self.fia else None,
                output_bytes=int(p.output_bytes),residual_bytes=int(p.residual_bytes),
                geometry=[int(p.output_m),int(p.output_n),int(p.residual_m),int(p.residual_k)] if not self.fia else list(p.shape),
                allocation=self.current['details'].get('allocation'))
        if self.fia:
            values=dict(FAULT26_FIA_WRITER_PROXY_ENABLED='1',FAULT26_FIA_WRITER_PROXY_SESSION=str(self.session),
                FAULT26_FIA_WRITER_PROXY_MODE='lossless_output_substitution' if arm=='baseline' else 'wrong_destination_residual_stomp',
                FAULT26_FIA_WRITER_PROXY_TARGET_DEVICE=str(p.device),FAULT26_FIA_WRITER_PROXY_TARGET_BYTES=str(p.output_bytes),
                FAULT26_FIA_WRITER_PROXY_TARGET_RESIDUAL_BYTES=str(p.residual_bytes),FAULT26_FIA_WRITER_PROXY_MAX_BYTES='8388608',
                FAULT26_FIA_WRITER_PROXY_ALL_HITS='1',FAULT26_FIA_WRITER_PROXY_MAX_HITS='4096')
        else:
            allocation=self.current['details']['allocation']
            values=dict(FAULT26_QMM_UNDERFLOW_ENABLED=str(int(arm=='fault')),FAULT26_QMM_UNDERFLOW_TARGET_DEVICE=str(p.device),
                FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_ADDRESS=str(p.original_output),FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_ADDRESS=str(p.residual),
                FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_BYTES=str(p.output_bytes),FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_BYTES=str(p.residual_bytes),
                FAULT26_QMM_UNDERFLOW_RESIDUAL_ALLOCATION_END=str(allocation['residual_block'][1]),
                FAULT26_QMM_UNDERFLOW_OUTPUT_ALLOCATION_BEGIN=str(allocation['output_block'][0]),
                FAULT26_QMM_UNDERFLOW_SHIFT_BYTES=str(allocation['shift']),FAULT26_QMM_UNDERFLOW_ALL_HITS='1',FAULT26_QMM_UNDERFLOW_MAX_HITS='4096')
        if any(k in os.environ for k in values):raise UnsupportedCall('Foreign native26 arm configuration')
        os.environ.update(values)
        try:claimed=invoke_once()
        finally:
            for k in values:os.environ.pop(k,None)
        expected=self.fia or arm=='fault'
        if claimed!=int(expected):raise RuntimeError('Actual native prepare did not match the bound writer')
        self.supported+=1;self.claims+=int(expected)
        self.last=dict(call_id=call.call_id,prepare_claimed=bool(claimed),actual_execution_pending=bool(claimed))
        self.events.append(self.last)
        return claimed
    def finish(self,call,mechanism,arm):return self.last
    def abort(self,call,mechanism,arm):
        if self.pending():raise RuntimeError('Native recovery pending; cannot unload provider')
    def verify_completion(self):
        lines=self.log.read_text().splitlines()[self.initial_log_lines:] if self.log.exists() else []
        rows=[json.loads(x) for x in lines]
        if self.fia:
            event='fault26_fia_lossless_output_substitution' if self.arm=='baseline' else 'fault26_fia_wrong_destination_restored'
            complete=[x for x in rows if x.get('event')==event and x.get('session')==self.session]
            valid=all(x.get('pass') is True for x in complete)
        else:
            complete=[x for x in rows if x.get('event')=='fault26_qmm_output_underflow_consumer_restored']
            valid=all(x.get('output_restored_exact') and x.get('padding_restored_exact') and x.get('residual_restored_exact')
                and x.get('changed_residual_bytes',0)>0 and x.get('consumer_status')==0 for x in complete)
        if len(complete)!=self.claims or not valid:raise RuntimeError('Deferred native writer/consumer/recovery evidence incomplete')
        self.completed=complete
