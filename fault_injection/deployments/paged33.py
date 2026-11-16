"""Structural HF decode adapter for the audited native Fault33 paged-IFA host.

Routing is model-independent; the native capability remains B8/Q1/KV16384,
H16/KV8/D128 FP16, page128, deterministic high precision. No output mutation.
The deployment provides the audited vendor bundle, not a layer name.
"""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import threading
import time
import uuid

from ..attention import AttentionSession
from ..core import Decision, UnsupportedCall
from ..native import NativeCall, NativeProof, REQUIREMENTS

VENDOR_VALIDATOR_SHA='75730d3cca29dd0ef2cb5d4e5112a99202fb149ba13e89d2ca5e9ab63d18cdb2'
VENDOR_MANIFEST_SHA='f48a82e89fdf7bbfafba5ac3e05334dbde01b422c55dc726f786152db5874cfe'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_bundle(root):
    root=Path(root).resolve(strict=True)
    path=root/'vendor_exec.py'
    if sha(path)!=VENDOR_VALIDATOR_SHA:raise UnsupportedCall('Paged33 validator identity differs')
    spec=importlib.util.spec_from_file_location('_dcat_paged33_vendor',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    case=dict(operator_variant='private_vendor',mode='off',
        approved_vendor_manifest_sha256=VENDOR_MANIFEST_SHA,
        approved_vendor_manifest_file=str(root/'vendor_stage_a4/vendor_manifest.json'),
        approved_custom_opp_path=str(root/'vendor_stage_a4/vendors/fault33_pa_transformer'))
    return mod.validate_vendor(case,root,root/'vendor_stage_a4')

def pack_pages(x,block):
    b,h,s,d=x.shape
    return x.reshape(b,h,s//block,block,d).permute(0,2,1,3,4).contiguous().view(b*(s//block),h,block,d)

def digest(x):
    # Bounded row transfers; all represented elements participate in the hash.
    import torch
    h=hashlib.sha256()
    for row in x.detach():h.update(row.cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class Paged33Runtime:
    validation_scope='Native paged IFA, audited CANN9.1 vendor, fixed geometry; HF shared AttentionInterface'

    def __init__(self,model,*,bundle,evidence_dir):
        self.model=model
        self.vendor=verify_bundle(bundle)
        if os.environ.get('ASCEND_CUSTOM_OPP_PATH')!=self.vendor['vendor_root'] or os.environ.get('ACLNN_CACHE_LIMIT')!='0':
            raise UnsupportedCall('Audited paged33 vendor must be selected before process import')
        if os.environ.get('TASK_QUEUE_ENABLE')!='0':raise UnsupportedCall('Synchronous native ownership required')
        self.output=Path(evidence_dir);self.output.mkdir(parents=True,exist_ok=False)
        self.pending=None;self.calls=[];self.cache=None;self.root_handle=None
        # Discover before an external collector wraps model.forward. This
        # retains actual object bindings, without treating arbitrary wrappers
        # as evidence of registry use.
        self.route=AttentionSession(self.model,self)

    def install(self,callback,candidates,mechanism,arm):
        if mechanism!='33.paged_decode':raise UnsupportedCall('Wrong paged33 mechanism')
        self.callback,self.arm=callback,arm
        def root(module,args,kwargs):
            ids=kwargs.get('input_ids',args[0] if args else None)
            cache=kwargs.get('past_key_values')
            if ids is None or tuple(ids.shape)!=(8,1) or cache is None or not hasattr(cache,'layers'):
                raise UnsupportedCall('Actual HF decode input and live DynamicCache required')
            if cache.get_seq_length()!=16383:raise UnsupportedCall('Unvalidated cache length')
            self.cache=cache;self.tokens=ids.detach().cpu().tolist()
        self.root_handle=self.model.register_forward_pre_hook(root,with_kwargs=True)
        try:self.route.__enter__()
        except BaseException:self.root_handle.remove();raise
        def remove():
            self.route.close();self.root_handle.remove()
            if self.pending is not None:raise RuntimeError('Unresolved paged33 invocation')
            if any(k.startswith('FAULT33_PA_') for k in os.environ):raise RuntimeError('Paged33 scope leaked')
        return remove

    def check(self,call,mechanism=None,arm=None):
        if isinstance(call,NativeCall):return Decision(self.pending is not None,'Live page mapping and native scope')
        import torch
        try:
            g=call.geometry()
            expected=dict(batch=8,query_heads=16,kv_heads=8,query_length=1,kv_length=16384,qk_dim=128,value_dim=128)
            if g!=expected:raise UnsupportedCall('Unvalidated native paged33 geometry: '+repr(g))
            if any(t.dtype!=torch.float16 or t.device.type!='npu' for t in (call.query,call.key,call.value)):
                raise UnsupportedCall('FP16 NPU Q/K/V required')
            if call.dropout or call.kwargs.get('output_attentions') or call.kwargs.get('sliding_window'):
                raise UnsupportedCall('Dropout, attention weights or sliding window not supported')
            if not torch.are_deterministic_algorithms_enabled():raise UnsupportedCall('Deterministic execution required')
            if call.scaling is None or not math.isclose(float(call.scaling),128**-.5,abs_tol=1e-12,rel_tol=0):
                raise UnsupportedCall('Unvalidated scale')
            if call.attention_mask is not None and not bool((call.attention_mask==0).all().item()):
                raise UnsupportedCall('Masked decode is unsupported')
            index=getattr(call.module,'layer_idx',None)
            if type(index) is not int or self.cache is None or not 0<=index<len(self.cache.layers):
                raise UnsupportedCall('Cannot bind actual attention to its DynamicCache layer')
            for x,y in ((call.key,self.cache.layers[index].keys),(call.value,self.cache.layers[index].values)):
                if x.data_ptr()!=y.data_ptr() or x.shape!=y.shape or x.stride()!=y.stride():
                    raise UnsupportedCall('Actual KV does not alias the declared live cache')
            return Decision(True,'Actual unmasked decode, live cache and supported native geometry')
        except (UnsupportedCall,AttributeError) as error:return Decision(False,str(error))

    def run(self,call):
        import torch,torch_npu
        pk,pv=pack_pages(call.key,128),pack_pages(call.value,128)
        mapping=[list(range(i*128,(i+1)*128)) for i in range(8)]
        table=torch.tensor(mapping,device=call.query.device,dtype=torch.int32)
        checks={}
        for name,logical,pages in (('key',call.key,pk),('value',call.value,pv)):
            # Invert every actual table row; equality is exact, not statistics.
            for i,ids in enumerate(mapping):
                inverse=pages[ids[0]:ids[-1]+1].permute(1,0,2,3).reshape(8,16384,128)
                if not torch.equal(inverse,logical[i]):raise UnsupportedCall('Page map is not lossless')
            checks[name]=dict(logical_sha256=digest(logical),physical_sha256=digest(pages))
        cid=uuid.uuid4().hex
        details=dict(phase='PRE_EXECUTION_BINDING',vendor_manifest_sha256=VENDOR_MANIFEST_SHA,
            geometry=call.geometry(),table=mapping,teacher_tokens=self.tokens,
            cache_layer=getattr(call.module,'layer_idx'),cache_object_id=id(self.cache),
            query_sha256=digest(call.query),kv=checks,actual_native_launch_pending=True)
        proofs=[]
        for kind in REQUIREMENTS['33.paged_decode']:
            receipt=dict(schema='demoncat.native.proof.v1',pid=os.getpid(),owner_id=id(call.module),
                call_id=cid,operator='torch_npu.npu_incre_flash_attention',mechanism='33.paged_decode',
                kind=kind,status='verified',details=details)
            raw=(json.dumps(receipt,sort_keys=True)+'\n').encode();p=self.output/f'{cid}-{kind}.json';p.write_bytes(raw)
            proofs.append(NativeProof(kind,str(p),hashlib.sha256(raw).hexdigest()))
        self.pending=dict(call_id=cid,query=call.query,key=pk,value=pv,table=table,details=details)
        native=NativeCall(call.module,cid,'torch_npu.npu_incre_flash_attention',proofs=tuple(proofs))
        try:
            result=self.callback(native,lambda:torch_npu.npu_incre_flash_attention(
                call.query,pk,pv,block_table=table,actual_seq_lengths=[16384]*8,num_heads=16,
                num_key_value_heads=8,input_layout='BNSD',scale_value=call.scaling,block_size=128,inner_precise=0))
            return result.transpose(1,2).contiguous(),None
        finally:self.pending=None

    def execute(self,call,invoke_once,mechanism,arm):
        import torch
        if any(k.startswith('FAULT33_PA_') for k in os.environ):raise UnsupportedCall('Foreign host scope')
        token=uuid.uuid4().hex;mode='off' if arm=='baseline' else 'inner1024'
        values=dict(FAULT33_PA_MODE=mode,FAULT33_PA_SCOPE_TOKEN=token,FAULT33_PA_SCOPE_PID=str(os.getpid()),
            FAULT33_PA_SCOPE_TID=str(threading.get_native_id()),FAULT33_PA_SCOPE_PHASE='decode',
            FAULT33_PA_CALLER_DETERMINISTIC='1',FAULT33_PA_BATCH='8',FAULT33_PA_LENGTH='16384',
            FAULT33_PA_BLOCK_SIZE='128',FAULT33_PA_CALL_ID=call.call_id,FAULT33_PA_LOG_PATH=str(self.output/'host_events.jsonl'))
        marker=dict(call_id=call.call_id,pid=os.getpid(),tid=threading.get_native_id(),block_size=128,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
        os.environ.update(values)
        try:
            torch.npu.synchronize();begin=time.monotonic_ns()
            print(json.dumps(dict(fault33='enter',monotonic_ns=begin,**marker)),flush=True)
            result=invoke_once();torch.npu.synchronize();end=time.monotonic_ns()
            print(json.dumps(dict(fault33='exit',monotonic_ns=end,**marker)),flush=True)
        finally:
            for k,v in values.items():
                if os.environ.get(k)!=v:raise RuntimeError('Host scope ownership changed')
                del os.environ[k]
        rows=[json.loads(line) for line in (self.output/'host_events.jsonl').read_text().splitlines()]
        matched=[r for r in rows if r.get('call_id')==call.call_id]
        if len(matched)!=1:raise RuntimeError('Expected exactly one actual native host callback')
        row=matched[0]
        if not (row['pid']==os.getpid() and row['tid']==threading.get_native_id() and row['token']==token
            and row['mode']==mode and row['phase']=='decode' and row['callback_status']==0
            and row['eligible'] is True and row['policy_applied']==(arm=='fault')
            and row['selector_checks']==1 and row['serialized_captures']==1
            and row['typed_matches_final_serialized_raw'] is True
            and begin<=row['callback_begin_monotonic_ns']<=row['callback_end_monotonic_ns']<=end):
            raise RuntimeError('Actual host/serialized-tiling verification failed')
        p=self.pending
        if digest(p['query'])!=p['details']['query_sha256'] or p['table'].cpu().tolist()!=p['details']['table']:
            raise RuntimeError('Native IFA changed query/table')
        for name in ('key','value'):
            if digest(p[name])!=p['details']['kv'][name]['physical_sha256']:raise RuntimeError('Native IFA changed KV pages')
        self.last=dict(call_id=call.call_id,host=row,inputs_unchanged=True,
            native_launch_independently_checked=False,scope_restored=True)
        self.calls.append(self.last)
        return result

    def finish(self,call,mechanism,arm):return self.last
    def abort(self,call,mechanism,arm):
        if any(k.startswith('FAULT33_PA_') for k in os.environ):raise RuntimeError('Host scope cleanup unresolved')
