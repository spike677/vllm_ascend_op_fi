"""Experimental direction-A split-fuse reference, not a frozen/public grant.

Fixed A2/CANN9.1 BF16 TND branch. Every eligible call must pass the 99%
control checkpoint. Entire output is replaced; residual control mismatch is
reported explicitly, never hidden with per-element native matching.
"""
import torch,os
from pathlib import Path
from .gqa28 import GQA28Site
from .mla28 import require,fingerprint
from fault_injection.engines.paged33_native_control import split_fuse_control,agreement

class ControlMismatch(RuntimeError):pass

class SplitFuse33Site(GQA28Site):
    reference_device="input"
    matmul_mode="fp32"
    def call_gqa(self,name,impl,original,args):
        actual=[]
        def observe(*values):
            result=original(*values);actual.append(result);return result
        try:return super().call_gqa(name,impl,observe,args)
        except ControlMismatch:
            require(len(actual)==1 and name not in self.pending,'SPLITFUSE33_REJECTION_AFTER_WRITE')
            event=self.events[-1]
            event.update(failed=True,non_target_exact=True,applied=False,consumed=False,restored=False)
            # Do not strand peer ranks in model collectives. The existing
            # collect/query boundary rejects eligible failed calls on all ranks.
            return actual[0]

    def eligible_call(self,count,length):
        if count!=1:return False,'PREFILL_NOT_PAGED_DECODE'
        require(2305<=length<=2307,'SPLITFUSE33_UNMEASURED_LENGTH')
        return True,None

    def geometry_and_spans(self,q,k,v,impl,event):
        require(tuple(q.shape)==(12,256) and tuple(k.shape[1:])==(2,256)
            and k.shape==v.shape and q.dtype==torch.bfloat16 and impl.scale==.0625,
            'SPLITFUSE33_UNMEASURED_GEOMETRY')
        require(event.get('page_mapping',{}).get('block_size')==128
            and event['page_mapping']['logical_keys']==len(k),'SPLITFUSE33_PAGE_MAPPING')
        return ([(i,min(i+512,len(k))) for i in range(0,len(k),512)],
                [(i,min(i+256,len(k))) for i in range(0,len(k),256)])

    def calculate(self,q,k,v,impl,backup,event):
        require(q.dtype==torch.bfloat16 and k.dtype==q.dtype and v.dtype==q.dtype,'SPLITFUSE33_DTYPE_UNADAPTED')
        control_spans,fault_spans=self.geometry_and_spans(q,k,v,impl,event)
        if self.reference_device=='cpu':
            rq,rk,rv=q.detach().cpu(),k.detach().cpu(),v.detach().cpu()
            native=backup.detach().cpu()
        else:
            require(self.reference_device=='input','SPLITFUSE33_REFERENCE_DEVICE')
            if q.device.type=='npu':require(not torch.npu.matmul.allow_hf32,'SPLITFUSE33_HF32_NOT_ALLOWED')
            rq,rk,rv,native=q,k,v,backup
        control=split_fuse_control(rq.unsqueeze(0),rk,rv,scale=impl.scale,split_spans=control_spans,matmul_mode=self.matmul_mode).reshape_as(backup)
        check=agreement(control,native)
        if not check['checkpoint_99pct_passed'] and os.environ.get('DCAT_PRIVATE_FAILED_CONTROL')=='1':
            path=Path('/evidence/private-failed-control-'+str(os.getpid())+'.pt')
            if not path.exists():
                torch.save(dict(q=q.detach().cpu().unsqueeze(0),k=k.detach().cpu(),v=v.detach().cpu(),
                    output=backup.detach().cpu().reshape(1,12,256),metadata=dict(site=event['site'],scale=impl.scale,length=len(k)),
                    observed_control=check),path)
        if not check['checkpoint_99pct_passed']:
            event['control_rejection']=check
            raise ControlMismatch('SPLITFUSE33_CONTROL_BELOW_99:'+event['site']+':'+str(check))
        alternate=split_fuse_control(rq.unsqueeze(0),rk,rv,scale=impl.scale,split_spans=fault_spans,matmul_mode=self.matmul_mode).reshape_as(backup)
        event['partition_reference']=dict(control_block=control_spans[0][1]-control_spans[0][0],fault_block=fault_spans[0][1]-fault_spans[0][0],same_legal_key_set=True,
            key_visits=len(k),denominator_replay=False,native_equal_control_elements=check['bitwise_equal'],
            control_agreement=check,control_spans=control_spans,fault_spans=fault_spans,
            partition_changed_elements=int((control!=alternate).sum().item()),
            selected_changed_elements=int((alternate!=native).sum().item()),nonselected_exact=True,
            replacement_policy='WHOLE_TARGET_ROW_NO_VALUE_SELECTION',control_sha256=fingerprint(control),
            alternate_sha256=fingerprint(alternate),source_model='A2_BF16_TND_SPLITFUSE_PBF16_LSE_FP32',reference_device=self.reference_device,matmul_mode=self.matmul_mode,
            native_control_residual_elements=check['elements']-check['bitwise_equal'])
        return alternate.to(device=backup.device)


class NativeBoundarySplitFuse33Site(SplitFuse33Site):
    """BF16 operands with FP32 output: match the observed native MM boundary."""
    reference_device='input'
    matmul_mode='bf16_out_fp32'
