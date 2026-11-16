"""Constructed NPU producer/writer/consumer model, not pretrained-model proof."""
import argparse,json,os
from pathlib import Path
import traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','fault'],required=True)
    p.add_argument('--kind',choices=['qmm','fia'],required=True);a=p.parse_args()
    import torch,torch_npu
    torch.npu.set_device(0);torch.npu.config.allow_internal_format=True;torch.manual_seed(37)
    import vllm_ascend.vllm_ascend_C
    from fault_injection import attach
    from fault_injection.deployments.residual26 import Residual26Runtime
    from capture import capture
    class AddNorm(torch.nn.Module):
        def __init__(self):
            super().__init__();self.register_buffer('gamma',torch.ones(4096,device='npu',dtype=torch.float16))
        def forward(self,x,r):return torch.ops._C_ascend.npu_add_rms_norm_bias(x,r,self.gamma,None,1e-6)
    class QuantMethod:
        def apply(self,layer,x):return torch_npu.npu_quant_matmul(x,layer.weight,layer.scale,output_dtype=torch.float16)
    class QuantWriter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.quant_method=QuantMethod()
            self.register_buffer('weight',torch_npu.npu_format_cast(torch.randint(-3,4,(128,4096),device='npu',dtype=torch.int8),29))
            self.register_buffer('scale',torch.ones(4096,device='npu',dtype=torch.float32)*.01)
        def forward(self,x):return self.quant_method.apply(self,x)
    class NativeImplementation:
        def forward(self,q,k,v):return torch_npu.npu_fused_infer_attention_score(q,k,v,num_heads=16,
            num_key_value_heads=16,input_layout='BNSD',scale=128**-.5)[0]
    class AttentionWriter(torch.nn.Module):
        def __init__(self):
            super().__init__();self.impl=NativeImplementation();self.num_heads=16;self.num_kv_heads=16;self.head_size=128
        def forward(self,q,k,v):return self.impl.forward(q,k,v)
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__();self.before=AddNorm();self.after=AddNorm()
            self.writer=QuantWriter() if a.kind=='qmm' else AttentionWriter()
            self.register_buffer('x',torch.randn(1,4096,device='npu',dtype=torch.float16))
            self.register_buffer('r',torch.randn_like(self.x))
            self.register_buffer('z',torch.zeros_like(self.x))
            self.register_buffer('qi',torch.randint(-8,9,(1,128),device='npu',dtype=torch.int8))
            self.register_buffer('q',torch.randn(1,16,1,128,device='npu',dtype=torch.float16))
            self.register_buffer('k',torch.randn(1,16,128,128,device='npu',dtype=torch.float16))
            self.register_buffer('v',torch.randn_like(self.k))
        def forward(self):
            normalized,inv,residual=self.before(self.x,self.r)
            writer=self.writer(self.qi) if a.kind=='qmm' else self.writer(self.q,self.k,self.v)
            consumed=self.after(self.z,residual)[0]
            return writer,consumed,residual
    model=Network().eval();mechanism='26.residual_stomp' if a.kind=='qmm' else '26.fia_destination'
    filename='libfault26_qmm_output_underflow_abi0.so' if a.kind=='qmm' else 'libfault26_fia_writer_proxy_abi0.so'
    result=dict(status='STARTED',arm=a.arm,mechanism=mechanism,scope='constructed_model_mechanism_only',full_pretrained_model=False)
    try:
        with torch.no_grad():
            # Baseline/OFF retain identical actual producer/writer/consumer code.
            healthy=tuple(t.cpu() for t in model())
            build=json.loads(Path('/native/build.json').read_text())
            runtime=Residual26Runtime(model,mechanism,library='/native/'+filename,sha256=build['files'][filename],
                evidence_dir='/evidence/native',log_path='/evidence/writer.jsonl')
            session=attach(model,mechanism,arm=a.arm,runtime=runtime)
            debugger=capture('/evidence/'+a.arm);outputs=[]
            with session:
                for step in range(3):
                    debugger.start(model=model,rank_id=0)
                    try:last=model()
                    finally:debugger.stop();debugger.step()
                    outputs.append(tuple(t.cpu() for t in last))
                session.assert_coverage()
            off=tuple(t.cpu() for t in model())
            equal=lambda x,y:all(torch.equal(a,b) for a,b in zip(x,y))
            result.update(off_exact=equal(healthy,off),repeat_stable=all(equal(outputs[0],x) for x in outputs),
                writer_exact=all(torch.equal(healthy[0],x[0]) for x in outputs),
                residual_restored=all(torch.equal(healthy[2],x[2]) for x in outputs),
                consumer_changed=not torch.equal(healthy[1],outputs[-1][1]),
                coverage=session.report(),native_completed=runtime.completed,supported=runtime.supported,claims=runtime.claims)
            if not all(result[k] for k in ('off_exact','repeat_stable','writer_exact','residual_restored')):
                raise RuntimeError('Native lifetime/OFF invariant failed')
            if result['consumer_changed']!=(a.arm=='fault'):raise RuntimeError('Unexpected consumer value')
            result['status']='PASS'
    except BaseException:result.update(status='FAIL',error=traceback.format_exc());raise
    finally:Path('/evidence/result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
