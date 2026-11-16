"""Real NPU smoke and B/F/OFF; arbitrary module names, native callbacks only."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','fault'],required=True)
    p.add_argument('--model',type=Path)
    a=p.parse_args()
    import torch,torch_npu
    from fault_injection import attach
    from fault_injection.deployments.buffer20 import Buffer20Runtime
    from capture import capture
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format=True
    torch.manual_seed(719)
    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=torch.nn.Linear(128,256,bias=False)
            self.finish=torch.nn.Linear(256,64,bias=False)
        def forward(self,x):return self.finish(self.encoder(x))
    if a.model:
        from transformers import AutoModelForCausalLM,AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,
            torch_dtype=torch.bfloat16,attn_implementation='eager').eval().to('npu')
    else:
        model=Network().eval().to(device='npu',dtype=torch.bfloat16)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module,torch.nn.Linear):
                module.weight=torch.nn.Parameter(torch_npu.npu_format_cast(module.weight,29),requires_grad=False)
    x=(torch.tensor([[tokenizer.encode('Hello',add_special_tokens=False)[0]],
                     [tokenizer.encode('World',add_special_tokens=False)[0]]],device='npu')
       if a.model else torch.randn(2,128,device='npu',dtype=torch.bfloat16))
    def execute():return model(x,use_cache=False).logits if a.model else model(x)
    def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    result={'stage':'native_provider_model_probe' if a.model else 'native_provider_mechanism_probe',
            'model':str(a.model) if a.model else 'two_actual_linear_modules',
            'full_pretrained_model':bool(a.model),'arm':a.arm,'status':'STARTED'}
    try:
        with torch.no_grad():
            for _ in range(3):baseline=execute()
            inputs=digest(x)
            identity=json.loads(Path('/native/build.json').read_text())
            runtime=Buffer20Runtime('/native/libdcat20.so',sha256=identity['files']['libdcat20.so'],
                                    evidence_dir='/evidence/native',keep_bits=0)
            session=attach(model,'20.buffer_substitute',arm=a.arm,runtime=runtime)
            debugger=capture('/evidence/'+a.arm)
            outputs=[]
            with session:
                for step in range(3):
                    debugger.start(model=model,rank_id=0)
                    try:out=execute()
                    finally:debugger.stop();debugger.step()
                    outputs.append(digest(out))
                session.assert_coverage()
            off=execute();torch.npu.synchronize()
            result.update(coverage=session.report(),output_sha256=outputs,baseline_sha256=digest(baseline),
                off_exact=torch.equal(baseline,off),input_unchanged=inputs==digest(x),
                changed=not torch.equal(baseline,out),repeat_stable=len(set(outputs))==1,
                native_error=runtime.error)
            if not result['off_exact'] or not result['input_unchanged'] or not result['repeat_stable']:
                raise RuntimeError('B/F/OFF invariant failed')
            if result['changed']!=(a.arm=='fault'):raise RuntimeError('Unexpected numeric result')
            result['status']='PASS'
    except BaseException:
        result.update(status='FAIL',error=traceback.format_exc());raise
    finally:
        Path('/evidence/result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
