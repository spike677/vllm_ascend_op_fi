"""Complete pretrained HF model; real prefill cache then fixed paged decode."""
import argparse
import json
import os
from pathlib import Path
import sys
import traceback

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','fault'],required=True)
    a=p.parse_args()
    from fault_injection.deployments.paged33 import verify_bundle
    verified=verify_bundle('/audit')
    if not os.environ.get('DCAT33_FRESH_PROCESS'):
        if os.environ.get('ASCEND_CUSTOM_OPP_PATH') or os.environ.get('LD_PRELOAD'):
            raise RuntimeError('Unexpected inherited custom native path')
        env=dict(os.environ,DCAT33_FRESH_PROCESS='1',ASCEND_CUSTOM_OPP_PATH=verified['vendor_root'],
                 ACLNN_CACHE_LIMIT='0',LD_LIBRARY_PATH=verified['op_api_library_directory']+':'+os.environ.get('LD_LIBRARY_PATH',''))
        os.execve(sys.executable,[sys.executable,__file__,'--arm',a.arm],env)
    import torch,torch_npu
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from fault_injection import attach
    from fault_injection.deployments.paged33 import Paged33Runtime,digest
    from capture import capture
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True)
    torch.npu.set_device(0)
    tokenizer=AutoTokenizer.from_pretrained('/model',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained('/model',local_files_only=True,torch_dtype=torch.float16,
        attn_implementation='sdpa').eval().to('npu')
    sentences=['Explain a compiler loop. ','Describe a library catalog. ','Solve elementary algebra. ',
        'Validate a Python configuration. ','Explain the water cycle. ','Describe a city journey. ',
        'Summarize plant growth observations. ','Compare sorting algorithms. ']
    ids=[tokenizer.encode(s*6000,add_special_tokens=False)[:16384] for s in sentences]
    if any(len(x)!=16384 for x in ids):raise RuntimeError('Input too short')
    prefix=torch.tensor([x[:-1] for x in ids],device='npu');teacher=torch.tensor([[x[-1]] for x in ids],device='npu')
    result=dict(status='STARTED',arm=a.arm,scope='full_model_one_teacher_decode_repeated_three_times',
                model_class=type(model).__qualname__,prefill_length=16383,decode_length=16384)
    try:
        with torch.no_grad(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            prefill=model(input_ids=prefix,use_cache=True,logits_to_keep=1)
        cache=prefill.past_key_values;del prefill,prefix
        def forward():
            cache.crop(16383)
            return model(input_ids=teacher,past_key_values=cache,use_cache=True,logits_to_keep=1).logits
        with torch.no_grad():
            warm=Paged33Runtime(model,bundle='/audit',evidence_dir='/evidence/warm_native')
            with attach(model,'33.paged_decode',arm='baseline',runtime=warm) as session:
                healthy=forward().cpu();session.assert_coverage()
            runtime=Paged33Runtime(model,bundle='/audit',evidence_dir='/evidence/native')
            session=attach(model,'33.paged_decode',arm=a.arm,runtime=runtime)
            off_runtime=Paged33Runtime(model,bundle='/audit',evidence_dir='/evidence/off_native')
            off_session=attach(model,'33.paged_decode',arm='baseline',runtime=off_runtime)
            debugger=capture('/evidence/'+a.arm);outputs=[];last=None
            with session:
                for step in range(3):
                    debugger.start(model=model,rank_id=0)
                    try:last=forward()
                    finally:debugger.stop();debugger.step()
                    outputs.append(digest(last))
                session.assert_coverage()
            last=last.cpu()
            with off_session:
                off=forward().cpu();off_session.assert_coverage()
            result.update(coverage=session.report(),native_calls=len(runtime.calls),
                actual_policy_applied=sum(c['host']['policy_applied'] for c in runtime.calls),
                off_exact=torch.equal(healthy,off),repeat_stable=len(set(outputs))==1,
                changed=not torch.equal(healthy,last),logits_max_abs=float((healthy-last).abs().max()),
                top1_equal=torch.equal(healthy.argmax(-1),last.argmax(-1)),output_sha256=outputs,
                inputs_unchanged=all(c['inputs_unchanged'] for c in runtime.calls))
            if not (result['off_exact'] and result['repeat_stable'] and result['inputs_unchanged']):
                raise RuntimeError('B/F/OFF invariant failed')
            if a.arm=='baseline' and result['changed']:raise RuntimeError('Baseline is unstable')
            result['status']='PASS'
    except BaseException:
        result.update(status='FAIL',error=traceback.format_exc());raise
    finally:Path('/evidence/result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
