"""TP2 pretrained W8A8 model, actual scheduler rows, separate healthy input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

def token_sha(x):return hashlib.sha256(json.dumps(list(x),separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','fault'],required=True);a=p.parse_args()
    os.environ['DCAT_QUANT_ARM']=a.arm
    from vllm import LLM,SamplingParams
    result=dict(status='STARTED',arm=a.arm)
    control=Path('/evidence/control.json');control.write_text(json.dumps({'active':False}))
    try:
        llm=LLM(model='/model',tensor_parallel_size=2,dtype='bfloat16',enforce_eager=True,
            max_model_len=256,max_num_batched_tokens=2048,max_num_seqs=8,block_size=128,
            gpu_memory_utilization=.8,enable_prefix_caching=False,enable_chunked_prefill=False,
            async_scheduling=False,distributed_executor_backend='mp',seed=42,
            additional_config={'enable_flashcomm1':False},trust_remote_code=False,disable_log_stats=True,
            worker_extension_cls='fault_injection.deployments.batch_scale.QuantWorkerExtension')
        rpc=llm.llm_engine.model_executor.collective_rpc
        result['install']=rpc('dcat_quant_install')
        tok=llm.get_tokenizer()
        texts=['Explain a compiler.','Describe the water cycle.','Count the letters in abracadabra.',
            'Write a short Python loop.','Compare two sorting methods.','Describe a museum.',
            'Summarize plant growth.','Explain a checksum.']
        prompts=[tok.encode(x,add_special_tokens=False) for x in texts]
        mapping={token_sha(x):str(i) for i,x in enumerate(prompts)}
        params=SamplingParams(temperature=0,seed=42,max_tokens=2,ignore_eos=True,logprobs=5)
        def run(mode,gid):
            control.write_text(json.dumps(dict(active=True,mode=mode,generation_id=gid,prompt_sha_to_logical=mapping)))
            replies=llm.generate([{'prompt_token_ids':x} for x in prompts],params,use_tqdm=False)
            control.write_text(json.dumps({'active':False}))
            return [dict(tokens=list(r.outputs[0].token_ids),text=r.outputs[0].text,
                logprobs=[{str(k):v.logprob for k,v in row.items()} for row in r.outputs[0].logprobs]) for r in replies]
        healthy=run('reference','reference')
        result['selection']=rpc('dcat_quant_report')
        runs=[run(a.arm,f'capture_{step}') for step in range(3)]
        result['native']=rpc('dcat_quant_report')
        result['removed']=rpc('dcat_quant_remove')
        off=run('off','off')
        result.update(off_exact=off==healthy,repeat_stable=runs[0]==runs[1]==runs[2],changed=runs[-1]!=healthy,
                      healthy=healthy,runs=runs,off=off)
        if not result['off_exact'] or not result['repeat_stable']:raise RuntimeError('B/F/OFF comparison failed')
        if any(x['events']!=3 or x['native_poisoned'] for x in result['native']):raise RuntimeError('Missing native executions')
        if a.arm=='baseline' and result['changed']:raise RuntimeError('Baseline changed')
        result['status']='PASS'
    except BaseException:result.update(status='FAIL',error=traceback.format_exc());raise
    finally:
        control.write_text(json.dumps({'active':False}))
        Path('/evidence/result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
