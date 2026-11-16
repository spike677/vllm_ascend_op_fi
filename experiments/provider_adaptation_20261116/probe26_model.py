"""Pretrained W8A8 TP2 B/F/OFF with actual residual lifetime provider."""
import argparse,json,os,traceback
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','fault'],required=True)
    p.add_argument('--kind',choices=['qmm','fia'],required=True);a=p.parse_args()
    os.environ.update(DCAT26_ARM=a.arm,DCAT26_KIND=a.kind,VLLM_ENABLE_V1_MULTIPROCESSING='0')
    from vllm import LLM,SamplingParams
    result=dict(status='STARTED',arm=a.arm,kind=a.kind,scope='pretrained_tp2_model')
    try:
        llm=LLM(model='/model',tensor_parallel_size=2,dtype='bfloat16',enforce_eager=True,
            max_model_len=256,max_num_batched_tokens=2048,max_num_seqs=8,gpu_memory_utilization=.8,
            enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,
            distributed_executor_backend='mp',seed=42,trust_remote_code=False,disable_log_stats=True,
            worker_extension_cls='fault_injection.deployments.residual26_worker.Residual26WorkerExtension')
        prompts=['Explain a compiler.','Describe the water cycle.','Count letters in abracadabra.',
            'Write a short Python loop.','Compare sorting methods.','Describe a museum.',
            'Summarize plant growth.','Explain a checksum.']
        params=SamplingParams(temperature=0,seed=42,max_tokens=2,ignore_eos=True,logprobs=5)
        def generate():
            return [dict(tokens=list(r.outputs[0].token_ids),text=r.outputs[0].text,
                logprobs=[{str(k):v.logprob for k,v in row.items()} for row in r.outputs[0].logprobs])
                for r in llm.generate(prompts,params,use_tqdm=False)]
        healthy=generate();rpc=llm.llm_engine.model_executor.collective_rpc
        result['install']=rpc('dcat26_install');fault=generate()
        result['native']=rpc('dcat26_remove');off=generate()
        result.update(healthy=healthy,observed=fault,off=off,off_exact=healthy==off,changed=healthy!=fault)
        if not result['off_exact']:raise RuntimeError('OFF did not restore exact generation/logprobs')
        if a.arm=='baseline' and result['changed']:raise RuntimeError('Baseline is unstable')
        result['status']='PASS'
    except BaseException:result.update(status='FAIL',error=traceback.format_exc());raise
    finally:Path('/evidence/result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
