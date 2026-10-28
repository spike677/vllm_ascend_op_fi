"""Frozen TP2 real-prompt matrix. Labels/truth are separate from observation/."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

CONDITIONS={'letter_counts_B2':['letter_counts','C14'],
            'even_count_B4':['even_count','C14','C21','C01','C06','C04','C19','C17']}

def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2))
    temporary.replace(path)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--backend',action='store_true')
    p.add_argument('--observe',action='store_true')
    p.add_argument('--allow-fault',action='store_true')
    p.add_argument('--reference',type=Path)
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--condition',choices=tuple(CONDITIONS))
    p.add_argument('--single-stage',choices=('B','H','H1','F','F2','R','X','X2','XR'))
    args=p.parse_args()
    if bool(args.condition)!=bool(args.single_stage): raise ValueError('Single stage requires a single condition')
    if args.allow_fault and not (args.backend and args.observe and args.reference):
        raise ValueError('Fault matrix needs backend, observer and independent reference')
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parent
    os.environ.update(DCAT_V2_HOOK='1',DCAT_V2_OUTPUT=str(args.output),
        DCAT_V2_CONTROL=str(args.output/'control.json'),DCAT_V2_BACKEND=str(int(args.backend)),
        DCAT_V2_OBSERVE=str(int(args.observe)),VLLM_WORKER_MULTIPROC_METHOD='spawn',
        PYTHONPATH=str(root)+os.pathsep+os.environ.get('PYTHONPATH',''))
    if args.backend:
        # vLLM prepends its vendor only when absent. Include it now, behind
        # ours, so its transitive libopapi dependency cannot shadow our symbols.
        production_vendor='/work/venv/lib/python3.11/site-packages/vllm_ascend/_cann_ops_custom/vendors/custom_transformer'
        assert Path(production_vendor).is_dir()
        os.environ['ASCEND_CUSTOM_OPP_PATH']=str(root/'runtime/vendor')+':'+production_vendor
        os.environ['DCAT_DQ_V2_ENABLE']='1'
    else:
        os.environ.pop('ASCEND_CUSTOM_OPP_PATH',None)
        os.environ.pop('DCAT_DQ_V2_ENABLE',None)
    import model_hook
    model_hook.install()
    frozen=json.loads(Path('/previous/prior_result.json').read_text())
    reference=json.loads(args.reference.read_text()) if args.reference else None
    if reference and reference['status']!='completed': raise RuntimeError('Reference incomplete')
    result={'status':'starting','backend':args.backend,'observe':args.observe,'runs':[],
            'max_tokens':args.max_tokens,'source_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()
                for f in root.glob('*.py')},'observation_window':[30,32]}
    save=lambda:write(args.output/'result.json',result)
    save()
    try:
        from vllm import LLM,SamplingParams
        llm=LLM(model='/models/DeepSeek-R1-Distill-Qwen-32B-W8A8',tensor_parallel_size=2,
            dtype='bfloat16',enforce_eager=True,max_model_len=2048,max_num_batched_tokens=4096,
            max_num_seqs=8,block_size=128,gpu_memory_utilization=.8,enable_prefix_caching=False,
            enable_chunked_prefill=False,async_scheduling=False,distributed_executor_backend='mp',
            seed=42,additional_config={'enable_flashcomm1':False},trust_remote_code=False,disable_log_stats=True)
        tok=llm.get_tokenizer()
        prompts={}
        for logical in dict.fromkeys(k for ids in CONDITIONS.values() for k in ids):
            item=frozen['prompts'][logical]
            rendered=tok.apply_chat_template([{'role':'user','content':item['text']}],tokenize=False,
                add_generation_prompt=True,enable_thinking=False)
            assert rendered==item['rendered'] and tok(rendered,add_special_tokens=False)['input_ids']==item['token_ids']
            prompts[logical]=item
        result['prompts']=prompts
        stages=[('H','healthy'),('H1','healthy'),('F','shared'),('F2','shared'),('R','healthy')] if args.allow_fault else [('B','healthy')]
        schedule=[(condition,ids,stages) for condition,ids in CONDITIONS.items()]
        if args.allow_fault and not args.single_stage:
            schedule.append(('letter_counts_B2',CONDITIONS['letter_counts_B2'],[('X','shared'),('X2','shared'),('XR','healthy')]))
        if args.single_stage:
            mode='shared' if args.single_stage in ('F','F2','X','X2') else 'healthy'
            if mode=='shared' and not args.allow_fault: raise ValueError('Fault stage requires --allow-fault')
            schedule=[(args.condition,CONDITIONS[args.condition],[(args.single_stage,mode)])]
        for condition,ids,run_stages in schedule:
            expected=None
            if reference:
                expected=next(r for r in reference['runs'] if r['condition']==condition)['contracts']
            for stage,mode in run_stages:
                generation=f'{condition}_{stage}'
                mapping={prompts[k]['token_sha256']:k for k in ids}
                control={'active':True,'mode':mode,'armed':mode=='shared' and args.allow_fault,
                    'generation_id':generation,'prompt_sha_to_logical':mapping}
                if stage in ('X','X2','XR'): control['fault_ranks']=[0]
                if expected: control['expected_by_rank']=expected
                write(args.output/'control.json',control)
                write(args.output/(generation+'.control.json'),control)
                start=time.time()
                params=[SamplingParams(temperature=0,seed=42,max_tokens=args.max_tokens if i==0 else 3,
                    ignore_eos=i!=0,logprobs=5) for i in range(len(ids))]
                replies=llm.generate([{'prompt_token_ids':prompts[k]['token_ids']} for k in ids],params,use_tqdm=False)
                write(args.output/'control.json',{'active':False,'armed':False})
                requests=[]
                for reply in replies:
                    value=reply.outputs[0]
                    logical=mapping[model_hook.token_sha(reply.prompt_token_ids)]
                    requests.append({'logical_id':logical,'token_ids':list(value.token_ids),'text':value.text,
                        'finish_reason':value.finish_reason,'logprobs':[{str(k):v.logprob for k,v in row.items()}
                            for row in value.logprobs]})
                events=[json.loads(line) for f in args.output.glob('pid*.jsonl') for line in f.read_text().splitlines()]
                calls=[e for e in events if e['event']=='selected_call' and e['generation_id']==generation]
                logits=[e for e in events if e['event']=='logits' and e['generation_id']==generation]
                assert len(calls)==2 and {e['rank'] for e in calls}=={0,1}
                assert len(logits)==2 and {e['rank'] for e in logits}=={0,1}
                contracts={str(e['rank']):e['contract'] for e in calls}
                if expected: assert contracts==expected
                expected=contracts
                canonical={'consumer':{str(e['rank']):e['consumer_output_sha256'] for e in calls},
                    'logits':{str(e['rank']):e['sha256'] for e in logits},
                    'requests':sorted(requests,key=lambda r:r['logical_id'])}
                run={'condition':condition,'stage':stage,'generation_id':generation,'contracts':contracts,
                    'canonical':canonical,'backend_hits':{str(e['rank']):e['backend'] for e in calls},
                    'observation_step':len(result['runs']),'seconds':time.time()-start}
                result['runs'].append(run);save()
                print(json.dumps({'generation_completed':generation,'seconds':run['seconds']}),flush=True)
                if mode=='healthy' and reference:
                    baseline=next(r for r in reference['runs'] if r['condition']==condition)
                    if canonical!=baseline['canonical']: raise RuntimeError('Healthy output differs from independent baseline')
        result['status']='completed';save();return 0
    except BaseException:
        result['status']='failed';result['traceback']=traceback.format_exc();save()
        print(result['traceback'],file=sys.stderr);return 1
    finally:
        write(args.output/'control.json',{'active':False,'armed':False})

if __name__=='__main__': raise SystemExit(main())
