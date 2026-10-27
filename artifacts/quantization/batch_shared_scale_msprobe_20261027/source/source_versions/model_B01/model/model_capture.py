"""Two proven batch conditions with independent msprobe-OFF baseline.

Baseline B uses the native binding observer for canonical hashes but neither
imports nor initializes msprobe. Capture uses real msprobe.save sessions for
H/H1/F/F2/R. Only the selected layer's first all-request pure-decode is saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from driver_helpers import write_json,certify,healthy_equal
from prompt_cases import CASES,score

SELECTION=['C14','C21','C01','C06','C04','C19','C17']
CONDITIONS={'letter_counts_B2':['letter_counts','C14'],
            'even_count_B4':['even_count']+SELECTION}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=('baseline','capture'),required=True)
    p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior-result',type=Path,required=True)
    p.add_argument('--baseline-result',type=Path)
    p.add_argument('--legacy-root',default='/audit/legacy')
    p.add_argument('--max-tokens',type=int,default=1024)
    p.add_argument('--condition',choices=tuple(CONDITIONS),action='append')
    p.add_argument('--allow-fault',action='store_true')
    args=p.parse_args()
    if not __debug__ or sys.flags.optimize:raise RuntimeError('Optimization disables binding guards')
    if not 4<=args.max_tokens<=1024:raise ValueError('Token budget outside bounded run')
    if args.mode=='capture' and args.baseline_result is None:raise ValueError('Capture requires independent baseline result')
    if args.allow_fault and (args.mode!='capture' or os.environ.get('BATCH_OUTLIER_ENABLED')!='1'):
        raise RuntimeError('Explicit capture mode, --allow-fault and BATCH_OUTLIER_ENABLED=1 are all required')
    conditions=list(dict.fromkeys(args.condition or CONDITIONS))
    prior=json.loads(args.prior_result.read_text(encoding='utf-8'))
    assert prior['status']=='completed' and prior['selection']['selected']==SELECTION
    assert prior['layer']=='model.layers.31.mlp.down_proj'
    baseline=json.loads(args.baseline_result.read_text(encoding='utf-8')) if args.baseline_result else None
    if baseline:
        assert baseline['status']=='completed' and baseline['capture_enabled'] is False
        assert baseline['max_tokens']==args.max_tokens and baseline['conditions']==conditions
    args.output.mkdir(parents=True,exist_ok=False);here=Path(__file__).resolve().parent
    os.environ.update(BATCH_OUTLIER_HOOK='1',BATCH_OUTLIER_OUTPUT=str(args.output/'operators'),
        BATCH_OUTLIER_CONTROL=str(args.output/'control.json'),BATCH_OUTLIER_LAYER=prior['layer'],
        BATCH_OUTLIER_LEGACY_ROOT=args.legacy_root,BATCH_MSPROBE_CAPTURE='1' if args.mode=='capture' else '0',
        BATCH_MSPROBE_DUMP_ROOT=str(args.output/'msprobe_dump'),VLLM_WORKER_MULTIPROC_METHOD='spawn',
        PYTHONPATH=os.pathsep.join([str(here),os.environ.get('PYTHONPATH','')]))
    if args.mode=='baseline':os.environ['BATCH_OUTLIER_ENABLED']='0'
    import native_capture_hook
    native_capture_hook.install()
    result={'status':'starting','start':time.time(),'runs':[],'model':args.model,'layer':prior['layer'],
        'mode':args.mode,'capture_enabled':args.mode=='capture','conditions':conditions,'max_tokens':args.max_tokens,
        'fault_permission':args.allow_fault,'no_artificial_amplitude':True,'default_disarmed':True,
        'baseline_scope':'Native binding canonical observer active; no msprobe import or initialization',
        'capture_format':'Native msprobe PrecisionDebugger.save debug.json + .debug.pt; not automatic API dump',
        'prior_result_sha256':hashlib.sha256(args.prior_result.read_bytes()).hexdigest(),
        'baseline_result_sha256':hashlib.sha256(args.baseline_result.read_bytes()).hexdigest() if args.baseline_result else None,
        'source_sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in here.glob('*.py')}}
    save=lambda:write_json(args.output/'result.json',result)
    save()
    try:
        from vllm import LLM,SamplingParams
        llm=LLM(model=args.model,tensor_parallel_size=2,dtype='bfloat16',enforce_eager=True,
            max_model_len=2048,max_num_batched_tokens=4096,max_num_seqs=8,block_size=128,
            gpu_memory_utilization=.8,enable_prefix_caching=False,enable_chunked_prefill=False,
            async_scheduling=False,distributed_executor_backend='mp',seed=42,
            additional_config={'enable_flashcomm1':False},trust_remote_code=False,disable_log_stats=True)
        tok=llm.get_tokenizer();prompts={}
        for logical in dict.fromkeys(k for c in conditions for k in CONDITIONS[c]):
            frozen=prior['prompts'][logical]
            rendered=tok.apply_chat_template([{'role':'user','content':frozen['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            tokens=tok(rendered,add_special_tokens=False)['input_ids']
            assert rendered==frozen['rendered'] and tokens==frozen['token_ids'],'Prior prompt/tokenizer changed'
            prompts[logical]=frozen
        if baseline:assert baseline['prompts']==prompts,'OFF/ON frozen prompts differ'
        result['prompts']=prompts;save()

        def run(condition,stage,mode='healthy',expected=None):
            generation=condition+'_'+stage;ids=CONDITIONS[condition];target=ids[0]
            mapping={prompts[k]['token_sha256']:k for k in ids}
            control={'active':True,'armed':mode=='shared' and args.allow_fault,'mode':mode,
                     'generation_id':generation,'target_logical_id':target,'prompt_sha_to_logical':mapping}
            if expected is not None:control['expected_by_rank']=expected
            write_json(args.output/'control.json',control);write_json(args.output/(generation+'.control.json'),control)
            start=time.time()
            params=[SamplingParams(temperature=0,seed=42,max_tokens=args.max_tokens if k==target else 3,
                                   ignore_eos=k!=target,logprobs=5) for k in ids]
            output=llm.generate([{'prompt_token_ids':prompts[k]['token_ids']} for k in ids],params,use_tqdm=False)
            write_json(args.output/'control.json',{'active':False,'armed':False})
            requests=[]
            for reply in output:
                item=reply.outputs[0];logical=mapping[native_capture_hook.token_sha(reply.prompt_token_ids)]
                requests.append({'logical_id':logical,'engine_request_id':reply.request_id,
                    'prompt_token_ids':reply.prompt_token_ids,'token_ids':list(item.token_ids),'text':item.text,
                    'finish_reason':item.finish_reason,'logprobs':[{str(k):{'logprob':v.logprob,'rank':v.rank,'decoded_token':v.decoded_token}
                        for k,v in row.items()} for row in item.logprobs]})
            proof=certify(args.output,generation,ids)
            if args.mode=='capture':
                sessions=[]
                for rank in (0,1):
                    path=args.output/'msprobe_dump'/('rank'+str(rank))/'sessions'/generation/'manifest.json'
                    manifest=json.loads(path.read_text(encoding='utf-8'))
                    assert manifest['status']=='PASS' and len(manifest['records'])==3
                    assert [r['stage'] for r in manifest['records']]==['pair','consumer','logits']
                    assert sum(len(r['tensors']) for r in manifest['records'])==7
                    sessions.append({'rank':rank,'manifest':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            else:
                assert not any(k=='msprobe' or k.startswith('msprobe.') for k in sys.modules),'Baseline driver imported msprobe'
                sessions=[]
            reply=next(x for x in requests if x['logical_id']==target)
            answer=next(c for c in CASES if c['id']==target)['expected']
            item={'generation_id':generation,'condition':condition,'case_id':target,'batch_id':condition.rsplit('_',1)[1],
                'phase':'answer','stage':stage,'target':target,'logical_ids':ids,'mode':mode,
                'requests':requests,'proof':proof,'msprobe_sessions':sessions,'seconds':time.time()-start,
                'score':score(reply['text'],reply['finish_reason'],answer)}
            result['runs'].append(item);save()
            print(json.dumps({'completed':generation,'seconds':item['seconds'],'msprobe_sessions':len(sessions)}),flush=True)
            return item

        if args.mode=='baseline':
            for condition in conditions:run(condition,'B')
        else:
            for condition in conditions:
                b=next(r for r in baseline['runs'] if r['generation_id']==condition+'_B')
                h=run(condition,'H','healthy',b['proof']['contracts'])
                h['baseline_transparency_exact']=healthy_equal(b,h);save()
                if not h['baseline_transparency_exact']:raise RuntimeError('msprobe healthy differs from independent native-observer baseline')
                h1=run(condition,'H1','healthy',h['proof']['contracts'])
                h1['healthy_native_repeat_exact']=healthy_equal(h,h1);save()
                if not h1['healthy_native_repeat_exact']:raise RuntimeError('Instrumented healthy repeats differ')
                qualified=all(c['binding_event']['target_calibration']['qualified'] for r in (h,h1) for c in r['proof']['calls'])
                external=any(c['binding_event']['external_range_increase'] for c in h['proof']['calls'])
                if not (qualified and external):raise RuntimeError('Actual target calibration or external-range requirement failed')
                if args.allow_fault:
                    f=run(condition,'F','shared',h['proof']['contracts'])
                    f2=run(condition,'F2','shared',h['proof']['contracts'])
                    f2['fault_repeat_exact']=healthy_equal(f,f2);save()
                    if not f2['fault_repeat_exact']:raise RuntimeError('Fault repetition differs')
                r=run(condition,'R','healthy',h['proof']['contracts'])
                r['recovery_exact']=healthy_equal(h,r);save()
                if not r['recovery_exact']:raise RuntimeError('Recovery differs from healthy msprobe run')
        result['status']='completed';result['end']=time.time();save();return 0
    except BaseException:
        result['status']='failed';result['traceback']=traceback.format_exc();result['end']=time.time();save()
        print(result['traceback'],file=sys.stderr);return 1
    finally:write_json(args.output/'control.json',{'active':False,'armed':False})


if __name__=='__main__':raise SystemExit(main())
