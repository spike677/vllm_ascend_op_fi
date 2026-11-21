"""Trusted real-model site acceptance, not a public deployment grant."""
import json,time,traceback,uuid
from pathlib import Path


def rpc(wrapper,action,owner=None,fault=False,kind='batch'):
    import torch,os
    from vllm.v1.worker.worker_base import WorkerWrapperBase
    from fault_injection.sites.mla28 import MLA28Site,fingerprint,require
    worker=wrapper.worker if isinstance(wrapper,WorkerWrapperBase) else wrapper
    runner=worker.model_runner;model=runner.model;key='_dcat_continuous_batch_site'
    held=getattr(worker,key,None)
    if action=='install':
        require(held is None,'PREVIOUS_SCOPE_NOT_RELEASED')
        from fault_injection.sites.batch_quant import BatchQuantSite
        site=BatchQuantSite(model,runner,fault=fault)
        rows=[];hooks=[]
        paths=site.install()
        def output_hook(mod,args,value):
            require(isinstance(value,torch.Tensor),'TENSOR_HIDDEN_REQUIRED')
            rows.append(dict(request_ids=list(runner.input_batch.req_ids),hidden=fingerprint(value),hidden_finite=bool(torch.isfinite(value).all().item())))
        hooks.append(model.register_forward_hook(output_hook))
        original=model.compute_logits
        require('compute_logits' not in vars(model),'LOGITS_ALREADY_WRAPPED')
        def logits(*args,**kwargs):
            value=original(*args,**kwargs)
            require(rows and 'logits' not in rows[-1],'LOGITS_ORDER')
            require(value is not None,'MISSING_LOGITS')
            rows[-1]['logits']=fingerprint(value)
            rows[-1]['logits_finite']=bool(torch.isfinite(value).all().item())
            return value
        model.compute_logits=logits
        setattr(worker,key,(owner,site,rows,hooks,logits))
        return dict(rank=worker.rank,pid=os.getpid(),model_id=id(model),sites=paths)
    require(held is not None and held[0]==owner,'OWNED_SITE_REQUIRED')
    _,site,rows,hooks,logits=held
    if action=='collect':
        require(len(rows)==4 and all('logits' in row for row in rows),'FOUR_ACTUAL_FORWARD_LOGITS_REQUIRED')
        require(len({tuple(row['request_ids']) for row in rows})==1,'REQUEST_MAPPING_CHANGED')
        require(len(site.events)==4*len(site.sites),'EVERY_LAYER_EVERY_FORWARD_REQUIRED')
        counts={p:sum(e['method']==p for e in site.events) for p in ('_forward_prefill','_forward_decode')}
        require(counts['_forward_prefill']==len(site.sites) and counts['_forward_decode']==3*len(site.sites),
                'PREFILL_DECODE_COVERAGE_REQUIRED')
        if site.fault:
            require(all(e['applied'] and e['consumed'] and e['restored'] and e['reference_exact'] and not e['failed'] for e in site.events),
                    'EFFECT_CONSUMPTION_RECOVERY_NOT_PROVEN')
            require(any(e['changed']>0 for e in site.events),'NO_NUMERICAL_EFFECT')
        require(not site.uncovered,'UNOBSERVED_DYNAMIC_QUANT_CALLS:'+str(site.uncovered[:3]))
        hits=sum(e['applied'] for e in site.events)
        coverage=dict(observed=len(site.events),eligible=len(site.events) if site.fault else 0,
            attempted=hits,applied=hits,changed=sum(e['changed']>0 for e in site.events),
            consumed=sum(e['consumed'] for e in site.events),restored=sum(e['restored'] for e in site.events),failed=0)
        return dict(rank=worker.rank,pid=os.getpid(),model_id=id(model),coverage=coverage,output=[{k:v for k,v in r.items() if k!='request_ids'} for r in rows],
                    requests=rows,events=site.events,site_count=len(site.sites),stage_counts=counts,uncovered=site.uncovered)
    require(action=='remove','UNKNOWN_ACTION')
    resources=site.close()
    require(vars(model).get('compute_logits') is logits,'LOGITS_WRAPPER_CHANGED')
    delattr(model,'compute_logits')
    for h in hooks:h.remove()
    delattr(worker,key)
    return dict(rank=worker.rank,resources=resources)


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--kind',choices=('qwen06','qwen38'),default='qwen38');parser.add_argument('--batch-size',type=int,default=2);opts=parser.parse_args();assert opts.batch_size>=2
    from vllm import LLM,SamplingParams
    report=dict(status='STARTED',scopes=[],warmup=[],qualification_granted=False,
                acceptance_scope='REAL_MODEL_SITE_NOT_PUBLIC_RUNTIME_ADMISSION',tp=2)
    out=Path('/evidence/continuous_batch.json')
    def save():out.write_text(json.dumps(report,indent=2)+'\n')
    try:
        extra=dict(limit_mm_per_prompt={'image':0,'video':0}) if opts.kind=='qwen38' else {}
        llm=LLM(model='/model',dtype='bfloat16',quantization='ascend',tensor_parallel_size=2,
            distributed_executor_backend='mp',enforce_eager=True,enable_prefix_caching=False,
            async_scheduling=False,enable_chunked_prefill=True,max_num_batched_tokens=1024,
            max_model_len=1024,max_num_seqs=opts.batch_size,gpu_memory_utilization=.8,seed=42,
            speculative_config=None,kv_transfer_config=None,trust_remote_code=True,
            worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension',**extra)
        seeds=['Explain in one sentence why the sky appears blue.','Explain in one sentence why the grass appears green.','Summarize the purpose of a computer in one sentence.','What makes a stone sink in water?','Explain why snow appears white in daylight.','Give one example of a simple machine.','What is the sum of four and five?','Name one planet in the solar system.']
        prompts=[dict(prompt_token_ids=llm.get_tokenizer().encode(seeds[i%len(seeds)],add_special_tokens=True)) for i in range(opts.batch_size)]
        params=SamplingParams(temperature=0,max_tokens=4,min_tokens=4,ignore_eos=True)
        def request(owner,phase):
            assert llm.reset_prefix_cache(reset_running_requests=False)
            llm.sleep(level=0,mode='wait')
            ids=list(llm.enqueue(prompts,params,use_tqdm=False))
            from demoncat.runtime.vllm_ascend.worker_extension.serving_context import actual_driver_request_map
            mapping=actual_driver_request_map(llm.llm_engine.output_processor)
            assert len(mapping)==len(ids)==opts.batch_size and {m['internal_request_id'] for m in mapping}==set(ids)
            installed=llm.collective_rpc(rpc,args=('install',owner,phase=='fault',opts.kind),timeout=120)
            start=time.monotonic()
            llm.wake_up(tags=['scheduling']);outputs=llm.wait_for_completion(use_tqdm=False)
            elapsed=time.monotonic()-start
            rows=llm.collective_rpc(rpc,args=('collect',owner),timeout=120)
            assert all(set(x['request_ids'])==set(ids) for r in rows for x in r['requests'])
            assert {o.request_id for o in outputs}=={m['external_request_id'] for m in mapping}
            cleanup=llm.collective_rpc(rpc,args=('remove',owner),timeout=120)
            assert len(rows)==2 and all(r['resources']==dict(leases=0,hooks=0,inflight=0) for r in cleanup)
            return dict(phase=phase,mapping=mapping,installed=installed,ranks=rows,cleanup=cleanup,seconds=elapsed,
                tokens=[list(o.outputs[0].token_ids) for o in outputs],text=[o.outputs[0].text for o in outputs],finite=all(v['hidden_finite'] and v['logits_finite'] for r in rows for v in r['output']))
        for i in range(6):
            report['warmup'].append(request('warm-'+str(i),'normal'));save()
            if i and report['warmup'][-1]['finite'] and report['warmup'][-2]['finite'] and all(report['warmup'][-1]['ranks'][r]['output']==report['warmup'][-2]['ranks'][r]['output'] for r in (0,1)):break
        else:raise RuntimeError('WARMUP_NOT_BIT_EXACT_WITHIN_SIX')
        for index in range(2):
            scope=dict(owner=str(uuid.uuid4()),requests=[]);report['scopes'].append(scope);save()
            for phase in ('normal','fault','off','health'):
                scope['requests'].append(request(scope['owner'],phase));save()
            n,f,o,h=scope['requests']
            assert all(r['finite'] for r in scope['requests']),'NONFINITE_ACCEPTANCE_BATCH'
            for rank in (0,1):
                assert n['ranks'][rank]['output']==o['ranks'][rank]['output']==h['ranks'][rank]['output'],'OFF_HEALTH_NOT_EXACT'
                assert n['ranks'][rank]['output']!=f['ranks'][rank]['output'],'NO_HIDDEN_LOGITS_PROPAGATION'
            scope.update(status='SITE_TWO_PHASE_RECOVERY_PASSED',fault_time_ratio=f['seconds']/n['seconds'])
            save()
        identities=[[ (r['pid'],r['model_id']) for r in s['requests'][0]['ranks']] for s in report['scopes']]
        assert identities[0]==identities[1],'WORKERS_CHANGED'
        report['status']='TWO_SCOPES_REAL_BATCH_SITE_PASSED_PENDING_RUNTIME_INTEGRATION'
    except BaseException as e:
        report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    save();print(json.dumps(dict(status=report['status'],error=report.get('error'))),flush=True)
    return 0 if report['status'].startswith('TWO_SCOPES') else 2

if __name__=='__main__':raise SystemExit(main())
