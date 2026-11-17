"""Real autoregressive tokens and KV growth; no teacher repetition or cache crop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from contextlib import ExitStack


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['baseline','fault'],required=True)
    p.add_argument('--steps',type=int,default=4);a=p.parse_args()
    if not 1<=a.steps<=8:raise ValueError('1..8 real decode steps required')
    from fault_injection.deployments.paged33 import verify_bundle
    vendor=verify_bundle('/audit')
    if not os.environ.get('DCAT33_FRESH_PROCESS'):
        if os.environ.get('ASCEND_CUSTOM_OPP_PATH') or os.environ.get('LD_PRELOAD'):raise RuntimeError('FOREIGN_VENDOR')
        env=dict(os.environ,DCAT33_FRESH_PROCESS='1',ASCEND_CUSTOM_OPP_PATH=vendor['vendor_root'],
            ACLNN_CACHE_LIMIT='0',LD_LIBRARY_PATH=vendor['op_api_library_directory']+':'+os.environ.get('LD_LIBRARY_PATH',''))
        os.execve(sys.executable,[sys.executable,__file__,'--arm',a.arm,'--steps',str(a.steps)],env)
    import torch,torch_npu
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from fault_injection import attach
    from fault_injection.deployments.paged33 import digest
    from fault_injection.deployments.paged33_continuous import ContinuousPaged33Runtime
    torch.set_num_threads(2);torch.manual_seed(42);torch.use_deterministic_algorithms(True);torch.npu.set_device(0)
    tok=AutoTokenizer.from_pretrained('/model',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained('/model',local_files_only=True,torch_dtype=torch.float16,
        attn_implementation='sdpa').eval().to('npu')
    seeds=['Explain a compiler loop. ','Describe a library catalog. ','Solve elementary algebra. ',
        'Validate a Python configuration. ','Explain the water cycle. ','Describe a city journey. ',
        'Summarize plant growth observations. ','Compare sorting algorithms. ']
    ids=[tok.encode(s*6000,add_special_tokens=False)[:16383] for s in seeds]
    if any(len(x)!=16383 for x in ids):raise RuntimeError('SHORT_INPUT')
    prefix=torch.tensor(ids,device='npu')
    result=dict(status='STARTED',arm=a.arm,actual_model_class=type(model).__qualname__,
        common_vllm_worker=False,scope='HF_REAL_AUTOREGRESSIVE_PAGED_DECODE_FIXED_FAULT_ELIGIBILITY',
        input_sha256=digest(prefix),prefill_length=16383,decode_steps=a.steps,phases=[])
    def save():Path('/evidence/result.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        runtimes={name:ContinuousPaged33Runtime(model,bundle='/audit',evidence_dir='/evidence/'+name+'_native',decode_steps=a.steps)
            for name in ('reference','comparison','off')}
        # Source discovery must see the actual implementation, before msprobe
        # installs forward wrappers. Lifetimes still enter/close independently.
        sessions={name:attach(model,'33.paged_decode',arm=a.arm if name=='comparison' else 'baseline',runtime=runtime)
            for name,runtime in runtimes.items()}
        reference=None
        for phase in ('reference','comparison','off'):
            with torch.no_grad(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                prefill=model(input_ids=prefix,use_cache=True,logits_to_keep=1)
            cache=prefill.past_key_values;current=prefill.logits[:,-1,:].argmax(-1,keepdim=True)
            generated=[row[:] for row in current.cpu().tolist()];del prefill
            row=dict(phase=phase,initial_tokens=current.cpu().tolist(),prefill_cache_length=cache.get_seq_length(),steps=[])
            result['phases'].append(row);save()
            runtime=runtimes[phase];arm=a.arm if phase=='comparison' else 'baseline'
            collector=None
            if phase=='comparison':
                import msprobe
                from msprobe.pytorch import PrecisionDebugger
                cap=Path('/evidence/serving/capture');cap.mkdir(parents=True)
                config=dict(task='statistics',level='mix',precision='high',rank=[],step=list(range(a.steps)),
                    dump_path=str(cap/'observation'),async_dump=False,
                    statistics=dict(scope=[],list=[],data_mode=['all'],summary_mode='statistics',tensor_list=[]))
                (cap/'config.json').write_text(json.dumps(config,indent=2)+'\n')
                root=Path(msprobe.__file__).resolve().parent
                files={p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.py','.json','.yaml')}
                (cap/'collector_identity.json').write_text(json.dumps(dict(files=files,modified=False),sort_keys=True,indent=2)+'\n')
                collector=PrecisionDebugger(config_path=str(cap/'config.json'))
            outputs=[]
            with torch.no_grad(),ExitStack() as stack:
                session=stack.enter_context(sessions[phase])
                for step in range(a.steps):
                    before=cache.get_seq_length();input_tokens=current.cpu().tolist()
                    if before!=16383+step:raise RuntimeError('CACHE_NOT_GROWING_BY_ONE')
                    if collector:collector.start(model=model,rank_id=0)
                    try:out=model(input_ids=current,past_key_values=cache,use_cache=True,logits_to_keep=1)
                    finally:
                        if collector:collector.stop();collector.step()
                    if out.past_key_values is not cache or cache.get_seq_length()!=before+1:raise RuntimeError('CACHE_IDENTITY_OR_LENGTH_CHANGED')
                    logits=out.logits.detach().cpu();del out
                    if not bool(torch.isfinite(logits).all()):raise RuntimeError('NONFINITE_LOGITS')
                    sampled=logits[:,-1,:].argmax(-1,keepdim=True)
                    for i,value in enumerate(sampled.tolist()):generated[i]+=value
                    current=sampled.to('npu');outputs.append(logits)
                    row['steps'].append(dict(step=step,input_tokens=input_tokens,output_tokens=sampled.tolist(),
                        cache_length_before=before,cache_length_after=before+1,logits_sha256=digest(logits),finite=True))
                    if step==0:
                        row['coverage']=session.report();save();session.assert_coverage()
                        session.close();row['coverage']=session.report()
                        if a.steps>1:stock=stack.enter_context(runtime.stock_followups())
                if a.steps>1:
                    row['stock_routing']=stock.report();stock.assert_coverage()
            row.update(native_policy_calls=len(runtime.calls),actual_policy_applied=sum(x['host']['policy_applied'] for x in runtime.calls),
                stock_followup_calls=len(runtime.followups),following_lengths=sorted({x['kv_length'] for x in runtime.followups}),
                inputs_unchanged=all(x['inputs_unchanged'] for x in runtime.calls+runtime.followups),
                generated_tokens=generated,generated_text=[tok.decode(x,skip_special_tokens=False) for x in generated])
            Path('/evidence/'+phase+'-followups.json').write_text(json.dumps(runtime.followups,indent=2)+'\n')
            if reference is None:reference=outputs;reference_tokens=generated
            else:
                row['complete_logits_equal_reference']=[torch.equal(x,y) for x,y in zip(reference,outputs)]
                row['per_request_max_abs']=[[float((x[i].float()-y[i].float()).abs().max()) for i in range(8)] for x,y in zip(reference,outputs)]
                row['generated_tokens_equal_reference']=[x==y for x,y in zip(reference_tokens,generated)]
                if (phase=='off' or arm=='baseline') and not all(row['complete_logits_equal_reference']):raise RuntimeError('COMPLETE_TRAJECTORY_NOT_RESTORED')
            layers=len(row['coverage']['candidates'])
            if row['native_policy_calls']!=layers or row['stock_followup_calls']!=layers*(a.steps-1) or not row['inputs_unchanged']:
                raise RuntimeError('ACTUAL_POLICY_OR_FOLLOWUP_COVERAGE_INCOMPLETE')
            if row['actual_policy_applied']!=(layers if arm=='fault' else 0):raise RuntimeError('POLICY_HIT_COUNT_DIFFERS')
            runtime.cache=runtime.previous_cache=None
            del cache,current,outputs;torch.npu.empty_cache();save()
        result['status']='PASS_HF_CONTINUOUS_PAGED_GENERATION_NOT_COMMON_WORKER'
    except BaseException as error:
        result.update(status='FAILED_WITH_EVIDENCE',error=repr(error),traceback=traceback.format_exc());save();return 2
    save();return 0


if __name__=='__main__':raise SystemExit(main())
