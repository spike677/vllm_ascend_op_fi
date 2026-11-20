"""One genuine stock TND Worker request for new-package numerical bridge.

Explicit zero-F research only. No custom OPP path or private loader is used.
"""
import argparse,json,os,sys,hashlib,traceback
from pathlib import Path
sys.path[:0]=['/repo','/repo/src','/repo/experiments/c2c3_closure_20261117']
from experiments.c2c3_closure_20261117.natural_attention_worker_core_probe import actual,remove as remove_binding
from experiments.worker_core_v11_20261118.tnd28_research_observation import ResearchObservationClient
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import read_enqueued_identity
M='28.fia.denominator_replay'
def need(v,why):
 if not v:raise RuntimeError(why)
def setup(wrapper,identity,tokens,observer,observer_sha,folder):
 from demoncat.runtime.vllm_ascend.worker_extension.attention28_stock_census import StockTndPrefillCensus
 from demoncat.runtime.vllm_ascend.worker_extension.serving_attention import install_attention_serving_binding
 w=actual(wrapper);need(w.demoncat_worker_identity()['worker']==identity,'WORKER_CHANGED')
 native=StockTndPrefillCensus(mechanism=M,observer_receipt=observer,observer_receipt_sha256=observer_sha,evidence_dir=folder)
 install_attention_serving_binding(w,M,prompt_token_ids=tokens,provenance={'explicit_research_stock_bridge':True},native=native)
 return dict(status='STOCK_RESEARCH_BOUND')
def main():
 p=argparse.ArgumentParser();p.add_argument('--observer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
 raw=a.observer.read_bytes();obs=json.loads(raw);library=obs['observer_library'];osh=hashlib.sha256(raw).hexdigest()
 need(hashlib.sha256(Path(library['path']).read_bytes()).hexdigest()==library['sha256'],'OBSERVER_BYTES')
 need(not os.environ.get('ASCEND_CUSTOM_OPP_PATH'),'STOCK_ONLY_NO_PRIVATE_VENDOR')
 if not a.child:
  need(not os.environ.get('LD_PRELOAD'),'CLEAN_START');a.out.mkdir();(a.out/'runtime').mkdir()
  env=dict(os.environ,LD_PRELOAD=library['path'],DCAT_ATTN_OBSERVER_DIR=str(a.out/'runtime'),ACLNN_CACHE_LIMIT='0',TASK_QUEUE_ENABLE='0')
  os.execve(sys.executable,[sys.executable,'-B',__file__,*sys.argv[1:],'--child'],env)
 d=dict(status='STARTED',protocol='EXPLICIT_RESEARCH_STOCK_ONLY',fault_authorized=False);client=None;identity=None
 def save():(a.out/'result.json').write_text(json.dumps(d,indent=2)+'\n')
 try:
  from vllm import LLM,SamplingParams
  llm=LLM(model='/model',dtype='float16',tensor_parallel_size=1,enforce_eager=True,max_model_len=2304,max_num_seqs=1,max_num_batched_tokens=2304,block_size=16,num_gpu_blocks_override=160,enable_prefix_caching=False,enable_chunked_prefill=False,async_scheduling=False,gpu_memory_utilization=.25,seed=42,disable_log_stats=True,worker_extension_cls='demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension')
  words=llm.get_tokenizer().encode('Explain how attention reads a sequence of tokens. ',add_special_tokens=True);tokens=(words*((2048+len(words)-1)//len(words)))[:2048]
  def bind(identity,config):
   result=llm.collective_rpc(setup,args=(identity,tokens,str(a.observer),osh,str(a.out/'native_calls')),timeout=300);need(len(result)==1,'ONE_WORKER');return {}
  client=ResearchObservationClient(llm,bind)
  def action(name,*args):
   reply=getattr(client,name)(*args);d['last_action']=reply;save();need(reply['status']==('QUERY_OK' if name=='query' else 'DETACHED' if name=='detach' else 'ACTION_COMPLETED'),'RESEARCH_ACTION_FAILED');return reply['response']['provider_state']
  for name in ('prepare','register','check','enable'):action(name)
  identity=client.worker
  need(llm.reset_prefix_cache(reset_running_requests=False),'CACHE_RESET')
  pending=list(llm.enqueue([dict(prompt_token_ids=tokens)],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True),use_tqdm=False));mapping=read_enqueued_identity(llm,pending,seen_internal_request_ids=[])
  action('advance',pending[0]);outputs=llm.wait_for_completion(use_tqdm=False);need(len(outputs)==1 and outputs[0].request_id==mapping['external_request_id'],'REQUEST_MAPPING')
  state=action('collect_replay');d['request']=state['evidence']['requests'][0];d['mapping']=mapping;d['tokens']=list(outputs[0].outputs[0].token_ids)
  action('disable');action('unregister');action('detach');d['binding_removal']=llm.collective_rpc(remove_binding,args=(identity,M),timeout=120)
  d['status']='STOCK_REQUEST_READY_FOR_FULL_HASH_BRIDGE'
 except BaseException as e:d.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
 finally:
  if client:
   if client.owned:
    for action_name in ('disable','unregister','detach'):
     try:getattr(client,action_name)()
     except BaseException as e:d.setdefault('cleanup_errors',[]).append(repr(e))
   d['host']=client.snapshot()
  save()
 print(json.dumps({k:d[k] for k in ('status','error') if k in d}));return 0 if d['status']=='STOCK_REQUEST_READY_FOR_FULL_HASH_BRIDGE' else 2
if __name__=='__main__':raise SystemExit(main())
