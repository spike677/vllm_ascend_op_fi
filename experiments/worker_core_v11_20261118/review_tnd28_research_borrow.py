"""Offline live-borrow review. Does not establish numerical bridge or F/OFF."""
import argparse,hashlib,json,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/c2c3_closure_20261117')]
from experiments.c2c3_closure_20261117 import review_tnd28_prepared_input as R

def need(v,why):
 if not v:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def review(root,source,build,libraries):
 d=read(root/'result.json');window=d['binding_removal']['input_window']
 need(d['entry_protocol']=='EXPLICIT_RESEARCH_PROVIDER_NOT_PUBLIC_RUNTIME' and not d['ownership_retained'] and not d.get('cleanup_errors'),'RESEARCH_RELEASE_REQUIRED')
 need(d['binding_removal']['first_error'] is None and window['window_complete'],'WINDOW_INCOMPLETE')
 host=root/d['host_receipt']['path'];need(sha(host)==d['host_receipt']['sha256'],'HOST_BYTES');h=read(host)
 need(h['protocol']==d['entry_protocol'] and not h['ownership_retained'],'HOST_RELEASE')
 actions=['prepare','register','check','enable','advance','query','collect_replay','disable','unregister','detach']
 need([r['action'] for r in h['history']]==actions,'RESEARCH_ACTION_ORDER')
 for pin,record in zip(d['rpc'],h['history']):
  p=root/pin['path'];need(sha(p)==pin['sha256'] and read(p)==record and record['research_direct_provider'] is True,'ORIGINAL_RESEARCH_RECEIPT')
 need(len(d['rpc'])==len(actions),'ALL_RECEIPTS')
 cleanup=d['cleanup'];need(all(cleanup[k]==0 for k in ('owned_hooks','owned_api_wrappers','inflight','leases')),'RESOURCES_REMAIN')
 def local(path):
  part=Path(path).relative_to('/evidence/probe');p=(root/part).resolve();need(p.is_relative_to(root.resolve()),'OUTSIDE_RUN');return p
 public=read(local(window['public_artifact']));returned=read(local(window['returned_artifact']));native=R.lines(local(window['native_artifact']),1024**2)
 # Validate the new borrow extension, then pass an in-memory base-schema view
 # to the preserved reader. Original files/hashes are never rewritten.
 borrow_event=[r for r in native if r['phase']=='execute_exit'][0]['data']['execute_borrow']
 expected=dict(first_error=0,child_started=1,child_consumed=1,parent_started=1,parent_returned=1,drained=1,detached=0,reset_attempted=0,reset_returned=0,cancelled=0,complete=1,terminal_uncertain=0,left=1,child_status=0,parent_status=0,drain_status=0,detach_status=0,reset_status=0,fault_authorized=False)
 need(all(borrow_event[k]==v for k,v in expected.items()),'NORMAL_BORROW_LIFETIME')
 need(borrow_event['pid']==public['identity']['pid'] and borrow_event['tid']==public['identity']['native_tid'],'BORROW_THREAD')
 base_events=deepcopy(native)
 [r for r in base_events if r['phase']=='execute_exit'][0]['data'].pop('execute_borrow')
 typed=R.protocol(public,returned,base_events)
 entries={r['recorded_path']:r for r in read(libraries)['files']}
 for data in typed.values():
  if 'function' not in data:continue
  row,_=R.function_library_entry(data['function'],entries)
  need(sha(Path(row['path']))==row['sha256'] and Path(row['path']).stat().st_size==row['bytes'],'FUNCTION_LIBRARY_BYTES')
 live=window['live_typed_plan'];need(live['first_error'] is None and [e['stage'] for e in live['events']]==[1,2],'TWO_TYPED_STAGES')
 need(live['events'][0]['typed_sha256']==live['events'][1]['typed_sha256'],'TYPED_VALUES_CHANGED')
 borrow=window['execute_borrow']['borrow']
 need(borrow==borrow_event,'BORROW_RETURN_JOIN')
 completion,files,metadata,abi=R.bundle(build/'evidence/completion.json',d['build_sha256'],build/'stage/vendors/dcat_f28_tnd_replay_transformer',source,d['source_manifest_sha256'])
 public['_return']=returned
 joined=R.runtime_join(root,R.lines(root/'runtime/runtime.jsonl',128*1024**2),public,base_events,abi,completion['kernel_binary_sha256'])
 calls=[c for f in d['request']['worker']['forwards'] for c in f['calls']]
 need(len(calls)==28 and all(c['original_calls']==1 and c['original_returned'] and c['inputs_unchanged'] for c in calls),'REAL_MODEL_CALLS')
 need(d['health']['output_tokens']==d['request']['output_tokens'],'POST_REMOVAL_TOKEN_HEALTH')
 return dict(status='LIVE_TWO_STAGE_BORROW_AND_RELEASE_REVIEWED_NO_BRIDGE_OR_F',source_manifest_sha256=d['source_manifest_sha256'],build_sha256=d['build_sha256'],result_sha256=sha(root/'result.json'),host_sha256=sha(host),actual_calls=28,selected_path=window['module_path'],native_events=len(native),typed_stages=live['events'],borrow=borrow,actual_launches=joined,resources_zero=True,post_removal_token_health=True,full_output_bridge_proven=False,fault_authorized=False,qualification_granted=False)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('root','source','build','libraries','out'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();v=review(a.root,a.source,a.build,a.libraries);v['reader_sha256']=sha(Path(__file__));a.out.write_text(json.dumps(v,indent=2)+'\n');print(v['status'])
