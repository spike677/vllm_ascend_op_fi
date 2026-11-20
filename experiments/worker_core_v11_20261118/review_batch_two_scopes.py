"""Read preserved ordinary Batch records; no model/native execution or top-level PASS trust."""
import argparse,hashlib,json
from pathlib import Path
from experiments.worker_core_v11_20261118.review_two_scope_requests import need
PHASES=('normal','reference','lossless','calibration_off','fault','off')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def review_health(data, required=False):
    before, after = data.get('health_hashes_before'), data.get('health_hashes_after')
    if before is None and after is None:
        need(not required, 'FULL_HEALTH_REQUIRED')
        return False
    need(before is not None and after is not None and before == after, 'FULL_HEALTH_CHANGED')
    need(before['complete'] and before['owned_hooks'] == 0, 'HEALTH_HOOKS_REMAIN')
    rows = before['rows']
    need([row['kind'] for row in rows] == ['hidden','logits','hidden','logits'], 'HEALTH_FORWARD_SEQUENCE')
    need(all(row['finite'] and isinstance(row['sha256'], str) and len(row['sha256']) == 64 for row in rows), 'HEALTH_FINITE_HASH_REQUIRED')
    need(rows[2]['shape'][0] == rows[3]['shape'][0] == 8, 'HEALTH_B8_REQUIRED')
    return True


def review(root, require_full_health=False):
    root=Path(root);top=read(root/'result.json');need(len(top['scopes'])==2,'TWO_SCOPES')
    a,b=top['scopes'];need(a['worker']==b['worker'] and a['owner']!=b['owner'],'SAME_WORKER_DISTINCT_OWNER')
    summaries=[];seen=set()
    for index,pin in enumerate(top['scopes']):
        folder=root/('scope'+str(index));path=folder/'result.json';host=folder/'host-core-receipts.json'
        need(sha(path)==pin['result_sha256'] and sha(host)==pin['host_sha256'],'ORIGINAL_DIGEST_CHANGED')
        d=read(path);h=read(host);s=d['serving_final'];rows=s['rows'];n,f,o=rows[0],rows[4],rows[5]
        need(h['worker']==pin['worker'] and h['owner']==pin['owner'] and h['ownership_retained'] is False,'OWNER_NOT_RELEASED')
        need([r['phase'] for r in rows]==list(PHASES) and not s['failed_batches'] and s['error'] is None,'PHASES_INCOMPLETE')
        need(d['model_before']==d['model_after'] and d['worker_after']['worker']==h['worker'],'MODEL_WORKER_CHANGED')
        need(d['healthy_before']==d['healthy_after'] and len(d['healthy_before'])==8,'HEALTH_TOKEN_RECOVERY')
        full_health = review_health(d, require_full_health)
        for i,p in enumerate(d['phases']):
            original=folder/('phase-'+PHASES[i]+'.json');need(sha(original)==p['original']['sha256'],'PHASE_BYTES_CHANGED');record=read(original)
            ids=record['request_ids'];need(record['completed'] and len(ids)==len(set(ids))==8 and not seen.intersection(ids),'ACTUAL_BATCH_IDS');seen.update(ids)
            need(record['serving']['rows']==rows[:i+1],'PHASE_ROW_HISTORY_CHANGED')
        for r in rows:
            need(r['projection']['hashes']['input']==n['projection']['hashes']['input'],'INPUT_CHANGED')
            need(r['projection']['consumer']['original_calls']==1 and r['projection']['execution']['original_model_calls']==1,'ACTUAL_QUANT_QMM_CHAIN')
            for stage in ('prefill','decode'):
                need(r[stage]['returned'] and r[stage]['hidden_finite'] and r[stage]['logits_finite'],'FINITE_REAL_FORWARD')
        for r in (rows[1],rows[2],rows[3],o):
            need(r['projection']['hashes']==n['projection']['hashes'],'CALIBRATION_OR_OFF_QUANT_DIFF')
            for stage in ('prefill','decode'):
                for field in ('hidden_sha256','logits_sha256'):need(r[stage][field]==n[stage][field],'CALIBRATION_OR_OFF_MODEL_DIFF')
        effect=f['projection']['effect'];need(s['actual_fault_hits']==1 and f['projection']['execution']['native_effect'] is True,'NATIVE_FAULT_REQUIRED')
        need(effect['phase']=='decode' and effect['quant']['changed_elements']>0 and effect['scale']['changed_elements']>0,'DOUBLE_NUMERIC_EFFECT')
        need(f['projection']['hashes']['quant']!=n['projection']['hashes']['quant'] and f['projection']['hashes']['scale']!=n['projection']['hashes']['scale'],'EFFECT_HASH_MISMATCH')
        for key in ('owned_hooks','owned_api_patches','owned_method_patches','owned_normal_readback_bytes','owned_row_modes'):need(s[key]==0,'RESOURCES_REMAIN')
        need(s['installed'] is False and s['native_registered'] is False and d['remove']['status']=='DEPLOYMENT_REMOVED','DEPLOYMENT_REMAINS')
        native=s['recovery'][-1]['native'];need(all(native[k]==0 for k in ('callback_active','completion_active','native_stack_active','orphan_descriptors','public_active','quarantined','unreturned_public_scope')),'NATIVE_ACTIVE')
        need(all(native['core'][k]==0 for k in ('borrowed_workspace_bytes','device_bytes','host_bytes','owned_descriptors','owned_workspace_bytes','pending_parent','quarantined_bytes','targets')),'NATIVE_ALLOCATION_LEAK')
        summaries.append(dict(selected_site=effect['semantic_position'],phase=effect['phase'],quant_changed=effect['quant']['changed_elements'],scale_changed=effect['scale']['changed_elements'],scale_max_abs_diff=effect['scale']['max_abs_diff_finite'],hidden_changed=n['decode']['hidden_sha256']!=f['decode']['hidden_sha256'],logits_changed=n['decode']['logits_sha256']!=f['decode']['logits_sha256'],top1_changed=n['decode']['top1']!=f['decode']['top1'],exact_off_quant_hidden_logits=True,external_health_tokens_exact=True,external_health_full_hidden_logits_observed=full_health,resources_zero=True,result_sha256=sha(path),host_sha256=sha(host)))
    return dict(status='TWO_SAME_WORKER_BATCH_SCOPES_REVIEWED',combination='A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B W8A8/TP1/B8',scope_count=2,actual_requests=128,actual_plan_batches=12,external_health_batches=4,scopes=summaries,qualification_granted=False,limitations=([] if all(s['external_health_full_hidden_logits_observed'] for s in summaries) else ['External health full hidden/logits missing in at least one scope.'])+['Complete hashes verify equality; not a statistics or all-target qualification.'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--require-full-health',action='store_true');a=p.parse_args();v=review(a.root,a.require_full_health);v.update(reader_sha256=sha(Path(__file__)),wrapper_sha256=sha(a.root/'result.json'));a.out.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(v['status'])
