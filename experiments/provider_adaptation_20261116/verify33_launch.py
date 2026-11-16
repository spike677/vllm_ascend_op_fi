"""Separate native launch audit, after raw-statistics first difference selection."""
import hashlib,importlib.util,json
from pathlib import Path
root=Path('/home/provider-adaptation-20261116/provider33-model-a2')
path=Path('/home/fault33-paged-tiling-20261105/analysis/native_evidence.py')
assert hashlib.sha256(path.read_bytes()).hexdigest()=='66eabe0e12fc4495889cc1d44a885dc962d149ac573de4735a952f73df0c901c'
spec=importlib.util.spec_from_file_location('native_evidence',path)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
reports=[]
for arm in ('baseline','fault'):
    case=root/f'provider33_{arm}'
    windows=mod.parse_log(case/'process.log')
    events=[]
    for stage in ('warm_native','native','off_native'):
        events.extend(json.loads(line) for line in (case/stage/'host_events.jsonl').read_text().splitlines())
    byid={x['call_id']:x for x in events}
    assert len(byid)==len(events)==len(windows)==140
    for w in windows:
        h=byid[w['call_id']];entry=w['enter']
        assert (h['pid'],h['tid'])==(entry['pid'],entry['tid'])
        assert entry['monotonic_ns']<=h['callback_begin_monotonic_ns']<=h['callback_end_monotonic_ns']
        assert h['callback_status']==0 and h['typed_matches_final_serialized_raw'] and h['serialized_captures']==1
        assert all(h[k]==w['launch'][k] for k in ('tiling_key','block_dim','schedule_mode'))
        assert h['raw_size']==w['native_raw_size']==len(bytes.fromhex(h['raw_hex']))
        for i in range(2):
            for field,value in w['sdk_bmm'][i].items():
                key=f'bmm{i+1}TilingData.{field}'
                if key in h['typed']:assert h['typed'][key]==value
    result=json.loads((case/'result.json').read_text())
    summaries=[]
    for stage in ('warm_native','native','off_native'):
        current=[json.loads(line) for line in (case/stage/'host_events.jsonl').read_text().splitlines()]
        summaries.append(dict(stage=stage,calls=len(current),applied=sum(x['policy_applied'] for x in current),
            sinner=sorted(set(x['typed']['increFlashAttentionSingleCoreParams.singleProcessSInnerSize'] for x in current))))
    reports.append(dict(arm=arm,status='PASS_NATIVE_LAUNCH',actual_launches=len(windows),stages=summaries,
        all_discovered_candidates_covered=result['coverage']['all_discovered_candidates_covered'],
        off_exact=result['off_exact'],repeat_stable=result['repeat_stable'],changed=result['changed'],
        logits_max_abs=result['logits_max_abs'],top1_equal=result['top1_equal'],
        matching_layers=len(result['coverage']['candidates']),requests_per_call=8,
        kernel_argument_raw_observed=False,public_host_raw_observed=True,
        independent_binding=['call_id','pid','tid','monotonic_begin','actual_launch','sdk_bmm'],
        enter_marker_scope_token_recorded=False))
out=root/'native_launch_summary.json'
with out.open('x') as f:json.dump(reports,f,indent=2)
print(json.dumps(reports,indent=2))
