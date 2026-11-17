"""Complete launch census after independent raw statistics comparison."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
source=Path('/home/fault33-paged-tiling-20261105/analysis/native_evidence.py')
if hashlib.sha256(source.read_bytes()).hexdigest()!='66eabe0e12fc4495889cc1d44a885dc962d149ac573de4735a952f73df0c901c':raise RuntimeError('PARSER_CHANGED')
spec=importlib.util.spec_from_file_location('native_evidence',source);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
windows=mod.parse_log(a.case/'process.log');result=json.loads((a.case/'result.json').read_bytes())
if result['status']!='PASS_HF_CONTINUOUS_PAGED_GENERATION_NOT_COMMON_WORKER':raise RuntimeError('RUN_NOT_COMPLETE')
exits={}
with (a.case/'process.log').open() as stream:
    for line in stream:
        if not line.startswith('{'):continue
        try:event=json.loads(line)
        except json.JSONDecodeError:continue
        if event.get('fault33')=='exit':
            if event['call_id'] in exits:raise RuntimeError('DUPLICATE_EXIT')
            exits[event['call_id']]=event
if set(exits)!={w['call_id'] for w in windows}:raise RuntimeError('EXIT_CENSUS_MISMATCH')
hosts={};following={};phases=[]
for phase in ('reference','comparison','off'):
    events=[json.loads(x) for x in (a.case/(phase+'_native')/'host_events.jsonl').read_text().splitlines()]
    follows=json.loads((a.case/(phase+'-followups.json')).read_bytes())
    for x in events:
        if x['call_id'] in hosts:raise RuntimeError('DUPLICATE_HOST_CALL')
        hosts[x['call_id']]=x
    for x in follows:
        if x['call_id'] in following:raise RuntimeError('DUPLICATE_FOLLOWUP')
        following[x['call_id']]=x
    phases.append(dict(phase=phase,qualified_geometry_calls=len(events),actual_policy_applied=sum(x['policy_applied'] for x in events),
        sinner=sorted({x['typed']['increFlashAttentionSingleCoreParams.singleProcessSInnerSize'] for x in events}),
        stock_followup_calls=len(follows),stock_lengths=sorted({x['kv_length'] for x in follows})))
if {w['call_id'] for w in windows}!=set(hosts)|set(following) or set(hosts)&set(following):raise RuntimeError('COMPLETE_LAUNCH_CENSUS_MISMATCH')
for window in windows:
    key=window['call_id'];entry=window['enter']
    if key in following:
        f=following[key]
        if entry.get('role')!='stock_followup' or entry['kv_length']!=f['kv_length'] or f['policy_applied']:raise RuntimeError('FOLLOWUP_SCOPE_DIFFERS')
        continue
    h=hosts[key]
    if (h['pid'],h['tid'])!=(entry['pid'],entry['tid']):raise RuntimeError('HOST_IDENTITY_DIFFERS')
    if not entry['monotonic_ns']<=h['callback_begin_monotonic_ns']<=h['callback_end_monotonic_ns']<=exits[key]['monotonic_ns']:raise RuntimeError('HOST_TIME_DIFFERS')
    if h['callback_status']!=0 or not h['typed_matches_final_serialized_raw'] or h['serialized_captures']!=1:raise RuntimeError('RAW_POD_MISMATCH')
    if not all(h[k]==window['launch'][k] for k in ('tiling_key','block_dim','schedule_mode')):raise RuntimeError('ACTUAL_LAUNCH_DIFFERS')
    if h['raw_size']!=window['native_raw_size'] or h['raw_size']!=len(bytes.fromhex(h['raw_hex'])):raise RuntimeError('RAW_SIZE_DIFFERS')
    for i in range(2):
        for field,value in window['sdk_bmm'][i].items():
            name=f'bmm{i+1}TilingData.{field}'
            if name in h['typed'] and h['typed'][name]!=value:raise RuntimeError('SDK_BMM_DIFFERS')
summary=dict(status='PASS_COMPLETE_NATIVE_LAUNCH_CENSUS',actual_launches=len(windows),phases=phases,
    qualified_host_pod_and_raw_calls=len(hosts),stock_launch_only_calls=len(following),
    kernel_argument_raw_observed=False,common_vllm_worker=False,raw_data_published=False,
    process_log_sha256=sha(a.case/'process.log'),
    result_sha256=sha(a.case/'result.json'),
    limits=['Following lengths have actual launch evidence but no new fault policy qualification',
        'Statistics first difference is computed separately without these logs'])
with a.out.open('x') as f:json.dump(summary,f,indent=2);f.write('\n')
print(json.dumps(summary))
