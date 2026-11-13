"""Create bounded cases from a frozen input bank and one verified private vendor."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/home/fault28-reduce-replay-20261113')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def create(case_id,input_id,scope,mode,*,qkv=None,batch=8,steps=1,capture=True):
    stage=ROOT/'vendor_visible_a1'
    base=json.loads(Path('/home/fault28-eager-20261110/setup/baseline.json').read_bytes())
    descriptions={'harvest':'Full-model baseline forward; private layer0 QKV harvest, not statistical acceptance.',
                  'calibration':'Fixed real-QKV native FIA replay only; not full-model acceptance.',
                  'model':'Full Qwen3 eager prefill through native FIA; captured steps are repeated prefill, not decode.'}
    base.update(schema='fault28.visible.case.v1',case_id=case_id,mode=mode,capture_scope=scope,batch=batch,
        scope=descriptions[scope],repeats=steps,request_ids=[f'request_{i:02d}' for i in range(batch)],
        model_family='qwen3_fullmodel',capture_msprobe=capture,statistics_steps=list(range(steps)),
        input_manifest='/audit/private/input_bank/'+input_id+'.json',
        input_manifest_sha256=sha(ROOT/'private/input_bank'/(input_id+'.json')),
        approved_vendor_manifest_file='/audit/vendor_visible_a1/vendor_manifest.json',
        approved_vendor_manifest_sha256=sha(stage/'vendor_manifest.json'),
        approved_custom_opp_path='/audit/vendor_visible_a1/vendors/fault28_fia_graph_transformer',
        statistics_variant='task_local_cpu_fp64_high_v1' if capture else 'official_low')
    if not capture:
        base.pop('approved_collector_manifest_file',None);base.pop('approved_collector_manifest_sha256',None)
    if qkv is not None:
        qkv=Path(qkv).resolve()
        assert qkv.is_relative_to(ROOT) and qkv.name=='qkv_manifest.json'
        base.update(qkv_manifest_file='/audit/'+qkv.relative_to(ROOT).as_posix(),qkv_manifest_sha256=sha(qkv))
    path=ROOT/'setup'/(case_id+'.json')
    if path.exists():
        assert json.loads(path.read_bytes())==base,'Refuse to overwrite a different case'
    else:
        with path.open('x',encoding='utf-8') as f:json.dump(base,f,indent=2,allow_nan=False);f.write('\n')
    return path
