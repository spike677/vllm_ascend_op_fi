"""CPU-only freeze, fixture creation, and exact execution-source preregistration."""
import hashlib
import json
from pathlib import Path
import os
import struct

ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')
def main():
    ev=ROOT/'build/kernel_a1/registration.json';assert json.loads(ev.read_bytes())['compiled_keys'][-1]==119000000000400000
    fixture=ROOT/'private/analytic_fixture';fixture.mkdir(exist_ok=False)
    manifest={'schema':'fault28.visible.qkv.v1','input_id':'analytic_Q0_K0_V1',
              'request_rows':[{'slot':i,'request_id':'analytic_'+str(i)} for i in range(8)]}
    for name,heads,value in [('query',16,0),('key',8,0),('value',8,1)]:
        path=fixture/(name+'.fp16.bin');block=struct.pack('<e',value)*(2048*128)
        with path.open('xb') as f:
            for i in range(8*heads):f.write(block)
        manifest[name]={'file':path.name,'sha256':sha(path),'shape':[8,heads,2048,128]}
    save(fixture/'qkv_manifest.json',manifest)
    (ROOT/'inventory_client.py').chmod(0o700)
    names=sorted(p.relative_to(ROOT).as_posix() for p in ROOT.glob('*.py'))+['env.sh','PLAN.md']
    names+=sorted(p.relative_to(ROOT).as_posix() for folder in ['analysis','source_policy','source_kernel'] for p in (ROOT/folder).rglob('*') if p.is_file())
    contract={'schema':'controlled.online_softmax_denominator_replay.preregister.v1',
        'source_files':{name:sha(ROOT/name) for name in names},
        'vendor_manifest_sha256':sha(ROOT/'vendor_visible_a1/vendor_manifest.json'),
        'input_bank_sha256':sha(ROOT/'private/input_bank/manifest.json'),
        'analytic_fixture_sha256':sha(fixture/'qkv_manifest.json'),
        'input_id':'input_00','candidate_count':1,'variant':'online_softmax_denominator_replay',
        'B_key':'109000000000400000','F_key':'119000000000400000','B_M_S2':[128,128],'F_M_S2':[128,128],
        'rtol':1e-5,'atol':1e-8,'reference':'baseline','minimum_reliable_statistic_relative_change':.001,
        'formal_capture_steps':[0,1,2],'formal_steps_meaning':'repeated 2048-token prefill, not decode',
        'historical_fault28_root_cause_claimed':False,'npu_executed_at_freeze':False}
    save(ROOT/'preregister.json',contract)
    print(json.dumps({'status':'FROZEN_REQUIRES_DEVICE_GATE','sha256':sha(ROOT/'preregister.json'),'source_files':len(names)}))
if __name__=='__main__':main()
