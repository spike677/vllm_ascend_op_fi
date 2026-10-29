#!/usr/bin/env python3
"""Rehash task-private build dependencies and immutable SDK image files, CPU only."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path('/home/fault28-causal-20261029')
SDK=Path('/home/docker-system/overlay2/7c31ac8b4081145ef06f7a036edef0df3ab4a9dac44d21f0d046c146cae91d13/diff/usr/local/Ascend/cann-9.1.0')

def main():
    expected=json.loads(Path(__file__).with_name('expected_dependencies.json').read_text())
    source=ROOT/'source/ops-transformer-v9.1.0'
    actual=[]
    for item in expected['files']:
        p=(SDK if item['kind']=='sdk' else source)/item['path']
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=item['sha256'] or p.stat().st_size!=item['bytes']:
            raise RuntimeError('Dependency changed: '+str(p))
        actual.append({**item,'actual_path':str(p),'matched':True})
    heads={}
    for k,p,exp in [('ops_transformer',source,expected['ops_transformer_commit']),
                    ('opbase',ROOT/'dependencies/opbase',expected['opbase_commit']),
                    ('ops_tensor',ROOT/'dependencies/ops-tensor',expected['ops_tensor_commit'])]:
        head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        if head!=exp:
            raise RuntimeError('Commit mismatch '+k)
        heads[k]=head
    result={'schema':'fault28.causal.build_dependencies.v1','status':'PASS_EXACT_DEPENDENCY_REHASH',
            'commits':heads,'files':actual,'npu_executed':False,'sdk_copied':False}
    out=ROOT/'build/evidence_a1/dependency_manifest.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(actual)}))

if __name__=='__main__':
    main()
