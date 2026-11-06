#!/usr/bin/env python3
"""Rehash task-private build dependencies and immutable SDK image files, CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from generation import layout

ROOT=Path('/audit')
SDK=Path('/usr/local/Ascend/cann-9.1.0')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',required=True)
    args=parser.parse_args()
    paths=layout(ROOT,args.generation)
    expected=json.loads(Path(__file__).with_name('expected_dependencies.json').read_text())
    source=paths['source']
    actual=[]
    for item in expected['files']:
        p=(SDK if item['kind']=='sdk' else source)/item['path']
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        if got!=item['sha256'] or p.stat().st_size!=item['bytes']:
            raise RuntimeError('Dependency changed: '+str(p))
        actual.append({**item,'actual_path':str(p),'matched':True})
    heads={}
    for k,p,exp in [('ops_transformer',source,expected['ops_transformer_commit']),
                    ('opbase',paths['dependencies']/'opbase',expected['opbase_commit']),
                    ('ops_tensor',paths['dependencies']/'ops-tensor',expected['ops_tensor_commit'])]:
        head=subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
        if head!=exp:
            raise RuntimeError('Commit mismatch '+k)
        heads[k]=head
    cann=json.loads((paths['evidence']/'cann_cmake_source.json').read_text())
    cann_root=paths['dependencies']/'cann-cmake'
    cann_head=subprocess.check_output(['git','-C',str(cann_root),'rev-parse','HEAD'],text=True).strip()
    if cann_head!=cann['source_commit']:
        raise RuntimeError('cann-cmake commit mismatch')
    for item in cann['files']:
        p=cann_root/item['path']
        if not p.is_file() or p.is_symlink() or p.stat().st_size!=item['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:
            raise RuntimeError('cann-cmake frozen file changed: '+str(p))
    heads['cann_cmake']=cann_head
    result={'schema':'fault33.pa.build_dependencies.v1','status':'PASS_EXACT_DEPENDENCY_REHASH',
            'commits':heads,'files':actual,'npu_executed':False,'sdk_headers_or_libraries_copied':False,
            'private_sdk_packaging_scripts':['makeself.sh','makeself-header.sh'],
            'cann_cmake_tree_sha256':cann['tree_sha256'],'generation':args.generation}
    out=paths['evidence']/'dependency_manifest.json'
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'path':str(out),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'files':len(actual)}))

if __name__=='__main__':
    main()
