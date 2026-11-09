"""Complete a fresh private JIT source include tree; stdlib, no device access.

Keeps original a2 stage/receipts immutable. Copies dependency source byte for
byte from the same frozen/patched release used to build the a2 host binary.
Does not alter an executable kernel algorithm, host binary or system SDK.
"""
import argparse
import re
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/fault28-fia-grain-20261109')



def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-generation',required=True)
    p.add_argument('--new-stage',required=True)
    p.add_argument('--approved-manifest-sha256',required=True)
    a=p.parse_args()
    assert re.fullmatch('a[1-9][0-9]{0,2}',a.source_generation)
    assert re.fullmatch('a[1-9][0-9]{0,2}',a.new_stage) and a.new_stage!=a.source_generation
    assert re.fullmatch('[0-9a-f]{64}',a.approved_manifest_sha256)
    OLD=ROOT/('vendor_stage_'+a.source_generation)
    NEW=ROOT/('vendor_stage_'+a.new_stage)
    SOURCE=ROOT/('source_'+a.source_generation)/'ops-transformer-v9.1.0'
    before=sha(OLD/'vendor_manifest.json')
    assert before==a.approved_manifest_sha256
    manifest=json.loads((OLD/'vendor_manifest.json').read_bytes())
    policy=ROOT/('build/evidence_'+a.source_generation)/'policy_source_manifest.json'
    assert sha(policy)==manifest['policy_source_manifest_sha256']
    committed=subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()
    assert committed==manifest['source_commit']
    for item in json.loads(policy.read_bytes())['files']:
        assert sha(SOURCE/item['path'])==item['after_sha256']
    old_vendor=ROOT/manifest['vendor_root']
    for row in manifest['artifacts']:
        assert sha(old_vendor/row['path'])==row['sha256']
    assert not NEW.exists()
    vendor=NEW/'vendors/fault28_fia_graph_transformer'
    shutil.copytree(old_vendor,vendor,symlinks=True)
    impl=vendor/'op_impl/ai_core/tbe/fault28_fia_graph_transformer_impl'
    copied=[]
    # Main installed ascendc/<op>/<file> retains ../../<dependency>/op_kernel.
    # The package omits this sibling include geometry. Restore the exact
    # original release paths, including common nested relative includes.
    for dirname in ('common','incre_flash_attention','prompt_flash_attention','fused_infer_attention_score'):
        source=SOURCE/'attention'/dirname/'op_kernel'
        destination=impl/dirname/'op_kernel'
        assert source.is_dir() and not destination.exists()
        for p in sorted(source.rglob('*')):
            if not p.is_file():continue
            assert not p.is_symlink() and p.stat().st_size<8*1024**2
            q=destination/p.relative_to(source);q.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,q)
            assert sha(p)==sha(q)
            copied.append({'source':p.relative_to(SOURCE).as_posix(),
                           'destination':q.relative_to(vendor).as_posix(),'sha256':sha(q),'bytes':q.stat().st_size})
    # Verify the previously failing include from the actual packaged main.
    main_file=impl/'ascendc/fused_infer_attention_score/fused_infer_attention_score.cpp'
    required=(main_file.parent/'../../incre_flash_attention/op_kernel/incre_flash_attention_arch22.h').resolve()
    assert required.is_relative_to(vendor) and required.is_file()
    artifacts=[]
    for p in sorted(vendor.rglob('*')):
        if p.is_file():
            assert p.resolve().is_relative_to(vendor)
            artifacts.append({'path':p.relative_to(vendor).as_posix(),'bytes':p.stat().st_size,
                              'sha256':sha(p),'symlink':str(p.readlink()) if p.is_symlink() else None})
    manifest.update(vendor_root=vendor.relative_to(ROOT).as_posix(),artifacts=artifacts,
                    generation=a.new_stage+'_stage_from_'+a.source_generation+'_build',status='CPU_INCLUDE_TREE_REPAIRED_NOT_NPU_VALIDATED')
    manifest['staging_repair']={'schema':'fault28.fia_graph.jit_include_repair.v1',
        'original_manifest_file':(OLD/'vendor_manifest.json').relative_to(ROOT).as_posix(),'original_manifest_sha256':before,
        'source_generation':a.source_generation,'source_commit':committed,'helper_sha256':sha(Path(__file__)),
        'copied':copied,'host_binary_changed':False,'original_stage_modified':False,
        'npu_executed':False,'finished_unix':time.time()}
    assert sha(OLD/'vendor_manifest.json')==before
    out=NEW/'vendor_manifest.json'
    with out.open('x') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':manifest['status'],'manifest_sha256':sha(out),
                      'copied_source_files':len(copied),'artifact_count':len(artifacts)}))


if __name__=='__main__':main()
