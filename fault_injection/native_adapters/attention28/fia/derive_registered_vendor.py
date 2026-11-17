"""Derive a new vendor by adding only its missing official binary summary index."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from binary_registration import register_index,require,sha

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-completion',type=Path,required=True)
    p.add_argument('--parent-sha256',required=True)
    p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[4]
    spec=importlib.util.spec_from_file_location('dcat_fia_experimental_verifier',
        repo/'experiments/engineering_20261117/fresh_fia_qualification.py')
    verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
    verified=verifier.verify_build(a.parent_completion,a.parent_sha256,a.vendor)
    parent=json.loads(a.parent_completion.read_bytes());source_vendor=Path(verified['vendor_root'])
    out=a.out.resolve()
    require(not out.exists() and not out.is_relative_to(source_vendor)
        and not source_vendor.is_relative_to(out),'FRESH_DISJOINT_DERIVATION_REQUIRED')
    require(not out.is_relative_to(a.source.resolve()),'OUTPUT_MUST_NOT_MUTATE_SOURCE_TREE')
    out.mkdir(parents=True)
    vendor=out/'vendors'/source_vendor.name;vendor.parent.mkdir()
    shutil.copytree(source_vendor,vendor,symlinks=True)
    verifier.verify_build(a.parent_completion,a.parent_sha256,vendor)
    registration=register_index(a.source,vendor,verified['variant'])
    for row in parent['artifacts']:
        path=vendor/row['path']
        require(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'PARENT_ARTIFACT_CHANGED')
    # Recheck the immutable source after copying/generation; this never edits it.
    verifier.verify_build(a.parent_completion,a.parent_sha256,source_vendor)
    files=[dict(path=p.relative_to(vendor).as_posix(),bytes=p.stat().st_size,sha256=sha(p))
           for p in sorted(vendor.rglob('*')) if p.is_file()]
    require({p['path'] for p in files}=={p['path'] for p in parent['artifacts']}|{registration['generated_index']['path']},
            'DERIVATION_ADDED_UNEXPECTED_ARTIFACT')
    receipt=dict(schema='demoncat.fia.registration_derivation.v1',
        parent_completion=dict(path=str(a.parent_completion.resolve()),sha256=a.parent_sha256),
        parent_vendor=str(source_vendor),derived_vendor=str(vendor),
        tool_sha256=sha(Path(__file__)),verifier_sha256=sha(Path(verifier.__file__)),
        registration=registration,parent_artifacts_unchanged=True,source_parent_rechecked=True,
        added_files=[registration['generated_index']['path']],host_kernel_bytes_changed=False,
        old_vendor_or_binary_copied=False,npu_executed=False,loading_allowed=False)
    receipt_path=out/'registration_derivation.json'
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    derived=copy.deepcopy(parent)
    derived.update(vendor_root=str(vendor),artifacts=files,
        parent_completion=receipt['parent_completion'],registration_derivation_sha256=sha(receipt_path),
        registration=registration,derivation_tool_sha256=sha(Path(__file__)),
        derivation_scope='ONLY_ADD_OFFICIAL_BINARY_SUMMARY_INDEX',
        status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',loading_allowed=False,npu_executed=False,actual_launch_verified=False)
    completion=out/'completion.json';completion.write_text(json.dumps(derived,indent=2)+'\n')
    verifier.verify_build(completion,sha(completion),vendor)
    print(json.dumps(dict(status=derived['status'],completion=str(completion),
        completion_sha256=sha(completion),vendor=str(vendor),host_kernel_bytes_changed=False)))
if __name__=='__main__':main()

