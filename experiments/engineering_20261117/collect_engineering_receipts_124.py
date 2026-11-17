"""Read this experiment's receipts; emit hashes and status, never statistics payloads."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path('/home/17-fault-engineering-20261117')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(path):
    return dict(path=str(path), sha256=sha(path))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    builds = []
    for name in ('build39-a1', 'build20-qmm-a1', 'build26-fia-a1', 'build26-qmm-a2', 'buildpost-a1'):
        receipts = list((ROOT/name/'cache').glob('*/attempt-*/receipt.json'))
        assert len(receipts) == 1, (name, receipts)
        path = receipts[0]
        data = json.loads(path.read_bytes())
        assert data['status'] == 'BUILD_VERIFIED_NO_NPU_QUALIFICATION'
        assert data['actual_npu_calls'] == 0 and not data['native_qualified']
        builds.append(dict(attempt=name, receipt=ref(path),
            **{k:data[k] for k in ('status','build_key','recipe','mechanism','operator','inputs_sha256','images')},
            inputs=ref(path.parent/'inputs.json'), actual_npu_calls=0, native_qualified=False))
    attention = []
    for name in ('fia-legal-fresh-a1','fia-legal-fresh-a2','fia-legal-fresh-a3','fia-replay-fresh-a1'):
        path = ROOT/name/'run/evidence/completion.json'
        if not path.exists():
            attention.append(dict(attempt=name, status='NO_COMPLETION_RECEIPT'))
            continue
        data = json.loads(path.read_bytes())
        keys=('schema','status','variant','source_commit','patch_receipt_sha256','contract_sha256',
            'build_tools_sha256','dependency_manifest_sha256','excluded_test_fixtures_sha256',
            'host_package_sha256','compiled_keys','kernel_binary_sha256','kernel_metadata_sha256',
            'generated_input_sha256','sdk_matcher_sha256','loading_allowed','qualification_missing','error')
        attention.append(dict(attempt=name, receipt=ref(path),
            **{k:data[k] for k in keys if k in data},
            artifacts=[row for row in data.get('artifacts',[]) if row['path'].endswith(('.so','.o'))],
            npu_executed=data['npu_executed'],actual_launch_verified=data['actual_launch_verified']))
    snapshots = [dict(name=p.parent.name,manifest=ref(p),base_commit=json.loads(p.read_bytes())['base_commit'])
        for p in sorted(ROOT.glob('source-a*/source_manifest.json'))]
    evroot=ROOT/'evidence-tensor15-a4'
    evidence=dict(status='SAVED_RUN_RECONCILED_NOT_NEW_ACCEPTANCE',
        files=[ref(p) for p in sorted(evroot.glob('*.json'))],
        original_run=ref(ROOT/'tensor-a4/tensor15/result.json'))
    report=dict(schema='demoncat.engineering.infrastructure_summary.v1',
        build_manager=builds, attention_builds=attention, source_snapshots=snapshots,
        evidence_manager_example=evidence,
        limitations=['Build success is not NPU qualification.',
          'Prior artifacts and qualification certificates are not inherited by new sources.',
          'New raw statistics, complete tensors, binaries and model weights remain on the validation host.'])
    with a.out.open('x',encoding='utf-8') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(build_manager_receipts=len(builds), attention_receipts=len(attention), output=str(a.out))))


if __name__ == '__main__':
    main()
