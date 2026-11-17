"""Export bounded build/test summaries, never vendor libraries or raw tensors."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/worker-positive-20261117')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = dict(schema='demoncat.worker_positive.build_summary.v1',
                  scope='CPU_BUILD_AND_TEST_ONLY_NOT_NPU_ACCEPTANCE', builds=[], packages=[])
    for label in ('build-a4', 'build-a7', 'build-a8'):
        row = dict(label=label, mechanisms={})
        for mechanism in ('build20', 'build39'):
            folder = ROOT / label / mechanism
            manifest = folder / 'build_manifest.json'
            if not manifest.exists():
                continue
            value = json.loads(manifest.read_text())
            row['mechanisms'][mechanism] = dict(manifest_sha256=sha(manifest),
                status=value.get('status'), profile_sha256=value.get('profile_sha256'),
                actual_npu_calls=value.get('actual_npu_calls'),
                artifact_sha256=value.get('files'),
                cpu_receipts={p.name: dict(sha256=sha(p), result=p.read_text()[:100]
                    if p.suffix == '.txt' else json.loads(p.read_text()).get('status'))
                    for p in folder.iterdir() if p.is_file()
                    and ('receipt' in p.name or p.name == 'serving_inputs_cpu.json')
                    and p.suffix in ('.txt', '.json')})
        result['builds'].append(row)
    for label in ('packages-a4', 'packages-a7', 'packages-a9', 'packages-a11'):
        folder = ROOT / label
        receipt = folder / 'receipt.json'
        value = json.loads(receipt.read_text())
        result['packages'].append(dict(label=label, receipt_sha256=sha(receipt),
            exit_code=value['exit_code'], actual_npu_calls=value['actual_npu_calls'],
            wheels={p.name: sha(p) for p in (folder / 'wheels').glob('*.whl')}))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
