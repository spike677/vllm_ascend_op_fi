"""Summarize all retained controller attempts, including unsuccessful ones."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/worker-positive-20261117')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    attempts = []
    for path in sorted(ROOT.glob('*/controller.json')):
        data = json.loads(path.read_text())
        cases = []
        for case in data['cases']:
            result_path = path.parent / case['id'] / 'result.json'
            value = json.loads(result_path.read_text()) if result_path.exists() else {}
            cases.append(dict(id=case['id'], exit_code=case.get('exit_code'),
                result_status=value.get('status'), result_sha256=sha(result_path)
                    if result_path.exists() else None,
                idle_gate_passed=case.get('gate', {}).get('idle'),
                image_id=case.get('image_id'), timeout=case.get('timeout', False),
                input_script_sha256=[row['sha256'] for row in case.get('frozen_script_inputs', [])]))
        attempts.append(dict(attempt=path.parent.name, status=data['status'], cards=data['cards'],
            controller_receipt_sha256=sha(path), controller_sha256=data.get('controller_sha256'),
            case_manifest_sha256=data.get('case_manifest_sha256'), cases=cases))
    args.out.write_text(json.dumps(dict(schema='demoncat.worker_positive.attempt_summary.v1',
        source_root=str(ROOT), raw_receipts_retained=True, attempts=attempts), indent=2) + '\n')


if __name__ == '__main__':
    main()
