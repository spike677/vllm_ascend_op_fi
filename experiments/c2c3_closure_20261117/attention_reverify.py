"""Offline final-launch verification; preserve every original runtime receipt."""
import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from fault_injection.native_adapters.attention28.launch import bundle_kernel_abi, verify_launch, sha, require


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--host-dir', type=Path, required=True)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--completion-sha256', required=True)
    parser.add_argument('--vendor', type=Path, required=True)
    parser.add_argument('--public-schema', type=Path, required=True)
    parser.add_argument('--schema-sha256', required=True)
    parser.add_argument('--expected-calls', type=int, choices=(2, 4), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(not args.out.exists(), 'FRESH_OFFLINE_SIDECAR_REQUIRED')
    report = dict(schema='demoncat.attention.offline_launch_reverification.v1',
        status='STARTED', original_evidence_modified=False, npu_calls=0,
        verifier_sha256=sha(sys.modules[verify_launch.__module__].__file__),
        completion_sha256=args.completion_sha256, public_schema_sha256=args.schema_sha256,
        public_worker_qualified=False, c2=False, c3=False, launches=[], failures=[],
        statistics='NOT_READ_NO_E5_CLAIM')
    try:
        require(sha(args.completion) == args.completion_sha256
                and sha(args.public_schema) == args.schema_sha256, 'OFFLINE_MANIFEST_OR_SCHEMA_CHANGED')
        completion = json.loads(args.completion.read_bytes())
        vendor = args.vendor.resolve(strict=True)
        files = {}
        for row in completion['artifacts']:
            relative = Path(row['path'])
            require(not relative.is_absolute() and '..' not in relative.parts and row['path'] not in files,
                    'OFFLINE_ARTIFACT_PATH_INVALID')
            path = vendor/relative
            require(path.resolve(strict=True).is_relative_to(vendor) and path.is_file()
                    and sha(path) == row['sha256'] and path.stat().st_size == row['bytes'],
                    'OFFLINE_FRESH_VENDOR_CHANGED')
            files[row['path']] = row
        metadata, abi = bundle_kernel_abi(vendor, files, completion['kernel_binary_sha256'])
        schema = json.loads(args.public_schema.read_bytes())
        report['raw_runtime_sha256'] = sha(args.runtime/'runtime.jsonl')
        report['host_inputs'] = []
        for path in sorted(args.host_dir.glob('host-*.jsonl')):
            report['host_inputs'].append(dict(path=str(path), sha256=sha(path)))
            for line in path.read_text().splitlines():
                host = json.loads(line)
                try:
                    report['launches'].append(verify_launch(args.runtime, host['token'],
                        kernel_sha256=completion['kernel_binary_sha256'], schema=schema, host=host,
                        kernel_metadata=metadata, kernel_abi=abi))
                except BaseException as error:
                    report['failures'].append(dict(call_id=host.get('token'), error=repr(error), traceback=traceback.format_exc()))
        require(len(report['launches']) == args.expected_calls and not report['failures'], 'OFFLINE_LAUNCH_SET_INCOMPLETE')
        require(len({row['call_id'] for row in report['launches']}) == args.expected_calls, 'OFFLINE_DUPLICATE_SCOPE')
        report['status'] = 'FINAL_ARGUMENTS_AND_LAUNCHES_VERIFIED_OFFLINE'
    except BaseException as error:
        report.update(status='OFFLINE_VERIFICATION_INCOMPLETE', error=repr(error), traceback=traceback.format_exc())
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=report['status'], launches=len(report['launches']), npu_calls=0)))
    return 0 if report['status'] == 'FINAL_ARGUMENTS_AND_LAUNCHES_VERIFIED_OFFLINE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
