"""Build TND/page16 policy from pinned sources in the actual 8.5.1 Worker image."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--interfaces', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    pristine = Path('/home/17-fault-engineering-20261117/ops-transformer-pristine-a1')
    official = a.out / 'official'
    subprocess.run(['git', 'clone', '--shared', '--no-hardlinks', str(pristine), str(official)], check=True)
    head = subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip()
    if head != 'c77c0829f1f0f0b8f882db1d9b051a758b605472':
        raise RuntimeError('PINNED_OFFICIAL_SOURCE_CHANGED')
    patch = a.out / 'patch'
    subprocess.run(['python3', str(a.source / 'fault_injection/native_adapters/attention28/fia/apply_tnd33_policy.py'),
                    '--source', str(official), '--output', str(patch)], check=True)
    output = a.out / 'build'
    output.mkdir()
    sdk_sha = hashlib.sha256(a.interfaces.read_bytes()).hexdigest()
    image = 'quay.io/ascend/vllm-ascend:v0.18.0'
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    mounts = [(a.source / 'fault_injection/native_adapters/attention28/fia', '/driver'),
              (official, '/input-source'), (patch / 'policy_source_manifest.json', '/policy.json'),
              (Path('/home/haoran/experiments/fault28/fault28-fia-graph-20261106/dependencies_a4'), '/input-dependencies'),
              (a.interfaces, '/sdk_interfaces.json'), (Path('/usr/bin/patch'), '/buildtools/patch')]
    cmd = ['docker', 'run', '--rm', '--runtime=runc', '--network=none', '--cpus=4', '--memory=16g',
           '-e', 'ASCEND_VISIBLE_DEVICES=void', '-e', 'PYTHONDONTWRITEBYTECODE=1']
    for src, dst in mounts:
        cmd += ['--mount', 'type=bind,src=' + str(src.resolve()) + ',dst=' + dst + ',readonly']
    cmd += ['--mount', 'type=bind,src=' + str(output) + ',dst=/build', '--entrypoint', 'bash', image_id,
            '-c', 'source /usr/local/Ascend/cann-8.5.1/set_env.sh\nexport PATH=/buildtools:$PATH\n'
            'exec python3 /driver/build_fresh.py --source /input-source --policy-receipt /policy.json '
            '--dependencies /input-dependencies --work /build/run --jobs 4 '
            '--sdk /usr/local/Ascend/cann-8.5.1 --sdk-interface-manifest /sdk_interfaces.json '
            '--sdk-interface-sha256 ' + sdk_sha + ' --opdef-compat ascend910b-cann851']
    receipt = dict(command=cmd, image_id=image_id, source_commit=head,
                   sdk_interfaces_sha256=sdk_sha, actual_npu_calls=0, status='BUILDING')
    record = a.out / 'controller.json'
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    with (a.out / 'process.log').open('xb') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
    receipt.update(exit_code=result.returncode, status='BUILD_COMMAND_COMPLETED' if result.returncode == 0 else 'BUILD_FAILED')
    record.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: receipt[k] for k in ('status', 'exit_code', 'actual_npu_calls')}))
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
