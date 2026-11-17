"""Root-owned device-free FA rebuild controller. Never run by an NPU subagent."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='New task repository snapshot')
    p.add_argument('--out', type=Path, required=True, help='New attempt directory')
    a = p.parse_args();a.source = a.source.resolve(strict=True);a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    driver = a.source / 'fault_injection/native_adapters/attention28/fa/build_fresh.py'
    spec = importlib.util.spec_from_file_location('fa_fresh_precheck', driver)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    imported = module.verify_imported()
    official = a.out / 'official'
    pristine = Path('/home/17-fault-engineering-20261117/ops-transformer-pristine-a1')
    module.common.local_clone(pristine, official, module.COMMIT)
    patch = a.out / 'patch'
    subprocess.run(['python3', '-B', str(a.source / 'native_execution_strategy/native/apply_fa_bridge.py'),
                    '--source', str(official), '--output', str(patch)], check=True)
    policy = patch / 'policy_source_manifest.json'
    module.validate_receipt(json.loads(policy.read_bytes()))
    output = a.out / 'build';output.mkdir()
    image = 'cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8'
    image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
    mounts = [(a.source, '/driver'), (official, '/input-source'), (policy, '/policy.json'),
        (Path('/home/haoran/experiments/fault28/fault28-fia-graph-20261106/dependencies_a4'), '/input-dependencies'),
        (Path('/usr/bin/patch'), '/buildtools/patch')]
    name = 'dcat-fa-cpu-' + hashlib.sha256(str(a.out).encode()).hexdigest()[:16]
    cmd = ['docker', 'run', '--rm', '--runtime=runc', '--network=none', '--cpus=4', '--memory=16g',
           '--pids-limit=2048', '--name', name, '-e', 'ASCEND_VISIBLE_DEVICES=void',
           '-e', 'PYTHONDONTWRITEBYTECODE=1']
    for source, target in mounts:
        cmd += ['--mount', 'type=bind,src=' + str(source.resolve(strict=True)) + ',dst=' + target + ',readonly']
    cmd += ['--mount', 'type=bind,src=' + str(output) + ',dst=/build', '--entrypoint', 'bash', image_id,
        '-c', 'source /usr/local/Ascend/cann-9.1.0/set_env.sh\nexport PATH=/buildtools:$PATH\n'
        'exec python3 -B /driver/fault_injection/native_adapters/attention28/fa/build_fresh.py '
        '--source /input-source --policy-receipt /policy.json --dependencies /input-dependencies '
        '--work /build/run --jobs 4']
    receipt = dict(schema='demoncat.fa.cpu_controller.v1', status='BUILDING', started_unix=time.time(),
        command=cmd, image_id=image_id, source_commit=module.COMMIT, imported_sources=imported,
        policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(), devices=[], network='none',
        actual_npu_calls=0, historical_qualification_inherited=False, production_loading_allowed=False)
    record = a.out / 'controller.json';save(record, receipt)
    try:
        with (a.out / 'process.log').open('xb') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=7200)
        receipt.update(exit_code=result.returncode,
            status='BUILD_COMMAND_COMPLETED' if result.returncode == 0 else 'BUILD_FAILED')
    except subprocess.TimeoutExpired:
        stopped = subprocess.run(['docker', 'stop', '--time', '5', name], capture_output=True, text=True)
        receipt.update(exit_code=124, status='CPU_BUILD_TIMEOUT', own_container_stop_returncode=stopped.returncode)
    except BaseException as error:
        receipt.update(exit_code=2, status='CONTROLLER_ERROR', error=repr(error))
        raise
    finally:
        receipt['finished_unix'] = time.time();save(record, receipt)
    print(json.dumps({k: receipt[k] for k in ('status', 'exit_code', 'actual_npu_calls')}))
    return receipt['exit_code']


if __name__ == '__main__':raise SystemExit(main())
