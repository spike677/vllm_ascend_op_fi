"""Device-free, image-specific compiler capability checks for Attention."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    results = []
    for name, image, sdk in (
        ('cann851', 'quay.io/ascend/vllm-ascend:v0.18.0', '/usr/local/Ascend/cann-8.5.1'),
        ('cann910', 'cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8', '/usr/local/Ascend/cann-9.1.0'),
    ):
        folder = a.out / name
        folder.mkdir()
        image_id = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', image], text=True).strip()
        cmd = ['docker', 'run', '--rm', '--runtime=runc', '--network=none',
               '--cpus=4', '--memory=8g', '-e', 'ASCEND_VISIBLE_DEVICES=void',
               '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'LD_PRELOAD=', '-e', 'ASCEND_CUSTOM_OPP_PATH=',
               '--mount', 'type=bind,src=' + str(a.source) + ',dst=/repo,readonly',
               '--mount', 'type=bind,src=' + str(a.official) + ',dst=/official,readonly',
               '--mount', 'type=bind,src=' + str(folder) + ',dst=/build',
               '--entrypoint', 'python3', image_id,
               '/repo/experiments/c2c3_closure_20261117/attention_cpu_preflight.py',
               '--repo', '/repo', '--sdk', sdk, '--source', '/official',
               '--out', '/build/attention-preflight']
        print(json.dumps(dict(status='CPU_PREFLIGHT_START', environment=name)), flush=True)
        with (folder / 'process.log').open('xb') as log:
            r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600)
        results.append(dict(environment=name, image=image, image_id=image_id,
                            command=cmd, exit_code=r.returncode, actual_npu_calls=0))
        (a.out / 'controller.json').write_text(json.dumps(results, indent=2) + '\n')
        print(json.dumps(dict(status='CPU_PREFLIGHT_END', environment=name, exit_code=r.returncode)), flush=True)
    return 0 if all(row['exit_code'] == 0 for row in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
