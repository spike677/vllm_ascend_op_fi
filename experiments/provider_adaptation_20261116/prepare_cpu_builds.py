"""Create owned device-free build containers; never attach to foreign jobs."""
import subprocess
from pathlib import Path

ROOT = Path('/home/provider-adaptation-20261116')
IMAGES = {
    '851': 'quay.io/ascend/vllm-ascend:v0.18.0',
    '90': 'cann:9.0.0-910b-torch2.10.0_npu-atk26.4.30',
    '91': 'cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8',
    '92': 'cann:9.2.0-910b-torch2.10.0_npu-atk26.7.8',
}
def main():
    for tag, image in IMAGES.items():
        subprocess.run(['docker', 'run', '-d', '--name', f'provider-build-{tag}-20261116',
            '--label', 'dcat.task=provider-adaptation-20261116', '--runtime=runc',
            '--network', 'none', '--cpus', '8', '--memory', '8g', '--pids-limit', '512',
            '-e', 'ASCEND_VISIBLE_DEVICES=void', '-v', str(ROOT)+':/work',
            '--entrypoint', 'sleep', image, '3600'], check=True)

if __name__=='__main__': main()
