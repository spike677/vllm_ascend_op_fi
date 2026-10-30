#!/usr/bin/env python3
"""Install a completed package only to this task's fresh private vendor stage."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT=Path('/home/native-execution-strategy-20261030')
IMAGE='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
NAME='nes-fa-cpu-stage-a1-20261030'

def main():
    ev=ROOT/'build/evidence_a1'
    done=json.loads((ev/'build_completion.json').read_text())
    if done['status']!='PASS_CPU_BUILD' or not done['container_removed']:
        raise RuntimeError('CPU build gate not satisfied')
    stage=ROOT/'native_vendor_stage_a1'
    if stage.exists():
        raise RuntimeError('Refuse existing stage')
    source=ROOT/'source/ops-transformer-v9.1.0'
    package=source/'build_out/cann-ops-transformer-native_execution_strategy_linux-aarch64.run'
    if not package.is_file():
        raise RuntimeError('Package missing')
    frozen=json.loads((ev/'source_freeze.json').read_text())['files']
    if any(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=s for p,s in frozen.items()):
        raise RuntimeError('Frozen build sources changed')
    script='''set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash /audit/source/ops-transformer-v9.1.0/build_out/cann-ops-transformer-native_execution_strategy_linux-aarch64.run --install-path=/audit/native_vendor_stage_a1 --quiet
'''
    argv=['docker','run','--rm','--network','none','--cpus','4','--memory','8g','--pids-limit','512','--name',NAME,
          '-v',str(ROOT)+':/audit','--entrypoint','bash',IMAGE,'-lc',script]
    (ev/'stage_start.json').write_text(json.dumps({'argv':argv,'devices':[],'privileged':False,'timeout_seconds':600,'started_unix':time.time()},indent=2)+'\n')
    with (ev/'stage.log').open('w') as log:
        try:
            rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=600).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','stop','--time','5',NAME],check=False)
            rc=124
    removed=subprocess.run(['docker','inspect',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    if rc or not removed:
        raise RuntimeError('Private install failed or container remains')
    manifest=stage/'vendor_manifest.json'
    subprocess.run(['python3',str(ROOT/'native/write_vendor_manifest.py'),'--task-root',str(ROOT),
                    '--vendor-root',str(stage/'vendors/native_execution_strategy_transformer'),'--source',str(source),
                    '--source-manifest',str(ev/'policy_source_manifest.json'),'--package',str(package),'--output',str(manifest)],check=True)
    result={'status':'PASS_CPU_PRIVATE_STAGE','returncode':rc,'container_removed':removed,'npu_executed':False,
            'global_sdk_modified':False,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
            'manifest':str(manifest),'finished_unix':time.time()}
    (ev/'stage_completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':
    main()
