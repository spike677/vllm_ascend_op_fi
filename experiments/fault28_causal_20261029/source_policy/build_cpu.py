#!/usr/bin/env python3
"""Bounded task-private CPU build; never passes NPU devices or installs globally."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path('/home/fault28-causal-20261029')
OLD = Path('/home/fault28-native-tiling-20261029')
IMAGE = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
NAME = 'fault28-causal-cpu-build-a1-20261029'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    source = ROOT/'source/ops-transformer-v9.1.0'
    evidence = ROOT/'build/evidence_a1'
    if source.exists():
        raise RuntimeError('Refuse existing source/build generation')
    source.parent.mkdir(exist_ok=True)
    subprocess.run(['git','clone','--local',str(OLD/'source_matching/ops-transformer-v9.1.0'),str(source)],check=True)
    subprocess.run(['git','-C',str(source),'checkout','--detach',COMMIT],check=True)
    subprocess.run(['python3',str(ROOT/'source_policy/apply_policy.py'),'--source',str(source),'--output',str(evidence)],check=True)
    # New dependency copies are independent; official CMake may postprocess them.
    deps = ROOT/'dependencies'
    deps.mkdir(exist_ok=False)
    for name in ['opbase','ops-tensor']:
        shutil.copytree(OLD/'source_matching/ops-transformer-fa-v9.1.0/third_party'/name,deps/name,symlinks=True)
    shutil.copytree(OLD/'build_off/python_deps',ROOT/'build/python_deps',symlinks=True)
    # Reuse pinned archives/header as new task-private files, not writable old mounts.
    oldtp=OLD/'source_matching/ops-transformer-fa-a2-v9.1.0/third_party'
    for name in ['pkg','protobuf','json']:
        shutil.copytree(oldtp/name,source/'third_party'/name,symlinks=True,dirs_exist_ok=True)
    for name in ['opbase','ops-tensor']:
        target=source/'third_party'/name
        if target.exists():
            raise RuntimeError('Unexpected checked-in dependency target')
        target.symlink_to('/deps/'+name, target_is_directory=True)
    frozen={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'source_policy').iterdir() if p.is_file()}
    manifest=json.loads((evidence/'policy_source_manifest.json').read_text())
    frozen.update({str((source/item['path']).relative_to(ROOT)):item['after_sha256'] for item in manifest['files']})
    (evidence/'source_freeze.json').write_text(json.dumps({'frozen_unix':time.time(),'files':frozen},indent=2)+'\n')
    script='''set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH=/audit/build/python_deps:${PYTHONPATH:-}
export PATH=/buildtools:${PATH}
export CMAKE_BUILD_PARALLEL_LEVEL=4
cd /audit/source/ops-transformer-v9.1.0
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -I/usr/local/Ascend/cann-9.1.0/include -x c++ -fsyntax-only attention/flash_attention_score/op_host/fault28_causal_policy.h
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -I/audit/source_policy -I/usr/local/Ascend/cann-9.1.0/include /audit/source_policy/scope_cpu_check.cpp -L/usr/local/Ascend/cann-9.1.0/lib64 -lregister -o /audit/build/evidence_a1/scope_cpu_check
/audit/build/evidence_a1/scope_cpu_check > /audit/build/evidence_a1/scope_cpu_check.json
g++ -std=c++17 -Iattention/flash_attention_score/op_kernel/arch22 -I/usr/local/Ascend/cann-9.1.0/aarch64-linux/ascendc/include/highlevel_api -I/usr/local/Ascend/cann-9.1.0/include /audit/source_policy/fa_pod_decoder.cpp -o /audit/build/evidence_a1/fa_pod_decoder
/audit/build/evidence_a1/fa_pod_decoder --schema > /audit/build/evidence_a1/fa_pod_schema.json
/audit/build/evidence_a1/fa_pod_decoder --self-test > /audit/build/evidence_a1/fa_pod_selftest.json
cmake -S . -B build -DENABLE_BUILD_PKG=ON -DBUILD_OPEN_PROJECT=ON -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=OFF -DENABLE_AICPU=OFF -DENABLE_AICPU_KERNEL=OFF -DENABLE_TEST=OFF -DENABLE_CCACHE=ON -DBUILD_OPS_RTY_KERNEL=OFF -DVENDOR_NAME=fault28_causal -DASCEND_OP_NAME=flash_attention_score -DASCEND_COMPUTE_UNIT=ascend910b -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 -DPython3_EXECUTABLE=/usr/bin/python3.11 -DHI_PYTHON=/usr/bin/python3.11 -DASCEND_PYTHON_EXECUTABLE=/usr/bin/python3.11 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/audit/source/ops-transformer-v9.1.0/build_out
cmake --build build --target package --parallel 4
'''
    argv=['docker','run','--rm','--cpus','4','--memory','16g','--pids-limit','512','--name',NAME,
          '-v',str(ROOT)+':/audit','-v',str(deps)+':/deps','-v','/usr/bin/patch:/buildtools/patch:ro',
          '--entrypoint','bash',IMAGE,'-lc',script]
    start={'schema':'fault28.causal.cpu_build.v1','argv':argv,'devices':[],'privileged':False,'timeout_seconds':1800,
           'started_unix':time.time(),'source_freeze_sha256':sha(evidence/'source_freeze.json')}
    (evidence/'build_start.json').write_text(json.dumps(start,indent=2)+'\n')
    with (evidence/'build.log').open('w') as log:
        try:
            proc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
            rc=proc.returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','stop','--time','5',NAME],check=False)
            rc=124
    changed=[p for p,s in frozen.items() if sha(ROOT/p)!=s]
    inspect=subprocess.run(['docker','inspect',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    done={'status':'PASS_CPU_BUILD' if rc==0 and not changed and inspect.returncode!=0 else 'FAIL_CPU_BUILD',
          'returncode':rc,'finished_unix':time.time(),'frozen_sources_changed':changed,'container_removed':inspect.returncode!=0,
          'npu_executed':False,'log_sha256':sha(evidence/'build.log')}
    (evidence/'build_completion.json').write_text(json.dumps(done,indent=2)+'\n')
    print(json.dumps(done),flush=True)
    return 0 if done['status']=='PASS_CPU_BUILD' else 1

if __name__=='__main__':
    raise SystemExit(main())
