"""Continue unchanged a3 CPU objects with a verified offline pip environment."""
import argparse
import json
from pathlib import Path
import time
from generation import layout
from resume_support import ROOT,attempt_paths,new_source_freeze,run_cpu,save,sha,verify_new_source_freeze,verify_original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    paths,ev=attempt_paths(args.generation,args.attempt)
    verify_original(paths)
    ev.mkdir(exist_ok=False)
    frozen=new_source_freeze()
    abi={name:sha(paths['evidence']/name) for name in ('ifa_abi_schema.json','ifa_abi_compile.json','ifa_abi_selftest.json')}
    save(ev/'continuation_source_freeze.json',{'files':frozen,'original_source_freeze_sha256':sha(paths['evidence']/'source_freeze.json'),'abi_files':abi})
    save(ev/'continuation_reason.json',{'schema':'fault33.pa.cpu_build_continuation.v1','generation':args.generation,'attempt':args.attempt,
        'reason':'Original pip wheel PEP517 isolation tried to fetch setuptools>=40.8.0; existing private setuptools84.2.1 will be used with isolation disabled and index disabled',
        'base_failure_receipt_sha256':sha(paths['evidence']/'build_completion.json'),'base_log_sha256':sha(paths['evidence']/'build.log'),
        'operator_source_changed':False,'source_policy_existing_files_changed':False,'started_unix':time.time()})
    c=layout(Path('/audit'),args.generation)
    script=f'''set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH={c['python_deps']}:${{PYTHONPATH:-}}
export PATH=/buildtools:${{PATH}}
export PIP_NO_INDEX=1 PIP_NO_BUILD_ISOLATION=0 PIP_DISABLE_PIP_VERSION_CHECK=1
export CMAKE_BUILD_PARALLEL_LEVEL=4
/usr/bin/python3.11 -B /audit/source_policy/packaging_preflight.py --generation {args.generation} --attempt {args.attempt}
cd {c['source']}
cmake --build build --target package --parallel 4
'''
    result=run_cpu(script,ev,f'fault33-pa-cpu-resume-{args.generation}-{args.attempt}-20261105',1800)
    verify_original(paths)
    verify_new_source_freeze(frozen)
    if any(sha(paths['evidence']/p)!=digest for p,digest in abi.items()):
        raise RuntimeError('Successful original ABI evidence changed')
    package=paths['source']/'build_out/cann-ops-transformer-fault33_pa_linux-aarch64.run'
    passed=result['returncode']==0 and result['container_removed'] and package.is_file()
    result.update(status='PASS_CPU_BUILD_CONTINUATION_PACKAGE' if passed else 'FAIL_CPU_BUILD_CONTINUATION',
        generation=args.generation,attempt=args.attempt,original_source_unchanged=True,
        package=str(package),package_present=package.is_file(),package_sha256=sha(package) if package.is_file() else None,
        original_build_completion_preserved=True,original_abi_evidence_preserved=True)
    save(ev/'build_completion.json',result)
    print(json.dumps(result),flush=True)
    return 0 if passed else 1


if __name__=='__main__':
    raise SystemExit(main())
