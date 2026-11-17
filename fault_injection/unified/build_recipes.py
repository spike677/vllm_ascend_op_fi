"""Explicit paths and ABI/version gates for existing FA build implementation."""
from pathlib import Path
import subprocess
import sys


CANN_RULES={
    '8.5.1':dict(native_recipe='fault_injection.native_build',fa_recipe=None,abi='exact deployment provider manifest required'),
    '9.1.0':dict(native_recipe='fault_injection.native_build',fa_recipe='fault_injection.native_adapters.attention28.fa.build_fresh',
                 abi='fresh FA POD/schema and registration identity required'),
}


def plan_fa_build(*,cann,sdk,source,policy_receipt,dependencies,work,jobs=1):
    version='9.1.0' if cann=='9.1' else cann
    if version not in CANN_RULES:raise ValueError('UNREGISTERED_CANN_VERSION')
    if type(jobs) is not int or not 1<=jobs<=4:raise ValueError('JOBS_OUT_OF_RANGE')
    rule=CANN_RULES[version]
    paths={name:Path(value).resolve() for name,value in dict(sdk=sdk,source=source,policy_receipt=policy_receipt,
            dependencies=dependencies,work=work).items()}
    missing=[name for name in ('sdk','source','policy_receipt','dependencies') if not paths[name].exists()]
    # Existing fresh builder verifies vendor identity and exact artifacts again.
    recipe=rule['fa_recipe']
    command=[sys.executable,'-m',recipe] if recipe else []
    if recipe:
        for name,path in paths.items():command.extend(['--'+name.replace('_','-'),str(path)])
        command.extend(['--jobs',str(jobs)])
    return dict(schema='demoncat.build_plan.v1',cann=version,abi=rule['abi'],command=command,
                status='NOT_IMPLEMENTED' if recipe is None else 'DEPENDENCY_MISSING' if missing else 'READY_FOR_BUILDER_VALIDATION',
                missing=missing,compiled=False,npu_qualified=False)


def execute_fa_build(**kwargs):
    plan=plan_fa_build(**kwargs)
    if plan['status']!='READY_FOR_BUILDER_VALIDATION':return plan
    result=subprocess.run(plan['command'],check=False,capture_output=True,text=True)
    return dict(plan,status='BUILD_SUCCEEDED' if result.returncode==0 else 'BUILD_FAILED',
                compiled=result.returncode==0,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
