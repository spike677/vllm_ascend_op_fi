"""No-device CANN8.5.1 SDK object compile and cancellation CPU contract.

This is a B1 subcheck, not a production FIA DSO or any NPU acceptance.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source-dir',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--sdk',type=Path,default=Path('/usr/local/Ascend/cann-8.5.1'))
    args=p.parse_args();args.out.mkdir(exist_ok=False)
    receipt=dict(schema='demoncat.tnd28.borrow_sdk_subcheck.v1',
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        npu_calls=0,production_fia_dso_built=False,qualification_granted=False,
        commands=[],source_files={})
    def run(name,argv):
        done=subprocess.run([str(x) for x in argv],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
        log=args.out/(name+'.log');log.write_bytes(done.stdout)
        receipt['commands'].append(dict(name=name,argv=list(map(str,argv)),exit_code=done.returncode,log_sha256=sha(log)))
        if done.returncode:raise RuntimeError(name+'_FAILED')
    try:
        source=args.source_dir.resolve(strict=True);sdk=args.sdk.resolve(strict=True)
        if sdk.name!='cann-8.5.1':raise RuntimeError('EXACT_CANN851_SDK_REQUIRED')
        for name in ('tnd28_execute_borrow.h','test_tnd28_execute_borrow.cpp'):
            receipt['source_files'][name]=sha(source/name)
        receipt['checker_sha256']=sha(__file__)
        unit=args.out/'borrow_sdk.cpp'
        unit.write_text('#include "tnd28_execute_borrow.h"\n'
            'bool dcat_sdk_borrow_cancel(aclOpExecutor* p,void* w,uint64_t n,aclrtStream s) {\n'
            ' tnd28_execute_borrow::ExecuteBorrow borrow(p,w,n,s); return borrow.CancelBeforeParent(); }\n')
        obj=args.out/'borrow_sdk.o';dep=args.out/'borrow_sdk.d'
        run('real_sdk_compile',['g++','-std=c++17','-fPIC','-MD','-MF',dep,
            '-I',sdk/'include/aclnn','-I',sdk/'include','-I',sdk/'pkg_inc','-I',source,
            '-c',unit,'-o',obj])
        names=shlex.split(dep.read_text().replace('\\\n',' ').split(':',1)[1])
        records=[]
        for name in sorted(set(names)):
            file=Path(name).resolve(strict=True)
            if '/dependencies/opbase/' in str(file) or '/include/nnopbase/' in str(file):
                raise RuntimeError('OPBASE_INTERNAL_HEADER_FORBIDDEN')
            records.append(dict(path=str(file),sha256=sha(file),bytes=file.stat().st_size))
        for suffix in ('include/aclnn/acl_meta.h','include/aclnn/aclnn_base.h'):
            if not any(row['path']==str((sdk/suffix).resolve()) for row in records):
                raise RuntimeError('REAL_SDK_HEADER_NOT_SELECTED:'+suffix)
        receipt['dependencies']=records;receipt['dependency_file_sha256']=sha(dep)
        receipt['object_sha256']=sha(obj)
        binary=args.out/'borrow_cpu_contract'
        run('cpu_contract_compile',['g++','-std=c++17','-pthread','-Wall','-Wextra','-Werror',
            '-Wno-misleading-indentation',source/'test_tnd28_execute_borrow.cpp','-o',binary])
        run('cpu_contract',[binary])
        receipt['cpu_executable_sha256']=sha(binary)
        receipt['status']='SDK_OBJECT_AND_CPU_CONTRACT_PASS_PRODUCTION_BUILD_PENDING'
    except Exception as error:
        receipt.update(status='FAILED_STOP',error=repr(error))
    receipt['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ('commands','dependencies')}))
    return 0 if receipt['status'].startswith('SDK_OBJECT_AND_CPU') else 2

if __name__=='__main__':raise SystemExit(main())
