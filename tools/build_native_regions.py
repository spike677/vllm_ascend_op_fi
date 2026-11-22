"""Build additive stream/object postwrite exports in an isolated CANN container.

No NPU calls and no deployment qualification. Output directory must be new.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source, sdk, output):
    source, sdk, output = (Path(p).resolve() for p in (source, sdk, output))
    if output == source or source in output.parents or output.exists():
        raise ValueError('FRESH_OUTPUT_OUTSIDE_SOURCE_REQUIRED')
    for name in ('postwrite.cpp', 'postwrite.h', 'test_bound_addresses.cpp',
                 'test_original_snapshot.cpp', 'test_stream_regions.cpp'):
        if not (source / name).is_file():
            raise ValueError('MISSING_BUILD_SOURCE:' + name)
    if not (sdk / 'include/acl/acl.h').is_file():
        raise ValueError('CANN_ACL_HEADERS_REQUIRED')
    output.mkdir(parents=True)
    receipt = dict(schema='demoncat.native_regions.build.v1', status='STARTED',
                   npu_calls=0, qualification_granted=False,
                   started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   machine=platform.machine(), sources={p.name:sha(p) for p in source.iterdir()
                       if p.suffix in ('.cpp', '.h')}, commands=[])
    def run(args):
        result=subprocess.run(args, text=True, capture_output=True, timeout=180)
        receipt['commands'].append(dict(args=args, returncode=result.returncode,
                                       stdout=result.stdout, stderr=result.stderr))
        result.check_returncode()
        return result.stdout
    try:
        receipt['compiler']=run(['g++', '--version']).splitlines()[0]
        receipt['sdk_version_files']={str(p.relative_to(sdk)):dict(sha256=sha(p),text=p.read_text())
            for p in (sdk/'version.info', sdk/'aarch64-linux/ascend_toolkit_install.info') if p.is_file()}
        base=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pthread','-I'+str(sdk/'include')]
        for name in ('test_bound_addresses','test_original_snapshot','test_stream_regions'):
            run(base+[str(source/(name+'.cpp')),'-o',str(output/name)])
            run([str(output/name)])  # ACL substitutes; no device access.
        library=output/'libdcat_postwrite.so'
        run(base+['-fPIC','-shared',str(source/'postwrite.cpp'),'-L'+str(sdk/'lib64'),
                  '-lascendcl','-Wl,-soname,libdcat_postwrite.so','-o',str(library)])
        symbols=run(['nm','-D','--defined-only',str(library)])
        for name in ('dcat_pw_open_stream_v2','dcat_pw_copy_region_v2','dcat_pw_restore','dcat_pw_release'):
            if name not in symbols:raise ValueError('NATIVE_EXPORT_MISSING:'+name)
        receipt['dependencies']=run(['readelf','-d',str(library)])
        receipt['library']=dict(file=library.name,bytes=library.stat().st_size,sha256=sha(library))
        receipt['status']='BUILD_AND_CPU_TESTS_ONLY'
    except BaseException as error:
        receipt.update(status='FAILED',error=repr(error))
        raise
    finally:
        receipt['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--sdk',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.source,args.sdk,args.output),indent=2))
