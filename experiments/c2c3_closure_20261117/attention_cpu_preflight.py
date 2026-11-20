"""CPU-only exact-image ABI/source probe; intended for the root build queue.

Never imports torch, initializes a device, downloads dependencies or uses an old
binary. A compile failure is preserved verbatim and is not a version-wide claim.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--sdk', type=Path, required=True)
    p.add_argument('--source', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise RuntimeError('FRESH_BUILD_OUTPUT_REQUIRED')
    a.out.mkdir(parents=True)
    report = dict(schema='demoncat.attention.cpu_preflight.v1', status='STARTED',
                  actual_npu_calls=0, native_launch_verified=False, loading_allowed=False,
                  machine=platform.machine(), sdk=str(a.sdk.resolve()), commands=[])
    def run(label, args, timeout=180):
        start = time.time()
        with (a.out / (label + '.log')).open('xb') as log:
            rc = subprocess.run([str(x) for x in args], cwd=a.out, stdout=log,
                                stderr=subprocess.STDOUT, timeout=timeout).returncode
        report['commands'].append(dict(label=label, argv=[str(x) for x in args],
            exit_code=rc, seconds=time.time()-start, log_sha256=sha(a.out / (label + '.log'))))
        return rc
    try:
        if list(Path('/dev').glob('davinci*')) or Path('/dev/devmm_svm').exists():
            raise RuntimeError('CPU_PREFLIGHT_MUST_HAVE_NO_NPU_DEVICES')
        if os.environ.get('LD_PRELOAD') or os.environ.get('ASCEND_CUSTOM_OPP_PATH'):
            raise RuntimeError('CPU_PREFLIGHT_REQUIRES_CLEAN_NATIVE_ENV')
        compiler = shutil.which('g++')
        if not compiler:
            raise RuntimeError('GXX_MISSING_IN_EXACT_IMAGE')
        report['compiler'] = dict(path=compiler, sha256=sha(Path(compiler).resolve()))
        report['packages'] = {}
        for name in ('torch', 'torch-npu', 'vllm', 'vllm-ascend'):
            try:
                report['packages'][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                report['packages'][name] = None
        sdk = a.sdk.resolve(strict=True)
        versions = [x for x in sdk.glob('**/version.info') if x.is_file() and len(x.parts)-len(sdk.parts)<5]
        report['sdk_versions'] = [dict(path=str(x), sha256=sha(x), text=x.read_text(errors='replace')) for x in versions]
        kernels = [x for x in sdk.rglob('kernel.h') if x.parent.name == 'runtime'
                   and b'rtKernelLaunchWithHandleV2' in x.read_bytes()]
        unique = {x.resolve() for x in kernels}
        if len(unique) != 1:
            raise RuntimeError('EXACT_RUNTIME_HEADER_UNAVAILABLE_OR_AMBIGUOUS:' + str(sorted(map(str, unique))))
        kernel = unique.pop()
        # Both SDK layouts are discovered from files, not selected by major version.
        include = {kernel.parent.parent, kernel.parent.parent.parent, sdk/'include', sdk/'aarch64-linux/include'}
        for name in ('rt_external_kernel.h', 'ascend_hal.h', 'securec.h', 'tilingdata_base.h', 'prof_common.h'):
            for header in sdk.rglob(name):
                include.add(header.parent)
                include.add(header.parent.parent)
        for base in (sdk/'include', sdk/'aarch64-linux/pkg_inc'):
            if base.is_dir():
                include.update(x for x in base.rglob('*') if x.is_dir() and len(x.parts)-len(base.parts) <= 4)
        include = sorted(x for x in include if x.is_dir())
        flags = ['-I' + str(x) for x in include]
        acl_header = sdk/'include/acl/acl_rt.h'
        require_acl = ('aclrtLaunchKernelWithHostArgs', 'aclrtBinaryLoadFromData', 'aclrtBinaryGetFunctionByEntry')
        acl_host_args = acl_header.is_file() and all(name in acl_header.read_text() for name in require_acl)
        if acl_host_args:
            flags.append('-DDCAT_OBSERVE_ACL_HOST_ARGS=1')
        report['launch_interfaces'] = dict(rtArgsEx=True, aclrt_host_args=acl_host_args)
        report['headers'] = [dict(path=str(x), sha256=sha(x)) for x in
            sorted({kernel, acl_header, *kernel.parent.glob('*.h'), *kernel.parent.parent.glob('rt_external*.h')})]
        source = a.repo/'fault_injection/native_adapters/attention28/launch_observer.cpp'
        report['observer_source'] = dict(path=str(source), sha256=sha(source))
        library = a.out/'libdcat_attention_observer.so'
        rc = run('observer_compile', [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-Wall',
            '-Wextra', *flags, source, '-o', library, '-ldl', '-pthread'])
        report['observer_compiled'] = rc == 0
        if rc == 0:
            report['observer_library'] = dict(path=str(library), sha256=sha(library), bytes=library.stat().st_size)
            run('observer_symbols', ['nm', '-D', '--defined-only', library])
            run('observer_elf', ['readelf', '-h', library])
            # Exercise the exact observer's bounded read under this container's
            # actual seccomp/procfs policy before allocating an NPU case.
            import ctypes
            observer = ctypes.CDLL(str(library))
            observer.dcat_attention_observer_memory_selftest_v1.restype = ctypes.c_int
            observer.dcat_attention_observer_memory_status_v1.restype = ctypes.c_char_p
            memory_rc = observer.dcat_attention_observer_memory_selftest_v1()
            report['observer_memory_selftest'] = dict(exit_code=memory_rc,
                **json.loads(observer.dcat_attention_observer_memory_status_v1().decode()))
            observer.dcat_attention_observer_launch_readback_selftest_v1.restype = ctypes.c_int
            readback_rc = observer.dcat_attention_observer_launch_readback_selftest_v1()
            report['observer_launch_readback_selftest'] = dict(exit_code=readback_rc,
                cpu_fixture_only=True,actual_launch_observed=False,fault_authorized=False)
            if readback_rc != 0:raise RuntimeError('OBSERVER_LIVE_READBACK_CPU_CONTRACT_FAILED')
            if memory_rc != 0:
                raise RuntimeError('OBSERVER_BOUNDED_SELF_MEMORY_READ_FAILED')
        probe = a.out/'abi.cpp'
        probe.write_text('#include <runtime/kernel.h>\n#include <cstddef>\n#include <iostream>\n'
            'int main(){std::cout << "{\\"rtArgsEx_size\\":" << sizeof(rtArgsEx_t)'
            ' << ",\\"args_offset\\":" << offsetof(rtArgsEx_t,args)'
            ' << ",\\"tiling_data_offset_offset\\":" << offsetof(rtArgsEx_t,tilingDataOffset)'
            ' << ",\\"rtTaskCfgInfo_size\\":" << sizeof(rtTaskCfgInfo_t) << "}\\n";}\n')
        abi = a.out/'abi'
        if run('abi_compile', [compiler, '-std=c++17', *flags, probe, '-o', abi]) == 0:
            report['abi_execution_success'] = run('abi_execute', [abi]) == 0
        if a.source:
            # A real translation-unit compile against the exact image SDK. No
            # library or cached object from another image enters this command.
            src = a.source.resolve(strict=True)
            report['official_commit'] = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip()
            target = src/'attention/common/op_host/arch22/fia_tiling_nonquant.cpp'
            report['official_translation_unit'] = dict(path=str(target), sha256=sha(target))
            source_dirs = {src/'attention/common/op_host', src/'attention/common/op_host/arch22'}
            for root in (src/'common', src/'3rdparty', src/'third_party', sdk/'include', sdk/'aarch64-linux/pkg_inc'):
                if root.is_dir():
                    source_dirs.add(root)
                    source_dirs.update(x for x in root.rglob('*') if x.is_dir() and len(x.parts)-len(root.parts) <= 4
                                       and '.git' not in x.parts and 'build' not in x.parts)
            rc_source = run('fia_source_compile', [compiler, '-std=c++17', '-fsyntax-only', *flags,
                *['-I'+str(x) for x in sorted(source_dirs)], target], timeout=300)
            report['fia_translation_unit_compiled'] = rc_source == 0
            report['source_compile_scope'] = 'EXACT_IMAGE_SYNTAX_AND_HEADERS_ONLY_NOT_COMPLETE_VENDOR_BUILD'
            contract = json.loads((a.repo/'fault_injection/native_adapters/attention28/fia/build_contract.json').read_bytes())
            sdk_files = [dict(path=x['path'], sha256=sha(sdk/x['path']), bytes=(sdk/x['path']).stat().st_size)
                         for x in contract['files'] if x['kind'] == 'sdk' and (sdk/x['path']).is_file()]
            manifest = dict(schema='demoncat.attention.sdk_source_probe.v1', source_commit=report['official_commit'],
                sdk=str(sdk), files=sdk_files, fia_translation_unit_compiled=rc_source == 0,
                actual_npu_calls=0, native_launch_verified=False, source_translation_unit_sha256=sha(target),
                required_paths_complete=len(sdk_files) == sum(x['kind'] == 'sdk' for x in contract['files']))
            (a.out/'sdk_interfaces.json').write_text(json.dumps(manifest, indent=2)+'\n')
            report['sdk_interface_manifest'] = dict(path=str(a.out/'sdk_interfaces.json'), sha256=sha(a.out/'sdk_interfaces.json'))
        report['status'] = ('CPU_OBSERVER_ABI_COMPILED_NOT_NPU_QUALIFIED' if report['observer_compiled']
                            else 'CPU_OBSERVER_BUILD_FAILED')
    except BaseException as error:
        report.update(status='CPU_PREFLIGHT_FAILED', error=repr(error), traceback=traceback.format_exc())
    (a.out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'observer_compiled', 'fia_translation_unit_compiled', 'error')}))
    return 0 if report.get('status') == 'CPU_OBSERVER_ABI_COMPILED_NOT_NPU_QUALIFIED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
