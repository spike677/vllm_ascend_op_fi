#!/usr/bin/env python3
"""Fresh task-private FIA host/JIT-source package; no device compilation here."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from generation import layout
import offline_dependencies

ROOT = Path('/home/fault28-fia-grain-20261109')
POLICY = Path(__file__).resolve().parent
OLD = Path('/home/fault28-native-tiling-20261029')
IMAGE = 'sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
CANN_CMAKE_COMMIT = '7e5691ce110367662017d9d94114845a7d6a8d8a'
CANN_CMAKE_SOURCE = Path('/home/fault28-causal-20261029/source/ops-transformer-v9.1.0/build/_deps/cann-cmake-src')
EIGEN_SOURCE = Path('/home/fault28-causal-20261029/source/ops-transformer-v9.1.0/build/third_party/src/external_eigen')
MAKESELF_SDK_SOURCE = '/usr/local/Ascend/cann-9.1.0/tools/op_project_templates/op_project_tmpl/cmake/util/makeself'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git_head(source):
    return subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()


def clone_pinned(source, destination, commit):
    if git_head(source) != commit:
        raise RuntimeError(f'Offline source HEAD mismatch: {source}')
    tracked_diff = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True)
    if tracked_diff.strip():
        raise RuntimeError(f'Offline source has tracked changes: {source}')
    subprocess.run(['git', 'clone', '--local', '--no-hardlinks', str(source), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
    if git_head(destination) != commit:
        raise RuntimeError('New private clone HEAD mismatch')


def content_files(root):
    files=[]
    for p in sorted(root.rglob('*')):
        if '.git' in p.relative_to(root).parts:
            continue
        if p.is_symlink():
            raise RuntimeError('Unexpected symlink in frozen source content')
        if p.is_file():
            files.append({'path': p.relative_to(root).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    if not files:
        raise RuntimeError('Empty frozen source directory')
    return files


def prepare_makeself(source,evidence,generation,local_source=None):
    destination=source/'third_party/makeself'
    if local_source is not None:
        shutil.copytree(local_source,destination,symlinks=True,dirs_exist_ok=True)
        return {'origin':'provided_local_source','source_path':str(local_source),'npu_executed':False}
    name=f'fault28-fia-cpu-tools-{generation}-20261106'
    container_destination=Path('/audit')/destination.relative_to(ROOT)
    script=f'''set -euo pipefail
test -f {MAKESELF_SDK_SOURCE}/makeself.sh
test -f {MAKESELF_SDK_SOURCE}/makeself-header.sh
test ! -e {container_destination}/makeself.sh
test ! -e {container_destination}/makeself-header.sh
mkdir -p {container_destination}
cp -p {MAKESELF_SDK_SOURCE}/makeself.sh {container_destination}/makeself.sh
cp -p {MAKESELF_SDK_SOURCE}/makeself-header.sh {container_destination}/makeself-header.sh
cmp {MAKESELF_SDK_SOURCE}/makeself.sh {container_destination}/makeself.sh
cmp {MAKESELF_SDK_SOURCE}/makeself-header.sh {container_destination}/makeself-header.sh
chmod 700 {container_destination}/makeself.sh {container_destination}/makeself-header.sh
sha256sum {MAKESELF_SDK_SOURCE}/makeself.sh {MAKESELF_SDK_SOURCE}/makeself-header.sh {container_destination}/makeself.sh {container_destination}/makeself-header.sh
'''
    argv=['docker','run','--rm','--network','none','--read-only','--cpus','1','--memory','512m',
        '--pids-limit','64','--name',name,'-v',str(ROOT)+':/audit','--entrypoint','bash',IMAGE,'-lc',script]
    with (evidence/'prepare_tools.log').open('w') as log:
        try:
            rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker','stop','--time','5',name],check=False)
            rc=124
    removed=subprocess.run(['docker','inspect',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
    receipt={'schema':'fault28.fia_graph.private_sdk_packager.v1','argv':argv,'origin':'immutable_image_sdk_packaging_scripts',
        'image':IMAGE,'source_path':MAKESELF_SDK_SOURCE,'returncode':rc,'container_removed':removed,
        'timeout_seconds':60,'log_sha256':sha(evidence/'prepare_tools.log'),'network_enabled':False,
        'npu_executed':False,'sdk_headers_or_libraries_copied':False,
        'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'mode':oct(p.stat().st_mode&0o777)} for p in destination.glob('makeself*.sh')]}
    (evidence/'prepare_tools_completion.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if rc or not removed:
        raise RuntimeError('Offline SDK packager preparation failed')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',required=True)
    parser.add_argument('--cann-cmake-source',type=Path,default=CANN_CMAKE_SOURCE)
    parser.add_argument('--eigen-source',type=Path,default=EIGEN_SOURCE)
    parser.add_argument('--makeself-source',type=Path)
    args=parser.parse_args()
    paths=layout(ROOT,args.generation)
    container_paths=layout(Path('/audit'),args.generation)
    NAME=f'fault28-fia-cpu-build-{args.generation}-20261106'
    source = paths['source']
    evidence = paths['evidence']
    deps = paths['dependencies']
    for p in [source, evidence, deps, paths['python_deps']]:
        if p.exists():
            raise RuntimeError(f'Refuse existing build generation: {p}')
    source.parent.mkdir(parents=True, exist_ok=True)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    clone_pinned(OLD/'source_matching/ops-transformer-v9.1.0',source,COMMIT)
    if not POLICY.is_relative_to(ROOT):
        raise RuntimeError('Policy directory must be within the task root')
    subprocess.run(['python3', '-B', str(POLICY/'apply_policy.py'), '--source', str(source), '--output', str(evidence)], check=True)
    # Independent new copies: official CMake may postprocess these dependencies.
    deps.mkdir(exist_ok=False)
    # Clone only committed content from the precise existing cache. Untracked
    # old cache material and Git object hardlinks are not carried into a2.
    clone_pinned(args.cann_cmake_source,deps/'cann-cmake',CANN_CMAKE_COMMIT)
    cmake_files=content_files(deps/'cann-cmake')
    cmake_tree_sha=hashlib.sha256(json.dumps(cmake_files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    cmake_receipt={'schema':'fault28.fia_graph.cann_cmake_source.v1','source_path':str(args.cann_cmake_source),
        'source_commit':CANN_CMAKE_COMMIT,'declared_tag':'master-026','private_path':str(deps/'cann-cmake'),
        'fetchcontent_override':'FETCHCONTENT_SOURCE_DIR_CANN-CMAKE=/deps/cann-cmake',
        'tree_sha256':cmake_tree_sha,'files':cmake_files,'git_clone_no_hardlinks':True,
        'untracked_old_cache_copied':False}
    (evidence/'cann_cmake_source.json').write_text(json.dumps(cmake_receipt,indent=2)+'\n')
    for name in ['opbase', 'ops-tensor']:
        shutil.copytree(OLD/'source_matching/ops-transformer-fa-v9.1.0/third_party'/name, deps/name, symlinks=True)
    shutil.copytree(OLD/'build_off/python_deps', paths['python_deps'], symlinks=True)
    oldtp = OLD/'source_matching/ops-transformer-fa-a2-v9.1.0/third_party'
    for name in ['pkg', 'protobuf', 'json']:
        shutil.copytree(oldtp/name, source/'third_party'/name, symlinks=True, dirs_exist_ok=True)
        shutil.copytree(oldtp/name, deps/name, symlinks=True)
    # Exact native ExternalProject cache locations. Abseil does not inspect pkg
    # when selecting its URL, so give it the directory explicitly supported by
    # the pinned module. Protobuf consumes its own local archive directory.
    (deps/'abseil-cpp').mkdir(exist_ok=False)
    shutil.copy2(deps/'pkg/abseil-cpp-20230802.1.tar.gz',deps/'abseil-cpp/abseil-cpp-20230802.1.tar.gz')
    shutil.copytree(args.eigen_source,deps/'eigen',symlinks=True)
    packager_receipt=prepare_makeself(source,evidence,args.generation,args.makeself_source)
    eigen_files=content_files(deps/'eigen')
    makeself_files=content_files(source/'third_party/makeself')
    # The sibling is the first native ops-tensor branch; unlike its generic
    # CANN_3RD_LIB_PATH branch it does not issue a git fetch.
    sibling=source.parent/'ops-tensor'
    sibling.symlink_to(os.path.relpath(deps/'ops-tensor',sibling.parent),target_is_directory=True)
    for name in ['opbase', 'ops-tensor']:
        target = source/'third_party'/name
        if target.exists() or target.is_symlink():
            raise RuntimeError('Unexpected dependency target')
        target.symlink_to('/deps/'+name, target_is_directory=True)
    offline=offline_dependencies.validate(source,deps,evidence)
    (evidence/'offline_source_copies.json').write_text(json.dumps({
        'schema':'fault28.fia_graph.offline_source_copies.v1','eigen_origin':str(args.eigen_source),
        'eigen_files':eigen_files,'makeself_origin':packager_receipt,'makeself_files':makeself_files,
        'enabled_preflight_status':offline['status'],'ops_tensor_sibling':str(sibling),
        'ops_tensor_sibling_target':os.readlink(sibling),'copied_built_third_party_libraries':False},indent=2)+'\n')
    frozen = {str(p.relative_to(ROOT)): sha(p) for p in POLICY.iterdir() if p.is_file()}
    manifest = json.loads((evidence/'policy_source_manifest.json').read_text())
    frozen.update({str((source/item['path']).relative_to(ROOT)): item['after_sha256'] for item in manifest['files']})
    frozen[str((evidence/'fia_schema_projection.h').relative_to(ROOT))]=sha(evidence/'fia_schema_projection.h')
    frozen[str((evidence/'typed_fields.json').relative_to(ROOT))]=sha(evidence/'typed_fields.json')
    frozen.update({str((deps/'cann-cmake'/item['path']).relative_to(ROOT)):item['sha256'] for item in cmake_files})
    frozen.update({str((deps/'eigen'/item['path']).relative_to(ROOT)):item['sha256'] for item in eigen_files})
    frozen.update({str((source/'third_party/makeself'/item['path']).relative_to(ROOT)):item['sha256'] for item in makeself_files})
    for p in [deps/'json/single_include/nlohmann/json.hpp',deps/'protobuf/protobuf-25.1.tar.gz',deps/'abseil-cpp/abseil-cpp-20230802.1.tar.gz']:
        frozen[str(p.relative_to(ROOT))]=sha(p)
    (evidence/'source_freeze.json').write_text(json.dumps({'frozen_unix': time.time(), 'generation':args.generation,'files': frozen}, indent=2)+'\n')
    script = '''set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=@PYDEPS@:${PYTHONPATH:-}
export PATH=/buildtools:${PATH}
export PIP_NO_INDEX=1 PIP_NO_BUILD_ISOLATION=0 PIP_DISABLE_PIP_VERSION_CHECK=1
export CMAKE_BUILD_PARALLEL_LEVEL=4
cd @SOURCE@
python3 -B @POLICY@/audit_dependencies.py --generation @GEN@
/usr/bin/python3.11 -B @POLICY@/check_pip_build_env.py --generation @GEN@
# Full host headers are compiled with their actual CMake include flags below.
g++ -std=c++17 -I@POLICY@ @POLICY@/selector_cpu_check.cpp -o @EVIDENCE@/selector_cpu_check
@EVIDENCE@/selector_cpu_check > @EVIDENCE@/selector_cpu_check.json
cmake -S . -B build -DCANN_3RD_LIB_PATH:PATH=/deps -DCANN_3RD_PKG_PATH:PATH=/deps/pkg -DFETCHCONTENT_SOURCE_DIR_CANN-CMAKE:PATH=/deps/cann-cmake -DFETCHCONTENT_SOURCE_DIR_OPBASE:PATH=/deps/opbase -DFETCHCONTENT_SOURCE_DIR_OPS-TENSOR:PATH=/deps/ops-tensor -DCMAKE_PROJECT_INCLUDE:FILEPATH=@POLICY@/offline_json.cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DENABLE_BUILD_PKG=ON -DBUILD_OPEN_PROJECT=ON -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=OFF -DENABLE_AICPU=OFF -DENABLE_AICPU_KERNEL=OFF -DENABLE_TEST=OFF -DENABLE_CCACHE=ON -DBUILD_OPS_RTY_KERNEL=OFF -DVENDOR_NAME=fault28_fia_graph -DASCEND_OP_NAME=fused_infer_attention_score -DASCEND_COMPUTE_UNIT=ascend910b -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 -DPython3_EXECUTABLE=/usr/bin/python3.11 -DHI_PYTHON=/usr/bin/python3.11 -DASCEND_PYTHON_EXECUTABLE=/usr/bin/python3.11 -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=@SOURCE@/build_out
python3 -B @POLICY@/verify_offline_configure.py --generation @GEN@
python3 -B @POLICY@/compile_abi_decoder.py --generation @GEN@
cmake --build build --target package --parallel 4
'''
    for token,value in {'@SOURCE@':str(container_paths['source']),'@EVIDENCE@':str(container_paths['evidence']),
                        '@PYDEPS@':str(container_paths['python_deps']),'@GEN@':args.generation,
                        '@POLICY@':str(Path('/audit')/POLICY.relative_to(ROOT))}.items():
        script=script.replace(token,value)
    argv = ['docker', 'run', '--rm', '--network', 'none', '--cpus', '4', '--memory', '16g', '--pids-limit', '512', '--name', NAME,
        '-v', str(ROOT)+':/audit', '-v', str(deps)+':/deps', '-v', '/usr/bin/patch:/buildtools/patch:ro',
        '--entrypoint', 'bash', IMAGE, '-lc', script]
    start = {'schema': 'fault28.fia_graph.cpu_build.v1', 'argv': argv, 'devices': [], 'privileged': False,
        'timeout_seconds': 1800, 'started_unix': time.time(), 'source_freeze_sha256': sha(evidence/'source_freeze.json'),
        'source_commit': COMMIT, 'old_materials_read_only': True, 'kernel_build_enabled': False,
        'modified_jit_kernel_source_packaged': True, 'new_kernel_device_compilation_pending': True,
        'generation':args.generation,'cann_cmake_source_receipt_sha256':sha(evidence/'cann_cmake_source.json')}
    (evidence/'build_start.json').write_text(json.dumps(start, indent=2)+'\n')
    with (evidence/'build.log').open('w') as log:
        try:
            rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=1800).returncode
        except subprocess.TimeoutExpired:
            subprocess.run(['docker', 'stop', '--time', '5', NAME], check=False)
            rc = 124
    changed = [p for p, digest in frozen.items() if not (ROOT/p).is_file() or sha(ROOT/p) != digest]
    cmake_tree_changed=content_files(deps/'cann-cmake')!=cmake_files
    removed = subprocess.run(['docker', 'inspect', NAME], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0
    package = source/'build_out/cann-ops-transformer-fault28_fia_graph_linux-aarch64.run'
    done = {'status': 'PASS_CPU_BUILD' if rc == 0 and not changed and not cmake_tree_changed and removed and package.is_file() else 'FAIL_CPU_BUILD',
        'returncode': rc, 'finished_unix': time.time(), 'frozen_sources_changed': changed, 'container_removed': removed,
        'npu_executed': False, 'log_sha256': sha(evidence/'build.log'), 'package_present': package.is_file(),
        'package_sha256': sha(package) if package.is_file() else None,
        'package': str(package), 'private_stage_required_before_runtime': True,'generation':args.generation,
        'cann_cmake_content_changed':cmake_tree_changed}
    (evidence/'build_completion.json').write_text(json.dumps(done, indent=2)+'\n')
    print(json.dumps(done), flush=True)
    return 0 if done['status'] == 'PASS_CPU_BUILD' else 1


if __name__ == '__main__':
    raise SystemExit(main())
