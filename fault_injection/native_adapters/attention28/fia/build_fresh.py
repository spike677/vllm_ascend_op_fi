"""Complete private FIA host/package and matching kernel build, without NPU.

Run inside an isolated aarch64 CANN 9.1 build container with no device mounts
or network. The source-specific recipe consumes committed dependency sources
and pinned archives, never a previous experiment's object file/lib_cache.
Success is CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION, not NPU acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback


HERE = Path(__file__).resolve().parent
COMMIT = 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
EXPECTED_BIN = 'FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347'
KEYS = [103000000000000000, 109000000000000000, 109000000000400000]
TND33_KEYS = [103000000010200003, 103000000010300003,
              104000000010200003, 104000000010300003]
TND33_FAI_KEYS = [5000000000000200100, 5000000000000200103,
                  5000000000010200100, 5000000000010200103,
                  5900000000010200100, 5900000000010200103]
TND33_FAI_FD_KEYS = [5000000000000200100, 5000000000000200103,
                     5000000000010200100, 5000000000010200103,
                     5100000000010200100, 5100000000010200103,
                     6000000000010200100, 6000000000010200103]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def require(condition, message):
    if not condition:raise RuntimeError(message)


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], text=True).strip()


def local_clone(source, destination, commit):
    require(Path(git(source, 'rev-parse', '--show-toplevel')).resolve() == source.resolve(),
            'DEPENDENCY_IS_NOT_ITS_OWN_GIT_REPOSITORY:' + str(source))
    require(git(source, 'rev-parse', commit + '^{commit}') == commit, 'MISSING_PINNED_COMMIT')
    subprocess.run(['git', 'clone', '--local', '--no-hardlinks', '--no-checkout', str(source), str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], check=True)
    require(git(destination, 'rev-parse', 'HEAD') == commit, 'PRIVATE_CLONE_IDENTITY_CHANGED')


def source_files(root):
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if '.git' in relative.parts:continue
        require(not path.is_symlink(), 'UNREVIEWED_SOURCE_SYMLINK:' + str(path))
        if not path.is_file():continue
        require(not re.search(r'\.(?:o|a|so(?:\.\d+)*)$', path.name), 'BUILT_OBJECT_IN_SOURCE:' + str(path))
        files.append(dict(path=relative.as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
    require(bool(files), 'EMPTY_SOURCE_DIRECTORY:' + str(root))
    return files


def remove_disabled_test_fixtures(private_dependencies, repository, commit, entries):
    """Exclude only individually pinned official test binaries in a new clone.

    ENABLE_TEST is OFF. This does not permit consuming a built object as a
    dependency: the matched files are removed before configure and inventory.
    Every path/hash/Git blob is validated before the first removal.
    """
    require(repository in ('opbase', 'ops-tensor', 'cann-cmake'), 'UNKNOWN_FIXTURE_REPOSITORY')
    parent = private_dependencies.resolve(strict=True)
    repo = (parent / repository).resolve(strict=True)
    require(repo.parent == parent and not (parent / repository).is_symlink(), 'FIXTURE_REPOSITORY_ESCAPES_PRIVATE_COPY')
    require(Path(git(repo, 'rev-parse', '--show-toplevel')).resolve() == repo
            and git(repo, 'rev-parse', 'HEAD') == commit, 'FIXTURE_REPOSITORY_COMMIT_MISMATCH')
    planned = []
    seen = set()
    for item in entries:
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and relative.parts[0] == 'tests'
                and relative.as_posix() not in seen, 'FIXTURE_EXCLUSION_PATH_INVALID')
        seen.add(relative.as_posix())
        require(re.fullmatch(r'[a-f0-9]{64}', item['sha256'])
                and re.fullmatch(r'[a-f0-9]{40}', item['git_blob']), 'FIXTURE_IDENTITY_INVALID')
        path = repo / relative
        require(path.resolve(strict=True).is_relative_to(repo) and not path.is_symlink()
                and re.search(r'\.(?:o|a|so(?:\.\d+)*)$', path.name), 'FIXTURE_NOT_A_CONFINED_BINARY')
        tracked = git(repo, 'ls-tree', '--full-tree', commit, '--', relative.as_posix())
        require(tracked in (f'100644 blob {item["git_blob"]}\t{relative.as_posix()}',
                            f'100755 blob {item["git_blob"]}\t{relative.as_posix()}'),
                'FIXTURE_NOT_EXACT_COMMITTED_BLOB')
        require(not git(repo, 'status', '--porcelain', '--untracked-files=all', '--', relative.as_posix()),
                'FIXTURE_WORKTREE_OR_INDEX_CHANGED')
        original = subprocess.check_output(['git', '-C', str(repo), 'cat-file', 'blob', item['git_blob']])
        require(hashlib.sha256(original).hexdigest() == item['sha256']
                and sha(path) == item['sha256'] and path.stat().st_size == item['bytes'] == len(original),
                'FIXTURE_CONTENT_HASH_MISMATCH')
        planned.append((path, dict(repository=repository, commit=commit, **item,
            disposition='REMOVED_FROM_NEW_CLONE_BEFORE_CONFIGURE', reason='Official test fixture; ENABLE_TEST=OFF')))
    for path, _ in planned:path.unlink()
    return [row for _, row in planned]


def verify_policy(source, receipt):
    require(receipt['source_commit'] == COMMIT
            and receipt['status'] == 'PATCHED_NOT_BUILT_NOT_NPU_VALIDATED', 'UNREVIEWED_POLICY_RECEIPT')
    require(git(source, 'rev-parse', 'HEAD') == COMMIT, 'SOURCE_COMMIT_CHANGED')
    for item in receipt['files']:
        relative = Path(item['path'])
        require(not relative.is_absolute() and '..' not in relative.parts and '.git' not in relative.parts,
                'INVALID_POLICY_PATH')
        path = source / relative
        require(not path.is_symlink() and sha(path) == item['after_sha256'], 'PATCHED_SOURCE_CHANGED:' + str(relative))


def sdk_interfaces(args, contract):
    """An explicit source compile probe may describe another exact SDK.

    This changes the new build inputs, never an existing artifact's admission.
    All source/dependency pins remain; host and kernels must rebuild together.
    """
    expected = [row for row in contract['files'] if row['kind'] == 'sdk']
    manifest = getattr(args, 'sdk_interface_manifest', None)
    digest = getattr(args, 'sdk_interface_sha256', None)
    require(bool(manifest) == bool(digest), 'SDK_PROBE_PATH_AND_SHA_REQUIRED_TOGETHER')
    if not manifest:return expected
    require(sha(manifest) == digest, 'SDK_SOURCE_PROBE_HASH_DIFFERS')
    data = json.loads(Path(manifest).read_bytes())
    require(data.get('schema') == 'demoncat.attention.sdk_source_probe.v1'
            and data.get('source_commit') == COMMIT and data.get('fia_translation_unit_compiled') is True
            and data.get('required_paths_complete') is True and data.get('actual_npu_calls') == 0
            and data.get('native_launch_verified') is False, 'COMPLETE_EXACT_SDK_SOURCE_PROBE_REQUIRED')
    rows = data['files']
    require(len(rows) == len(expected) and {row['path'] for row in rows} == {row['path'] for row in expected},
            'SDK_SOURCE_PROBE_INTERFACE_SET_DIFFERS')
    require(all(type(row.get('bytes')) is int and row['bytes'] > 0
                and re.fullmatch('[0-9a-f]{64}', row.get('sha256', '')) for row in rows), 'SDK_PROBE_FILE_IDENTITY_INVALID')
    return [dict(row, kind='sdk') for row in rows]


class Builder:
    operator = 'fused_infer_attention_score'
    include_components = ('common', 'incre_flash_attention', 'prompt_flash_attention', 'fused_infer_attention_score')

    def __init__(self, args):
        self.args = args
        self.root = args.work.resolve()
        require(not self.root.exists(), 'FRESH_BUILD_DIRECTORY_REQUIRED')
        self.root.mkdir(parents=True)
        self.ev = self.root / 'evidence';self.ev.mkdir()
        self.commands = []
        self.report = dict(schema='demoncat.fia.fresh_build.v1', status='STARTED',
                           npu_executed=False, actual_launch_verified=False,
                           previous_experiment_objects_used=False, historical_default_policy_claimed=False)

    def run(self, label, argv, *, cwd=None, env=None, timeout=1800):
        record = dict(label=label, argv=[str(value) for value in argv], cwd=str(cwd or self.root),
                      started_unix=time.time())
        self.commands.append(record);save(self.ev / 'commands.json', self.commands)
        log = self.ev / (label + '.log')
        try:
            with log.open('x', encoding='utf-8') as stream:
                result = subprocess.run(record['argv'], cwd=cwd or self.root, env=env,
                                        stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
            record['returncode'] = result.returncode
            require(result.returncode == 0, 'BUILD_COMMAND_FAILED:' + label)
        finally:
            record.update(finished_unix=time.time(), log_sha256=sha(log) if log.is_file() else None)
            save(self.ev / 'commands.json', self.commands)

    def preflight_tools(self):
        tools = {}
        for name in ('patch', 'cmake', 'g++', 'make', 'git'):
            located = shutil.which(name)
            require(located is not None, 'REQUIRED_BUILD_TOOL_MISSING:' + name)
            path = Path(located).resolve(strict=True)
            require(path.is_file(), 'REQUIRED_BUILD_TOOL_NOT_FILE:' + name)
            tools[name] = dict(path=str(path), sha256=sha(path))
            self.run('preflight_' + name.replace('+', 'x'), [path, '--version'], timeout=30)
        # The image previously lacked GNU patch. Exercise its actual parser
        # before building anything; do not bypass official dependency patches.
        probe = self.ev / 'patch_probe';probe.mkdir()
        (probe / 'input.txt').write_text('before\n')
        patch_file = probe / 'tool.patch'
        patch_file.write_text('--- input.txt\n+++ input.txt\n@@ -1 +1 @@\n-before\n+after\n')
        self.run('preflight_patch_dry_run', [tools['patch']['path'], '--batch', '--dry-run', '-p0', '-i', patch_file],
                 cwd=probe, timeout=30)
        require((probe / 'input.txt').read_text() == 'before\n', 'PATCH_DRY_RUN_MUTATED_INPUT')
        self.run('preflight_compiler_target', [tools['g++']['path'], '-dumpmachine'], timeout=30)
        target = (self.ev / 'preflight_compiler_target.log').read_text().strip()
        require(target.startswith('aarch64'), 'MATCHED_AARCH64_HOST_COMPILER_REQUIRED')
        tools['compiler_target'] = target
        save(self.ev / 'build_tools.json', tools)
        self.report['build_tools_sha256'] = sha(self.ev / 'build_tools.json')

    def prepare(self):
        args = self.args
        require(platform.machine().lower() in ('aarch64', 'arm64'), 'AARCH64_BUILD_HOST_REQUIRED')
        require(not list(Path('/dev').glob('davinci*')), 'CPU_BUILD_CONTAINER_MUST_NOT_EXPOSE_NPU_DEVICES')
        self.preflight_tools()
        source_in, dep_in = args.source.resolve(strict=True), args.dependencies.resolve(strict=True)
        self.sdk = args.sdk.resolve(strict=True)
        require(source_in != self.root and source_in not in self.root.parents
                and dep_in != self.root and dep_in not in self.root.parents, 'OUTPUT_MUST_NOT_MUTATE_INPUT_TREE')
        self.receipt = json.loads(args.policy_receipt.read_bytes())
        verify_policy(source_in, self.receipt)
        self.configure_policy(source_in)
        self.contract = json.loads((HERE / 'build_contract.json').read_bytes())
        contract = self.contract
        interfaces = sdk_interfaces(args, contract)
        self.report['sdk_interface_files'] = interfaces
        self.report['sdk_interface_manifest_sha256'] = getattr(args, 'sdk_interface_sha256', None)
        self.source, self.deps = self.root / 'source', self.root / 'dependencies'
        self.deps.mkdir()
        local_clone(source_in, self.source, COMMIT)
        # Only source bytes explicitly included in the reviewed patch receipt.
        for item in self.receipt['files']:
            target = self.source / item['path']
            require(target.parent.is_dir(), 'PATCH_TARGET_PARENT_MISSING')
            target.write_bytes((source_in / item['path']).read_bytes())
        excluded = []
        for name, pin in (('cann-cmake', contract['cann_cmake_commit']),
                          ('opbase', contract['opbase_commit']), ('ops-tensor', contract['ops_tensor_commit'])):
            local_clone(dep_in / name, self.deps / name, pin)
            entries = [row for row in contract.get('disabled_test_binary_fixtures', []) if row['repository'] == name]
            entries = [{key: value for key, value in row.items() if key != 'repository'} for row in entries]
            if entries:excluded.extend(remove_disabled_test_fixtures(self.deps, name, pin, entries))
        save(self.ev / 'excluded_test_fixtures.json', excluded)
        self.excluded_test_fixtures = excluded
        for item in contract['reviewed_files']:
            root = self.source if item['repository'] == 'ops-transformer' else self.deps / 'cann-cmake'
            require(sha(root / item['path']) == item['sha256'], 'REVIEWED_CMAKE_SOURCE_CHANGED:' + item['path'])
        for item in [row for row in contract['files'] if row['kind'] != 'sdk'] + interfaces:
            if item['kind'] == 'sdk':
                require(sha(self.sdk / item['path']) == item['sha256'], 'MATCHED_SDK_INTERFACE_DIFFERS:' + item['path'])
                continue
            relative = Path(item['path']).relative_to('third_party')
            original = dep_in / relative
            require(original.is_file() and sha(original) == item['sha256']
                    and original.stat().st_size == item['bytes'], 'OFFLINE_SOURCE_PIN_DIFFERS:' + str(original))
            target = self.deps / relative
            target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(original, target)
        compatibility = None
        if getattr(args, 'opdef_compat', None):
            require(self.sdk.name == 'cann-8.5.1' and args.sdk_interface_manifest
                    and self.variant in ('tnd_page16_s2_256', 'tnd_page16_fai_s2_256',
                                         'tnd_page16_fai_fd_s2_256'), 'EXACT_TND33_CANN851_BUILD_REQUIRED')
            from opdef_compat import adapt
            compatibility = adapt(self.source)
            save(self.ev / 'opdef_compat.json', compatibility)
            self.report['opdef_compat_sha256'] = sha(self.ev / 'opdef_compat.json')
        (self.deps / 'abseil-cpp').mkdir()
        shutil.copyfile(self.deps / 'pkg/abseil-cpp-20230802.1.tar.gz',
                        self.deps / 'abseil-cpp/abseil-cpp-20230802.1.tar.gz')
        eigen_files = source_files(dep_in / 'eigen')
        for item in eigen_files:
            target = self.deps / 'eigen' / item['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dep_in / 'eigen' / item['path'], target)
        packager = self.sdk / 'tools/op_project_templates/op_project_tmpl/cmake/util/makeself'
        for name in ('makeself.sh', 'makeself-header.sh'):
            target = self.source / 'third_party/makeself' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            require(not target.exists(), 'PRISTINE_PACKAGER_TARGET_REQUIRED')
            shutil.copyfile(packager / name, target);target.chmod(0o700)
        # Use the official local dependency branches. The previous lib_cache is
        # neither inspected nor copied; protobuf/abseil rebuild from archives.
        (self.root / 'ops-tensor').symlink_to(self.deps / 'ops-tensor', target_is_directory=True)
        for name in ('opbase', 'ops-tensor'):
            target = self.source / 'third_party' / name
            require(not target.exists(), 'PRISTINE_DEPENDENCY_TARGET_REQUIRED')
            target.symlink_to(self.deps / name, target_is_directory=True)
        json_root = self.deps / 'json'
        require(not any(char in str(json_root) for char in (';', '"', '\n', '\r')), 'UNSUPPORTED_BUILD_PATH')
        offline = self.ev / 'offline_json.cmake'
        offline.write_text('if(NOT TARGET json)\n'
            '  add_library(json INTERFACE)\n'
            f'  target_include_directories(json INTERFACE "{json_root}/single_include")\n'
            '  target_compile_definitions(json INTERFACE nlohmann=ascend_nlohmann)\n'
            f'  set(JSON_INCLUDE_DIR "{json_root}/single_include")\n'
            f'  set(JSON_SOURCE_PATH "{json_root}")\nendif()\n')
        self.frozen = {str(self.source / row['path']): row['after_sha256'] for row in self.receipt['files']}
        if compatibility:
            self.frozen.update({str(self.source / row['path']): row['after_sha256']
                                for row in compatibility['files']})
        deps_inventory = {name: source_files(self.deps / name) for name in
                          ('cann-cmake', 'opbase', 'ops-tensor', 'eigen', 'json', 'pkg', 'protobuf', 'abseil-cpp')}
        save(self.ev / 'dependency_sources.json', deps_inventory)
        self.report.update(variant=self.variant, source_commit=COMMIT,
                           patch_receipt_sha256=sha(args.policy_receipt), contract_sha256=sha(HERE / 'build_contract.json'),
                           dependency_manifest_sha256=sha(self.ev / 'dependency_sources.json'),
                           excluded_test_fixtures_sha256=sha(self.ev / 'excluded_test_fixtures.json'))

    def configure_policy(self, source_in):
        self.variant = (self.receipt.get('variant') if self.receipt['mechanism'] == '33.paged_decode'
                        else self.receipt['mechanism'].removeprefix('28.fia.'))
        require(self.variant in ('legal_tiling', 'denominator_replay', 'tnd_page16_s2_256',
                                'tnd_page16_fai_s2_256', 'tnd_page16_fai_fd_s2_256'), 'UNREVIEWED_VARIANT')
        self.vendor_name = {'legal_tiling': 'dcat_f28_legal', 'denominator_replay': 'dcat_f28_replay',
                            'tnd_page16_s2_256': 'dcat_f33_tnd',
                            'tnd_page16_fai_s2_256': 'dcat_f33_fai',
                            'tnd_page16_fai_fd_s2_256': 'dcat_f33_fai_fd'}[self.variant]

    def cpu_checks(self):
        self.run('geometry_compile', ['g++', '-std=c++17', '-I', HERE, HERE / 'geometry_check.cpp',
                                      '-o', self.ev / 'geometry_check'], env=self.env)
        self.run('geometry_check', [self.ev / 'geometry_check'], env=self.env)

    def host(self):
        args, source, deps = self.args, self.source, self.deps
        self.env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PIP_NO_INDEX='1',
                        PIP_NO_BUILD_ISOLATION='0', PIP_DISABLE_PIP_VERSION_CHECK='1',
                        CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs), CCACHE_DISABLE='1')
        self.run('python_build_tools', [args.python, '-c',
            'import importlib.metadata as m; from pip._internal.commands import create_command; '
            'o,_=create_command("wheel").parse_args([".","--no-deps"]); '
            'assert o.no_index and not o.build_isolation; '
            'print({n:m.version(n) for n in ("pip","setuptools","wheel")})'], env=self.env)
        self.cpu_checks()
        build = source / 'build'
        flags = {'CANN_3RD_LIB_PATH:PATH': deps, 'CANN_3RD_PKG_PATH:PATH': deps / 'pkg',
            'FETCHCONTENT_SOURCE_DIR_CANN-CMAKE:PATH': deps / 'cann-cmake',
            'FETCHCONTENT_SOURCE_DIR_OPBASE:PATH': deps / 'opbase',
            'FETCHCONTENT_SOURCE_DIR_OPS-TENSOR:PATH': deps / 'ops-tensor',
            'CMAKE_PROJECT_INCLUDE:FILEPATH': self.ev / 'offline_json.cmake',
            'CMAKE_EXPORT_COMPILE_COMMANDS': 'ON', 'ENABLE_BUILD_PKG': 'ON', 'BUILD_OPEN_PROJECT': 'ON',
            'BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG': 'ON', 'ENABLE_BUILT_IN': 'OFF',
            'ENABLE_OPS_HOST': 'ON', 'ENABLE_OPS_KERNEL': 'OFF', 'ENABLE_AICPU': 'OFF',
            'ENABLE_AICPU_KERNEL': 'OFF', 'ENABLE_TEST': 'OFF', 'ENABLE_CCACHE': 'OFF',
            'BUILD_OPS_RTY_KERNEL': 'OFF', 'VENDOR_NAME': self.vendor_name,
            'ASCEND_OP_NAME': self.operator, 'ASCEND_COMPUTE_UNIT': 'ascend910b',
            'CUSTOM_ASCEND_CANN_PACKAGE_PATH': self.sdk, 'Python3_EXECUTABLE': args.python,
            'HI_PYTHON': args.python, 'ASCEND_PYTHON_EXECUTABLE': args.python,
            'CMAKE_BUILD_TYPE': 'Release', 'CMAKE_INSTALL_PREFIX': source / 'build_out'}
        self.run('configure', ['cmake', '-S', source, '-B', build] + ['-D' + key + '=' + str(value) for key, value in flags.items()], env=self.env)
        cache = (build / 'CMakeCache.txt').read_text()
        require(re.search(r'^ENABLE_TEST:[^=\n]+=OFF$', cache, re.MULTILINE) is not None,
                'ACTUAL_CONFIGURE_DID_NOT_DISABLE_TESTS')
        self.check_excluded_fixtures()
        acquisition = []
        for path in sorted(build.rglob('*')):
            if path.is_file() and path.name.endswith(('-urlinfo.txt', '-gitinfo.txt')):
                content = path.read_text()
                require(not re.search(r'https?://|ssh://|git@', content), 'CONFIGURED_REMOTE_DEPENDENCY:' + str(path))
                acquisition.append(dict(path=str(path), sha256=sha(path)))
        save(self.ev / 'configured_acquisition.json', acquisition)
        self.run('complete_host_package', ['cmake', '--build', build, '--target', 'package', '--parallel', str(args.jobs)], env=self.env, timeout=3600)
        package = source / f'build_out/cann-ops-transformer-{self.vendor_name}_linux-aarch64.run'
        require(package.is_file(), 'NEW_HOST_PACKAGE_MISSING')
        self.stage = self.root / 'stage'
        self.run('private_stage', ['bash', package, '--install-path=' + str(self.stage), '--quiet'], env=self.env, timeout=600)
        self.vendor = self.stage / 'vendors' / (self.vendor_name + '_transformer')
        require(self.vendor.is_dir(), 'NEW_PRIVATE_VENDOR_MISSING')
        # This release's package omits sibling source include paths referenced
        # by ascendc/<op> (../../incre_flash_attention/op_kernel, etc.). Restore
        # those paths solely from this freshly patched checkout, never an old
        # vendor. Keep every existing byte if present and require equality.
        impl_root = self.vendor / ('op_impl/ai_core/tbe/' + self.vendor_name + '_transformer_impl')
        copied = []
        for name in self.include_components:
            origin = self.source / 'attention' / name / 'op_kernel'
            for item in source_files(origin):
                original = origin / item['path']
                target = impl_root / name / 'op_kernel' / item['path']
                require(item['bytes'] < 8 * 1024**2, 'UNEXPECTED_JIT_SOURCE_SIZE')
                if target.exists():
                    require(not target.is_symlink() and sha(target) == item['sha256'], 'PACKAGED_SOURCE_BYTES_DIFFER')
                else:
                    target.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(original, target)
                copied.append(dict(source=original.relative_to(self.source).as_posix(),
                                   destination=target.relative_to(self.vendor).as_posix(), sha256=item['sha256']))
        save(self.ev / 'jit_include_sources.json', copied)
        self.report.update(host_package_sha256=sha(package), status='CPU_HOST_BUILT_KERNEL_BUILD_PENDING')
        save(self.ev / 'completion.json', self.report)
        if self.variant in ('tnd_page16_fai_s2_256', 'tnd_page16_fai_fd_s2_256'):
            compiler = 'compile_tnd33_fai_fd_schema.py' if self.variant == 'tnd_page16_fai_fd_s2_256' else 'compile_tnd33_fai_schema.py'
            self.run('actual_fai_geometry_and_schema', [self.args.python, HERE/compiler,
                '--source', self.source, '--sdk', self.sdk, '--out', self.ev/'fai_schema'], env=self.env)
            self.report['fai_public_schema_sha256'] = sha(self.ev/'fai_schema/fai_abi_schema.json')
            self.report['fai_cpu_geometry_receipt_sha256'] = sha(self.ev/'fai_schema/compile_receipt.json')
            if self.variant == 'tnd_page16_fai_fd_s2_256':
                self.report['fai_fd_cpu_matrix_sha256'] = sha(self.ev/'fai_schema/fai_fd_geometry_matrix.json')
                self.report['fai_fd_geometry_validator_sha256'] = sha(HERE/'fai_fd_geometry.py')
                self.report['actual_worker_platform_observed'] = False

    def kernel(self):
        params = list((self.source / 'build').rglob(EXPECTED_BIN + '_param.json'))
        require(len(params) == 1, 'NEW_BUILD_DID_NOT_GENERATE_UNIQUE_SUPPORTED_KERNEL_INPUT')
        payload = json.loads(params[0].read_bytes())
        require(payload['op_list'][0]['bin_filename'] == EXPECTED_BIN, 'GENERATED_BINARY_IDENTITY_DIFFERS')
        kernel = self.root / 'kernel';kernel.mkdir()
        input_param = kernel / 'input_param.json';shutil.copyfile(params[0], input_param)
        dynamic = self.vendor / ('op_impl/ai_core/tbe/' + self.vendor_name + '_transformer_impl/dynamic/fused_infer_attention_score.py')
        require(dynamic.is_file(), 'NEW_PACKAGE_JIT_SOURCE_MISSING')
        opc = self.sdk / 'python/site-packages/opc_tool/opc.py'
        require(opc.is_file(), 'MATCHED_OPC_COMPILER_MISSING')
        keys = (TND33_FAI_FD_KEYS if self.variant == 'tnd_page16_fai_fd_s2_256' else
                TND33_FAI_KEYS if self.variant == 'tnd_page16_fai_s2_256' else
                TND33_KEYS if self.variant == 'tnd_page16_s2_256' else
                KEYS + ([119000000000400000] if self.variant == 'denominator_replay' else []))
        output = kernel / 'output';output.mkdir()
        env = dict(self.env, ASCEND_CUSTOM_OPP_PATH=str(self.vendor), OMP_NUM_THREADS='2')
        self.run('new_matching_kernel', [self.args.python, opc, dynamic,
            '--main_func=fused_infer_attention_score', '--input_param=' + str(input_param),
            '--soc_version=Ascend910B4', '--output=' + str(output), '--impl_mode=high_performance,optional',
            '--simplified_key_mode=0', '--op_mode=dynamic', '--tiling_key=' + ','.join(map(str, keys)),
            '--op_debug_config=dump_cce'], env=env, timeout=1800)
        matches = list(output.rglob(EXPECTED_BIN + '.json'))
        require(len(matches) == 1, 'UNIQUE_NEW_KERNEL_METADATA_REQUIRED')
        metadata = matches[0];binary = metadata.with_suffix('.o')
        compiled = json.loads(metadata.read_bytes())
        require(compiled['sha256'] == sha(binary) and compiled['binFileName'] == EXPECTED_BIN
                and {int(row['tilingKey']) for row in compiled['kernelList']} == set(keys),
                'ACTUAL_COMPILED_KEY_SET_OR_BINARY_HASH_MISMATCH')
        sdk_matcher = self.sdk / 'opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_transformer/fused_infer_attention_score.json'
        rows = [row for row in json.loads(sdk_matcher.read_bytes())['binList']
                if Path(row['binInfo']['jsonFilePath']).stem == EXPECTED_BIN]
        require(len(rows) == 1 and rows[0]['outputs'][0]['dtype'] == 'float16', 'MATCHED_SDK_FP16_MATCHER_REQUIRED')
        target_root = self.vendor / 'op_impl/ai_core/tbe/kernel'
        target = target_root / 'ascend910b/fused_infer_attention_score';target.mkdir(parents=True, exist_ok=True)
        for path in (metadata, binary):shutil.copyfile(path, target / path.name)
        rows[0]['binInfo']['jsonFilePath'] = 'ascend910b/fused_infer_attention_score/' + metadata.name
        registry = target_root / 'config/ascend910b/fused_infer_attention_score.json'
        registry.parent.mkdir(parents=True, exist_ok=True);save(registry, dict(binList=rows))
        # The summary index is independently required for ACLNN to discover
        # this private matching kernel. The per-operator JSON alone is not enough.
        from binary_registration import register_index
        registration = register_index(self.source, self.vendor, self.variant)
        save(self.ev / 'binary_summary_registration.json', registration)
        self.report.update(compiled_keys=keys, kernel_binary_sha256=sha(binary),
                           kernel_metadata_sha256=sha(metadata), generated_input_sha256=sha(input_param),
                           sdk_matcher_sha256=sha(sdk_matcher),
                           binary_summary_registration_sha256=sha(self.ev / 'binary_summary_registration.json'))

    def finish(self):
        self.check_excluded_fixtures()
        require(all(sha(path) == digest for path, digest in self.frozen.items()), 'PATCHED_SOURCE_CHANGED_DURING_BUILD')
        artifacts = []
        for path in sorted(self.vendor.rglob('*')):
            if path.is_file():
                artifacts.append(dict(path=path.relative_to(self.vendor).as_posix(), bytes=path.stat().st_size, sha256=sha(path)))
        require(any(row['path'].endswith('libcust_opapi.so') for row in artifacts), 'NEW_OPAPI_LIBRARY_MISSING')
        self.report.update(status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION', artifacts=artifacts,
                           vendor_root=str(self.vendor), finished_unix=time.time(),
                           loading_allowed=False, qualification_missing=['new build artifact binding',
                            'actual serialized raw/POD/launch equality', 'lossless bridge', 'B/F/OFF',
                            'independent statistics', 'public Worker boundary'])
        save(self.ev / 'completion.json', self.report)

    def check_excluded_fixtures(self):
        require(all(not (self.deps / row['repository'] / row['path']).exists()
                    for row in self.excluded_test_fixtures), 'DISABLED_TEST_BINARY_REAPPEARED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--policy-receipt', type=Path, required=True)
    parser.add_argument('--dependencies', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--sdk', type=Path, default=Path('/usr/local/Ascend/cann-9.1.0'))
    parser.add_argument('--sdk-interface-manifest', type=Path)
    parser.add_argument('--sdk-interface-sha256')
    parser.add_argument('--opdef-compat', choices=['ascend910b-cann851'])
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    args = parser.parse_args()
    builder = Builder(args)
    try:
        builder.prepare();builder.host();builder.kernel();builder.finish()
    except BaseException as error:
        builder.report.update(status='CPU_BUILD_FAILED', error=repr(error), traceback=traceback.format_exc(), finished_unix=time.time())
        save(builder.ev / 'completion.json', builder.report)
        print(json.dumps({key: builder.report[key] for key in ('status', 'error')}))
        return 2
    print(json.dumps({key: builder.report[key] for key in ('status', 'vendor_root', 'variant')}))
    return 0


if __name__ == '__main__':raise SystemExit(main())
