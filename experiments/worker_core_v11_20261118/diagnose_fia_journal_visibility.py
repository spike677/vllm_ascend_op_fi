"""Read loader visibility before/after torch_npu import; never call an operator.

Run only in a root-controlled, device-free container. Import/ELF constructors
can call driver discovery code: this script does not attest device isolation.
"""
import argparse
import ctypes as C
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import threading
import time


PROFILE = '33102edbd8c3a8b98f410123a67e07ba0655431b2cfdf215b8927e085801a31c'
SYMBOLS = ('aclnnFusedInferAttentionScoreV3GetWorkspaceSize', 'aclnnFusedInferAttentionScoreV3',
           'aclCreateTensorList', 'aclDestroyTensorList', 'aclCreateIntArray', 'aclDestroyIntArray')
# Visibility candidates, not a claim about the executing/cached resolver order.
NAMES = ('libcust_opapi.so', 'libopapi_math.so', 'libopapi_nn.so', 'libopapi_cv.so',
         'libopapi_transformer.so', 'libopapi_oam.so', 'libopapi_legacy.so', 'libopapi.so',
         'libaclnn_ops_infer.so', 'libaclnn_ops_train.so', 'libaclnn_math.so',
         'libaclnn_sparse.so', 'libaclnn_fft.so', 'libaclnn_rand.so', 'libnnopbase.so', 'libascendcl.so')
MAX_HANDLES, MAX_FILE, MAX_CODE, CHUNK = 96, 256 << 20, 128 << 20, 64 << 10
MAX_STAGE_CODE = 512 << 20
ENV_KEYS = ('LD_PRELOAD', 'LD_LIBRARY_PATH', 'ASCEND_OPP_PATH',
            'ASCEND_CUSTOM_OPP_PATH', 'TORCH_DEVICE_BACKEND_AUTOLOAD')


def need(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def process_identity():
    start = Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]
    return dict(pid=os.getpid(), thread=threading.get_native_id(), start_ticks=start,
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    need(path.is_file() and 0 <= before.st_size <= MAX_FILE, 'FILE_SIZE_OR_TYPE_REFUSED:' + str(path))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    after = path.stat()
    keys = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    need(all(getattr(before, key) == getattr(after, key) for key in keys), 'FILE_CHANGED_DURING_READ:' + str(path))
    return dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest(),
                stat={key: getattr(before, key) for key in keys})


def child(root, name):
    path = (root / name).resolve(strict=True)
    need(path.is_relative_to(root), 'PATH_ESCAPES_BOUND_SOURCE')
    return path


def validate_bundle(args):
    root = args.journal.resolve(strict=True)
    build_file = file_identity(root / 'build.json')
    need(build_file['sha256'] == args.build_sha256, 'BUILD_SHA_DIFFERS')
    build = json.loads((root / 'build.json').read_bytes())
    need(build.get('candidate') == 'fia_input_journal' and build.get('qualified') is False
         and build.get('actual_npu_calls') == 0, 'EXACT_READONLY_JOURNAL_BUILD_REQUIRED')
    need(isinstance(build.get('files'), dict) and 1 <= len(build['files']) <= 128, 'BUILD_FILES_REQUIRED')
    files = {}
    for name, digest in build['files'].items():
        files[name] = file_identity(child(root, name))
        need(files[name]['sha256'] == digest, 'BUILD_FILE_DIFFERS:' + name)
    need('profile.json' in files and build.get('library') in files, 'PROFILE_AND_DSO_MUST_BE_BUILD_BOUND')
    profile = json.loads((root / 'profile.json').read_bytes())
    payload = dict(profile); ident = payload.pop('profile_sha256')
    need(ident == PROFILE == build.get('profile_sha256') == hashlib.sha256(canonical(payload)).hexdigest(),
         'FROZEN_SOURCE111_PROFILE_REQUIRED')
    source = args.native_source_root.resolve(strict=True)
    manifest = file_identity(source / 'source_manifest.json')
    need(manifest['sha256'] == args.native_source_manifest_sha256, 'NATIVE_SOURCE_MANIFEST_DIFFERS')
    rows = json.loads((source / 'source_manifest.json').read_bytes())['files']
    need(isinstance(rows, list) and len({row['path'] for row in rows}) == len(rows), 'SOURCE_MANIFEST_DUPLICATE_PATH')
    index = {row['path']: row for row in rows}
    need(len(profile['sources']) == 17, 'SOURCE111_COMPLETE_17_SOURCE_CLOSURE_REQUIRED')
    source_files = {}
    for name, digest in profile['sources'].items():
        value = source_files[name] = file_identity(child(source, name))
        need(name in index and index[name]['sha256'] == value['sha256'] == digest
             and index[name]['bytes'] == value['bytes'], 'COMPILE_SOURCE_DIFFERS:' + name)
    originals = {}
    for role, expected in profile['originals'].items():
        value = originals[role] = file_identity(expected['path'])
        need(value['sha256'] == expected['file_sha256'] and value['bytes'] == expected['file_bytes'],
             'ORIGINAL_DSO_DIFFERS:' + role)
    library = files[build['library']]
    return dict(root=str(root), build=build_file, profile_file=files['profile.json'],
                profile=profile, library=library, bundle_files=files, originals=originals,
                native_source_manifest=manifest, source_files=source_files)


def parse_maps(text):
    rows = []
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        need(len(fields) >= 5, 'PROC_MAPS_MALFORMED')
        start, end = (int(x, 16) for x in fields[0].split('-'))
        rows.append(dict(start=start, end=end, permissions=fields[1], offset=int(fields[2], 16),
                         device=fields[3], inode=int(fields[4]), path=fields[5] if len(fields) == 6 else '', raw=line))
    return rows


def executable_segments(path):
    with Path(path).open('rb') as stream:
        head = stream.read(64)
        need(len(head) == 64 and head[:6] == b'\x7fELF\x02\x01', 'ELF64_LE_REQUIRED')
        phoff = struct.unpack_from('<Q', head, 32)[0]
        entsize, count = struct.unpack_from('<HH', head, 54)
        need(entsize == 56 and 0 < count <= 128, 'ELF_PROGRAM_HEADER_BOUND')
        stream.seek(phoff); headers = stream.read(entsize * count)
        need(len(headers) == entsize * count, 'ELF_PROGRAM_HEADERS_TRUNCATED')
        result = []
        for i in range(count):
            kind, flags, offset, vaddr, _, filesz, memsz, _ = struct.unpack_from('<IIQQQQQQ', headers, i * entsize)
            if kind != 1 or not flags & 1:
                continue
            need(flags & 4 and not flags & 2 and 0 < filesz <= memsz <= MAX_CODE, 'EXECUTABLE_SEGMENT_REFUSED')
            stream.seek(offset); digest = hashlib.sha256(); remaining = filesz
            while remaining:
                chunk = stream.read(min(CHUNK, remaining)); need(chunk, 'ELF_EXECUTABLE_BYTES_TRUNCATED')
                digest.update(chunk); remaining -= len(chunk)
            remaining = memsz - filesz
            while remaining:
                size = min(CHUNK, remaining); digest.update(b'\0' * size); remaining -= size
            result.append(dict(vaddr=vaddr, size=memsz, sha256=digest.hexdigest()))
        need(result and sum(row['size'] for row in result) <= MAX_CODE, 'EXECUTABLE_IMAGE_BOUND')
        return result


def covered_maps(rows, start, size, path):
    selected, end = [], start + size
    for row in sorted(rows, key=lambda item: item['start']):
        if row['start'] <= start < row['end']:
            need('r' in row['permissions'] and 'x' in row['permissions'] and 'w' not in row['permissions'],
                 'FUNCTION_IMAGE_NOT_READONLY_EXECUTABLE')
            need(row['path'] and not row['path'].endswith(' (deleted)')
                 and str(Path(row['path']).resolve(strict=True)) == path, 'FUNCTION_IMAGE_MAPPING_DIFFERS')
            selected.append(row); start = min(end, row['end'])
            if start == end:
                return selected
    raise RuntimeError('FUNCTION_IMAGE_RANGE_NOT_MAPPED')


class DlInfo(C.Structure):
    _fields_ = [('fname', C.c_char_p), ('base', C.c_void_p), ('sname', C.c_char_p), ('saddr', C.c_void_p)]


class Loader:
    """Only libc loader functions are invoked; looked-up functions are never called."""
    def __init__(self):
        need(sys.platform == 'linux' and hasattr(os, 'RTLD_NOLOAD'), 'LINUX_NOLOAD_REQUIRED')
        self.libc = C.CDLL(None)
        specs = dict(dlopen=([C.c_char_p, C.c_int], C.c_void_p), dlclose=([C.c_void_p], C.c_int),
                     dlsym=([C.c_void_p, C.c_char_p], C.c_void_p), dlerror=([], C.c_char_p),
                     dladdr=([C.c_void_p, C.POINTER(DlInfo)], C.c_int))
        for name, (args, result) in specs.items():
            fn = getattr(self.libc, name); fn.argtypes, fn.restype = args, result

    def error(self):
        value = self.libc.dlerror()
        return value.decode(errors='replace') if value else None

    def maps(self):
        return Path('/proc/self/maps').read_text()

    def open(self, name):
        self.error()
        handle = self.libc.dlopen(os.fsencode(name), os.RTLD_NOLOAD | os.RTLD_LAZY)
        return handle, self.error()

    def close(self, handle):
        self.error(); code = self.libc.dlclose(handle)
        return code, self.error()

    def lookup(self, handle, name):
        self.error(); address = self.libc.dlsym(handle, name.encode())
        return address, self.error()

    def origin(self, address):
        info = DlInfo()
        need(self.libc.dladdr(address, C.byref(info)) == 1 and info.fname and info.base, 'DLADDR_FAILED')
        return dict(path=os.fsdecode(info.fname), base=int(info.base),
                    symbol=os.fsdecode(info.sname) if info.sname else None,
                    symbol_address=int(info.saddr) if info.saddr else None)

    def memory(self, address, size):
        return C.string_at(address, size)


def handle_names(bundle, rows):
    names = [bundle['library']['path'], *(value['path'] for value in bundle['originals'].values()), *NAMES]
    # Include actual mapped aliases, never enumerate directories or load candidates.
    names.extend(row['path'] for row in rows if row['path'] and Path(row['path']).name in NAMES)
    names = list(dict.fromkeys(names))
    need(len(names) <= MAX_HANDLES, 'HANDLE_CENSUS_BOUND_EXCEEDED')
    return names


def observe(loader, bundle, stage):
    raw = loader.maps(); maps = parse_maps(raw)
    result = dict(stage=stage, started_monotonic_ns=time.monotonic_ns(), maps_before=raw,
                  environment={name: os.environ.get(name) for name in ENV_KEYS},
                  handles=[], owners=[], errors=[], complete=False, cached_execution_owner_proven=False)
    owners = {}; code_bytes = 0

    def origin(address):
        nonlocal code_bytes
        location = loader.origin(address)
        path = str(Path(location['path']).resolve(strict=True)); key = (path, location['base'])
        if key not in owners:
            identity = file_identity(path); segments = executable_segments(path)
            need(code_bytes + sum(s['size'] for s in segments) <= MAX_STAGE_CODE, 'STAGE_CODE_READ_BOUND')
            code_bytes += sum(s['size'] for s in segments)
            # Keep the actual dladdr owner alive during memory reads, including
            # RTLD_DEFAULT and dependency lookups. This never loads a new image.
            pin, pin_error = loader.open(location['path'])
            need(pin, 'ALREADY_LOADED_SYMBOL_OWNER_PIN_REQUIRED:' + str(pin_error))
            verified, failures = [], []
            try:
                need(not pin_error, 'SYMBOL_OWNER_PIN_LOADER_ERROR:' + str(pin_error))
                current_maps = parse_maps(loader.maps())
                for segment in segments:
                    start, size = location['base'] + segment['vaddr'], segment['size']
                    spans = covered_maps(current_maps, start, size, path); digest = hashlib.sha256()
                    for offset in range(0, size, CHUNK):
                        digest.update(loader.memory(start + offset, min(CHUNK, size - offset)))
                    verified.append(dict(**segment, mapped_sha256=digest.hexdigest(), mappings=spans))
                    need(digest.hexdigest() == segment['sha256'], 'MAPPED_EXECUTABLE_BYTES_DIFFER')
            except Exception as error:
                failures.append(repr(error))
            finally:
                try:
                    code, error = loader.close(pin)
                    if code or error:
                        failures.append('SYMBOL_OWNER_NOLOAD_REFERENCE_RELEASE_FAILED:' + str((code, error)))
                except Exception as error:
                    failures.append('SYMBOL_OWNER_NOLOAD_REFERENCE_RELEASE_FAILED:' + repr(error))
            need(not failures, 'SYMBOL_OWNER_ERRORS:' + json.dumps(failures))
            need(file_identity(path) == identity, 'OWNER_FILE_CHANGED_DURING_OBSERVATION')
            roles = [role for role, value in bundle['originals'].items()
                     if value['sha256'] == identity['sha256'] and value['bytes'] == identity['bytes']]
            if identity['sha256'] == bundle['library']['sha256']:
                roles.append('journal')
            for role in roles:
                if role != 'journal':
                    need(bundle['profile']['originals'][role]['segments'] == segments, 'PINNED_SEGMENTS_DIFFER')
            owners[key] = len(result['owners'])
            result['owners'].append(dict(file=identity, base=location['base'], matched_profile_roles=roles,
                                         segments=verified, actual_loaded_code_bytes_rehashed=True,
                                         owner_noload_reference_released=True))
        return dict(**location, canonical_path=path, offset=address-location['base'], owner_index=owners[key])

    names = [None, *handle_names(bundle, maps)]
    for name in names:
        row = dict(handle='RTLD_DEFAULT' if name is None else name, loaded=name is None,
                   lookup_mode='RTLD_DEFAULT' if name is None else 'RTLD_NOLOAD|RTLD_LAZY', symbols={})
        result['handles'].append(row); handle = None
        try:
            if name is not None:
                handle, row['open_error'] = loader.open(name); row['loaded'] = bool(handle)
            for symbol in SYMBOLS:
                value = row['symbols'][symbol] = dict(address=None, lookup_error=None, owner_error=None)
                if name is not None and not handle:
                    value['lookup_error'] = 'HANDLE_NOT_ALREADY_LOADED'; continue
                try:
                    address, error = loader.lookup(handle, symbol)
                    value.update(address=address, lookup_error=error)
                    if address:
                        value['origin'] = origin(address)
                except Exception as error:
                    value['owner_error'] = repr(error)
                    result['errors'].append(dict(handle=row['handle'], symbol=symbol, error=repr(error)))
        except Exception as error:
            row['error'] = repr(error); result['errors'].append(dict(handle=row['handle'], error=repr(error)))
        finally:
            if handle:
                try:
                    code, error = loader.close(handle); row.update(close_code=code, close_error=error)
                    if code or error:
                        result['errors'].append(dict(handle=row['handle'], error='NOLOAD_REFERENCE_RELEASE_FAILED', code=code, detail=error))
                except Exception as error:
                    row['close_error'] = repr(error)
                    result['errors'].append(dict(handle=row['handle'], error='NOLOAD_REFERENCE_RELEASE_FAILED', detail=repr(error)))
    result['maps_after'] = loader.maps()
    result['executable_bytes_read_budget_charged'] = code_bytes
    result['complete'] = not result['errors'] and all(set(row['symbols']) == set(SYMBOLS) for row in result['handles'])
    result['ended_monotonic_ns'] = time.monotonic_ns()
    return result


def save(folder, name, value):
    raw = canonical(value)
    with (folder / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return dict(path=name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def run(args, *, loader_factory=Loader, importer=importlib.import_module):
    args.out.mkdir(parents=True, exist_ok=False)
    report = dict(schema='demoncat.fia_journal.lookup_visibility.v1', status='INCOMPLETE', observations=[], errors=[],
                  explicit_operator_calls=0, explicit_device_api_calls=0, journal_install_calls=0,
                  worker_calls=0, device_isolation='REQUIRES_EXTERNAL_CONTROLLER_EVIDENCE',
                  constructors_and_import_runtime_calls='NOT_INTERCEPTED_MAY_ATTEMPT_DRIVER_DISCOVERY',
                  actual_cached_resolver_or_execution_owner_proven=False, qualified=False,
                  C1=False, C2=False, C3=False, E5=False)
    try:
        need(not any(name in sys.modules for name in ('torch', 'torch_npu', 'vllm')), 'CLEAN_PRE_IMPORT_PROCESS_REQUIRED')
        report['probe'] = file_identity(__file__)
        need(report['probe']['sha256'] == args.probe_sha256, 'PROBE_SOURCE_DIFFERS')
        bundle = validate_bundle(args); report['bundle'] = bundle
        report['environment'] = {name: os.environ.get(name) for name in ENV_KEYS}
        preload = (os.environ.get('LD_PRELOAD') or '').split(os.pathsep)
        need(len(preload) == 2 and all(preload), 'EXACT_OBSERVER_JOURNAL_PRELOAD_REQUIRED')
        report['preload'] = [file_identity(path) for path in preload]
        need([row['sha256'] for row in report['preload']] ==
             [bundle['originals']['observer']['sha256'], bundle['library']['sha256']], 'PRELOAD_BYTES_OR_ORDER_DIFFERS')
        need(os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') == '0', 'EXPLICIT_AUTOLOAD_ZERO_REQUIRED')
        spec = importlib.util.find_spec('torch_npu')
        need(spec is not None and spec.origin, 'TORCH_NPU_SPEC_REQUIRED')
        initial = file_identity(spec.origin); expected_init = Path(bundle['originals']['torch_npu']['path']).parent.parent/'__init__.py'
        need(initial['path'] == str(expected_init.resolve(strict=True)) and initial['sha256'] == args.torch_npu_init_sha256,
             'EXACT_TORCH_NPU_IMPORT_SOURCE_REQUIRED')
        report['torch_npu_init_before'] = initial
        report['process'] = process_identity()
        loader = loader_factory()
        first = observe(loader, bundle, 'before_torch_npu_import')
        report['observations'].append(save(args.out, '01-before-import.json', first))
        required = [bundle['library']['path'], *(bundle['originals'][key]['path'] for key in ('nn','opapi','device_runtime','observer','custom'))]
        need(first['complete'] and all(any(row['handle'] == path and row['loaded'] for row in first['handles']) for path in required),
             'STARTUP_REQUIRED_HANDLES_OR_IDENTITIES_UNOBSERVED')
        imported = False
        try:
            module = importer('torch_npu'); imported = True
            report['torch_npu_init_after'] = file_identity(module.__file__)
            need(report['torch_npu_init_after'] == initial, 'TORCH_NPU_IMPORT_SOURCE_CHANGED')
            version = sys.modules.get('torch_npu.version')
            report['version_fields'] = {key: getattr(version, key, None) for key in ('__version__','git_version','cann_version')}
        except Exception as error:
            report['errors'].append(dict(stage='torch_npu_import', error=repr(error)))
        second = observe(loader, bundle, 'after_torch_npu_import' if imported else 'after_failed_torch_npu_import')
        report['observations'].append(save(args.out, '02-after-import.json', second))
        need(process_identity() == report['process'],
             'PROCESS_OR_THREAD_CHANGED')
        need(second['complete'] and not report['errors'], 'IMPORT_OR_VISIBILITY_INCOMPLETE')
        report['status'] = 'LOADER_VISIBILITY_CAPTURED_NOT_CACHED_EXECUTION_PROOF'
    except Exception as error:
        report['status'] = 'REFUSED_OR_INCOMPLETE'
        report['errors'].append(dict(stage='diagnostic', error=repr(error)))
    finally:
        save(args.out, 'result.json', report)
    return 0 if report['status'] == 'LOADER_VISIBILITY_CAPTURED_NOT_CACHED_EXECUTION_PROOF' else 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('journal', 'native-source-root', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    for name in ('build-sha256', 'native-source-manifest-sha256', 'probe-sha256', 'torch-npu-init-sha256'):
        p.add_argument('--'+name, required=True)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
