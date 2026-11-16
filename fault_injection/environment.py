"""Read worker environment identity without importing device libraries."""
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import re
import stat
import struct
import sys


_ASCEND = Path('/usr/local/Ascend')
_LIBRARIES = re.compile(
    r'^lib(?:nnopbase|opapi\w*|ascendcl|acl_rt|ascend_hal|cust_opapi\w*|torch_npu|'
    r'fault\d+\w*|dcat\d+\w*)\.so(?:\.[\w.]+)?$'
)
_ABI_HEADERS = ('acl/acl.h', 'aclnn/acl_meta.h', 'aclnn/opdev/op_cache.h')
_SOC_REFERENCE = 'https://docs.vllm.ai/projects/ascend/en/v0.18.0/faqs.html'


def classify_soc(observed_name):
    """Classify an observed SoC family, never certify a provider or a product SKU."""
    normalized = re.sub(r'\s+', '', observed_name).lower() if isinstance(observed_name, str) else ''
    if re.fullmatch(r'(?:ascend)?910b\d*', normalized):
        family = 'A2'
    elif re.fullmatch(r'(?:ascend)?910_93\d{2}', normalized):
        family = 'A3'
    else:
        family = None
    return dict(observed_name=observed_name, family=family,
                status='SOC_FAMILY_CLASSIFIED' if family else 'UNCLASSIFIED',
                source=_SOC_REFERENCE if family else None,
                product_sku=None, provider_compatibility='UNVERIFIED')


def _live_soc():
    """Query only an already initialized backend; never import or initialize NPU."""
    backend = sys.modules.get('torch_npu')
    npu = getattr(backend, 'npu', None) if backend is not None else None
    ready = getattr(npu, 'is_initialized', None)
    if not callable(ready):
        return None, 'REQUIRES_LIVE_WORKER_OBSERVATION'
    try:
        if ready() is not True:
            return None, 'BACKEND_NOT_INITIALIZED_NO_QUERY'
        index = npu.current_device()
        name = npu.get_device_name(index)
        if type(index) is not int or not isinstance(name, str) or not name:
            return None, 'INVALID_LIVE_DEVICE_OBSERVATION'
        return dict(classify_soc(name), logical_device=index), 'INITIALIZED_TORCH_NPU_DEVICE_QUERY'
    except (RuntimeError, AttributeError, TypeError):
        return None, 'LIVE_DEVICE_QUERY_UNAVAILABLE'


def _digest(stream):
    digest = hashlib.sha256()
    stream.seek(0)
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def _manifest(path):
    try:
        with path.open('rb') as stream:
            data = stream.read(64 * 1024 + 1)
    except OSError:
        return None
    if len(data) > 64 * 1024:
        return None
    entries = {}
    for line in data.decode(errors='replace').splitlines():
        key, separator, value = line.partition('=')
        if separator and key.strip():
            entries[key.strip().lower()] = value.strip().strip("'\"")
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(),
            'fields': {k: entries[k] for k in ('version', 'innerversion', 'arch', 'package_name')
                       if k in entries}}


def _toolkit():
    # The selected environment wins over a convenience latest symlink.
    configured = os.environ.get('ASCEND_HOME_PATH')
    roots = [Path(configured)] if configured else [
        _ASCEND / 'ascend-toolkit' / 'latest', _ASCEND / 'cann' / 'latest',
    ]
    installations = {}
    machine = platform.machine().lower()
    architecture = {'arm64': 'aarch64', 'amd64': 'x86_64'}.get(machine, machine)
    for root in roots:
        # Several releases expose both a toolkit root and an architecture
        # subdirectory. Select the directory used for headers on this machine.
        for directory in (root / (architecture + '-linux'), root):
            info = _manifest(directory / 'ascend_toolkit_install.info')
            if not info:
                continue
            resolved = directory.resolve()
            headers = {}
            for name in _ABI_HEADERS:
                path = resolved / 'include' / name
                try:
                    with path.open('rb') as stream:
                        headers[name] = _digest(stream)
                except OSError:
                    headers[name] = None
            installations[str(resolved)] = {**info, 'toolkit': str(resolved), 'abi_headers': headers}
            break
    status = 'AMBIGUOUS' if len(installations) > 1 else 'FOUND' if installations else 'UNAVAILABLE'
    return {'status': status, 'ascend_home_path': configured, 'installations': list(installations.values())}


def discover_installations(root=None, *, fingerprint_libraries=False):
    """Read installed toolkit metadata without loading a library or a device.

    This is an inventory, not an ABI compatibility decision. Full library hashes
    are opt-in for an offline inventory because a worker's status RPC should
    not reread gigabytes of unused installations on every call.
    """
    root = Path(root) if root is not None else _ASCEND
    machine = platform.machine().lower()
    arch = {'arm64': 'aarch64', 'amd64': 'x86_64'}.get(machine, machine)
    candidates = set()
    errors = []
    for base in (root, root / 'ascend-toolkit', root / 'cann'):
        try:
            if base.is_dir():
                candidates.add(base)
                candidates.update(p for p in base.iterdir() if p.is_dir())
        except OSError as error:
            errors.append(dict(path=str(base), error=type(error).__name__))
    if len(candidates) > 512:
        raise RuntimeError('Toolkit inventory exceeds bounded directory budget')
    records = {}
    for base in sorted(candidates):
        for directory in (base / (arch + '-linux'), base):
            info = _manifest(directory / 'ascend_toolkit_install.info')
            if info is None:
                continue
            directory = directory.resolve()
            if str(directory) in records:
                break
            row = dict(info, toolkit=str(directory), abi_headers={}, libraries={},
                       native_loaded=False, compatibility='UNVERIFIED')
            for name in _ABI_HEADERS:
                path = directory / 'include' / name
                try:
                    with path.open('rb') as stream:
                        row['abi_headers'][name] = _digest(stream)
                except OSError:
                    row['abi_headers'][name] = None
            if fingerprint_libraries:
                for name in ('libopapi.so', 'libopapi_nn.so', 'libnnopbase.so', 'libascendcl.so'):
                    path = directory / 'lib64' / name
                    try:
                        with path.open('rb') as stream:
                            length = os.fstat(stream.fileno()).st_size
                            row['libraries'][name] = dict(path=str(path.resolve()), bytes=length,
                                sha256=_digest(stream), elf_build_id=_elf_build_id(stream, length))
                    except OSError:
                        row['libraries'][name] = None
            records[str(directory)] = row
            break
    return dict(status='INSTALLED_METADATA_FOUND' if records else 'UNAVAILABLE',
                root=str(root), installations=list(records.values()),
                library_fingerprints_collected=fingerprint_libraries,
                loaded_library_identity_proven=False, scan_errors=errors)


def loaded_installation_links(libraries, installations):
    """Associate verified mapped file paths with installation metadata, not ABI.

    A proxy outside the toolkit stays unresolved. Multiple installations remain
    visible. No version is inferred from an arbitrary directory-name substring.
    """
    rows = []
    for library in libraries:
        path = Path(library['path'])
        matches = []
        if library.get('status') == 'MAPPED_FILE_VERIFIED':
            for installation in installations:
                root = Path(installation['toolkit'])
                try:
                    path.relative_to(root)
                except ValueError:
                    continue
                matches.append(dict(toolkit=str(root), version=installation['fields'].get('version'),
                                    metadata_sha256=installation['sha256']))
        rows.append(dict(name=library['name'], path=str(path), matches=matches,
                         status='PATH_LINKED_TO_INSTALLATION' if len(matches) == 1 else
                                'AMBIGUOUS_INSTALLATION' if matches else 'UNRESOLVED'))
    versions = {m['toolkit'] for row in rows for m in row['matches']}
    return dict(status='MULTIPLE_INSTALLATIONS_LOADED' if len(versions) > 1 else
                'PARTIALLY_LINKED' if versions else 'UNRESOLVED', libraries=rows,
                abi_compatibility_proven=False)


def _elf_build_id(stream, length):
    """Return a GNU build ID from ELF program notes. Malformed entries are ignored."""
    stream.seek(0)
    header = stream.read(64)
    if len(header) < 52 or header[:4] != b'\x7fELF' or header[4] not in (1, 2) or header[5] not in (1, 2):
        return None
    endian = '<' if header[5] == 1 else '>'
    if header[4] == 2:
        if len(header) < 64:
            return None
        table = struct.unpack_from(endian + 'Q', header, 32)[0]
        entry_size, count = struct.unpack_from(endian + 'HH', header, 54)
        fmt = endian + 'IIQQQQQQ'
        offset_index, size_index = 2, 5
    else:
        table = struct.unpack_from(endian + 'I', header, 28)[0]
        entry_size, count = struct.unpack_from(endian + 'HH', header, 42)
        fmt = endian + 'IIIIIIII'
        offset_index, size_index = 1, 4
    size = struct.calcsize(fmt)
    if entry_size < size or count > 1024 or table + entry_size * count > length:
        return None
    for index in range(count):
        stream.seek(table + index * entry_size)
        raw_entry = stream.read(size)
        if len(raw_entry) != size:
            return None
        entry = struct.unpack(fmt, raw_entry)
        offset, amount = entry[offset_index], entry[size_index]
        if entry[0] != 4 or amount > 1024 * 1024 or offset + amount > length:  # PT_NOTE
            continue
        stream.seek(offset)
        notes = stream.read(amount)
        cursor = 0
        while cursor + 12 <= len(notes):
            name_size, value_size, note_type = struct.unpack_from(endian + 'III', notes, cursor)
            name = cursor + 12
            value = name + ((name_size + 3) & ~3)
            cursor = value + ((value_size + 3) & ~3)
            if cursor > len(notes):
                break
            if note_type == 3 and notes[name:name + name_size] == b'GNU\0' and 0 < value_size <= 128:
                return notes[value:value + value_size].hex()
    return None


def _mapped_images(content, additional_names):
    images = {}
    for line in content.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or 'x' not in fields[1] or not fields[5].startswith('/'):
            continue
        address, _, offset, device, inode, mapped = fields
        deleted = mapped.endswith(' (deleted)')
        path = mapped.removesuffix(' (deleted)')
        name = Path(path).name
        if not _LIBRARIES.fullmatch(name) and name not in additional_names:
            continue
        key = (path, device, inode, deleted)
        offsets = images.setdefault(key, [])
        offsets.append((address, offset))
    return images


def _loaded_library(key, offsets):
    path, device, inode, deleted = key
    row = {'path': path, 'name': Path(path).name, 'status': 'UNAVAILABLE',
           'sha256': None, 'elf_build_id': None}
    if deleted:
        return dict(row, status='MAPPING_DELETED')
    if len(offsets) != len({offset for _, offset in offsets}):
        return dict(row, status='DUPLICATE_EXECUTABLE_MAPPING')
    try:
        with open(path, 'rb') as stream:
            before = os.fstat(stream.fileno())
            mapped_device = tuple(int(part, 16) for part in device.split(':'))
            if (not stat.S_ISREG(before.st_mode) or before.st_ino != int(inode)
                    or mapped_device != (os.major(before.st_dev), os.minor(before.st_dev))):
                return dict(row, status='MAPPED_FILE_IDENTITY_CHANGED')
            if stream.read(4) != b'\x7fELF':
                return dict(row, status='MAPPED_FILE_NOT_ELF')
            build_id = _elf_build_id(stream, before.st_size)
            digest = _digest(stream)
            after = os.fstat(stream.fileno())
            current = os.stat(path)
            identity = lambda info: (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
            if identity(before) != identity(after) or identity(before) != identity(current):
                return dict(row, status='MAPPED_FILE_IDENTITY_CHANGED')
    except (OSError, ValueError, struct.error):
        return row
    return dict(row, status='MAPPED_FILE_VERIFIED', sha256=digest, elf_build_id=build_id)


def _loaded_libraries(additional_names):
    maps = Path('/proc/self/maps')
    try:
        inventory = _mapped_images(maps.read_text(errors='surrogateescape'), additional_names)
    except OSError:
        return [], 'UNAVAILABLE'
    rows = [_loaded_library(key, offsets) for key, offsets in sorted(inventory.items())]
    # A worker can load more than one implementation with the same filename.
    # Keep all records, so the resolver cannot silently pick the first one.
    counts = {}
    for row in rows:
        counts[row['name']] = counts.get(row['name'], 0) + 1
    for row in rows:
        if counts[row['name']] > 1:
            row['status'] = 'AMBIGUOUS_MAPPING'
    try:
        final_inventory = _mapped_images(maps.read_text(errors='surrogateescape'), additional_names)
    except OSError:
        return rows, 'UNAVAILABLE'
    if final_inventory != inventory:
        return rows, 'MAPPING_CHANGED_DURING_INSPECTION'
    return rows, 'OBSERVED' if rows else 'NONE'


def _package(name, module):
    try:
        installed = metadata.version(name)
    except metadata.PackageNotFoundError:
        installed = None
    loaded = sys.modules.get(module)
    version = vars(loaded).get('__version__') if loaded is not None else None
    return {'installed': installed, 'loaded': str(version) if isinstance(version, str) else None}


def inspect_environment(*, additional_libraries=()):
    """Inventory the current process. Reading a mapped file does not hash loaded code."""
    additional_libraries = frozenset(additional_libraries)
    if any(not isinstance(name, str) or not name or Path(name).name != name for name in additional_libraries):
        raise ValueError('Additional libraries must be basenames')
    libraries, maps_status = _loaded_libraries(additional_libraries)
    abi = getattr(sys.modules.get('torch._C'), '_GLIBCXX_USE_CXX11_ABI', None)
    packages = {name: _package(name, module) for name, module in (
        ('torch', 'torch'), ('torch-npu', 'torch_npu'), ('vllm', 'vllm'), ('vllm-ascend', 'vllm_ascend'),
    )}
    installations = discover_installations()
    soc, soc_source = _live_soc()
    driver_file = _manifest(_ASCEND / 'driver' / 'version.info')
    driver = {'installation': driver_file, 'running_module_versions': {}}
    for module in ('drv_pcie_host', 'drv_devdrv_host'):
        try:
            version = (Path('/sys/module') / module / 'version').read_text().strip()
            driver['running_module_versions'][module] = version
        except OSError:
            pass
    return dict(schema='demoncat.environment.v1', pid=os.getpid(), machine=platform.machine(),
                python=platform.python_version(), packages=packages, toolkit=_toolkit(), driver=driver,
                torch_cxx11_abi=abi if type(abi) is bool else None,
                soc=soc, soc_source=soc_source,
                installed_toolkits=installations,
                loaded_toolkit_links=loaded_installation_links(libraries, installations['installations']),
                maps_status=maps_status, loaded_libraries=libraries,
                has_native_code_attestation=False)


def environment_fingerprint(environment, required_libraries):
    """Fingerprint exactly the named mapped files for a reviewed deployment profile.

    Install records, path spellings and process IDs are diagnostic metadata.
    Callers still need native attestation and live slot/storage proofs.
    """
    if environment is None or environment['maps_status'] != 'OBSERVED':
        return None
    names = sorted(set(required_libraries))
    if not names:
        raise ValueError('A profile must identify required mapped libraries')
    if any(row['status'] != 'MAPPED_FILE_VERIFIED' for row in environment['loaded_libraries']):
        return None
    mapped = {row['name']: (row['sha256'], row['elf_build_id']) for row in environment['loaded_libraries']}
    if not set(names) <= mapped.keys():
        return None
    driver = environment['driver']
    material = dict(machine=environment['machine'], python=environment['python'],
                    packages=environment['packages'], torch_cxx11_abi=environment['torch_cxx11_abi'],
                    libraries={name: mapped[name] for name in names},
                    driver={'running_module_versions': driver['running_module_versions'],
                            'installation_sha256': (driver['installation'] or {}).get('sha256')})
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
