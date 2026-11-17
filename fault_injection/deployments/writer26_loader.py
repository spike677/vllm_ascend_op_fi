"""Explicit pre-model loading of the reviewed writer26 split dependency.

This deployment bootstrap is separate from the read-only metadata observer.
It does not import Torch/vLLM, query a device, or install a writer callback.
RTLD_LOCAL preserves the existing global interposition/RTLD_NEXT order.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys

from ..metadata_arrays import (LoadedWriter26MetadataArrays, MetadataArraySession,
    WRITER26_SYMBOLS, SCHEMA, _mapped_proxies, _sha, _stamp, writer26_mapping_diagnostic)


class StartupError(RuntimeError):
    def __init__(self, reason, evidence):
        super().__init__('WRITER26_STARTUP_' + reason)
        self.evidence = evidence


class StartupDependencies:
    def __init__(self, *, bundle, manifest_sha256):
        self.evidence = dict(schema='demoncat.writer26.startup_dependencies.v1', status='STARTING',
            pid=os.getpid(), mechanism_qualification=False, callback_installed=False,
            device_api_calls=0, global_symbol_scope_changed=False, images_loaded=[])
        self.handles = []
        self.reader = None
        try:
            self._initialize(bundle, manifest_sha256)
        except BaseException as error:
            self.evidence.update(status='FAILED_STARTUP_DEPENDENCY_CHECK', error=repr(error))
            raise StartupError(str(error), self.evidence) from error

    def _initialize(self, bundle, manifest_sha256):
        imported = [name for name in ('torch', 'torch_npu', 'vllm', 'vllm_ascend') if name in sys.modules]
        self.evidence['model_runtime_imports_before'] = imported
        if imported or not hasattr(os, 'RTLD_NOLOAD'):
            raise RuntimeError('EXPLICIT_LINUX_STARTUP_BEFORE_MODEL_IMPORT_REQUIRED')
        root = Path(bundle).resolve(strict=True)
        manifest = root / 'build.json'
        raw = manifest.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha256:
            raise RuntimeError('MANIFEST_SHA256_CHANGED')
        build = json.loads(raw)
        spec = build.get('metadata_array_ownership', {})
        if (build.get('schema') != 'demoncat.writer26.group_deployment.v2'
                or build.get('legacy_writers_absent') is not True or build.get('actual_npu_calls') != 0
                or spec.get('schema') != SCHEMA or spec.get('state_version') != 1
                or spec.get('images') != WRITER26_SYMBOLS or spec.get('reset_supported') is not False
                or spec.get('runtime_quiescent_readback_required') is not True
                or not isinstance(spec.get('parent_receipt_sha256'), str) or len(spec['parent_receipt_sha256']) != 64
                or spec.get('parent_receipt_sha256') != build.get('files', {}).get('metadata-array-ownership.json')
                or not set(WRITER26_SYMBOLS).issubset(build.get('files', {}))):
            raise RuntimeError('FRESH_COMPLETE_FOUR_IMAGE_BUILD_REQUIRED')
        files = {manifest: _stamp(manifest)}
        for name, expected in build['files'].items():
            path = (root / name).resolve(strict=True)
            if not path.is_relative_to(root) or _sha(path) != expected:
                raise RuntimeError('DEPENDENCY_BYTES_CHANGED:' + name)
            files[path] = _stamp(path)
        if any((root / name).exists() for name in
               ('libfault26_qmm_output_underflow_abi0.so', 'libfault26_fia_writer_proxy_abi0.so')):
            raise RuntimeError('LEGACY_WRITER_PRESENT')
        before = _mapped_proxies(tuple(WRITER26_SYMBOLS))
        self.evidence.update(bundle=str(root), manifest_sha256=manifest_sha256,
                             before=writer26_mapping_diagnostic(before))
        if set(before) != set(WRITER26_SYMBOLS) - {'libopapi_nn.so'} and set(before) != set(WRITER26_SYMBOLS):
            raise RuntimeError('EXPECTED_CORE_TRACE_MAIN_STARTUP_PRELOAD_REQUIRED')
        for name, aliases in before.items():
            for alias in aliases:
                if not Path(alias).samefile(root / name) or _sha(Path(alias)) != build['files'][name]:
                    raise RuntimeError('FOREIGN_PREEXISTING_IMAGE:' + name)
        if 'libopapi_nn.so' not in before:
            path = root / 'libopapi_nn.so'
            self.handles.append(ctypes.CDLL(str(path), mode=os.RTLD_NOW | os.RTLD_LOCAL))
            self.evidence['images_loaded'].append(dict(path=str(path), sha256=build['files'][path.name],
                mode='RTLD_NOW|RTLD_LOCAL', before_model_import=True))
        if any(_stamp(path) != stamp for path, stamp in files.items()):
            raise RuntimeError('BUILD_CHANGED_DURING_STARTUP')
        self.reader = LoadedWriter26MetadataArrays(bundle=root, manifest_sha256=manifest_sha256)
        session = MetadataArraySession(self.reader.read, manifest_sha256)
        initial = session.read()
        self.evidence.update(status='LOADED_FOUR_IMAGE_CONTRACT_ONLY_NOT_NATIVE_ACCEPTANCE',
            after=writer26_mapping_diagnostic(self.reader.paths), metadata_arrays=initial,
            activity_required=False, native_qualified_targets=0, c2=False, c3=False,
            implementation_sha256=_sha(Path(__file__)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--observe-only', action='store_true', help='Record existing maps and refuse missing images; do not load')
    args = parser.parse_args()
    report = dict(schema='demoncat.writer26.loader_preflight.v1', status='FAILED_WITH_EVIDENCE',
        actual_npu_calls=0, model_created=False, c2=False, c3=False, observe_only=args.observe_only)
    try:
        report['before'] = writer26_mapping_diagnostic(_mapped_proxies(tuple(WRITER26_SYMBOLS)))
        if args.observe_only:
            reader = LoadedWriter26MetadataArrays(bundle=args.bundle, manifest_sha256=args.manifest_sha256)
            report['metadata_arrays'] = MetadataArraySession(reader.read, args.manifest_sha256).read()
        else:
            dependencies = StartupDependencies(bundle=args.bundle, manifest_sha256=args.manifest_sha256)
            report['startup'] = dependencies.evidence
        report['status'] = 'CPU_LOADED_FOUR_IMAGE_CONTRACT_PASS_NOT_NATIVE_ACCEPTANCE'
    except BaseException as error:
        report['error'] = repr(error)
        if isinstance(error, StartupError):
            report['startup'] = error.evidence
    with args.out.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return 0 if report['status'].startswith('CPU_LOADED_') else 2


if __name__ == '__main__':
    raise SystemExit(main())
