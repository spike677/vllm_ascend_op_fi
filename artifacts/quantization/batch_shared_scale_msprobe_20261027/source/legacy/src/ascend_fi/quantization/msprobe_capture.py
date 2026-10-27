"""Bounded msProbe 26.1 single-point tensor capture for paired quantization.

Use ``create_capture`` before installing NativePairBinding. One debugger is
created per process, in ``debug``/``tensor`` mode, and reused across sessions.
No model/API dump hooks or debugger.start() are requested. msProbe itself has
process-wide initialization effects, so observer OFF/ON runs must use separate
processes and verify numerical transparency. This adapter is not a fault backend.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import threading


class CaptureError(RuntimeError):
    pass


SOURCE_HASHES = {
    'msprobe.pytorch.dump.debugger.precision_debugger':
        '43bb9d834c736ebeb5c7c89e4770c23e1c6fc813000d12aeee8c2b72753a02fb',
    'msprobe.pytorch.dump.pytorch_service':
        '12799d280f17c82cefeb8463e272376828e2239e6e057dfd4ca8e4cf62ce9036',
    'msprobe.core.dump.service':
        'd51804b819a07453733e52eb81ec132f2cfa9934bfa5b243aaf7ceecd8ad97b3',
}
_RUNTIME = None
_RUNTIME_LOCK = threading.Lock()
_PAIR_KEYS = {'input_x', 'native_q', 'native_scale', 'returned_q', 'returned_scale'}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json_copy(value):
    # Reject non-finite and opaque metadata instead of silently stringifying it.
    return json.loads(json.dumps(deepcopy(value), allow_nan=False))


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value


def _tensor_info(tensor):
    import torch
    from .batch_binding import tensor_digest
    if not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided:
        raise CaptureError('ONLY_DENSE_TENSORS_SUPPORTED')
    return {
        'dtype': str(tensor.dtype), 'shape': list(tensor.shape),
        'tensor_bytes': tensor.numel() * tensor.element_size(),
        'tensor_sha256': tensor_digest(tensor),
    }


def _debug_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _debug_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _debug_keys(item)


class _MsprobeRuntime:
    def __init__(self, dump_root):
        self.dump_root = Path(dump_root).resolve()
        if self.dump_root.exists() and any(self.dump_root.iterdir()):
            raise CaptureError('DUMP_ROOT_MUST_BE_EMPTY_FOR_NEW_PROCESS')
        import torch
        import torch_npu
        self._api_slots = [(torch_npu, 'npu_dynamic_quant'), (torch_npu, 'npu_quant_matmul'),
                           (torch, 'add'), (torch.Tensor, '__add__'),
                           (torch.nn.Module, '__call__')]
        before = [getattr(module, name) for module, name in self._api_slots]
        version = importlib.metadata.version('mindstudio-probe')
        if version != '26.1.0':
            raise CaptureError('UNREVIEWED_MSPROBE_VERSION:' + version)
        source_info = {}
        for name, expected in SOURCE_HASHES.items():
            module = importlib.import_module(name)
            path = Path(inspect.getsourcefile(module)).resolve()
            actual = _sha(path)
            if actual != expected:
                raise CaptureError('UNREVIEWED_MSPROBE_SOURCE:' + name)
            source_info[name] = {'path': str(path), 'sha256': actual}
        from msprobe.pytorch import PrecisionDebugger
        existing = getattr(PrecisionDebugger, '_instance', None)
        if existing is not None and getattr(existing, 'initialized', False):
            raise CaptureError('MSPROBE_ALREADY_INITIALIZED_OUTSIDE_ADAPTER')
        # This reviewed release unconditionally loads custom_wrap_ops.yaml even
        # in debug mode. Refuse registrations instead of modifying that package.
        import yaml
        package = importlib.import_module('msprobe.pytorch.dump.api_dump')
        custom_path = Path(package.__file__).parent / 'custom_wrap_ops.yaml'
        custom = yaml.safe_load(custom_path.read_text(encoding='utf-8')) if custom_path.exists() else None
        if custom:
            raise CaptureError('CUSTOM_API_REGISTRATION_CONFIG_NOT_EMPTY')
        self.dump_root.mkdir(parents=True, exist_ok=True)
        config = {'task': 'tensor', 'level': 'debug', 'dump_path': str(self.dump_root),
                  'rank': [], 'step': [], 'async_dump': False,
                  'custom_op_namespaces': [], 'tensor': {}}
        self.config_path = self.dump_root / 'capture_config.json'
        self.config_path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
        self.debugger = PrecisionDebugger(config_path=str(self.config_path))
        self._baseline = before
        self.assert_api_identity()
        if getattr(self.debugger.config, 'level', None) != 'debug' or self.debugger.task != 'tensor':
            raise CaptureError('DEBUG_TENSOR_CONFIGURATION_NOT_APPLIED')
        if getattr(self.debugger.service.api_register, 'all_api_registered', False):
            raise CaptureError('AUTOMATIC_API_HOOKS_WERE_REGISTERED')
        if getattr(self.debugger.config, 'custom_op_namespaces', None) != []:
            raise CaptureError('CUSTOM_OP_AUTO_REGISTRATION_NOT_DISABLED')
        self.source_info = {'version': version, 'sources': source_info,
                            'configuration': config, 'config_sha256': _sha(self.config_path),
                            'lifecycle': 'singleton; save only; step at session exit',
                            'automatic_model_api_dump_requested': False}
        self.active = False
        self.session_count = 0

    def assert_api_identity(self):
        if any(getattr(module, name) is not original
               for (module, name), original in zip(self._api_slots, self._baseline)):
            raise CaptureError('OBSERVER_API_IDENTITY_CHANGED')

    @staticmethod
    def describe(tensor):
        return _tensor_info(tensor)

    def save(self, tensor, name, expected):
        import torch
        # Debug save has no useful return value and can silently skip an invalid
        # call. Confirm the actual file and its original dtype/content instead.
        self.debugger.save(tensor, name, save_backward=False)
        matches = list(self.dump_root.rglob(name + '.*.pt'))
        if len(matches) != 1:
            raise CaptureError('EXPECTED_ONE_MSPROBE_TENSOR_FILE:' + name)
        path = matches[0].resolve()
        if not path.is_relative_to(self.dump_root):
            raise CaptureError('DUMP_FILE_OUTSIDE_ROOT')
        restored = torch.load(path, map_location='cpu', weights_only=True)
        if _tensor_info(restored) != expected:
            raise CaptureError('MSPROBE_TENSOR_READBACK_MISMATCH:' + name)
        if _tensor_info(tensor) != expected:
            raise CaptureError('MSPROBE_MUTATED_OBSERVER_SNAPSHOT:' + name)
        return {'file': str(path), 'file_sha256': _sha(path), 'file_bytes': path.stat().st_size,
                'debug_json': str(path.parent.parent / 'debug.json'),
                'debug_key': name + '.0.debug', 'readback_verified': True}

    def flush(self):
        # In this release stop() is a no-op at debug level; step() flushes JSON
        # and advances the next session's directory without starting API hooks.
        self.debugger.step()
        self.assert_api_identity()


class MsprobeCapture:
    """One run's bounded observer, sharing a process-owned debugger runtime."""
    def __init__(self, runtime, *, output_dir, module_name, run_metadata=None,
                 max_bytes=268435456, max_disk_bytes=536870912, max_records=513):
        self.runtime = runtime
        self.output_dir = Path(output_dir).resolve()
        self.module_name = module_name
        self.run_metadata = _json_copy(run_metadata or {})
        self.max_bytes = _positive(max_bytes, 'max_bytes')
        self.max_disk_bytes = _positive(max_disk_bytes, 'max_disk_bytes')
        self.max_records = _positive(max_records, 'max_records')
        self.records = []
        self.tensor_bytes = self.file_bytes = 0
        self._entered = self._closed = self._failed = False

    def __enter__(self):
        if self._entered or self._closed or self.runtime.active:
            raise CaptureError('CAPTURE_SESSION_ALREADY_ACTIVE_OR_CLOSED')
        if (self.output_dir / 'manifest.json').exists():
            raise CaptureError('CAPTURE_MANIFEST_ALREADY_EXISTS')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.assert_api_identity()
        self.runtime.active = True
        self.runtime.session_count += 1
        self.session_id = self.runtime.session_count
        self._entered = True
        try:
            self._write_manifest('RUNNING')
        except BaseException:
            self.runtime.active = False
            self._closed = True
            raise
        return self

    def record(self, stage, *, metadata, tensors):
        return self._record(stage, metadata, tensors)

    def record_logits(self, logits, *, metadata=None):
        """Save one or more mapped logits tensors; caller supplies row metadata."""
        return self._record('logits', metadata or {}, {'logits': logits})

    def _record(self, stage, metadata, tensors):
        if not self._entered or self._closed or self._failed:
            raise CaptureError('CAPTURE_SESSION_NOT_WRITABLE')
        event = {'stage': stage, 'metadata': {}, 'tensors': {}, 'status': 'RUNNING'}
        try:
            if stage not in ('pair', 'consumer', 'logits'):
                raise CaptureError('UNSUPPORTED_CAPTURE_STAGE')
            required = _PAIR_KEYS if stage == 'pair' else {'output'} if stage == 'consumer' else {'logits'}
            if set(tensors) != required:
                raise CaptureError('UNEXPECTED_STAGE_TENSORS')
            if stage != 'logits' and (not isinstance(metadata, dict) or not metadata.get('call_id')):
                raise CaptureError('CAPTURE_REQUIRES_CALL_ID')
            event['metadata'] = _json_copy(metadata)
            if len(self.records) >= self.max_records:
                raise CaptureError('CAPTURE_RECORD_BUDGET_EXHAUSTED')
            descriptions = {name: self.runtime.describe(tensor) for name, tensor in tensors.items()}
            payload_bytes = sum(info['tensor_bytes'] for info in descriptions.values())
            if self.tensor_bytes + payload_bytes > self.max_bytes:
                raise CaptureError('CAPTURE_TENSOR_BYTE_BUDGET_EXHAUSTED')
            self.records.append(event)
            for tensor_name, tensor in tensors.items():
                name = f'bsc_{self.session_id:04d}_{len(self.records):04d}_{tensor_name}'
                info = descriptions[tensor_name]
                saved = self.runtime.save(tensor, name, info)
                event['tensors'][tensor_name] = {'save_name': name, **info, **saved}
                self.tensor_bytes += info['tensor_bytes']
                self.file_bytes += saved['file_bytes']
                if self.file_bytes > self.max_disk_bytes:
                    raise CaptureError('CAPTURE_DISK_BYTE_BUDGET_EXHAUSTED')
            event['status'] = 'SAVED'
            self._write_manifest('RUNNING')
        except BaseException as error:
            self._failed = True
            event['status'], event['error'] = 'FAILED', type(error).__name__ + ':' + str(error)
            if not any(item is event for item in self.records):
                self.records.append(event)
            self._write_manifest('FAILED')
            raise

    def __exit__(self, exc_type, exc, traceback):
        cleanup_error = None
        try:
            self.runtime.flush()
            for record in self.records:
                for saved in record['tensors'].values():
                    path = Path(saved['debug_json'])
                    if not path.is_file():
                        raise CaptureError('MSPROBE_DEBUG_JSON_MISSING:' + str(path))
                    data = json.loads(path.read_text(encoding='utf-8'))
                    if saved['debug_key'] not in set(_debug_keys(data)):
                        raise CaptureError('MSPROBE_DEBUG_ENTRY_MISSING:' + saved['debug_key'])
                    saved['debug_json_sha256'] = _sha(path)
            if not self.records and exc is None:
                raise CaptureError('NO_CAPTURE_RECORDS')
        except BaseException as error:
            cleanup_error = error
        finally:
            self._closed = True
            self.runtime.active = False
            failed = self._failed or exc is not None or cleanup_error is not None
            self._write_manifest('FAILED' if failed else 'PASS',
                                 cleanup_error=str(cleanup_error) if cleanup_error else None)
        if cleanup_error is not None:
            raise CaptureError('CAPTURE_FINALIZATION_FAILED:' + str(cleanup_error)) from cleanup_error
        return False

    def _write_manifest(self, status, *, cleanup_error=None):
        data = {'schema': 'ascend_fi.msprobe_capture.v1', 'status': status,
                'module': self.module_name, 'run_metadata': self.run_metadata,
                'session_id': self.session_id, 'dump_root': str(self.runtime.dump_root),
                'msprobe': self.runtime.source_info, 'tensor_bytes': self.tensor_bytes,
                'file_bytes': self.file_bytes, 'max_bytes': self.max_bytes,
                'max_disk_bytes': self.max_disk_bytes, 'max_records': self.max_records,
                'records': self.records, 'cleanup_error': cleanup_error}
        target = self.output_dir / 'manifest.json'
        temporary = target.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(target)


def create_capture(*, model, layer, module_name, output_dir, run_metadata, config=None):
    """Observer-factory contract used by validate_batch_scale_checkpoint.py.

    ``max_bytes`` counts raw tensor payload; ``max_disk_bytes`` counts saved .pt
    files. Required ``dump_root`` must be identical for every session in this process.
    The supplied model/layer are intentionally not modified or registered.
    """
    del model, layer
    config = dict(config or {})
    allowed = {'dump_root', 'max_bytes', 'max_disk_bytes', 'max_records'}
    if set(config) - allowed:
        raise ValueError('unknown capture configuration: ' + ','.join(sorted(set(config) - allowed)))
    if not isinstance(config.get('dump_root'), str) or not config['dump_root'].strip():
        raise ValueError('capture configuration requires an explicit shared dump_root')
    output_dir = Path(output_dir).resolve()
    dump_root = Path(config.pop('dump_root')).resolve()
    global _RUNTIME
    with _RUNTIME_LOCK:
        if _RUNTIME is None:
            _RUNTIME = _MsprobeRuntime(dump_root)
        elif _RUNTIME.dump_root != dump_root:
            raise CaptureError('MSPROBE_SINGLETON_DUMP_ROOT_CHANGED')
    return MsprobeCapture(_RUNTIME, output_dir=output_dir, module_name=module_name,
                          run_metadata=run_metadata, **config)
