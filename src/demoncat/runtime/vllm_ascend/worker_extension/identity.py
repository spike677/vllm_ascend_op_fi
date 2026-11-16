"""Read an initialized Worker's context without importing a serving or NPU package."""

from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import re
import secrets
import sys
import weakref


EXTENSION_VERSION = '1'


def _version(distribution):
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _process_start(pid):
    try:
        stat = Path('/proc/self/stat').read_text(encoding='utf-8')
        boot = Path('/proc/sys/kernel/random/boot_id').read_text(encoding='utf-8').strip()
        start = stat.rsplit(') ', 1)[1].split()[19]
        if stat.split(' ', 1)[0] == str(pid) and boot and start.isdecimal():
            return f'proc:{boot}:{pid}:{start}'
    except (OSError, ValueError, IndexError):
        pass
    return f'process_local:{pid}:{secrets.token_hex(16)}'


def _visible_devices():
    for name in ('ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
        value = os.environ.get(name)
        if value is not None:
            if re.fullmatch(r'[0-9]+(?:,[0-9]+)*', value) is None:
                return None, name
            return [int(index) for index in value.split(',')], name
    return None, None


def _current_device():
    torch = sys.modules.get('torch')
    npu = vars(torch).get('npu') if torch is not None else None
    initialized = getattr(npu, 'is_initialized', None)
    current = getattr(npu, 'current_device', None)
    if not callable(initialized) or not callable(current):
        return None
    try:
        if initialized() is True:
            device = current()
            if type(device) is int and device >= 0:
                return device
    except (RuntimeError, OSError, ValueError):
        pass
    return None


@dataclass(frozen=True)
class WorkerContext:
    worker: dict
    model: object
    ready: bool


class IdentityTracker:
    def __init__(self, pid):
        self.pid = pid
        self.process_start_identity = _process_start(pid)
        self.worker_start_identity = secrets.token_hex(16)
        self.vllm_version = _version('vllm')
        self.vllm_ascend_version = _version('vllm-ascend')
        self._runner = self._model = None
        self._model_instance_id = None

    def read(self, worker):
        runner = getattr(worker, 'model_runner', None)
        model = getattr(runner, 'model', None)
        if (self._runner is None or self._runner() is not runner
                or self._model is None or self._model() is not model):
            self._model_instance_id = None
            self._runner = self._model = None
            if model is not None:
                try:
                    self._runner, self._model = weakref.ref(runner), weakref.ref(model)
                except TypeError:
                    pass
                else:
                    self._model_instance_id = secrets.token_hex(16)

        rank = getattr(worker, 'rank', None)
        if type(rank) is not int or rank < 0:
            rank = None
        device = getattr(worker, 'device', None)
        configured = getattr(device, 'index', None)
        if getattr(device, 'type', None) != 'npu' or type(configured) is not int or configured < 0:
            configured = None
        current = _current_device() if configured is not None and model is not None else None
        logical = current if current is not None and current == configured else None
        visible, source = _visible_devices()
        identity = {
            'pid': self.pid,
            'rank': rank,
            'logical_device': logical,
            'configured_logical_device': configured,
            'observed_logical_device': current,
            'visible_devices': visible,
            'visible_devices_source': source,
            'process_start_identity': self.process_start_identity,
            'worker_start_identity': self.worker_start_identity,
            'model_instance_id': self._model_instance_id,
            'worker_class': f'{type(worker).__module__}.{type(worker).__qualname__}',
            'vllm_version': self.vllm_version,
            'vllm_ascend_version': self.vllm_ascend_version,
            'extension_version': EXTENSION_VERSION,
        }
        ready = rank is not None and logical is not None and self._model_instance_id is not None
        return WorkerContext(identity, model, ready)
