"""Explicit CPU fixture identity, separate from immutable production pins.

Not a public config option, environment override or NPU deployment receipt.
It only describes actual temporary source files and CPU dispatch test doubles.
Readers must call require_cpu_values before using it with tensor arguments.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
import threading

_CURRENT = ContextVar('demoncat_cpu_build_identity', default=None)

@dataclass(frozen=True)
class CpuBuildIdentity:
    name: str
    torch_version: str
    torch_npu_version: str
    source_hashes: object
    registrations: object
    thread: int
    test_only: bool = True

    def report(self):
        return dict(profile=self.name, kind='CPU_TEST_DOUBLE_ONLY', test_only=True,
                    deployment_qualification=False, torch_version=self.torch_version,
                    torch_npu_version=self.torch_npu_version)


def current_cpu_identity():
    value = _CURRENT.get()
    if value is not None and value.thread != threading.get_ident():
        raise RuntimeError('CPU_TEST_IDENTITY_FOREIGN_THREAD')
    return value


def require_cpu_values(values):
    if current_cpu_identity() is None:
        return
    import torch
    def walk(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu':
                raise RuntimeError('CPU_TEST_BUILD_CANNOT_ATTEST_DEVICE_TENSOR')
        elif isinstance(value, (tuple, list)):
            for item in value: walk(item)
        elif isinstance(value, dict):
            for item in value.values(): walk(item)
    walk(values)


@contextmanager
def cpu_test_build_identity(*, name, torch_version, torch_npu_version, source_hashes, registrations):
    if not isinstance(name, str) or not name.startswith('CPU_FIXTURE_') or current_cpu_identity() is not None:
        raise RuntimeError('EXPLICIT_UNNESTED_CPU_BUILD_IDENTITY_REQUIRED')
    if not source_hashes or any(not isinstance(v,str) or len(v)!=64
                               or any(c not in '0123456789abcdef' for c in v) for v in source_hashes.values()):
        raise ValueError('CPU_FIXTURE_ACTUAL_SOURCE_HASHES_REQUIRED')
    profile = CpuBuildIdentity(name, str(torch_version), str(torch_npu_version),
        MappingProxyType(dict(source_hashes)), MappingProxyType(dict(registrations)), threading.get_ident())
    token = _CURRENT.set(profile)
    try: yield profile
    finally: _CURRENT.reset(token)
