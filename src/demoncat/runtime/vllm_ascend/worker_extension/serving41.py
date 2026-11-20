"""Inert host-owned deployment of the existing source-bound KV41 adapter."""
from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import threading

from .serving_context import require, require_serial_eager, ContextRefused
from fault_injection.prerequisites import require_identity


_ATTRIBUTE = '_demoncat_kv41_serving_binding'


@dataclass(frozen=True)
class KV41ServingBinding:
    worker: object
    runner: object
    model: object
    transport: object
    prompt: tuple
    scope: object
    max_seconds: float
    pid: int
    thread: int
    native_sha256: str
    native_library_identity: int
    simulation: bool = False
    provider_factory: object = None
    native_readback: object = None
    runtime_only: bool = False
    group_context: object = None
    leases: set = field(default_factory=set, compare=False, repr=False)


def read_native_binding(binding):
    """Read loaded ABI identity; this library has no callback registration API."""
    transport = binding.transport
    if binding.simulation:
        result = binding.native_readback(transport)
        require(isinstance(result, dict) and result.get('abi') == 1
            and result.get('sha256') == binding.native_sha256, 'KV41_CPU_NATIVE_READBACK_REQUIRED')
        return dict(result, simulation=True, qualification_granted=False)
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    require(type(transport) is NativePostwriteLibrary and id(transport.lib) == binding.native_library_identity,
            'KV41_NATIVE_LIBRARY_OBJECT_CHANGED')
    path = Path(transport.path).resolve(strict=True)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    require_identity(transport.sha256==binding.native_sha256 and digest==binding.native_sha256,
        'postwrite','current_library_sha256',binding.native_sha256,
        dict(held=transport.sha256,file=digest),'KV41_NATIVE_LIBRARY_BYTES_CHANGED',ContextRefused)
    transport._verify_loaded_file(path)
    abi=int(transport.lib.dcat_pw_abi())
    require_identity(abi==1 and transport._snapshot is not None,'postwrite','current_abi_and_read_span',
        dict(abi=1,read_span=True),dict(abi=abi,read_span=transport._snapshot is not None),
        'KV41_NATIVE_ABI_OR_READ_SPAN_UNAVAILABLE',ContextRefused)
    return dict(abi=1, sha256=binding.native_sha256, loaded_mapping_verified=True,
        read_span_available=True, simulation=False, qualification_granted=False)


def install_fault41_serving_binding(worker, *, transport, prompt_token_ids, scope,
        max_seconds=600, first_live_validation=False, _simulation=False,
        _provider_factory=None, _native_readback=None, runtime_only=False, group_contract=None):
    """Trusted existing-serving host only; acquire no hooks, leases or workers."""
    from fault_injection.coverage_session import CoverageScope
    from fault_injection.deployments.postwrite import NativePostwriteLibrary
    require(not hasattr(worker, _ATTRIBUTE), 'SERVING_DEPLOYMENT_ALREADY_BOUND')
    require(type(runtime_only) is bool, 'KV41_BOOLEAN_RUNTIME_ONLY_REQUIRED')
    require(first_live_validation is True and type(_simulation) is bool, 'EXPLICIT_FIRST_LIVE_VALIDATION_REQUIRED')
    require((_provider_factory is None and _native_readback is None) or _simulation,
            'KV41_DEPLOYMENT_OVERRIDES_CPU_ONLY')
    require(not _simulation or callable(_provider_factory) and callable(_native_readback),
            'EXPLICIT_CPU_KV41_FIXTURE_REQUIRED')
    require(_simulation or type(transport) is NativePostwriteLibrary, 'KV41_VERIFIED_NATIVE_TRANSPORT_REQUIRED')
    require(type(scope) is CoverageScope and scope.mechanism == '41.tail_column'
        and scope.evidence_kind == ('cpu_contract' if _simulation else 'natural_worker')
        and scope.model_weights_sha256 is not None and scope.model_source_sha256 is not None,
        'KV41_REVIEWED_COMPLETE_MODEL_SCOPE_REQUIRED')
    require(isinstance(prompt_token_ids, (list, tuple)) and 2 <= len(prompt_token_ids) <= 4096
        and all(type(token) is int and token >= 0 for token in prompt_token_ids), 'KV41_BOUNDED_PROMPT_REQUIRED')
    require(type(max_seconds) in (int, float) and 0 < max_seconds <= 1500, 'KV41_BOUNDED_GROUP_TIME_REQUIRED')
    sha = getattr(transport, 'sha256', None)
    require(isinstance(sha, str) and len(sha) == 64 and all(c in '0123456789abcdef' for c in sha),
            'KV41_REVIEWED_NATIVE_HASH_REQUIRED')
    runner, model = worker.model_runner, worker.model_runner.model
    context=None
    if group_contract is None:require_serial_eager(runner, expected_max_seqs=1)
    else:
        from .rms18_tp2 import validate_tp2_group
        require(runtime_only and not _simulation,'TP2_KV41_ORDINARY_REAL_BINDING_REQUIRED')
        context=validate_tp2_group(worker,'41.tail_column',group_contract)
    require(model.training is False, 'INFERENCE_MODEL_REQUIRED')
    binding = KV41ServingBinding(worker, runner, model, transport, tuple(prompt_token_ids), scope,
        max_seconds, os.getpid(), threading.get_ident(), sha, id(getattr(transport, 'lib', None)),
        _simulation, _provider_factory, _native_readback, runtime_only, context)
    evidence = read_native_binding(binding)
    setattr(worker, _ATTRIBUTE, binding)
    return dict(status='SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED', mechanism='41.tail_column',
        workers_created=0, hooks_installed=0, native_leases_opened=0, native=evidence,
        qualification_granted=False, simulation=_simulation)


def remove_fault41_serving_binding(worker):
    binding = getattr(worker, _ATTRIBUTE, None)
    require(type(binding) is KV41ServingBinding and binding.worker is worker, 'SERVING_DEPLOYMENT_NOT_BOUND')
    require(not binding.leases, 'OWNER_CLEAN_REQUIRED')
    runtime = getattr(worker, '_demoncat_worker_fault_runtime', None)
    lease = getattr(runtime, '_leases', {}).get('41.tail_column')
    require(lease is None or getattr(lease.provider, 'binding', None) is not binding,
            'WORKER_RPC_OWNER_STILL_RETAINED')
    delattr(worker, _ATTRIBUTE)
    return dict(status='SERVING_DEPLOYMENT_REMOVED')


def install_fault41_task_binding(worker, **deployment):
    """Trusted ordinary deployment; no RPC factory or research fallback."""
    require('runtime_only' not in deployment, 'KV41_TASK_MODE_IS_FIXED')
    return install_fault41_serving_binding(worker, runtime_only=True, **deployment)
