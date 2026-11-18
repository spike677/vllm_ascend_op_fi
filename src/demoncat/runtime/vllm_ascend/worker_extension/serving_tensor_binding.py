"""Inert trusted natural-Worker deployment for the existing Fault8/18 groups."""
from dataclasses import dataclass, field
import json
import os
import threading

from .serving_context import require, require_serial_eager


_ATTRIBUTE = '_demoncat_natural_tensor_serving_bindings'
MECHANISMS = ('8.restore_offset', '18.api_half')


@dataclass(frozen=True)
class TensorServingBinding:
    mechanism: str
    worker: object
    runner: object
    model: object
    prompt: tuple
    weight_manifest_json: str
    workload_sha256: str
    environment_sha256: str
    pid: int
    thread: int
    simulation: bool = False
    synchronize: object = None
    trace_factory: object = None
    leases: set = field(default_factory=set, compare=False, repr=False)


def install_tensor_serving_binding(worker, mechanism, *, prompt_token_ids, weight_manifest,
        workload_sha256, environment_sha256, first_live_validation=False,
        _simulation=False, _synchronize=None, _trace_factory=None):
    require(mechanism in MECHANISMS, 'REVIEWED_NATURAL_TENSOR_MECHANISM_REQUIRED')
    bindings = getattr(worker, _ATTRIBUTE, {})
    require(type(bindings) is dict and mechanism not in bindings, 'SERVING_DEPLOYMENT_ALREADY_BOUND')
    require(first_live_validation is True and type(_simulation) is bool,
            'EXPLICIT_FIRST_LIVE_VALIDATION_REQUIRED')
    require((_synchronize is None and _trace_factory is None) or _simulation,
            'TENSOR_DEPLOYMENT_OVERRIDES_CPU_ONLY')
    require(not _simulation or callable(_synchronize), 'EXPLICIT_CPU_SYNCHRONIZER_REQUIRED')
    require((_trace_factory is None if mechanism == '18.api_half' else
             not _simulation or callable(_trace_factory)), 'EXPLICIT_CPU_MOE_TRACE_REQUIRED')
    require(isinstance(prompt_token_ids, (list, tuple)) and 2 <= len(prompt_token_ids) <= 4096
            and all(type(token) is int and token >= 0 for token in prompt_token_ids),
            'BOUNDED_FRESH_PREFILL_PROMPT_REQUIRED')
    require(all(isinstance(value, str) and len(value) == 64
                and all(char in '0123456789abcdef' for char in value)
                for value in (workload_sha256, environment_sha256)), 'REVIEWED_SCOPE_HASHES_REQUIRED')
    require(isinstance(weight_manifest, dict) and weight_manifest, 'WEIGHT_MANIFEST_REQUIRED')
    manifest = json.dumps(weight_manifest, sort_keys=True, allow_nan=False)
    runner, model = worker.model_runner, worker.model_runner.model
    require_serial_eager(runner)
    require(model.training is False, 'INFERENCE_MODEL_REQUIRED')
    binding = TensorServingBinding(mechanism, worker, runner, model, tuple(prompt_token_ids), manifest,
        workload_sha256, environment_sha256, os.getpid(), threading.get_ident(),
        _simulation, _synchronize, _trace_factory)
    if not hasattr(worker, _ATTRIBUTE):
        setattr(worker, _ATTRIBUTE, bindings)
    bindings[mechanism] = binding
    return dict(status='SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED', mechanism=mechanism,
        hooks_installed=0, workers_created=0, qualification_granted=False, simulation=_simulation,
        execution='REAL_ENQUEUE_REQUIRED', constructed_replay_fallback=False)


def remove_tensor_serving_binding(worker, mechanism):
    bindings = getattr(worker, _ATTRIBUTE, None)
    binding = bindings.get(mechanism) if type(bindings) is dict else None
    require(type(binding) is TensorServingBinding and binding.worker is worker
            and binding.mechanism == mechanism, 'SERVING_DEPLOYMENT_NOT_BOUND')
    require(not binding.leases, 'OWNER_CLEAN_REQUIRED')
    del bindings[mechanism]
    if not bindings:
        delattr(worker, _ATTRIBUTE)
    return dict(status='SERVING_DEPLOYMENT_REMOVED', mechanism=mechanism)
