"""Inert, reviewed fused-RoPE deployment on an existing serving Worker.

The public config selects Fault22. Actual source-bound calls select its fused
positions-input variant; this binding never falls back to a lookup proxy.
"""
from dataclasses import dataclass, field
import json
import os
import threading

from .serving_context import require, require_serial_eager


_ATTRIBUTE = '_demoncat_rope22_serving_binding'


@dataclass(frozen=True)
class RoPE22ServingBinding:
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


def install_fault22_serving_binding(worker, *, prompt_token_ids, weight_manifest,
        workload_sha256, environment_sha256, first_live_validation=False,
        _simulation=False, _synchronize=None, _trace_factory=None):
    """Trusted host deployment only; installing this object acquires no hooks.

    Prompt and reviewed provenance belong to the serving adapter, not user RPC
    settings. CPU seams cannot produce natural-Worker qualification.
    """
    require(not hasattr(worker, _ATTRIBUTE), 'SERVING_DEPLOYMENT_ALREADY_BOUND')
    require(first_live_validation is True and type(_simulation) is bool,
            'EXPLICIT_FIRST_LIVE_VALIDATION_REQUIRED')
    require((_synchronize is None and _trace_factory is None) or _simulation,
            'ROPE_DEPLOYMENT_OVERRIDES_CPU_ONLY')
    require(not _simulation or callable(_synchronize) and callable(_trace_factory),
            'EXPLICIT_CPU_ROPE_FIXTURE_REQUIRED')
    require(isinstance(prompt_token_ids, (list, tuple)) and 2 <= len(prompt_token_ids) <= 4096
            and all(type(token) is int and token >= 0 for token in prompt_token_ids),
            'BOUNDED_FRESH_PREFILL_PROMPT_REQUIRED')
    require(all(isinstance(value, str) and len(value) == 64
                and all(char in '0123456789abcdef' for char in value)
                for value in (workload_sha256, environment_sha256)), 'REVIEWED_SCOPE_HASHES_REQUIRED')
    require(isinstance(weight_manifest, dict) and bool(weight_manifest), 'WEIGHT_MANIFEST_REQUIRED')
    manifest = json.dumps(weight_manifest, sort_keys=True, allow_nan=False)
    runner, model = worker.model_runner, worker.model_runner.model
    require_serial_eager(runner)
    require(model.training is False, 'INFERENCE_MODEL_REQUIRED')
    binding = RoPE22ServingBinding(worker, runner, model, tuple(prompt_token_ids), manifest,
        workload_sha256, environment_sha256, os.getpid(), threading.get_ident(),
        _simulation, _synchronize, _trace_factory)
    setattr(worker, _ATTRIBUTE, binding)
    return dict(status='SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED', mechanism='22.wrong_row',
        variant='ACTUAL_FUSED_ROPE_CALLS_REQUIRED', hooks_installed=0, workers_created=0,
        legacy_lookup_fallback=False, qualification_granted=False, simulation=_simulation)


def remove_fault22_serving_binding(worker):
    binding = getattr(worker, _ATTRIBUTE, None)
    require(type(binding) is RoPE22ServingBinding and binding.worker is worker,
            'SERVING_DEPLOYMENT_NOT_BOUND')
    require(not binding.leases, 'OWNER_CLEAN_REQUIRED')
    delattr(worker, _ATTRIBUTE)
    return dict(status='SERVING_DEPLOYMENT_REMOVED')
