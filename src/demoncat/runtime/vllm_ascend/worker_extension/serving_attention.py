"""Inert trusted deployment of actual Attention discovery/native adapters."""
from dataclasses import dataclass, field
import json
import os
import threading

from .attention_native import MECHANISMS, FaNativeCall, FiaNativeCall, PagedNativeCall
from .paged33_arch22_bridge import Arch22NormalNativeCall
from .attention28_stock_census import StockTndPrefillCensus
from .serving_context import require, require_serial_eager

ATTRIBUTE = '_demoncat_attention_serving_bindings'


class ReadOnlyAttentionCall:
    simulation = False
    fingerprint = {'scope': 'ACTUAL_CALL_DISCOVERY_ONLY'}

    def state(self):
        return dict(native_loaded=False, scope_active=False, scope_owned=False,
                    environment_owned=0, calls=0, failures=[], build=dict(self.fingerprint), simulation=False)

    def stop(self):
        return self.state()

    def execute(self, *args):
        raise RuntimeError('ATTENTION_MATCHING_NATIVE_DEPLOYMENT_REQUIRED')


@dataclass(frozen=True)
class AttentionDeployment:
    worker: object
    runner: object
    model: object
    mechanism: str
    prompt: tuple
    native: object
    provenance_json: str
    pid: int
    thread: int
    max_calls: int
    simulation: bool = False
    test_context: object = None
    leases: set = field(default_factory=set, compare=False)


def install_attention_serving_binding(worker, mechanism, *, prompt_token_ids, provenance,
                                      native=None, max_calls=224, _simulation=False, _test_context=None):
    """Trusted host only; RPC never accepts modules, callbacks, paths or QKV.

Omitting Native enables useful zero-F discovery on the loaded Worker. It is
reported as Python observation, never a loaded Native provider or qualification.
"""
    require(mechanism in MECHANISMS and type(_simulation) is bool, 'ATTENTION_DEPLOYMENT_MECHANISM_REQUIRED')
    require(type(prompt_token_ids) in (tuple, list) and prompt_token_ids
        and all(type(x) is int and x >= 0 for x in prompt_token_ids), 'ATTENTION_ACTUAL_PROMPT_REQUIRED')
    require(type(max_calls) is int and 1 <= max_calls <= 224, 'ATTENTION_CALL_BUDGET_INVALID')
    runner, model = worker.model_runner, worker.model_runner.model
    require(not model.training and all(name not in vars(obj) for obj in (worker, runner)
            for name in ('execute_model', '_prepare_inputs')), 'ATTENTION_ORIGINAL_INFERENCE_WORKER_REQUIRED')
    if not _simulation:
        require_serial_eager(runner, expected_max_seqs=1)
        require(_test_context is None, 'ATTENTION_CPU_SEAM_ON_REAL_WORKER')
        require(native is None or type(native) in (FaNativeCall, FiaNativeCall, PagedNativeCall,
                Arch22NormalNativeCall, StockTndPrefillCensus),
                'ATTENTION_REVIEWED_NATIVE_ADAPTER_REQUIRED')
    else:
        require(type(_test_context) is dict, 'ATTENTION_EXPLICIT_CPU_CONTEXT_REQUIRED')
    if native is not None:
        require(native.mechanism == mechanism, 'ATTENTION_NATIVE_MECHANISM_DIFFERS')
        state = native.state()
        require(state['native_loaded'] is True and not state['scope_active'] and not state['scope_owned']
            and state['environment_owned'] == 0 and not state.get('pending_drains')
            and not state.get('pending_references'), 'ATTENTION_NATIVE_NOT_INERT')
    require(type(provenance) is dict, 'ATTENTION_DEPLOYMENT_PROVENANCE_REQUIRED')
    canonical = json.dumps(provenance, allow_nan=False, sort_keys=True)
    bindings = getattr(worker, ATTRIBUTE, None)
    require(bindings is None or type(bindings) is dict, 'ATTENTION_BINDING_TABLE_CHANGED')
    if bindings is None:
        bindings = {};setattr(worker, ATTRIBUTE, bindings)
    require(mechanism not in bindings and not any(item.leases for item in bindings.values()),
            'ATTENTION_DEPLOYMENT_ALREADY_OWNED')
    binding = AttentionDeployment(worker, runner, model, mechanism, tuple(prompt_token_ids),
        native or ReadOnlyAttentionCall(), canonical, os.getpid(), threading.get_ident(), max_calls,
        _simulation, _test_context)
    bindings[mechanism] = binding
    return binding


def remove_attention_serving_binding(worker, mechanism):
    bindings = getattr(worker, ATTRIBUTE, None)
    require(type(bindings) is dict and type(bindings.get(mechanism)) is AttentionDeployment,
            'ATTENTION_DEPLOYMENT_BINDING_MISSING')
    binding = bindings[mechanism]
    require(not binding.leases and binding.worker is worker and binding.runner is worker.model_runner
        and binding.model is worker.model_runner.model and binding.pid == os.getpid()
        and binding.thread == threading.get_ident(), 'ATTENTION_BINDING_STILL_OWNED_OR_CHANGED')
    state = binding.native.state()
    require(not state['scope_active'] and not state['scope_owned'] and state['environment_owned'] == 0
            and not state.get('pending_drains') and not state.get('pending_references'),
            'ATTENTION_NATIVE_RECOVERY_PENDING')
    del bindings[mechanism]
    return dict(status='SERVING_DEPLOYMENT_REMOVED', mechanism=mechanism, native_scope_active=False)
