"""Bounded real-worker msProbe capture; native identity observer, no fault injection.

The old observer's scoped TorchDispatchMode is deliberately not entered: msProbe
mix mode owns Python/API hooks. Native LD_PRELOAD and RecordFunction identity
records remain enabled. This does not preserve the old Python operator coverage.
"""
from __future__ import annotations

import atexit
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import threading
import time
import traceback

REFERENCE_DEBUGGER_SHA256 = "43bb9d834c736ebeb5c7c89e4770c23e1c6fc813000d12aeee8c2b72753a02fb"
TARGET_MODULE = "vllm_ascend.worker.model_runner_v1"
_state = None
_debugger = None
_observer = None
_owner_thread = None


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _get_state():
    global _state
    pid = os.getpid()
    if _state is None or _state["pid"] != pid:
        _state = {"schema": "as_strided.msprobe_real_execute_bridge.v1", "pid": pid,
                  "status": "READY_NOT_CAPTURED", "task": "statistics", "level": "mix",
                  "capture_enabled": os.environ.get("MSPROBE_CAPTURE_ENABLED", "0") == "1",
                  "requested_steps": [0, 1, 2], "rank": 0, "completed_steps": [],
                  "events": [], "failures": [], "python_dispatch_mode_enabled": False,
                  "native_identity_observer_preserved": True,
                  "boundary": "Native API/schema metadata identity plus native msProbe statistics. No fault injection; statistics are not tensor-value dump or writeback oracle."}
    return _state


def _persist():
    state = _get_state()
    directory = Path(os.environ.get("MSPROBE_BRIDGE_DIR", "/evidence/msprobe_bridge"))
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("pid%d.bridge.json" % os.getpid())
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _event(kind, **fields):
    _get_state()["events"].append({"event": kind, "epoch_ns": time.time_ns(),
                                  "mono_ns": time.monotonic_ns(), "tid": threading.get_native_id(), **fields})
    _persist()


def _fail(stage, error):
    state = _get_state()
    state["status"] = "FAILED_CLOSED"
    state["failures"].append({"stage": stage, "type": type(error).__name__,
                              "error": str(error), "traceback": traceback.format_exc()})
    _persist()


def _initialize(runner):
    global _debugger, _owner_thread
    if _debugger is not None:
        if _owner_thread != threading.get_ident():
            raise RuntimeError("MSPROBE_CAPTURE_THREAD_CHANGED")
        return _debugger
    import torch
    if not torch.distributed.is_initialized():
        raise RuntimeError("REAL_WORKER_DISTRIBUTED_STATE_NOT_INITIALIZED")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    if (rank, world) != (0, 1):
        raise RuntimeError("CAPTURE_REQUIRES_ACTUAL_RANK0_WORLD1")
    if getattr(runner, "debugger", None) is not None:
        raise RuntimeError("VLLM_ALREADY_HAS_DEBUGGER")
    model = getattr(runner, "model", None)
    if not isinstance(model, torch.nn.Module):
        raise RuntimeError("ACTUAL_RUNNER_MODEL_IS_NOT_NN_MODULE")
    config_path = Path(os.environ.get("MSPROBE_CONFIG_PATH", "/audit/setup/msprobe_config.json"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    required = {"task": "statistics", "level": "mix", "rank": [0], "step": [0, 1, 2],
                "async_dump": False, "extra_info": True}
    if any(config.get(k) != v for k, v in required.items()):
        raise RuntimeError("UNEXPECTED_MSPROBE_CONFIG")
    if config.get("statistics") != {"scope": [], "list": [], "data_mode": ["all"], "summary_mode": "statistics"}:
        raise RuntimeError("UNEXPECTED_STATISTICS_CONFIG")
    dump_root = Path(config["dump_path"])
    if os.environ.get("MSPROBE_DUMP_ROOT") and Path(os.environ["MSPROBE_DUMP_ROOT"]) != dump_root:
        raise RuntimeError("CONFIG_AND_REQUESTED_DUMP_ROOT_DIFFER")
    if dump_root.exists() and any(dump_root.iterdir()):
        raise RuntimeError("MSPROBE_DUMP_ROOT_NOT_EMPTY")
    version = importlib.metadata.version("mindstudio-probe")
    if version != "26.1.0":
        raise RuntimeError("UNREVIEWED_MSPROBE_VERSION:" + version)
    from msprobe.pytorch import PrecisionDebugger
    source_path = Path(inspect.getsourcefile(PrecisionDebugger))
    if _sha(source_path) != REFERENCE_DEBUGGER_SHA256:
        raise RuntimeError("MSPROBE_DEBUGGER_SOURCE_CHANGED")
    existing = getattr(PrecisionDebugger, "_instance", None)
    if existing is not None and getattr(existing, "initialized", False):
        raise RuntimeError("MSPROBE_ALREADY_INITIALIZED")
    signatures = {name: str(inspect.signature(getattr(PrecisionDebugger, name)))
                  for name in ("start", "stop", "step")}
    if "model" not in inspect.signature(PrecisionDebugger.start).parameters or "rank_id" not in inspect.signature(PrecisionDebugger.start).parameters:
        raise RuntimeError("MSPROBE_START_SIGNATURE_CHANGED")
    _debugger = PrecisionDebugger(config_path=str(config_path))
    _owner_thread = threading.get_ident()
    if _debugger.task != "statistics" or _debugger.config.level != "mix":
        raise RuntimeError("CONFIG_NOT_APPLIED")
    state = _get_state()
    state.update({"versions": {name: importlib.metadata.version(name) for name in
                               ("torch", "torch-npu", "vllm", "vllm-ascend", "mindstudio-probe")},
                  "configuration": config, "config_path": str(config_path),
                  "config_sha256": _sha(config_path), "debugger_source": str(source_path),
                  "debugger_source_sha256": _sha(source_path), "signatures": signatures,
                  "actual_rank": rank, "actual_world_size": world,
                  "actual_model_type": type(model).__module__ + "." + type(model).__name__,
                  "actual_model_python_id": id(model), "bridge_source_sha256": _sha(__file__)})
    _event("debugger_initialized_only_at_real_cold_execute")
    return _debugger


def _scheduled(original, runner, args, kwargs):
    bound = inspect.signature(original).bind(runner, *args, **kwargs)
    scheduled = bound.arguments.get("scheduler_output")
    if scheduled is None:
        raise RuntimeError("EXECUTE_MODEL_HAS_NO_SCHEDULER_OUTPUT")
    counts = getattr(scheduled, "num_scheduled_tokens", None)
    total = getattr(scheduled, "total_num_scheduled_tokens", None)
    if not isinstance(counts, dict) or not counts or type(total) is not int or total <= 0:
        raise RuntimeError("NOT_A_NONEMPTY_REAL_SCHEDULE")
    if any(not isinstance(k, str) or type(v) is not int or v <= 0 for k, v in counts.items()) or sum(counts.values()) != total:
        raise RuntimeError("INVALID_REAL_SCHEDULE_COUNTS")
    return {"scheduler_type": type(scheduled).__module__ + "." + type(scheduled).__name__,
            "scheduled_request_ids": sorted(counts), "num_scheduled_tokens": counts,
            "total_num_scheduled_tokens": total}


def _run_target(original, runner, args, kwargs, scope):
    state = _get_state()
    if not state["capture_enabled"]:
        first_off = state["status"] != "OFF_BASELINE_NATIVE_ONLY_NO_MSPROBE_INITIALIZATION"
        state["status"] = "OFF_BASELINE_NATIVE_ONLY_NO_MSPROBE_INITIALIZATION"
        if first_off:
            _event("off_baseline_no_msprobe_initialization")
        return original(runner, *args, **kwargs)
    if state["status"] == "FAILED_CLOSED":
        raise RuntimeError("PREVIOUS_MSPROBE_FAILURE_STOPPED_RUN")
    phase = scope["phase"]
    eligible = (original.__module__ == TARGET_MODULE and original.__name__ == "execute_model"
                and phase.get("phase") == "cold_generate" and phase.get("request_id") == "cold-1")
    if not eligible or len(state["completed_steps"]) >= 3:
        return original(runner, *args, **kwargs)
    if any(x["method"] in {"_dummy_run", "profile_run", "compile_or_warm_up_model", "load_model"}
           for x in getattr(_observer._tls, "stack", [])):
        raise RuntimeError("DUMMY_OR_INITIALIZATION_ANCESTOR_IN_CAPTURE")
    step = len(state["completed_steps"])
    started = False
    try:
        schedule = _scheduled(original, runner, args, kwargs)
        debugger = _initialize(runner)
        _event("capture_start_requested", step=step, scope=scope, schedule=schedule,
               actual_model_python_id=id(runner.model))
        state["status"] = "CAPTURING"
        # Native scope is already active, Python TorchDispatchMode is absent.
        debugger.start(model=runner.model, rank_id=0)
        started = True
        _event("capture_started", step=step)
        result = original(runner, *args, **kwargs)
        debugger.stop()
        started = False
        _event("capture_stopped", step=step, returned_type=type(result).__module__ + "." + type(result).__name__)
        debugger.step()
        state["completed_steps"].append(step)
        state["status"] = "THREE_EXECUTES_CAPTURED_AWAITING_ARTIFACT_REVIEW" if step == 2 else "WAITING_NEXT_COLD_EXECUTE"
        _event("capture_step_advanced", completed_step=step, next_step=step + 1)
        return result
    except BaseException as error:
        if started or _debugger is not None:
            try:
                _debugger.stop()
            except BaseException as stop_error:
                _event("emergency_stop_failed", error=repr(stop_error))
        _fail("real_execute_capture", error)
        raise


def _wrap_native_only(original):
    @functools.wraps(original)
    def observed(runner, *args, **kwargs):
        obs = _observer
        obs.reset_pid()
        stack = getattr(obs._tls, "stack", [])
        scope = {"id": obs.next_id("SCOPE"), "parent_id": stack[-1]["id"] if stack else None,
                 "method": original.__name__, "phase": obs.phase_read()}
        obs._tls.stack = stack + [scope]
        previous_native = getattr(obs._tls, "native_state", None)
        obs.emit("runner_scope_enter", control=True, scope=scope,
                 owner=type(runner).__module__ + "." + type(runner).__name__, config=obs.option_state(runner),
                 source=inspect.getsourcefile(original), source_line=original.__code__.co_firstlineno,
                 python_dispatch_mode=False, native_identity_observer=True)
        obs.native_scope_sync(scope)
        try:
            obs.sample_weights(runner)
            if original.__name__ == "execute_model":
                obs.save_maps("first_execute")
            return _run_target(original, runner, args, kwargs, scope)
        except BaseException as error:
            obs.emit("runner_scope_exception", control=True, scope_id=scope["id"], error=repr(error))
            raise
        finally:
            obs.native_restore(previous_native)
            if original.__name__ == "load_model":
                obs.sample_weights(runner)
                obs.save_maps("after_load_model")
            obs.emit("runner_scope_exit", control=True, scope=scope, config=obs.option_state(runner))
            obs.summary()
            obs._tls.stack = stack
    observed._chain_observed = True
    observed._msprobe_native_only = True
    return observed


def install(observer):
    global _observer
    if os.environ.get("AS_STRIDED_OBS_MAIN_THREAD_MODE", "0") != "0":
        raise RuntimeError("BROAD_TORCHDISPATCH_MODE_FORBIDDEN_WITH_MSPROBE")
    if os.environ.get("MSPROBE_CAPTURE_ENABLED") not in {"0", "1"}:
        raise RuntimeError("MSPROBE_CAPTURE_ENABLED_MUST_BE_EXPLICIT_0_OR_1")
    _observer = observer
    observer.wrap = _wrap_native_only
    atexit.register(_persist)
    _event("bridge_installed", scope="Only real cold_generate first three Ascend runner executes; no dummy/load/warm capture")
