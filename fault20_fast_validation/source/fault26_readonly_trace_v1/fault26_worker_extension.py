"""Developer-mode, read-only typed FIA probe for Fault26.

The extension is deliberately inert when imported.  A vLLM collective RPC must
call ``fault26_install_typed_fia_probe`` before it wraps the public torch_npu
``npu_fused_infer_attention_score.out`` overload.  The wrapper only records
tensor metadata and addresses; it forwards the original arguments unchanged.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import threading
import time
from typing import Any


_STATE_LOCK = threading.Lock()
_LOG_LOCK = threading.Lock()
_REAL_PACKET: Any = None
_PROXY: Any = None
_REAL_QMM: Any = None
_QMM_PROXY: Any = None
_QMM_CALL_INDEX = 0
_REAL_ADD_RMS_NORM: Any = None
_ADD_RMS_NORM_PROXY: Any = None
_REAL_ADD_RMS_NORM_BIAS: Any = None
_ADD_RMS_NORM_BIAS_PROXY: Any = None
_CALL_INDEX = 0
_RESIDUAL_CALL_INDEX = 0
_RESIDUAL_BIAS_CALL_INDEX = 0
_PRODUCER_HOOKS: list[Any] = []
_PRODUCER_MODULES: list[dict[str, Any]] = []
_PRODUCER_CALL_INDEX = 0
_CURRENT_RESIDUAL_METADATA: dict[str, Any] | None = None
_CURRENT_RESIDUAL_GENERATION = 0
_LIVE_ALLOCATOR_ARMED = False
_LIVE_ALLOCATOR_MAX_HITS = 1
_LIVE_ALLOCATOR_HITS = 0
_LIVE_ALLOCATOR_TARGETS: list[dict[str, Any]] = []
_LIVE_ALLOCATOR_CAPTURES: list[dict[str, Any]] = []
_LIVE_ALLOCATOR_MIN_RESIDUAL_GENERATION = 0
_LIVE_ALLOCATOR_TARGET_WRITER = "FIA"
_LIVE_ALLOCATOR_TARGET_OUTPUT_BYTES = 0
_LIVE_ALLOCATOR_TARGET_RESIDUAL_BYTES = 0

_LIBC = ctypes.CDLL(None)
_LIBC.getenv.argtypes = [ctypes.c_char_p]
_LIBC.getenv.restype = ctypes.c_char_p


def _process_getenv(name: str) -> str | None:
    """Read the real process environment, including native setenv changes."""
    value = _LIBC.getenv(name.encode("utf-8"))
    return None if value is None else value.decode("utf-8")


def _developer_mode_enabled() -> bool:
    return os.environ.get("VLLM_SERVER_DEV_MODE") == "1"


def _probe_enabled() -> bool:
    return os.environ.get("FAULT26_TYPED_PROBE_ALLOWED") == "1"


def _writer_audit_allowed() -> bool:
    return os.environ.get("FAULT26_WRITER_AUDIT_ALLOWED") == "1"


def _fia_writer_proxy_allowed() -> bool:
    return os.environ.get("FAULT26_FIA_WRITER_PROXY_ALLOWED") == "1"


def _allocator_audit_allowed() -> bool:
    return os.environ.get("FAULT26_ALLOCATOR_AUDIT_ALLOWED") == "1"


def _log_path() -> str:
    return os.environ.get("FAULT26_TYPED_LOG_PATH", "")


def _append_event(event: dict[str, Any]) -> None:
    path = _log_path()
    if not path:
        raise RuntimeError("FAULT26_TYPED_LOG_PATH is not configured")
    payload = (json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    with _LOG_LOCK:
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)


def _tensor_metadata(value: Any) -> dict[str, Any] | None:
    try:
        import torch

        if not isinstance(value, torch.Tensor):
            return None
        storage = value.untyped_storage()
        device = value.device
        return {
            "data_ptr": f"0x{value.data_ptr():x}",
            "storage_ptr": f"0x{storage.data_ptr():x}",
            "storage_nbytes": int(storage.nbytes()),
            "storage_offset": int(value.storage_offset()),
            "view_nbytes": int(value.numel() * value.element_size()),
            "shape": [int(item) for item in value.shape],
            "stride": [int(item) for item in value.stride()],
            "dtype": str(value.dtype),
            "device": str(device),
            "device_index": device.index,
            "contiguous": bool(value.is_contiguous()),
            "requires_grad": bool(value.requires_grad),
        }
    except Exception as exc:  # Metadata failure must never alter inference.
        return {"metadata_error": f"{type(exc).__name__}: {exc}"}


def _last_tensor(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in reversed(value):
                tensor = _last_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for key in ("x", "residual", "output_x"):
                tensor = _last_tensor(value.get(key))
                if tensor is not None:
                    return tensor
    except Exception:
        pass
    return None


def _allocator_map_targets(targets: list[dict[str, Any]]) -> dict[str, Any]:
    import torch_npu

    segments = torch_npu.npu.memory_snapshot()
    results: list[dict[str, Any]] = []
    for target in targets:
        name = str(target["name"])[:128]
        address = int(target["address"])
        size = int(target["bytes"])
        target_end = address + size
        allocation: dict[str, Any] | None = None
        for segment_index, segment in enumerate(segments):
            segment_begin = int(segment.get("address", 0))
            segment_size = int(segment.get("total_size", 0))
            segment_end = segment_begin + segment_size
            if not (segment_begin <= address < segment_end):
                continue
            blocks = segment.get("blocks", [])
            begins: list[int] = []
            cursor = segment_begin
            for block in blocks:
                begins.append(cursor)
                cursor += int(block.get("size", 0))
            block_match: dict[str, Any] | None = None
            for block_index, (block, block_begin) in enumerate(zip(blocks, begins)):
                block_size = int(block.get("size", 0))
                block_end = block_begin + block_size
                if not (block_begin <= address < block_end):
                    continue
                neighbors = []
                for neighbor_index in range(
                    max(0, block_index - 1), min(len(blocks), block_index + 2)
                ):
                    neighbor = blocks[neighbor_index]
                    neighbor_begin = begins[neighbor_index]
                    neighbor_size = int(neighbor.get("size", 0))
                    neighbors.append(
                        {
                            "index": neighbor_index,
                            "begin": f"0x{neighbor_begin:x}",
                            "end": f"0x{neighbor_begin + neighbor_size:x}",
                            "size": neighbor_size,
                            "requested_size": int(neighbor.get("requested_size", 0)),
                            "state": str(neighbor.get("state", "")),
                        }
                    )
                block_match = {
                    "index": block_index,
                    "begin": f"0x{block_begin:x}",
                    "end": f"0x{block_end:x}",
                    "size": block_size,
                    "requested_size": int(block.get("requested_size", 0)),
                    "state": str(block.get("state", "")),
                    "address_offset": address - block_begin,
                    "target_fits_block": target_end <= block_end,
                    "neighbors": neighbors,
                }
                break
            allocation = {
                "segment_index": segment_index,
                "segment_begin": f"0x{segment_begin:x}",
                "segment_end": f"0x{segment_end:x}",
                "segment_size": segment_size,
                "segment_device": segment.get("device"),
                "segment_stream": str(segment.get("stream", "")),
                "segment_type": str(segment.get("segment_type", "")),
                "block": block_match,
            }
            break
        results.append(
            {
                "name": name,
                "address": f"0x{address:x}",
                "bytes": size,
                "end": f"0x{target_end:x}",
                "found": allocation is not None,
                "allocation": allocation,
            }
        )
    return {"segment_count": len(segments), "targets": results}


def _maybe_capture_live_allocator(writer: str, writer_output: Any) -> None:
    global _LIVE_ALLOCATOR_ARMED, _LIVE_ALLOCATOR_HITS
    with _STATE_LOCK:
        if not _LIVE_ALLOCATOR_ARMED or _LIVE_ALLOCATOR_HITS >= _LIVE_ALLOCATOR_MAX_HITS:
            return
        if writer != _LIVE_ALLOCATOR_TARGET_WRITER:
            return
        residual_generation = _CURRENT_RESIDUAL_GENERATION
        if residual_generation <= _LIVE_ALLOCATOR_MIN_RESIDUAL_GENERATION:
            return
        residual = dict(_CURRENT_RESIDUAL_METADATA or {})
        output_metadata = _tensor_metadata(_first_tensor(writer_output)) or {}
        if (
            _LIVE_ALLOCATOR_TARGET_RESIDUAL_BYTES > 0
            and int(residual.get("storage_nbytes", 0))
            != _LIVE_ALLOCATOR_TARGET_RESIDUAL_BYTES
        ):
            return
        if (
            _LIVE_ALLOCATOR_TARGET_OUTPUT_BYTES > 0
            and int(output_metadata.get("storage_nbytes", 0))
            != _LIVE_ALLOCATOR_TARGET_OUTPUT_BYTES
        ):
            return
        _LIVE_ALLOCATOR_HITS += 1
        hit = _LIVE_ALLOCATOR_HITS
        configured = [dict(item) for item in _LIVE_ALLOCATOR_TARGETS]
        if _LIVE_ALLOCATOR_HITS >= _LIVE_ALLOCATOR_MAX_HITS:
            _LIVE_ALLOCATOR_ARMED = False

    targets = configured
    for name, metadata in (
        ("live_residual", residual),
        (f"live_{writer.lower()}_output", output_metadata),
    ):
        try:
            address = int(str(metadata["storage_ptr"]), 0)
            size = int(metadata["storage_nbytes"])
        except (KeyError, TypeError, ValueError):
            continue
        targets.append({"name": name, "address": address, "bytes": size})

    event: dict[str, Any] = {
        "event": "fault26_live_allocator_snapshot",
        "pid": os.getpid(),
        "tid": threading.get_native_id(),
        "monotonic_ns": time.monotonic_ns(),
        "hit": hit,
        "writer": writer,
        "residual_generation": residual_generation,
        "residual_metadata": residual,
        "writer_output_metadata": output_metadata,
        "device_memory_modified": False,
    }
    if writer == "FIA":
        event["fia_output_metadata"] = output_metadata
    try:
        event.update({"ok": True, **_allocator_map_targets(targets)})
    except Exception as exc:
        event.update({"ok": False, "reason": f"{type(exc).__name__}: {exc}"})
    with _STATE_LOCK:
        _LIVE_ALLOCATOR_CAPTURES.append(event)
    try:
        _append_event(event)
    except Exception:
        pass


def _capture_named_tensors(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    captured: dict[str, Any] = {}
    positional_names = ("query", "key", "value")
    for index, value in enumerate(args[:3]):
        metadata = _tensor_metadata(value)
        if metadata is not None:
            captured[positional_names[index]] = metadata

    tensor_names = (
        "query",
        "key",
        "value",
        "pse_shift",
        "atten_mask",
        "dequant_scale1",
        "quant_scale1",
        "dequant_scale2",
        "quant_scale2",
        "quant_offset2",
        "antiquant_scale",
        "antiquant_offset",
        "key_antiquant_scale",
        "key_antiquant_offset",
        "value_antiquant_scale",
        "value_antiquant_offset",
        "block_table",
        "query_padding_size",
        "kv_padding_size",
        "key_shared_prefix",
        "value_shared_prefix",
        "query_rope",
        "key_rope",
        "key_rope_antiquant_scale",
        "workspace",
    )
    for name in tensor_names:
        if name not in kwargs:
            continue
        metadata = _tensor_metadata(kwargs[name])
        if metadata is not None:
            captured[name] = metadata

    outputs = kwargs.get("out")
    if isinstance(outputs, (tuple, list)):
        for index, value in enumerate(outputs):
            metadata = _tensor_metadata(value)
            if metadata is not None:
                captured[f"out_{index}"] = metadata
    else:
        metadata = _tensor_metadata(outputs)
        if metadata is not None:
            captured["out"] = metadata
    return captured


def _capture_scalars(kwargs: dict[str, Any]) -> dict[str, Any]:
    names = (
        "input_layout",
        "block_size",
        "num_key_value_heads",
        "num_heads",
        "sparse_mode",
        "scale",
        "pre_tokens",
        "next_tokens",
        "inner_precise",
        "antiquant_mode",
        "key_antiquant_mode",
        "value_antiquant_mode",
        "softmax_lse_flag",
    )
    result: dict[str, Any] = {}
    for name in names:
        value = kwargs.get(name)
        if isinstance(value, (str, int, float, bool)) or value is None:
            result[name] = value
    for name in ("actual_seq_lengths", "actual_seq_lengths_kv", "actual_shared_prefix_len"):
        value = kwargs.get(name)
        if isinstance(value, (tuple, list)):
            result[name] = [int(item) for item in value]
    return result


def _first_tensor(value: Any) -> Any:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value
        if isinstance(value, (tuple, list)):
            for item in value:
                tensor = _first_tensor(item)
                if tensor is not None:
                    return tensor
        if isinstance(value, dict):
            for item in value.values():
                tensor = _first_tensor(item)
                if tensor is not None:
                    return tensor
    except Exception:
        pass
    return None


def _producer_hook(stage: str, layer: int, module_name: str):
    def hook(module: Any, args: tuple[Any, ...], output: Any) -> None:
        global _PRODUCER_CALL_INDEX
        with _STATE_LOCK:
            _PRODUCER_CALL_INDEX += 1
            call_index = _PRODUCER_CALL_INDEX
        event: dict[str, Any] = {
            "event": "typed_final_producer_module_exit",
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "call_index": call_index,
            "monotonic_ns": time.monotonic_ns(),
            "stage": stage,
            "layer": layer,
            "module_name": module_name,
            "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
            "input": _tensor_metadata(_first_tensor(args)),
            "output": _tensor_metadata(_first_tensor(output)),
            "weight": _tensor_metadata(getattr(module, "weight", None)),
            "reduce_results": getattr(module, "reduce_results", None),
            "tp_size": getattr(module, "tp_size", None),
            "tp_rank": getattr(module, "tp_rank", None),
        }
        try:
            _append_event(event)
        except Exception:
            pass

    return hook


def _resolve_worker_model(worker: Any) -> Any:
    candidates = [
        getattr(worker, "model_runner", None),
        getattr(worker, "runner", None),
    ]
    for candidate in candidates:
        model = getattr(candidate, "model", None)
        if model is not None:
            return model
    return None


class _FiaPacketProxy:
    def __init__(self, real_packet: Any):
        self._real_packet = real_packet

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        global _CALL_INDEX
        with _STATE_LOCK:
            _CALL_INDEX += 1
            call_index = _CALL_INDEX

        started_ns = time.monotonic_ns()
        common = {
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "call_index": call_index,
            "monotonic_ns": started_ns,
        }
        try:
            _append_event(
                {
                    **common,
                    "event": "typed_fia_call_enter",
                    "tensors": _capture_named_tensors(args, kwargs),
                    "scalars": _capture_scalars(kwargs),
                }
            )
        except Exception:
            pass

        try:
            result = self._real_packet(*args, **kwargs)
        except BaseException as exc:
            try:
                _append_event(
                    {
                        **common,
                        "event": "typed_fia_call_exception",
                        "elapsed_ns": time.monotonic_ns() - started_ns,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception:
                pass
            raise

        _maybe_capture_live_allocator("FIA", result)
        try:
            _append_event(
                {
                    **common,
                    "event": "typed_fia_call_exit",
                    "elapsed_ns": time.monotonic_ns() - started_ns,
                    "outputs": _capture_named_tensors((), {"out": result}),
                }
            )
        except Exception:
            pass
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_packet, name)

    def out(self, *args: Any, **kwargs: Any) -> Any:
        global _CALL_INDEX
        with _STATE_LOCK:
            _CALL_INDEX += 1
            call_index = _CALL_INDEX

        started_ns = time.monotonic_ns()
        common = {
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "call_index": call_index,
            "monotonic_ns": started_ns,
        }
        try:
            _append_event(
                {
                    **common,
                    "event": "typed_fia_out_enter",
                    "tensors": _capture_named_tensors(args, kwargs),
                    "scalars": _capture_scalars(kwargs),
                }
            )
        except Exception as exc:
            # The probe is fail-open: logging must never block the real operator.
            try:
                _append_event(
                    {
                        **common,
                        "event": "typed_fia_probe_error",
                        "stage": "enter",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception:
                pass

        try:
            result = self._real_packet.out(*args, **kwargs)
        except BaseException as exc:
            try:
                _append_event(
                    {
                        **common,
                        "event": "typed_fia_out_exception",
                        "elapsed_ns": time.monotonic_ns() - started_ns,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception:
                pass
            raise

        _maybe_capture_live_allocator("FIA", kwargs.get("out", result))
        try:
            _append_event(
                {
                    **common,
                    "event": "typed_fia_out_exit",
                    "elapsed_ns": time.monotonic_ns() - started_ns,
                    "outputs": _capture_named_tensors((), {"out": kwargs.get("out")}),
                }
            )
        except Exception:
            pass
        return result


class _QmmCallableProxy:
    def __init__(self, real_callable: Any):
        self._real_callable = real_callable

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_callable, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        global _QMM_CALL_INDEX
        with _STATE_LOCK:
            _QMM_CALL_INDEX += 1
            call_index = _QMM_CALL_INDEX
        started_ns = time.monotonic_ns()
        try:
            result = self._real_callable(*args, **kwargs)
        except BaseException as exc:
            try:
                _append_event(
                    {
                        "event": "typed_qmm_call_exception",
                        "pid": os.getpid(),
                        "tid": threading.get_native_id(),
                        "call_index": call_index,
                        "monotonic_ns": started_ns,
                        "elapsed_ns": time.monotonic_ns() - started_ns,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception:
                pass
            raise

        _maybe_capture_live_allocator("QMM", result)
        try:
            tensors: dict[str, Any] = {}
            for index, name in enumerate(("x1", "x2", "scale")):
                if index < len(args):
                    metadata = _tensor_metadata(args[index])
                    if metadata is not None:
                        tensors[name] = metadata
            for name in ("bias",):
                metadata = _tensor_metadata(kwargs.get(name))
                if metadata is not None:
                    tensors[name] = metadata
            _append_event(
                {
                    "event": "typed_qmm_call_exit",
                    "pid": os.getpid(),
                    "tid": threading.get_native_id(),
                    "call_index": call_index,
                    "monotonic_ns": started_ns,
                    "elapsed_ns": time.monotonic_ns() - started_ns,
                    "tensors": tensors,
                    "output": _tensor_metadata(_first_tensor(result)),
                }
            )
        except Exception:
            pass
        return result


def _capture_add_rms_norm_inputs(
    args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    tensors: dict[str, Any] = {}
    positional_names = ("x1", "x2", "gamma")
    for index, value in enumerate(args[:3]):
        metadata = _tensor_metadata(value)
        if metadata is not None:
            tensors[positional_names[index]] = metadata
    for name in positional_names:
        if name in kwargs:
            metadata = _tensor_metadata(kwargs[name])
            if metadata is not None:
                tensors[name] = metadata

    epsilon: Any = kwargs.get("epsilon")
    if epsilon is None and len(args) >= 4:
        epsilon = args[3]
    scalars = {"epsilon": epsilon} if isinstance(epsilon, (int, float)) else {}
    return tensors, scalars


class _AddRmsNormProxy:
    def __init__(self, real_callable: Any):
        self._real_callable = real_callable

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_callable, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        global _RESIDUAL_CALL_INDEX
        with _STATE_LOCK:
            _RESIDUAL_CALL_INDEX += 1
            call_index = _RESIDUAL_CALL_INDEX

        started_ns = time.monotonic_ns()
        common = {
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "call_index": call_index,
            "monotonic_ns": started_ns,
        }
        try:
            tensors, scalars = _capture_add_rms_norm_inputs(args, kwargs)
            _append_event(
                {
                    **common,
                    "event": "typed_add_rms_norm_enter",
                    "tensors": tensors,
                    "scalars": scalars,
                }
            )
        except Exception:
            pass

        try:
            result = self._real_callable(*args, **kwargs)
        except BaseException as exc:
            try:
                _append_event(
                    {
                        **common,
                        "event": "typed_add_rms_norm_exception",
                        "elapsed_ns": time.monotonic_ns() - started_ns,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception:
                pass
            raise

        try:
            _append_event(
                {
                    **common,
                    "event": "typed_add_rms_norm_exit",
                    "elapsed_ns": time.monotonic_ns() - started_ns,
                    "outputs": _capture_named_tensors((), {"out": result}),
                }
            )
        except Exception:
            pass
        return result


class _AddRmsNormBiasProxy:
    def __init__(self, real_callable: Any):
        self._real_callable = real_callable

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real_callable, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        global _RESIDUAL_BIAS_CALL_INDEX
        with _STATE_LOCK:
            _RESIDUAL_BIAS_CALL_INDEX += 1
            call_index = _RESIDUAL_BIAS_CALL_INDEX

        started_ns = time.monotonic_ns()
        common = {
            "pid": os.getpid(),
            "tid": threading.get_native_id(),
            "call_index": call_index,
            "monotonic_ns": started_ns,
        }
        try:
            tensors: dict[str, Any] = {}
            for index, name in enumerate(("x1", "x2", "gamma", "beta")):
                value = args[index] if index < len(args) else kwargs.get(name)
                metadata = _tensor_metadata(value)
                if metadata is not None:
                    tensors[name] = metadata
            epsilon: Any = kwargs.get("epsilon")
            if epsilon is None and len(args) >= 5:
                epsilon = args[4]
            _append_event(
                {
                    **common,
                    "event": "typed_add_rms_norm_bias_enter",
                    "tensors": tensors,
                    "scalars": {"epsilon": epsilon}
                    if isinstance(epsilon, (int, float))
                    else {},
                }
            )
        except Exception:
            pass

        try:
            result = self._real_callable(*args, **kwargs)
        except BaseException as exc:
            try:
                _append_event(
                    {
                        **common,
                        "event": "typed_add_rms_norm_bias_exception",
                        "elapsed_ns": time.monotonic_ns() - started_ns,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            except Exception:
                pass
            raise

        global _CURRENT_RESIDUAL_METADATA, _CURRENT_RESIDUAL_GENERATION
        residual_metadata = _tensor_metadata(_last_tensor(result))
        if residual_metadata is not None:
            with _STATE_LOCK:
                _CURRENT_RESIDUAL_METADATA = residual_metadata
                _CURRENT_RESIDUAL_GENERATION = call_index
        try:
            _append_event(
                {
                    **common,
                    "event": "typed_add_rms_norm_bias_exit",
                    "elapsed_ns": time.monotonic_ns() - started_ns,
                    "outputs": _capture_named_tensors((), {"out": result}),
                }
            )
        except Exception:
            pass
        return result


class Fault26ReadOnlyWorkerExtension:
    """Methods exposed only through vLLM's developer-mode collective RPC."""

    def fault26_install_typed_fia_probe(self) -> dict[str, Any]:
        global _REAL_PACKET, _PROXY, _REAL_QMM, _QMM_PROXY, _QMM_CALL_INDEX
        global _REAL_ADD_RMS_NORM
        global _ADD_RMS_NORM_PROXY, _REAL_ADD_RMS_NORM_BIAS
        global _ADD_RMS_NORM_BIAS_PROXY, _CALL_INDEX, _RESIDUAL_CALL_INDEX
        global _RESIDUAL_BIAS_CALL_INDEX
        if not _developer_mode_enabled() or not _probe_enabled():
            return {
                "installed": False,
                "reason": "developer mode or FAULT26_TYPED_PROBE_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }
        if not _log_path():
            return {
                "installed": False,
                "reason": "FAULT26_TYPED_LOG_PATH is empty",
                "pid": os.getpid(),
            }

        import torch
        import torch_npu

        with _STATE_LOCK:
            if _PROXY is not None:
                return {
                    "installed": True,
                    "already_installed": True,
                    "pid": os.getpid(),
                    "log_path": _log_path(),
                    "call_count": _CALL_INDEX,
                }
            _REAL_PACKET = torch_npu.npu_fused_infer_attention_score
            _PROXY = _FiaPacketProxy(_REAL_PACKET)
            _REAL_QMM = torch_npu.npu_quant_matmul
            _QMM_PROXY = _QmmCallableProxy(_REAL_QMM)
            _REAL_ADD_RMS_NORM = torch_npu.npu_add_rms_norm
            _ADD_RMS_NORM_PROXY = _AddRmsNormProxy(_REAL_ADD_RMS_NORM)
            _REAL_ADD_RMS_NORM_BIAS = torch.ops._C_ascend.npu_add_rms_norm_bias
            _ADD_RMS_NORM_BIAS_PROXY = _AddRmsNormBiasProxy(_REAL_ADD_RMS_NORM_BIAS)
            torch_npu.npu_fused_infer_attention_score = _PROXY
            torch_npu.npu_quant_matmul = _QMM_PROXY
            torch_npu.npu_add_rms_norm = _ADD_RMS_NORM_PROXY
            torch.ops._C_ascend.npu_add_rms_norm_bias = _ADD_RMS_NORM_BIAS_PROXY
            _CALL_INDEX = 0
            _QMM_CALL_INDEX = 0
            _RESIDUAL_CALL_INDEX = 0
            _RESIDUAL_BIAS_CALL_INDEX = 0

        _append_event(
            {
                "event": "typed_fia_probe_installed",
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "monotonic_ns": time.monotonic_ns(),
                "wrapped": [
                    "npu_fused_infer_attention_score",
                    "npu_quant_matmul",
                    "npu_add_rms_norm",
                    "_C_ascend.npu_add_rms_norm_bias",
                ],
            }
        )
        return {
            "installed": True,
            "already_installed": False,
            "pid": os.getpid(),
            "log_path": _log_path(),
        }

    def fault26_remove_typed_fia_probe(self) -> dict[str, Any]:
        global _REAL_PACKET, _PROXY, _REAL_QMM, _QMM_PROXY
        global _REAL_ADD_RMS_NORM, _ADD_RMS_NORM_PROXY
        global _REAL_ADD_RMS_NORM_BIAS, _ADD_RMS_NORM_BIAS_PROXY
        import torch
        import torch_npu

        with _STATE_LOCK:
            if _PROXY is None:
                return {"removed": False, "reason": "not installed", "pid": os.getpid()}
            current = torch_npu.npu_fused_infer_attention_score
            current_qmm = torch_npu.npu_quant_matmul
            current_add_rms_norm = torch_npu.npu_add_rms_norm
            current_add_rms_norm_bias = torch.ops._C_ascend.npu_add_rms_norm_bias
            if (
                current is not _PROXY
                or current_qmm is not _QMM_PROXY
                or current_add_rms_norm is not _ADD_RMS_NORM_PROXY
                or current_add_rms_norm_bias is not _ADD_RMS_NORM_BIAS_PROXY
            ):
                return {
                    "removed": False,
                    "reason": "a wrapped torch_npu callable changed after installation",
                    "pid": os.getpid(),
                }
            torch_npu.npu_fused_infer_attention_score = _REAL_PACKET
            torch_npu.npu_quant_matmul = _REAL_QMM
            torch_npu.npu_add_rms_norm = _REAL_ADD_RMS_NORM
            torch.ops._C_ascend.npu_add_rms_norm_bias = _REAL_ADD_RMS_NORM_BIAS
            fia_calls = _CALL_INDEX
            qmm_calls = _QMM_CALL_INDEX
            residual_calls = _RESIDUAL_CALL_INDEX
            residual_bias_calls = _RESIDUAL_BIAS_CALL_INDEX
            _PROXY = None
            _REAL_PACKET = None
            _QMM_PROXY = None
            _REAL_QMM = None
            _ADD_RMS_NORM_PROXY = None
            _REAL_ADD_RMS_NORM = None
            _ADD_RMS_NORM_BIAS_PROXY = None
            _REAL_ADD_RMS_NORM_BIAS = None

        _append_event(
            {
                "event": "typed_fia_probe_removed",
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "monotonic_ns": time.monotonic_ns(),
                "fia_call_count": fia_calls,
                "qmm_call_count": qmm_calls,
                "add_rms_norm_call_count": residual_calls,
                "add_rms_norm_bias_call_count": residual_bias_calls,
            }
        )
        return {
            "removed": True,
            "pid": os.getpid(),
            "fia_call_count": fia_calls,
            "qmm_call_count": qmm_calls,
            "add_rms_norm_call_count": residual_calls,
            "add_rms_norm_bias_call_count": residual_bias_calls,
        }

    def fault26_query_typed_fia_probe(self) -> dict[str, Any]:
        return {
            "installed": _PROXY is not None,
            "pid": os.getpid(),
            "log_path": _log_path(),
            "call_count": _CALL_INDEX,
            "fia_call_count": _CALL_INDEX,
            "qmm_call_count": _QMM_CALL_INDEX,
            "add_rms_norm_call_count": _RESIDUAL_CALL_INDEX,
            "add_rms_norm_bias_call_count": _RESIDUAL_BIAS_CALL_INDEX,
            "developer_mode": _developer_mode_enabled(),
            "allowed": _probe_enabled(),
        }

    def fault26_install_final_producer_probe(self) -> dict[str, Any]:
        """Install read-only hooks on each Qwen3 o_proj and down_proj module."""
        global _PRODUCER_HOOKS, _PRODUCER_MODULES, _PRODUCER_CALL_INDEX
        if not _developer_mode_enabled() or not _probe_enabled():
            return {
                "installed": False,
                "reason": "developer mode or FAULT26_TYPED_PROBE_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }
        if not _log_path():
            return {
                "installed": False,
                "reason": "FAULT26_TYPED_LOG_PATH is empty",
                "pid": os.getpid(),
            }
        model = _resolve_worker_model(self)
        if model is None:
            return {
                "installed": False,
                "reason": "worker model_runner.model is unavailable",
                "pid": os.getpid(),
            }

        with _STATE_LOCK:
            if _PRODUCER_HOOKS:
                return {
                    "installed": True,
                    "already_installed": True,
                    "pid": os.getpid(),
                    "modules": list(_PRODUCER_MODULES),
                    "call_count": _PRODUCER_CALL_INDEX,
                }
            modules: list[dict[str, Any]] = []
            hooks: list[Any] = []
            for name, module in model.named_modules():
                stage = None
                if name.endswith(".self_attn.o_proj"):
                    stage = "attention"
                elif name.endswith(".mlp.down_proj"):
                    stage = "mlp"
                if stage is None:
                    continue
                match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
                if match is None:
                    continue
                layer = int(match.group(1))
                handle = module.register_forward_hook(
                    _producer_hook(stage, layer, name)
                )
                hooks.append(handle)
                modules.append(
                    {
                        "stage": stage,
                        "layer": layer,
                        "module_name": name,
                        "module_type": (
                            f"{type(module).__module__}.{type(module).__qualname__}"
                        ),
                        "reduce_results": getattr(module, "reduce_results", None),
                        "tp_size": getattr(module, "tp_size", None),
                        "tp_rank": getattr(module, "tp_rank", None),
                        "weight": _tensor_metadata(getattr(module, "weight", None)),
                    }
                )
            if not hooks:
                return {
                    "installed": False,
                    "reason": "no .self_attn.o_proj or .mlp.down_proj modules found",
                    "pid": os.getpid(),
                }
            _PRODUCER_HOOKS = hooks
            _PRODUCER_MODULES = modules
            _PRODUCER_CALL_INDEX = 0

        _append_event(
            {
                "event": "typed_final_producer_probe_installed",
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "monotonic_ns": time.monotonic_ns(),
                "module_count": len(_PRODUCER_MODULES),
                "modules": _PRODUCER_MODULES,
            }
        )
        return {
            "installed": True,
            "already_installed": False,
            "pid": os.getpid(),
            "module_count": len(_PRODUCER_MODULES),
            "modules": list(_PRODUCER_MODULES),
            "log_path": _log_path(),
        }

    def fault26_remove_final_producer_probe(self) -> dict[str, Any]:
        global _PRODUCER_HOOKS, _PRODUCER_MODULES
        with _STATE_LOCK:
            if not _PRODUCER_HOOKS:
                return {
                    "removed": False,
                    "reason": "not installed",
                    "pid": os.getpid(),
                }
            handles = list(_PRODUCER_HOOKS)
            modules = list(_PRODUCER_MODULES)
            _PRODUCER_HOOKS = []
            _PRODUCER_MODULES = []
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        _append_event(
            {
                "event": "typed_final_producer_probe_removed",
                "pid": os.getpid(),
                "tid": threading.get_native_id(),
                "monotonic_ns": time.monotonic_ns(),
                "module_count": len(modules),
                "call_count": _PRODUCER_CALL_INDEX,
            }
        )
        return {
            "removed": True,
            "pid": os.getpid(),
            "module_count": len(modules),
            "call_count": _PRODUCER_CALL_INDEX,
        }

    def fault26_query_final_producer_probe(self) -> dict[str, Any]:
        return {
            "installed": bool(_PRODUCER_HOOKS),
            "pid": os.getpid(),
            "module_count": len(_PRODUCER_MODULES),
            "modules": list(_PRODUCER_MODULES),
            "call_count": _PRODUCER_CALL_INDEX,
            "developer_mode": _developer_mode_enabled(),
            "allowed": _probe_enabled(),
            "log_path": _log_path(),
        }

    def fault26_arm_residual_writer_audit(
        self,
        session: int,
        max_qmm_windows: int = 4096,
        max_bytes: int = 4194304,
    ) -> dict[str, Any]:
        """Arm the native P0-P4 D2H-only audit for one explicit session."""
        if not _developer_mode_enabled() or not _writer_audit_allowed():
            return {
                "armed": False,
                "reason": "developer mode or FAULT26_WRITER_AUDIT_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }
        if not isinstance(session, int) or session <= 0:
            return {"armed": False, "reason": "session must be positive", "pid": os.getpid()}
        if not isinstance(max_qmm_windows, int) or not 1 <= max_qmm_windows <= 1_000_000:
            return {
                "armed": False,
                "reason": "max_qmm_windows must be in [1, 1000000]",
                "pid": os.getpid(),
            }
        if not isinstance(max_bytes, int) or not 1 <= max_bytes <= 1_073_741_824:
            return {
                "armed": False,
                "reason": "max_bytes must be in [1, 1073741824]",
                "pid": os.getpid(),
            }
        with _STATE_LOCK:
            os.environ["FAULT26_WRITER_AUDIT_SESSION"] = str(session)
            os.environ["FAULT26_WRITER_AUDIT_MAX_QMM_WINDOWS"] = str(max_qmm_windows)
            os.environ["FAULT26_WRITER_AUDIT_MAX_BYTES"] = str(max_bytes)
            os.environ["FAULT26_WRITER_AUDIT_ENABLED"] = "1"
        return {
            "armed": True,
            "pid": os.getpid(),
            "session": session,
            "max_qmm_windows": max_qmm_windows,
            "max_bytes": max_bytes,
            "device_memory_modified": False,
        }

    def fault26_query_allocator_snapshot(
        self, targets: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Map named device addresses to caching-allocator segment/blocks.

        This is a developer-mode, metadata-only query.  It returns allocator
        addresses, extents, states and requested sizes; it never reads or
        modifies tensor contents.
        """
        if not _developer_mode_enabled() or not _allocator_audit_allowed():
            return {
                "ok": False,
                "reason": "developer mode or FAULT26_ALLOCATOR_AUDIT_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }
        if not isinstance(targets, list) or not 1 <= len(targets) <= 128:
            return {
                "ok": False,
                "reason": "targets must be a list with 1..128 entries",
                "pid": os.getpid(),
            }

        def as_int(value: Any) -> int:
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                return int(value, 0)
            raise ValueError("address/bytes must be int or integer string")

        try:
            import torch_npu

            segments = torch_npu.npu.memory_snapshot()
            normalized = []
            for target in targets:
                name = str(target.get("name", ""))[:128]
                address = as_int(target.get("address"))
                size = as_int(target.get("bytes", 1))
                if address <= 0 or size <= 0:
                    raise ValueError("address and bytes must be positive")
                normalized.append((name, address, size))

            results: list[dict[str, Any]] = []
            for name, address, size in normalized:
                target_end = address + size
                match: dict[str, Any] | None = None
                for segment_index, segment in enumerate(segments):
                    segment_begin = int(segment.get("address", 0))
                    segment_size = int(segment.get("total_size", 0))
                    segment_end = segment_begin + segment_size
                    if not (segment_begin <= address < segment_end):
                        continue
                    block_begin = segment_begin
                    blocks = segment.get("blocks", [])
                    block_match: dict[str, Any] | None = None
                    for block_index, block in enumerate(blocks):
                        block_size = int(block.get("size", 0))
                        block_end = block_begin + block_size
                        if block_begin <= address < block_end:
                            neighbors = []
                            for neighbor_index in range(
                                max(0, block_index - 1),
                                min(len(blocks), block_index + 2),
                            ):
                                neighbor_begin = segment_begin + sum(
                                    int(item.get("size", 0))
                                    for item in blocks[:neighbor_index]
                                )
                                neighbor = blocks[neighbor_index]
                                neighbor_size = int(neighbor.get("size", 0))
                                neighbors.append(
                                    {
                                        "index": neighbor_index,
                                        "begin": f"0x{neighbor_begin:x}",
                                        "end": f"0x{neighbor_begin + neighbor_size:x}",
                                        "size": neighbor_size,
                                        "requested_size": int(
                                            neighbor.get("requested_size", 0)
                                        ),
                                        "state": str(neighbor.get("state", "")),
                                    }
                                )
                            block_match = {
                                "index": block_index,
                                "begin": f"0x{block_begin:x}",
                                "end": f"0x{block_end:x}",
                                "size": block_size,
                                "requested_size": int(
                                    block.get("requested_size", 0)
                                ),
                                "state": str(block.get("state", "")),
                                "address_offset": address - block_begin,
                                "target_fits_block": target_end <= block_end,
                                "neighbors": neighbors,
                            }
                            break
                        block_begin = block_end
                    match = {
                        "segment_index": segment_index,
                        "segment_begin": f"0x{segment_begin:x}",
                        "segment_end": f"0x{segment_end:x}",
                        "segment_size": segment_size,
                        "segment_device": segment.get("device"),
                        "segment_stream": str(segment.get("stream", "")),
                        "segment_type": str(segment.get("segment_type", "")),
                        "block": block_match,
                    }
                    break
                results.append(
                    {
                        "name": name,
                        "address": f"0x{address:x}",
                        "bytes": size,
                        "end": f"0x{target_end:x}",
                        "found": match is not None,
                        "allocation": match,
                    }
                )
            return {
                "ok": True,
                "pid": os.getpid(),
                "device": str(getattr(self, "device", "")),
                "segment_count": len(segments),
                "targets": results,
                "device_memory_modified": False,
            }
        except Exception as exc:
            return {
                "ok": False,
                "pid": os.getpid(),
                "reason": f"{type(exc).__name__}: {exc}",
                "device_memory_modified": False,
            }

    def fault26_arm_live_allocator_snapshot(
        self,
        targets: list[dict[str, Any]] | None = None,
        max_hits: int = 1,
        target_writer: str = "FIA",
        target_output_bytes: int = 0,
        target_residual_bytes: int = 0,
    ) -> dict[str, Any]:
        """Capture allocator blocks from inside a real FIA or QMM live window.

        The configured targets are augmented at capture time with the current
        AddRmsNormBias residual and the selected real writer output.  Only allocator
        metadata is inspected; tensor contents and device memory are untouched.
        """
        global _LIVE_ALLOCATOR_ARMED, _LIVE_ALLOCATOR_MAX_HITS
        global _LIVE_ALLOCATOR_HITS, _LIVE_ALLOCATOR_TARGETS
        global _LIVE_ALLOCATOR_CAPTURES, _LIVE_ALLOCATOR_MIN_RESIDUAL_GENERATION
        global _LIVE_ALLOCATOR_TARGET_WRITER
        global _LIVE_ALLOCATOR_TARGET_OUTPUT_BYTES
        global _LIVE_ALLOCATOR_TARGET_RESIDUAL_BYTES
        if not _developer_mode_enabled() or not _allocator_audit_allowed():
            return {
                "armed": False,
                "reason": "developer mode or FAULT26_ALLOCATOR_AUDIT_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }
        if targets is None:
            targets = []
        if not isinstance(targets, list) or len(targets) > 128:
            return {
                "armed": False,
                "reason": "targets must be a list with at most 128 entries",
                "pid": os.getpid(),
            }
        if not isinstance(max_hits, int) or not 1 <= max_hits <= 8:
            return {
                "armed": False,
                "reason": "max_hits must be in [1, 8]",
                "pid": os.getpid(),
            }
        target_writer = str(target_writer).upper()
        if target_writer not in {"FIA", "QMM"}:
            return {
                "armed": False,
                "reason": "target_writer must be FIA or QMM",
                "pid": os.getpid(),
            }
        if (
            not isinstance(target_output_bytes, int)
            or not 0 <= target_output_bytes <= 8_388_608
            or not isinstance(target_residual_bytes, int)
            or not 0 <= target_residual_bytes <= 8_388_608
        ):
            return {
                "armed": False,
                "reason": "target output/residual bytes must be in [0, 8388608]",
                "pid": os.getpid(),
            }

        normalized: list[dict[str, Any]] = []
        try:
            for target in targets:
                if not isinstance(target, dict):
                    raise ValueError("each target must be an object")
                name = str(target.get("name", ""))[:128]
                raw_address = target.get("address")
                raw_size = target.get("bytes", 1)
                address = (
                    raw_address
                    if isinstance(raw_address, int)
                    else int(str(raw_address), 0)
                )
                size = raw_size if isinstance(raw_size, int) else int(str(raw_size), 0)
                if not name or address <= 0 or size <= 0:
                    raise ValueError("name, address and bytes must be positive/non-empty")
                normalized.append({"name": name, "address": address, "bytes": size})
        except (TypeError, ValueError) as exc:
            return {
                "armed": False,
                "reason": f"invalid target: {exc}",
                "pid": os.getpid(),
            }

        with _STATE_LOCK:
            _LIVE_ALLOCATOR_ARMED = True
            _LIVE_ALLOCATOR_MAX_HITS = max_hits
            _LIVE_ALLOCATOR_HITS = 0
            _LIVE_ALLOCATOR_TARGETS = normalized
            _LIVE_ALLOCATOR_CAPTURES = []
            _LIVE_ALLOCATOR_MIN_RESIDUAL_GENERATION = _CURRENT_RESIDUAL_GENERATION
            _LIVE_ALLOCATOR_TARGET_WRITER = target_writer
            _LIVE_ALLOCATOR_TARGET_OUTPUT_BYTES = target_output_bytes
            _LIVE_ALLOCATOR_TARGET_RESIDUAL_BYTES = target_residual_bytes
        return {
            "armed": True,
            "pid": os.getpid(),
            "max_hits": max_hits,
            "target_writer": target_writer,
            "target_output_bytes": target_output_bytes,
            "target_residual_bytes": target_residual_bytes,
            "configured_target_count": len(normalized),
            "minimum_new_residual_generation": _LIVE_ALLOCATOR_MIN_RESIDUAL_GENERATION,
            "device_memory_modified": False,
        }

    def fault26_query_live_allocator_snapshot(self) -> dict[str, Any]:
        with _STATE_LOCK:
            captures = [dict(item) for item in _LIVE_ALLOCATOR_CAPTURES]
            return {
                "ok": True,
                "pid": os.getpid(),
                "armed": _LIVE_ALLOCATOR_ARMED,
                "hits": _LIVE_ALLOCATOR_HITS,
                "max_hits": _LIVE_ALLOCATOR_MAX_HITS,
                "target_writer": _LIVE_ALLOCATOR_TARGET_WRITER,
                "target_output_bytes": _LIVE_ALLOCATOR_TARGET_OUTPUT_BYTES,
                "target_residual_bytes": _LIVE_ALLOCATOR_TARGET_RESIDUAL_BYTES,
                "minimum_new_residual_generation": _LIVE_ALLOCATOR_MIN_RESIDUAL_GENERATION,
                "current_residual_generation": _CURRENT_RESIDUAL_GENERATION,
                "configured_targets": [dict(item) for item in _LIVE_ALLOCATOR_TARGETS],
                "captures": captures,
                "developer_mode": _developer_mode_enabled(),
                "allowed": _allocator_audit_allowed(),
                "device_memory_modified": False,
            }

    def fault26_disarm_live_allocator_snapshot(self) -> dict[str, Any]:
        global _LIVE_ALLOCATOR_ARMED
        with _STATE_LOCK:
            was_armed = _LIVE_ALLOCATOR_ARMED
            _LIVE_ALLOCATOR_ARMED = False
            hits = _LIVE_ALLOCATOR_HITS
            capture_count = len(_LIVE_ALLOCATOR_CAPTURES)
        return {
            "disarmed": True,
            "was_armed": was_armed,
            "pid": os.getpid(),
            "hits": hits,
            "capture_count": capture_count,
            "device_memory_modified": False,
        }

    def fault26_arm_qmm_output_base_underflow(
        self,
        target_device: int,
        output_address: int | str,
        output_bytes: int,
        residual_address: int | str,
        residual_bytes: int,
        residual_allocation_end: int | str,
        output_allocation_begin: int | str,
        shift_bytes: int,
        max_hits: int = 1,
        target_pairs: list[dict[str, Any]] | None = None,
        all_hits: bool = False,
    ) -> dict[str, Any]:
        """Arm bounded exact-address QMM allocation-boundary underflow targets."""
        if not _developer_mode_enabled() or os.environ.get(
            "FAULT26_QMM_UNDERFLOW_ALLOWED"
        ) != "1":
            return {
                "armed": False,
                "reason": "developer mode or FAULT26_QMM_UNDERFLOW_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }

        def integer(value: int | str) -> int:
            return value if isinstance(value, int) else int(value, 0)

        try:
            output_address_i = integer(output_address)
            residual_address_i = integer(residual_address)
            residual_allocation_end_i = integer(residual_allocation_end)
            output_allocation_begin_i = integer(output_allocation_begin)
        except (TypeError, ValueError) as exc:
            return {"armed": False, "reason": f"invalid address: {exc}", "pid": os.getpid()}
        checks = (
            (isinstance(target_device, int) and 0 <= target_device <= 65535,
             "target_device must be in [0, 65535]"),
            (1 <= output_bytes <= 8_388_608, "output_bytes must be in [1, 8388608]"),
            (1 <= residual_bytes <= 8_388_608,
             "residual_bytes must be in [1, 8388608]"),
            (1 <= shift_bytes <= 4096, "shift_bytes must be in [1, 4096]"),
            (isinstance(max_hits, int) and 1 <= max_hits <= 4096,
             "max_hits must be in [1, 4096]"),
            (isinstance(all_hits, bool), "all_hits must be boolean"),
            (output_address_i == output_allocation_begin_i,
             "output must begin at its proven allocator block boundary"),
            (residual_allocation_end_i == output_allocation_begin_i,
             "residual and output allocator blocks must be adjacent"),
            (residual_address_i + residual_bytes <= residual_allocation_end_i,
             "residual declared span exceeds its allocator block"),
            (shift_bytes > residual_allocation_end_i -
             (residual_address_i + residual_bytes),
             "shift does not cross padding into residual data"),
            (output_address_i > shift_bytes and
             output_address_i - shift_bytes >= residual_address_i,
             "replacement output start is outside the proven residual span"),
        )
        for valid, reason in checks:
            if not valid:
                return {"armed": False, "reason": reason, "pid": os.getpid()}

        normalized_pairs = [
            {
                "output_address": output_address_i,
                "residual_address": residual_address_i,
                "residual_allocation_end": residual_allocation_end_i,
                "output_allocation_begin": output_allocation_begin_i,
            }
        ]
        if target_pairs is not None:
            if not isinstance(target_pairs, list) or not 1 <= len(target_pairs) <= 8:
                return {
                    "armed": False,
                    "reason": "target_pairs must contain 1..8 proven pairs",
                    "pid": os.getpid(),
                }
            normalized_pairs = []
            try:
                for index, pair in enumerate(target_pairs):
                    if not isinstance(pair, dict):
                        raise ValueError(f"target_pairs[{index}] must be an object")
                    output_i = integer(pair["output_address"])
                    residual_i = integer(pair["residual_address"])
                    residual_end_i = integer(pair["residual_allocation_end"])
                    output_begin_i = integer(pair["output_allocation_begin"])
                    pair_checks = (
                        (output_i == output_begin_i,
                         "output must begin at its proven allocator block"),
                        (residual_end_i == output_begin_i,
                         "allocator blocks must be exactly adjacent"),
                        (residual_i + residual_bytes <= residual_end_i,
                         "residual declared span exceeds its allocator block"),
                        (shift_bytes > residual_end_i -
                         (residual_i + residual_bytes),
                         "shift does not cross padding into residual data"),
                        (output_i > shift_bytes and
                         output_i - shift_bytes >= residual_i,
                         "replacement output is outside the proven residual"),
                    )
                    for valid, reason in pair_checks:
                        if not valid:
                            raise ValueError(f"target_pairs[{index}]: {reason}")
                    normalized_pairs.append(
                        {
                            "output_address": output_i,
                            "residual_address": residual_i,
                            "residual_allocation_end": residual_end_i,
                            "output_allocation_begin": output_begin_i,
                        }
                    )
            except (KeyError, TypeError, ValueError) as exc:
                return {
                    "armed": False,
                    "reason": f"invalid target_pairs: {exc}",
                    "pid": os.getpid(),
                }
            unique_pairs = {
                (item["output_address"], item["residual_address"])
                for item in normalized_pairs
            }
            if len(unique_pairs) != len(normalized_pairs):
                return {
                    "armed": False,
                    "reason": "target_pairs contains duplicate address pairs",
                    "pid": os.getpid(),
                }

        pair_spec = ";".join(
            ",".join(
                hex(item[key])
                for key in (
                    "output_address",
                    "residual_address",
                    "residual_allocation_end",
                    "output_allocation_begin",
                )
            )
            for item in normalized_pairs
        ) if target_pairs is not None else ""

        with _STATE_LOCK:
            os.environ["FAULT26_QMM_UNDERFLOW_ENABLED"] = "0"
            os.environ["FAULT26_QMM_UNDERFLOW_TARGET_DEVICE"] = str(target_device)
            os.environ["FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_ADDRESS"] = hex(output_address_i)
            os.environ["FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_BYTES"] = str(output_bytes)
            os.environ["FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_ADDRESS"] = hex(residual_address_i)
            os.environ["FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_BYTES"] = str(residual_bytes)
            os.environ["FAULT26_QMM_UNDERFLOW_RESIDUAL_ALLOCATION_END"] = hex(
                residual_allocation_end_i
            )
            os.environ["FAULT26_QMM_UNDERFLOW_OUTPUT_ALLOCATION_BEGIN"] = hex(
                output_allocation_begin_i
            )
            os.environ["FAULT26_QMM_UNDERFLOW_SHIFT_BYTES"] = str(shift_bytes)
            os.environ["FAULT26_QMM_UNDERFLOW_MAX_HITS"] = str(max_hits)
            os.environ["FAULT26_QMM_UNDERFLOW_TARGET_PAIRS"] = pair_spec
            os.environ["FAULT26_QMM_UNDERFLOW_ALL_HITS"] = "1" if all_hits else "0"
            os.environ["FAULT26_QMM_UNDERFLOW_ENABLED"] = "1"
        return {
            "armed": True,
            "pid": os.getpid(),
            "target_device": target_device,
            "output_address": hex(output_address_i),
            "output_bytes": output_bytes,
            "residual_address": hex(residual_address_i),
            "residual_bytes": residual_bytes,
            "residual_allocation_end": hex(residual_allocation_end_i),
            "output_allocation_begin": hex(output_allocation_begin_i),
            "allocator_padding_bytes": residual_allocation_end_i
            - (residual_address_i + residual_bytes),
            "shift_bytes": shift_bytes,
            "max_hits": max_hits,
            "all_hits": all_hits,
            "target_pair_count": len(normalized_pairs),
            "target_pairs": [
                {key: hex(value) for key, value in item.items()}
                for item in normalized_pairs
            ],
            "residual_overlap_bytes": shift_bytes
            - (residual_allocation_end_i - (residual_address_i + residual_bytes)),
        }

    def fault26_disarm_qmm_output_base_underflow(self) -> dict[str, Any]:
        with _STATE_LOCK:
            was_armed = _process_getenv("FAULT26_QMM_UNDERFLOW_ENABLED") == "1"
            os.environ["FAULT26_QMM_UNDERFLOW_ENABLED"] = "0"
        return {"disarmed": True, "was_armed": was_armed, "pid": os.getpid()}

    def fault26_query_qmm_output_base_underflow(self) -> dict[str, Any]:
        keys = (
            "FAULT26_QMM_UNDERFLOW_TARGET_DEVICE",
            "FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_ADDRESS",
            "FAULT26_QMM_UNDERFLOW_TARGET_OUTPUT_BYTES",
            "FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_ADDRESS",
            "FAULT26_QMM_UNDERFLOW_TARGET_RESIDUAL_BYTES",
            "FAULT26_QMM_UNDERFLOW_RESIDUAL_ALLOCATION_END",
            "FAULT26_QMM_UNDERFLOW_OUTPUT_ALLOCATION_BEGIN",
            "FAULT26_QMM_UNDERFLOW_SHIFT_BYTES",
            "FAULT26_QMM_UNDERFLOW_MAX_HITS",
            "FAULT26_QMM_UNDERFLOW_TARGET_PAIRS",
            "FAULT26_QMM_UNDERFLOW_ALL_HITS",
        )
        return {
            "armed": _process_getenv("FAULT26_QMM_UNDERFLOW_ENABLED") == "1",
            "allowed": os.environ.get("FAULT26_QMM_UNDERFLOW_ALLOWED") == "1",
            "developer_mode": _developer_mode_enabled(),
            "pid": os.getpid(),
            "configuration": {key: _process_getenv(key) or "" for key in keys},
        }

    def fault26_disarm_residual_writer_audit(self) -> dict[str, Any]:
        with _STATE_LOCK:
            was_armed = os.environ.get("FAULT26_WRITER_AUDIT_ENABLED") == "1"
            os.environ["FAULT26_WRITER_AUDIT_ENABLED"] = "0"
            session = int(os.environ.get("FAULT26_WRITER_AUDIT_SESSION", "0"))
        return {
            "disarmed": True,
            "was_armed": was_armed,
            "pid": os.getpid(),
            "session": session,
        }

    def fault26_query_residual_writer_audit(self) -> dict[str, Any]:
        return {
            "armed": os.environ.get("FAULT26_WRITER_AUDIT_ENABLED") == "1",
            "allowed": _writer_audit_allowed(),
            "developer_mode": _developer_mode_enabled(),
            "pid": os.getpid(),
            "session": int(os.environ.get("FAULT26_WRITER_AUDIT_SESSION", "0")),
            "max_qmm_windows": int(
                os.environ.get("FAULT26_WRITER_AUDIT_MAX_QMM_WINDOWS", "4096")
            ),
            "max_bytes": int(os.environ.get("FAULT26_WRITER_AUDIT_MAX_BYTES", "4194304")),
            "device_memory_modified": False,
        }

    def fault26_arm_fia_lossless_output_substitution(
        self,
        session: int,
        target_device: int,
        target_bytes: int,
        target_occurrence: int = 1,
        max_bytes: int = 4194304,
    ) -> dict[str, Any]:
        """Arm one FIA output-to-temporary lossless substitution."""
        if not _developer_mode_enabled() or not _fia_writer_proxy_allowed():
            return {
                "armed": False,
                "reason": "developer mode or FAULT26_FIA_WRITER_PROXY_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }
        checks = (
            (isinstance(session, int) and session > 0, "session must be positive"),
            (
                isinstance(target_device, int) and 0 <= target_device <= 65535,
                "target_device must be in [0, 65535]",
            ),
            (
                isinstance(target_bytes, int) and 1 <= target_bytes <= 1_073_741_824,
                "target_bytes must be in [1, 1073741824]",
            ),
            (
                isinstance(target_occurrence, int)
                and 1 <= target_occurrence <= 1_000_000,
                "target_occurrence must be in [1, 1000000]",
            ),
            (
                isinstance(max_bytes, int) and 1 <= max_bytes <= 1_073_741_824,
                "max_bytes must be in [1, 1073741824]",
            ),
            (target_bytes <= max_bytes, "target_bytes exceeds max_bytes"),
        )
        for valid, reason in checks:
            if not valid:
                return {"armed": False, "reason": reason, "pid": os.getpid()}
        with _STATE_LOCK:
            os.environ["FAULT26_FIA_WRITER_PROXY_ENABLED"] = "0"
            os.environ["FAULT26_FIA_WRITER_PROXY_MODE"] = (
                "lossless_output_substitution"
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_SESSION"] = str(session)
            os.environ["FAULT26_FIA_WRITER_PROXY_TARGET_DEVICE"] = str(target_device)
            os.environ["FAULT26_FIA_WRITER_PROXY_TARGET_BYTES"] = str(target_bytes)
            os.environ["FAULT26_FIA_WRITER_PROXY_TARGET_OCCURRENCE"] = str(
                target_occurrence
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_MAX_BYTES"] = str(max_bytes)
            os.environ["FAULT26_FIA_WRITER_PROXY_ENABLED"] = "1"
        return {
            "armed": True,
            "mode": "lossless_output_substitution",
            "pid": os.getpid(),
            "session": session,
            "target_device": target_device,
            "target_bytes": target_bytes,
            "target_occurrence": target_occurrence,
            "max_bytes": max_bytes,
            "max_hits": 1,
            "residual_target_supported": False,
        }

    def fault26_arm_fia_wrong_destination_residual_stomp(
        self,
        session: int,
        target_device: int,
        target_output_bytes: int,
        target_residual_bytes: int,
        target_occurrence: int = 1,
        victim_offset_bytes: int = 0,
        max_bytes: int = 4194304,
    ) -> dict[str, Any]:
        """Arm one bounded FIA output binding to a proven live residual."""
        if not _developer_mode_enabled() or not _fia_writer_proxy_allowed():
            return {
                "armed": False,
                "reason": "developer mode or FAULT26_FIA_WRITER_PROXY_ALLOWED gate is disabled",
                "pid": os.getpid(),
            }
        checks = (
            (isinstance(session, int) and session > 0, "session must be positive"),
            (
                isinstance(target_device, int) and 0 <= target_device <= 65535,
                "target_device must be in [0, 65535]",
            ),
            (
                isinstance(target_output_bytes, int)
                and 1 <= target_output_bytes <= 1_073_741_824,
                "target_output_bytes must be in [1, 1073741824]",
            ),
            (
                isinstance(target_residual_bytes, int)
                and 1 <= target_residual_bytes <= 1_073_741_824,
                "target_residual_bytes must be in [1, 1073741824]",
            ),
            (
                isinstance(target_occurrence, int)
                and 1 <= target_occurrence <= 1_000_000,
                "target_occurrence must be in [1, 1000000]",
            ),
            (
                isinstance(victim_offset_bytes, int)
                and 0 <= victim_offset_bytes <= 1_073_741_824,
                "victim_offset_bytes must be in [0, 1073741824]",
            ),
            (
                isinstance(max_bytes, int) and 1 <= max_bytes <= 1_073_741_824,
                "max_bytes must be in [1, 1073741824]",
            ),
            (
                isinstance(target_output_bytes, int)
                and isinstance(target_residual_bytes, int)
                and isinstance(victim_offset_bytes, int)
                and victim_offset_bytes <= target_residual_bytes
                and target_output_bytes
                <= target_residual_bytes - victim_offset_bytes,
                "output binding does not fit inside residual storage",
            ),
            (
                target_residual_bytes <= max_bytes,
                "target_residual_bytes exceeds max_bytes",
            ),
        )
        for valid, reason in checks:
            if not valid:
                return {"armed": False, "reason": reason, "pid": os.getpid()}
        with _STATE_LOCK:
            os.environ["FAULT26_FIA_WRITER_PROXY_ENABLED"] = "0"
            os.environ["FAULT26_FIA_WRITER_PROXY_MODE"] = (
                "wrong_destination_residual_stomp"
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_SESSION"] = str(session)
            os.environ["FAULT26_FIA_WRITER_PROXY_TARGET_DEVICE"] = str(
                target_device
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_TARGET_BYTES"] = str(
                target_output_bytes
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_TARGET_RESIDUAL_BYTES"] = str(
                target_residual_bytes
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_TARGET_OCCURRENCE"] = str(
                target_occurrence
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_VICTIM_OFFSET_BYTES"] = str(
                victim_offset_bytes
            )
            os.environ["FAULT26_FIA_WRITER_PROXY_MAX_BYTES"] = str(max_bytes)
            os.environ["FAULT26_FIA_WRITER_PROXY_ENABLED"] = "1"
        return {
            "armed": True,
            "mode": "wrong_destination_residual_stomp",
            "pid": os.getpid(),
            "session": session,
            "target_device": target_device,
            "target_output_bytes": target_output_bytes,
            "target_residual_bytes": target_residual_bytes,
            "target_occurrence": target_occurrence,
            "victim_offset_bytes": victim_offset_bytes,
            "max_bytes": max_bytes,
            "max_hits": 1,
            "writer": "real_aclnnFusedInferAttentionScoreV3_kernel",
        }

    def fault26_disarm_fia_lossless_output_substitution(self) -> dict[str, Any]:
        with _STATE_LOCK:
            was_armed = os.environ.get("FAULT26_FIA_WRITER_PROXY_ENABLED") == "1"
            os.environ["FAULT26_FIA_WRITER_PROXY_ENABLED"] = "0"
            session = int(os.environ.get("FAULT26_FIA_WRITER_PROXY_SESSION", "0"))
        return {
            "disarmed": True,
            "was_armed": was_armed,
            "pid": os.getpid(),
            "session": session,
        }

    def fault26_query_fia_lossless_output_substitution(self) -> dict[str, Any]:
        mode = os.environ.get(
            "FAULT26_FIA_WRITER_PROXY_MODE", "lossless_output_substitution"
        )
        return {
            "armed": os.environ.get("FAULT26_FIA_WRITER_PROXY_ENABLED") == "1",
            "allowed": _fia_writer_proxy_allowed(),
            "developer_mode": _developer_mode_enabled(),
            "mode": mode,
            "pid": os.getpid(),
            "session": int(os.environ.get("FAULT26_FIA_WRITER_PROXY_SESSION", "0")),
            "target_device": int(
                os.environ.get("FAULT26_FIA_WRITER_PROXY_TARGET_DEVICE", "0")
            ),
            "target_bytes": int(
                os.environ.get("FAULT26_FIA_WRITER_PROXY_TARGET_BYTES", "0")
            ),
            "target_occurrence": int(
                os.environ.get("FAULT26_FIA_WRITER_PROXY_TARGET_OCCURRENCE", "1")
            ),
            "max_bytes": int(
                os.environ.get("FAULT26_FIA_WRITER_PROXY_MAX_BYTES", "4194304")
            ),
            "max_hits": 1,
            "residual_target_supported": (
                mode == "wrong_destination_residual_stomp"
            ),
            "target_residual_bytes": int(
                os.environ.get(
                    "FAULT26_FIA_WRITER_PROXY_TARGET_RESIDUAL_BYTES", "0"
                )
            ),
            "victim_offset_bytes": int(
                os.environ.get(
                    "FAULT26_FIA_WRITER_PROXY_VICTIM_OFFSET_BYTES", "0"
                )
            ),
        }
