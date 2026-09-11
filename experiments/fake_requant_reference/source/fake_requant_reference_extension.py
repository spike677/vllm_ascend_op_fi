"""Developer-mode Torch-module fake-requant reference experiment.

This module is intentionally experiment-local.  It is not a DemonCAT runtime,
capability registry, production CLI, or direct ``torch_npu`` patch.  When used
as a vLLM worker extension, its methods are available only through vLLM's
developer-mode collective RPC surface.
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


RESEARCH_REFERENCE_ONLY = "RESEARCH_REFERENCE_ONLY"
NOT_A_CANONICAL_CAPABILITY = "NOT_A_CANONICAL_CAPABILITY"

IDENTITY = "IDENTITY"
REFERENCE_REQUANT = "REFERENCE_REQUANT"
_SUPPORTED_MODES = frozenset({IDENTITY, REFERENCE_REQUANT})
_SUPPORTED_SELECTORS = frozenset({"first_float", "tensor_index"})
_SUPPORTED_SCALE_PROVIDERS = frozenset(
    {"fixed", "tensor_absmax", "module_runtime_scale", "per_token_absmax"}
)


class ReferenceConfigurationError(ValueError):
    """Raised for an invalid experiment control request."""


class ReferenceTransformError(RuntimeError):
    """Raised when one hook invocation must fail closed."""


@dataclass(frozen=True)
class ScalarTransformStats:
    values: tuple[float, ...]
    changed_elements: int
    clip_count: int
    mean_abs_error: float
    max_abs_error: float


def _finite_positive(value: Any, *, name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceConfigurationError(f"{name} must be numeric") from exc
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ReferenceConfigurationError(f"{name} must be finite and > 0")
    return numeric


def reference_requant_scalars(
    values: Iterable[float],
    *,
    base_scale: float,
    scale_multiplier: float = 1.0,
    eps: float = 1e-12,
) -> ScalarTransformStats:
    """Pure-Python signed symmetric INT8 quantize/dequantize reference.

    This helper exists for device-independent unit tests.  PyTorch's and
    Python's round-to-nearest-even behavior agree for the values exercised by
    the tests; model execution uses the corresponding tensor operation below.
    """

    base = _finite_positive(base_scale, name="base_scale")
    multiplier = _finite_positive(scale_multiplier, name="scale_multiplier")
    epsilon = _finite_positive(eps, name="eps")
    fault_scale = max(base * multiplier, epsilon)
    original = tuple(float(item) for item in values)
    if not original:
        raise ReferenceTransformError("empty tensor/value sequence")
    if not all(math.isfinite(item) for item in original):
        raise ReferenceTransformError("non-finite input")

    output: list[float] = []
    clip_count = 0
    errors: list[float] = []
    changed = 0
    for value in original:
        unbounded = round(value / fault_scale)
        quantized = min(127, max(-128, unbounded))
        if quantized != unbounded:
            clip_count += 1
        restored = quantized * fault_scale
        output.append(restored)
        error = abs(restored - value)
        errors.append(error)
        if restored != value:
            changed += 1
    return ScalarTransformStats(
        values=tuple(output),
        changed_elements=changed,
        clip_count=clip_count,
        mean_abs_error=sum(errors) / len(errors),
        max_abs_error=max(errors),
    )


def _developer_mode_enabled() -> bool:
    return os.environ.get("VLLM_SERVER_DEV_MODE") == "1"


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised on target workers.
        raise ReferenceTransformError("torch is unavailable") from exc
    return torch


def _module_layer(name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match is not None else None


def _normalize_layers(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)):
        raise ReferenceConfigurationError("layer_filter must be a list of non-negative integers")
    layers: list[int] = []
    for item in value:
        if not isinstance(item, int) or item < 0:
            raise ReferenceConfigurationError("layer_filter must be a list of non-negative integers")
        layers.append(item)
    return tuple(sorted(set(layers)))


def _module_metadata(name: str, module: Any, layer: int | None) -> dict[str, Any]:
    return {
        "module": name,
        "layer": layer,
        "module_type": f"{type(module).__module__}.{type(module).__qualname__}",
        "tp_rank": getattr(module, "tp_rank", None),
        "tp_size": getattr(module, "tp_size", None),
        "reduce_results": getattr(module, "reduce_results", None),
    }


def _resolve_worker_model(worker: Any) -> Any | None:
    """Resolve the model without importing or depending on vLLM internals."""

    candidates = (
        (worker, "model_runner", "model"),
        (getattr(worker, "worker", None), "model_runner", "model"),
        (getattr(worker, "model_runner", None), "model"),
    )
    for root, *attrs in candidates:
        current = root
        for attr in attrs:
            current = getattr(current, attr, None)
        if current is not None and callable(getattr(current, "named_modules", None)):
            return current
    return None


def _rank_for(module: Any) -> tuple[int | None, str]:
    for env_name in ("RANK", "LOCAL_RANK", "VLLM_RANK"):
        raw = os.environ.get(env_name)
        if raw is None:
            continue
        try:
            return int(raw), f"env:{env_name}"
        except ValueError:
            continue
    module_rank = getattr(module, "tp_rank", None)
    if isinstance(module_rank, int):
        return module_rank, "module:tp_rank"
    return None, "unresolved"


@dataclass(frozen=True)
class ArmConfig:
    mode: str
    module_regex: str
    layer_filter: tuple[int, ...] | None
    rank_filter: tuple[int, ...] | None
    tensor_selector: str
    tensor_index: int | None
    kwarg_name: str | None
    scale_provider: str
    scale_value: float | None
    runtime_scale_attr: str | None
    scale_multiplier: float
    max_hits: int
    interval: int
    request_generation: int | None
    eps: float


@dataclass
class HookBinding:
    module_name: str
    module: Any
    layer: int | None
    handle: Any
    metadata: dict[str, Any]


@dataclass
class ExperimentState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    bindings: list[HookBinding] = field(default_factory=list)
    install_pattern: str | None = None
    config: ArmConfig | None = None
    enabled: bool = False
    hook_seen: int = 0
    candidate_seen: int = 0
    hit_count: int = 0
    reject_reasons: Counter[str] = field(default_factory=Counter)
    per_module: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    per_rank: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    last_tensor_effect: dict[str, dict[str, Any]] = field(default_factory=dict)
    tensor_effect_history: list[dict[str, Any]] = field(default_factory=list)


class FakeRequantReferenceExtension:
    """Developer-mode collective-RPC extension for an experiment-local hook."""

    @property
    def _state(self) -> ExperimentState:
        """Lazily attach state to the vLLM worker/extension instance.

        Worker-extension loading may bind methods onto a worker rather than
        construct this class in the ordinary Python way. Avoid relying on an
        extension ``__init__`` being called.
        """

        state = getattr(self, "_fake_requant_reference_state", None)
        if state is None:
            state = ExperimentState()
            setattr(self, "_fake_requant_reference_state", state)
        return state

    def fake_requant_install(
        self,
        module_regex: str = r"(?:^|\.)mlp\.down_proj$",
        layer_filter: list[int] | None = None,
    ) -> dict[str, Any]:
        """Resolve modules and install disabled forward-pre-hooks."""

        if not _developer_mode_enabled():
            return self._result(False, reason="VLLM_SERVER_DEV_MODE=1 is required")
        try:
            pattern = re.compile(module_regex)
            layers = _normalize_layers(layer_filter)
        except (re.error, ReferenceConfigurationError) as exc:
            return self._result(False, reason=str(exc))
        model = _resolve_worker_model(self)
        if model is None:
            return self._result(False, reason="worker model_runner.model is unavailable")

        with self._state.lock:
            if self._state.bindings:
                return self._result(
                    True,
                    already_installed=True,
                    matched_modules=[binding.metadata for binding in self._state.bindings],
                )

            bindings: list[HookBinding] = []
            try:
                for name, module in model.named_modules():
                    layer = _module_layer(name)
                    if pattern.search(name) is None or (layers is not None and layer not in layers):
                        continue
                    handle = module.register_forward_pre_hook(
                        self._make_pre_hook(name, module, layer), with_kwargs=True
                    )
                    bindings.append(
                        HookBinding(name, module, layer, handle, _module_metadata(name, module, layer))
                    )
            except Exception as exc:
                for binding in bindings:
                    binding.handle.remove()
                return self._result(False, reason=f"hook installation failed: {type(exc).__name__}: {exc}")
            if not bindings:
                return self._result(False, reason="no modules matched module_regex/layer_filter")
            self._state.bindings = bindings
            self._state.install_pattern = module_regex
            self._state.config = None
            self._state.enabled = False
            self._reset_counters_locked()
        return self._result(
            True,
            already_installed=False,
            matched_modules=[binding.metadata for binding in bindings],
        )

    def fake_requant_arm(
        self,
        mode: str,
        module_regex: str = r"(?:^|\.)mlp\.down_proj$",
        layer_filter: list[int] | None = None,
        rank_filter: list[int] | None = None,
        tensor_selector: str = "first_float",
        scale_provider: str = "tensor_absmax",
        scale_multiplier: float = 1.0,
        max_hits: int = 1,
        interval: int = 1,
        tensor_index: int | None = None,
        runtime_scale_attr: str | None = None,
        fixed_scale: float | None = None,
        request_generation: int | None = None,
        eps: float = 1e-12,
    ) -> dict[str, Any]:
        """Arm IDENTITY or REFERENCE_REQUANT over an installed module subset."""

        if not _developer_mode_enabled():
            return self._result(False, reason="VLLM_SERVER_DEV_MODE=1 is required")
        try:
            config = self._make_config(
                mode=mode,
                module_regex=module_regex,
                layer_filter=layer_filter,
                rank_filter=rank_filter,
                tensor_selector=tensor_selector,
                scale_provider=scale_provider,
                scale_multiplier=scale_multiplier,
                max_hits=max_hits,
                interval=interval,
                tensor_index=tensor_index,
                runtime_scale_attr=runtime_scale_attr,
                fixed_scale=fixed_scale,
                request_generation=request_generation,
                eps=eps,
            )
            pattern = re.compile(config.module_regex)
        except (re.error, ReferenceConfigurationError) as exc:
            return self._result(False, reason=str(exc))

        with self._state.lock:
            if not self._state.bindings:
                return self._result(False, reason="install must succeed before arm")
            matched = [
                binding.metadata
                for binding in self._state.bindings
                if pattern.search(binding.module_name) is not None
                and (config.layer_filter is None or binding.layer in config.layer_filter)
            ]
            if not matched:
                return self._result(False, reason="arm target does not match any installed module")
            self._state.config = config
            self._state.enabled = True
            self._reset_counters_locked()
        return self._result(True, armed_config=dataclasses.asdict(config), armed_modules=matched)

    def fake_requant_disarm(self) -> dict[str, Any]:
        """Disable mutation without removing the installed hooks."""

        with self._state.lock:
            was_enabled = self._state.enabled
            self._state.enabled = False
            self._state.config = None
        return self._result(True, disarmed=True, was_enabled=was_enabled)

    def fake_requant_remove(self) -> dict[str, Any]:
        """Remove hook handles; no model restart is required."""

        with self._state.lock:
            bindings = list(self._state.bindings)
            self._state.bindings = []
            self._state.install_pattern = None
            self._state.config = None
            self._state.enabled = False
        failures: list[str] = []
        for binding in bindings:
            try:
                binding.handle.remove()
            except Exception as exc:  # pragma: no cover - defensive worker cleanup.
                failures.append(f"{binding.module_name}: {type(exc).__name__}")
        return self._result(
            not failures,
            removed=True,
            removed_modules=[binding.metadata for binding in bindings],
            remove_failures=failures,
        )

    def fake_requant_query(self) -> dict[str, Any]:
        with self._state.lock:
            return {
                **self._base_metadata(),
                "installed": bool(self._state.bindings),
                "matched_modules": [binding.metadata for binding in self._state.bindings],
                "enabled": self._state.enabled,
                "armed_config": None
                if self._state.config is None
                else dataclasses.asdict(self._state.config),
                "hook_seen": self._state.hook_seen,
                "candidate_seen": self._state.candidate_seen,
                "hit_count": self._state.hit_count,
                "budget_exhausted": bool(
                    self._state.config is not None
                    and self._state.hit_count >= self._state.config.max_hits
                ),
                "reject_count": sum(self._state.reject_reasons.values()),
                "reject_reasons": dict(sorted(self._state.reject_reasons.items())),
                "per_module": {
                    key: dict(sorted(value.items())) for key, value in sorted(self._state.per_module.items())
                },
                "per_rank": {
                    key: dict(sorted(value.items())) for key, value in sorted(self._state.per_rank.items())
                },
                "last_tensor_effect": dict(self._state.last_tensor_effect),
                "tensor_effect_history": list(self._state.tensor_effect_history),
            }

    def _make_pre_hook(self, name: str, module: Any, layer: int | None):
        def pre_hook(_module: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]):
            return self._apply_pre_hook(name, module, layer, args, kwargs)

        return pre_hook

    def _apply_pre_hook(
        self,
        name: str,
        module: Any,
        layer: int | None,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        with self._state.lock:
            self._state.hook_seen += 1
            self._state.per_module[name]["hook_seen"] += 1
            config = self._state.config
            enabled = self._state.enabled
        if not enabled or config is None:
            return None

        rank, rank_source = _rank_for(module)
        rank_key = "unresolved" if rank is None else str(rank)
        with self._state.lock:
            self._state.per_rank[rank_key]["hook_seen"] += 1
        if config.rank_filter is not None and rank not in config.rank_filter:
            self._reject(name, rank_key, "rank_filter")
            return None
        if re.search(config.module_regex, name) is None:
            self._reject(name, rank_key, "module_regex")
            return None
        if config.layer_filter is not None and layer not in config.layer_filter:
            self._reject(name, rank_key, "layer_filter")
            return None

        try:
            location, tensor = self._select_tensor(args, kwargs, config)
        except ReferenceTransformError as exc:
            self._reject(name, rank_key, str(exc))
            return None
        with self._state.lock:
            self._state.candidate_seen += 1
            self._state.per_module[name]["candidate_seen"] += 1
            candidate_index = self._state.candidate_seen
            if (candidate_index - 1) % config.interval != 0:
                self._reject_locked(name, rank_key, "interval")
                return None
            if self._state.hit_count >= config.max_hits:
                self._reject_locked(name, rank_key, "max_hits")
                return None
            hit_index = self._state.hit_count + 1

        if config.mode == IDENTITY:
            effect = self._identity_effect(name, layer, rank, rank_source, hit_index, tensor, config)
            with self._state.lock:
                self._state.hit_count += 1
                self._state.per_module[name]["hit_count"] += 1
                self._state.per_rank[rank_key]["hit_count"] += 1
                self._state.last_tensor_effect[name] = effect
                self._state.tensor_effect_history.append(effect)
            return None

        try:
            faulty, effect = self._requant_tensor(
                module, tensor, name, layer, rank, rank_source, hit_index, config
            )
        except ReferenceTransformError as exc:
            self._reject(name, rank_key, str(exc))
            return None
        with self._state.lock:
            if self._state.hit_count >= config.max_hits:
                self._reject_locked(name, rank_key, "max_hits_race")
                return None
            self._state.hit_count += 1
            self._state.per_module[name]["hit_count"] += 1
            self._state.per_rank[rank_key]["hit_count"] += 1
            self._state.last_tensor_effect[name] = effect
            self._state.tensor_effect_history.append(effect)
        return self._replace_tensor(args, kwargs, location, faulty)

    def _select_tensor(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any], config: ArmConfig
    ) -> tuple[tuple[str, int | str], Any]:
        torch = _import_torch()
        selector = config.tensor_selector
        if selector == "tensor_index":
            assert config.tensor_index is not None
            if config.tensor_index >= len(args):
                raise ReferenceTransformError("tensor_index_out_of_range")
            tensor = args[config.tensor_index]
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise ReferenceTransformError("tensor_index_not_floating_tensor")
            return ("args", config.tensor_index), tensor
        if selector.startswith("kwarg:"):
            key = selector.split(":", 1)[1]
            tensor = kwargs.get(key)
            if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
                raise ReferenceTransformError("kwarg_not_floating_tensor")
            return ("kwargs", key), tensor
        if selector != "first_float":
            raise ReferenceTransformError("unsupported_tensor_selector")
        candidates: list[tuple[tuple[str, int | str], Any]] = []
        for index, value in enumerate(args):
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                candidates.append((("args", index), value))
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                candidates.append((("kwargs", key), value))
        if not candidates:
            raise ReferenceTransformError("no_floating_tensor")
        if len(candidates) != 1:
            raise ReferenceTransformError("ambiguous_float_tensor")
        return candidates[0]

    def _scale_for(self, module: Any, tensor: Any, config: ArmConfig) -> Any:
        torch = _import_torch()
        if tensor.numel() == 0:
            raise ReferenceTransformError("empty_tensor")
        if not bool(torch.isfinite(tensor).all().item()):
            raise ReferenceTransformError("non_finite_input")
        if config.scale_provider == "fixed":
            assert config.scale_value is not None
            base = torch.as_tensor(config.scale_value, device=tensor.device, dtype=torch.float32)
        elif config.scale_provider == "tensor_absmax":
            base = tensor.detach().abs().amax().to(dtype=torch.float32) / 127.0
        elif config.scale_provider == "per_token_absmax":
            if tensor.ndim < 1 or tensor.shape[-1] == 0:
                raise ReferenceTransformError("per_token_requires_nonempty_last_dimension")
            base = tensor.detach().abs().amax(dim=-1, keepdim=True).to(dtype=torch.float32) / 127.0
        elif config.scale_provider == "module_runtime_scale":
            assert config.runtime_scale_attr is not None
            raw = getattr(module, config.runtime_scale_attr, None)
            if raw is None:
                raise ReferenceTransformError("module_runtime_scale_unresolved")
            if isinstance(raw, torch.Tensor):
                base = raw.detach().to(device=tensor.device, dtype=torch.float32)
            elif isinstance(raw, (int, float)):
                base = torch.as_tensor(raw, device=tensor.device, dtype=torch.float32)
            else:
                raise ReferenceTransformError("module_runtime_scale_unresolved")
        else:  # guarded by _make_config; retained for defensive fail-closed behavior.
            raise ReferenceTransformError("unsupported_scale_provider")
        if not bool(torch.isfinite(base).all().item()):
            raise ReferenceTransformError("non_finite_scale")
        return base.clamp_min(config.eps) * config.scale_multiplier

    def _requant_tensor(
        self,
        module: Any,
        tensor: Any,
        module_name: str,
        layer: int | None,
        rank: int | None,
        rank_source: str,
        hit_index: int,
        config: ArmConfig,
    ) -> tuple[Any, dict[str, Any]]:
        torch = _import_torch()
        fault_scale = self._scale_for(module, tensor, config)
        working = tensor.detach().to(dtype=torch.float32)
        unbounded = torch.round(working / fault_scale)
        quantized = unbounded.clamp(-128, 127)
        faulty = (quantized * fault_scale).to(dtype=tensor.dtype)
        if faulty.shape != tensor.shape or faulty.dtype != tensor.dtype:
            raise ReferenceTransformError("dtype_or_shape_preservation_failed")
        difference = (faulty - tensor).abs()
        changed = torch.count_nonzero(faulty != tensor)
        clipped = torch.count_nonzero((unbounded < -128) | (unbounded > 127))
        total = tensor.numel()
        effect = {
            "mode": REFERENCE_REQUANT,
            "module": module_name,
            "layer": layer,
            "rank": rank,
            "rank_source": rank_source,
            "hit_index": hit_index,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "scale_provider": config.scale_provider,
            "scale_multiplier": config.scale_multiplier,
            "scale_min": float(fault_scale.min().item()),
            "scale_max": float(fault_scale.max().item()),
            "scale_mean": float(fault_scale.float().mean().item()),
            "input_max_abs": float(working.abs().max().item()),
            "changed_elements": int(changed.item()),
            "total_elements": int(total),
            "changed_rate": float(changed.item()) / total,
            "clip_count": int(clipped.item()),
            "clip_rate": float(clipped.item()) / total,
            "mean_abs_error": float(difference.float().mean().item()),
            "max_abs_error": float(difference.max().item()),
            "output_max_abs": float(faulty.detach().abs().max().item()),
        }
        return faulty, effect

    def _identity_effect(
        self,
        module_name: str,
        layer: int | None,
        rank: int | None,
        rank_source: str,
        hit_index: int,
        tensor: Any,
        config: ArmConfig,
    ) -> dict[str, Any]:
        return {
            "mode": IDENTITY,
            "module": module_name,
            "layer": layer,
            "rank": rank,
            "rank_source": rank_source,
            "hit_index": hit_index,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "scale_provider": config.scale_provider,
            "scale_multiplier": config.scale_multiplier,
            "changed_elements": 0,
            "total_elements": int(tensor.numel()),
            "changed_rate": 0.0,
            "clip_count": 0,
            "clip_rate": 0.0,
            "mean_abs_error": 0.0,
            "max_abs_error": 0.0,
            "identity_returns_original_tensor": True,
        }

    @staticmethod
    def _replace_tensor(
        args: tuple[Any, ...], kwargs: Mapping[str, Any], location: tuple[str, int | str], tensor: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        destination, key = location
        if destination == "args":
            replaced = list(args)
            assert isinstance(key, int)
            replaced[key] = tensor
            return tuple(replaced), dict(kwargs)
        replaced_kwargs = dict(kwargs)
        assert isinstance(key, str)
        replaced_kwargs[key] = tensor
        return args, replaced_kwargs

    def _make_config(self, **kwargs: Any) -> ArmConfig:
        mode = kwargs["mode"]
        if mode not in _SUPPORTED_MODES:
            raise ReferenceConfigurationError(f"mode must be one of {sorted(_SUPPORTED_MODES)}")
        selector = kwargs["tensor_selector"]
        index = kwargs["tensor_index"]
        kwarg_name: str | None = None
        if selector.startswith("kwarg:"):
            kwarg_name = selector.split(":", 1)[1]
            if not kwarg_name:
                raise ReferenceConfigurationError("kwarg selector requires a name")
        elif selector not in _SUPPORTED_SELECTORS:
            raise ReferenceConfigurationError("unsupported tensor_selector")
        if selector == "tensor_index":
            if not isinstance(index, int) or index < 0:
                raise ReferenceConfigurationError("tensor_index requires a non-negative integer")
        elif index is not None:
            raise ReferenceConfigurationError("tensor_index is only valid with tensor_index selector")
        provider = kwargs["scale_provider"]
        if provider not in _SUPPORTED_SCALE_PROVIDERS:
            raise ReferenceConfigurationError("unsupported scale_provider")
        fixed_scale = kwargs["fixed_scale"]
        if provider == "fixed":
            fixed_scale = _finite_positive(fixed_scale, name="fixed_scale")
        elif fixed_scale is not None:
            raise ReferenceConfigurationError("fixed_scale is only valid with fixed provider")
        runtime_scale_attr = kwargs["runtime_scale_attr"]
        if provider == "module_runtime_scale":
            if not isinstance(runtime_scale_attr, str) or not runtime_scale_attr:
                raise ReferenceConfigurationError("module_runtime_scale requires runtime_scale_attr")
        elif runtime_scale_attr is not None:
            raise ReferenceConfigurationError("runtime_scale_attr is only valid with module_runtime_scale")
        max_hits = kwargs["max_hits"]
        interval = kwargs["interval"]
        if not isinstance(max_hits, int) or max_hits < 1:
            raise ReferenceConfigurationError("max_hits must be a positive integer")
        if not isinstance(interval, int) or interval < 1:
            raise ReferenceConfigurationError("interval must be a positive integer")
        request_generation = kwargs["request_generation"]
        if request_generation is not None and (
            not isinstance(request_generation, int) or request_generation < 1
        ):
            raise ReferenceConfigurationError("request_generation must be a positive integer or null")
        ranks = _normalize_layers(kwargs["rank_filter"])
        return ArmConfig(
            mode=mode,
            module_regex=str(kwargs["module_regex"]),
            layer_filter=_normalize_layers(kwargs["layer_filter"]),
            rank_filter=ranks,
            tensor_selector=selector,
            tensor_index=index,
            kwarg_name=kwarg_name,
            scale_provider=provider,
            scale_value=fixed_scale,
            runtime_scale_attr=runtime_scale_attr,
            scale_multiplier=_finite_positive(kwargs["scale_multiplier"], name="scale_multiplier"),
            max_hits=max_hits,
            interval=interval,
            request_generation=request_generation,
            eps=_finite_positive(kwargs["eps"], name="eps"),
        )

    def _reject(self, module_name: str, rank_key: str, reason: str) -> None:
        with self._state.lock:
            self._reject_locked(module_name, rank_key, reason)

    def _reject_locked(self, module_name: str, rank_key: str, reason: str) -> None:
        self._state.reject_reasons[reason] += 1
        self._state.per_module[module_name][f"reject:{reason}"] += 1
        self._state.per_rank[rank_key][f"reject:{reason}"] += 1

    def _reset_counters_locked(self) -> None:
        self._state.hook_seen = 0
        self._state.candidate_seen = 0
        self._state.hit_count = 0
        self._state.reject_reasons.clear()
        self._state.per_module.clear()
        self._state.per_rank.clear()
        self._state.last_tensor_effect.clear()
        self._state.tensor_effect_history.clear()

    def _base_metadata(self) -> dict[str, Any]:
        return {
            "research_status": RESEARCH_REFERENCE_ONLY,
            "canonical_capability": NOT_A_CANONICAL_CAPABILITY,
            "developer_mode": _developer_mode_enabled(),
            "pid": os.getpid(),
        }

    def _result(self, ok: bool, **payload: Any) -> dict[str, Any]:
        return {**self._base_metadata(), "ok": ok, **payload}


__all__ = [
    "FakeRequantReferenceExtension",
    "IDENTITY",
    "NOT_A_CANONICAL_CAPABILITY",
    "REFERENCE_REQUANT",
    "RESEARCH_REFERENCE_ONLY",
    "ReferenceConfigurationError",
    "ReferenceTransformError",
    "ScalarTransformStats",
    "reference_requant_scalars",
]
