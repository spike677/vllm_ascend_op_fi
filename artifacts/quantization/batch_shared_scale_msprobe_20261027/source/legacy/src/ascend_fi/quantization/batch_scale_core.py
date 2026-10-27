"""Paired INT8/FP32 reference for a controlled change of reduction scope.

This core reuses the pinned FakeRequant scale providers, but returns an actual
INT8 payload with its matching per-token scale. It does not patch torch_npu or
claim equivalence to a native quantizer. The caller must calibrate the healthy
pair against that quantizer before using a shared pair in a real consumer.

Invalid rows retain *reference* per-token values here. An adapter that has the
native pair must clone it and replace only valid rows; that preserves native
padding and zero-row conventions independently of this reference calculation.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any


REFERENCE_SHA256 = "56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7"
REFERENCE_EPS = 1e-12


class BatchScaleCoreError(ValueError):
    """An unsupported or invalid request; the caller must not inject a pair."""


@dataclass(frozen=True)
class Pair:
    """q is INT8 [T,H], scale is FP32 [T], dequant has the input dtype."""

    q: Any
    scale: Any
    dequant: Any


def validate_metadata(
    shape: Any,
    *,
    shared: bool,
    valid_rows: list[bool],
    smooth_scales: Any = None,
    group_index: Any = None,
) -> tuple[bool, ...]:
    """Validate the reduction domain without importing torch or touching a device."""
    if type(shared) is not bool:
        raise BatchScaleCoreError("SHARED_MUST_BE_BOOL")
    if smooth_scales is not None:
        raise BatchScaleCoreError("SMOOTH_SCALES_UNSUPPORTED")
    if group_index is not None:
        raise BatchScaleCoreError("GROUP_INDEX_UNSUPPORTED")
    if not isinstance(shape, (tuple, list)) or len(shape) != 2:
        raise BatchScaleCoreError("REQUIRES_TWO_DIMENSIONAL_INPUT")
    if any(type(size) is not int or size <= 0 for size in shape):
        raise BatchScaleCoreError("REQUIRES_POSITIVE_CONCRETE_DIMENSIONS")
    if type(valid_rows) is not list or any(type(value) is not bool for value in valid_rows):
        raise BatchScaleCoreError("VALID_ROWS_MUST_BE_BOOL_LIST")
    if len(valid_rows) != shape[0]:
        raise BatchScaleCoreError("VALID_ROW_COUNT_MISMATCH")
    if not any(valid_rows):
        raise BatchScaleCoreError("NO_VALID_ROWS")
    return tuple(valid_rows)


@lru_cache(maxsize=1)
def _load_reference() -> Any:
    """Enforce the existing startup contract without importing unrelated owners.

    The caller configures PYTHONPATH and ASCEND_FI_REFERENCE_SHA256 just as for
    the existing inspect.combined entry point. There is no download, alternate
    checkout search, or implicit fallback. One worker epoch pins one instance.
    """
    if os.environ.get("ASCEND_FI_REFERENCE_SHA256") != REFERENCE_SHA256:
        raise BatchScaleCoreError("PINNED_REFERENCE_SHA256_REQUIRED")
    try:
        spec = importlib.util.find_spec('fake_requant_reference_extension')
        if spec is None or not spec.origin:
            raise BatchScaleCoreError('PINNED_REFERENCE_UNAVAILABLE')
        path = Path(spec.origin)
        if hashlib.sha256(path.read_bytes()).hexdigest() != REFERENCE_SHA256:
            raise BatchScaleCoreError("PINNED_REFERENCE_SOURCE_MISMATCH")
        module = importlib.import_module('fake_requant_reference_extension')
        FakeRequantReferenceExtension = module.FakeRequantReferenceExtension
        if Path(inspect.getfile(FakeRequantReferenceExtension)).resolve() != path.resolve():
            raise BatchScaleCoreError('PINNED_REFERENCE_SOURCE_MISMATCH')
    except BatchScaleCoreError:
        raise
    except (ImportError, OSError, TypeError, ValueError, RuntimeError) as exc:
        raise BatchScaleCoreError("PINNED_REFERENCE_UNAVAILABLE") from exc
    return FakeRequantReferenceExtension()


def _scale(reference: Any, tensor: Any, provider: str) -> Any:
    # These are all fields read by the two selected _scale_for branches. Keep
    # multiplier fixed: otherwise this becomes a second, unrelated perturbation.
    config = SimpleNamespace(
        scale_provider=provider, scale_multiplier=1.0, eps=REFERENCE_EPS
    )
    try:
        return reference._scale_for(None, tensor, config)
    except Exception as exc:
        raise BatchScaleCoreError("REFERENCE_SCALE_FAILED") from exc


def compute_pair(
    x: Any,
    *,
    shared: bool,
    valid_rows: list[bool],
    reference: Any = None,
    smooth_scales: Any = None,
    group_index: Any = None,
) -> Pair:
    """Compute a fresh paired payload, with no mutation of x or model state.

    Inputs are dense strided two-dimensional FP16/BF16/FP32 tensors with finite
    values and an explicit, nonempty bool row mask. In shared mode only valid
    rows contribute to the shared absmax; invalid rows keep per-token reference
    quantization. An explicitly supplied reference is a caller-verified instance
    exposing the pinned extension's _scale_for method.

    Zero rows use the pinned provider's eps floor. This is an explicit reference
    convention, not a statement about a native quantizer's zero scale policy.
    smooth_scales and group_index are rejected even if they would be identities.
    """
    mask_values = validate_metadata(
        tuple(getattr(x, "shape", ())), shared=shared, valid_rows=valid_rows,
        smooth_scales=smooth_scales, group_index=group_index,
    )
    try:
        import torch
    except ImportError as exc:
        raise BatchScaleCoreError("TORCH_REQUIRED") from exc
    if not isinstance(x, torch.Tensor):
        raise BatchScaleCoreError("REQUIRES_TORCH_TENSOR")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise BatchScaleCoreError("REQUIRES_FP16_BF16_OR_FP32")
    if x.layout != torch.strided or x.is_meta:
        raise BatchScaleCoreError("REQUIRES_DENSE_MATERIALIZED_INPUT")
    if not bool(torch.isfinite(x).all().item()):
        raise BatchScaleCoreError("NON_FINITE_INPUT")
    if reference is None:
        reference = _load_reference()
    if not callable(getattr(reference, "_scale_for", None)):
        raise BatchScaleCoreError("REFERENCE_SCALE_PROVIDER_REQUIRED")

    with torch.no_grad():
        scale = _scale(reference, x, "per_token_absmax")
        if (not isinstance(scale, torch.Tensor) or tuple(scale.shape) != (x.shape[0], 1)
                or scale.dtype != torch.float32 or scale.device != x.device):
            raise BatchScaleCoreError("HEALTHY_SCALE_CONTRACT_MISMATCH")
        scale = scale.reshape(x.shape[0]).clone()
        if shared:
            mask = torch.tensor(mask_values, device=x.device, dtype=torch.bool)
            shared_scale = _scale(reference, x.detach()[mask], "tensor_absmax")
            if (not isinstance(shared_scale, torch.Tensor) or shared_scale.ndim != 0
                    or shared_scale.dtype != torch.float32 or shared_scale.device != x.device):
                raise BatchScaleCoreError("SHARED_SCALE_CONTRACT_MISMATCH")
            scale[mask] = shared_scale
        if not bool((torch.isfinite(scale) & (scale > 0)).all().item()):
            raise BatchScaleCoreError("INVALID_SCALE")
        working = x.detach().to(dtype=torch.float32)
        q = torch.round(working / scale[:, None]).clamp(-128, 127).to(dtype=torch.int8)
        dequant = (q.to(dtype=torch.float32) * scale[:, None]).to(dtype=x.dtype)
        if not bool(torch.isfinite(dequant).all().item()):
            raise BatchScaleCoreError("NON_FINITE_DEQUANT")
    return Pair(q=q, scale=scale, dequant=dequant)


__all__ = [
    "BatchScaleCoreError", "Pair", "REFERENCE_EPS", "REFERENCE_SHA256",
    "compute_pair", "validate_metadata",
]
