"""Composable Torch actions; callers retain target, request and lease ownership.

No hooks, model execution, diagnostics, device synchronization or recovery
state machine live here. Native implementations keep their own typed ABI.
"""
import math

from .core import UnsupportedCall


def scale_value(value, gain):
    if type(gain) not in (int, float) or not math.isfinite(gain):
        raise ValueError('A finite real gain is required')
    return value * gain


def independent_storage(value, converter=None):
    """Retain logical tensor type/shape but require distinct backing storage.

    An NPU format adapter supplies its real converter. This helper does not
    impose a format or stride, emulate format conversion or claim value equality.
    The mechanism still checks values and its actual subsequent write/consumer.
    """
    result = value.clone() if converter is None else converter(value)
    if (result.shape != value.shape or result.dtype != value.dtype
            or result.device != value.device):
        raise UnsupportedCall('Storage transformation changed shape, dtype or device')
    source_storage, result_storage = value.untyped_storage(), result.untyped_storage()
    if result_storage._cdata == source_storage._cdata:
        raise UnsupportedCall('Storage transformation did not create independent storage')
    # Different StorageImpl objects can still wrap overlapping external memory.
    # Compare storage ranges, not offset-adjusted tensor data pointers.
    source_start, result_start = source_storage.data_ptr(), result_storage.data_ptr()
    if source_storage.nbytes() and result_storage.nbytes() and (
            source_start < result_start + result_storage.nbytes()
            and result_start < source_start + source_storage.nbytes()):
        raise UnsupportedCall('Storage transformation retained overlapping backing memory')
    return result


def replace_row_(target, index, values):
    """One actual copy_ into a caller-owned row; no hidden clone or restore."""
    if type(index) is not int or target.ndim < 1 or not -target.shape[0] <= index < target.shape[0]:
        raise ValueError('A valid integer row index is required')
    row = target[index]
    if row.shape != values.shape or row.dtype != values.dtype or row.device != values.device:
        raise ValueError('Row replacement requires identical shape, dtype and device')
    row.copy_(values)
    return target
