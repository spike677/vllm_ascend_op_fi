"""CPU reference oracle for one input-value transformation, never an output proxy.

Native execution owns device data. These helpers operate only on supplied local
sequences for contract tests; no serializer or uploader accepts tensor contents.
"""
from dataclasses import dataclass
import math
from .schema import PATTERNS

BF16_MAX = math.ldexp(2 - 2**-7, 127)
FP32_MAX_EXACT = (2**24 - 1) * 2**104


@dataclass(frozen=True)
class TensorLayout:
    shape: tuple[int, int]
    stride: tuple[int, int]
    storage_elements: int
    offset: int = 0
    dtype: str = "BF16"
    format: str = "ND"

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "stride", tuple(self.stride))
        if self.dtype != "BF16" or self.format != "ND":
            raise ValueError("UNSUPPORTED_DTYPE_OR_FORMAT")
        if len(self.shape) != 2 or len(self.stride) != 2:
            raise ValueError("Only rank-two layouts are supported")
        if any(type(v) is not int or v <= 0 for v in self.shape + self.stride):
            raise ValueError("Positive integer shape and strides are required")
        if type(self.offset) is not int or self.offset < 0:
            raise ValueError("Invalid view offset")
        if type(self.storage_elements) is not int or self.storage_elements <= 0:
            raise ValueError("Invalid storage size")
        if self.offset + sum((n - 1) * s for n, s in zip(self.shape, self.stride)) >= self.storage_elements:
            raise ValueError("LAYOUT_OUT_OF_BOUNDS")
        n0, n1 = self.shape
        s0, s1 = self.stride
        divisor = math.gcd(s0, s1)
        if n0 > s1 // divisor and n1 > s0 // divisor:
            raise ValueError("OVERLAPPING_LAYOUT")

    def index(self, i, j):
        return self.offset + i * self.stride[0] + j * self.stride[1]


def dimensions(self_layout, mat2_layout, output_layout=None):
    m, k = self_layout.shape
    kb, n = mat2_layout.shape
    if k != kb or (output_layout is not None and output_layout.shape != (m, n)):
        raise ValueError("MATMUL_DIMENSION_MISMATCH")
    if k % 2:
        raise ValueError("ODD_K_UNSUPPORTED")
    return m, k, n


def amplitude_exponent(k, policy="historical58"):
    if type(k) is not int or k <= 0 or k % 2:
        raise ValueError("POSITIVE_EVEN_K_REQUIRED")
    if policy == "historical58":
        return 58
    if policy != "auto":
        raise ValueError("UNKNOWN_AMPLITUDE_POLICY")
    # The amplitude is calibrated from the half-pattern pressure for the whole
    # five-pattern comparison. Controls must retain that exact same amplitude.
    for exponent in range(0, 64):
        product = 2 ** (2 * exponent)
        if product <= FP32_MAX_EXACT and (k // 2) * product > FP32_MAX_EXACT:
            return exponent
    raise ValueError("NO_OVERFLOW_SOLUTION")


def pressure_assessment(k, exponent):
    product = 2 ** (2 * exponent)
    return {"input_finite": math.ldexp(1.0, exponent) <= BF16_MAX,
            "single_product_finite": product <= FP32_MAX_EXACT,
            "half_pattern_pressure": (k // 2) * product > FP32_MAX_EXACT,
            "pressure_reference": "FP32_MAX_FINITE_NOT_ACCUMULATOR_PROOF",
            "theoretical_output_exact_zero": k > 0 and k % 2 == 0}


def cancellation_sign(k_index, k, pattern):
    if pattern not in PATTERNS or k <= 0 or k % 2 or not 0 <= k_index < k:
        raise ValueError("INVALID_CANCELLATION_PARAMETERS")
    if pattern == "half":
        return 1 if k_index < k // 2 else -1
    if pattern == "alternating":
        return 1 if k_index % 2 == 0 else -1
    block = int(pattern.removeprefix("block"))
    local = k_index % (k // 2)
    sign = 1 if (local // block) % 2 == 0 else -1
    return sign if k_index < k // 2 else -sign


def input_value_transform(self_storage, mat2_storage, self_layout, mat2_layout,
                          pattern="half", amplitude_policy="historical58"):
    m, k, n = dimensions(self_layout, mat2_layout)
    if len(self_storage) != self_layout.storage_elements or len(mat2_storage) != mat2_layout.storage_elements:
        raise ValueError("STORAGE_SIZE_MISMATCH")
    exponent = amplitude_exponent(k, amplitude_policy)
    a = math.ldexp(1.0, exponent)
    left, right = list(self_storage), list(mat2_storage)
    for row in range(m):
        for col in range(k):
            left[self_layout.index(row, col)] = a
    for row in range(k):
        value = a * cancellation_sign(row, k, pattern)
        for col in range(n):
            right[mat2_layout.index(row, col)] = value
    return left, right
