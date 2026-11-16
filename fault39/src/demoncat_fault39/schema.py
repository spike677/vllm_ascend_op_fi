"""Public, model-independent Fault39 configuration. No tensor payload fields."""
from dataclasses import asdict, dataclass
from typing import Literal

PATTERNS = ("half", "block256", "block512", "block1024", "alternating")
PRIMITIVE = "FLOAT_MATMUL_CANCELLATION"
INTEGRATION_STATUS = "INNER_FRAMEWORK_INTEGRATION_PENDING"
REQUEST_MAPPING = "UNPROVEN_REQUEST_MAPPING"


@dataclass(frozen=True)
class Fault39Config:
    fault_id: int = 39
    pattern: str = "half"
    amplitude_policy: Literal["historical58", "auto"] = "historical58"
    target: str = "auto"
    target_policy: str = "unique"
    ranks: tuple[int, ...] = (0,)
    devices: tuple[int, ...] = ()
    max_hits: int = 1
    max_bytes: int = 512 * 1024 * 1024
    schema_version: int = 1

    def __post_init__(self):
        if type(self.fault_id) is not int or self.fault_id != 39:
            raise ValueError("Only Fault39 is supported")
        if self.schema_version != 1 or self.target != "auto":
            raise ValueError("schema_version=1 and target=auto are required")
        if self.target_policy not in ('unique', 'max_k_min_m_unique'):
            raise ValueError('Unsupported target policy')
        if self.pattern not in PATTERNS:
            raise ValueError("Unsupported cancellation pattern")
        if self.amplitude_policy not in ("historical58", "auto"):
            raise ValueError("amplitude_policy must be historical58 or auto")
        for key in ("ranks", "devices"):
            value = tuple(getattr(self, key))
            if any(type(v) is not int or not 0 <= v < 2**31 for v in value):
                raise ValueError(f"{key} must contain nonnegative int32 values")
            if len(set(value)) != len(value):
                raise ValueError(f"{key} must not contain duplicates")
            object.__setattr__(self, key, value)
        if not self.ranks:
            raise ValueError("At least one explicit rank is required")
        if type(self.max_hits) is not int or not 1 <= self.max_hits < 2**32:
            raise ValueError("max_hits must be a positive uint32")
        if type(self.max_bytes) is not int or not 1 <= self.max_bytes < 2**64:
            raise ValueError("max_bytes must be a positive uint64")

    @property
    def exponent(self):
        return 58 if self.amplitude_policy == "historical58" else -1

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        # Dataclass construction deliberately rejects unknown/ignored fields.
        return cls(**value)
