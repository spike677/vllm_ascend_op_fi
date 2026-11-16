"""Model-independent contracts. Importing this module never initializes a device."""
from dataclasses import dataclass, field
from typing import Any


class UnsupportedCall(RuntimeError):
    """A discovered call cannot be handled without changing its semantics."""


@dataclass(frozen=True)
class Decision:
    supported: bool
    reason: str


@dataclass(frozen=True)
class AttentionCall:
    path: str
    module: Any
    query: Any
    key: Any
    value: Any
    attention_mask: Any
    dropout: float = 0.0
    scaling: Any = None
    kwargs: dict = field(default_factory=dict)

    def geometry(self):
        """QK and V dimensions are deliberately independent (e.g. MLA)."""
        q, k, v = (tuple(x.shape) for x in (self.query, self.key, self.value))
        if any(len(s) != 4 for s in (q, k, v)):
            raise UnsupportedCall('Expected actual BNSD tensors; packed/TND needs its own adapter')
        if q[0] != k[0] or k[0] != v[0] or k[:3] != v[:3] or q[3] != k[3]:
            raise UnsupportedCall('Inconsistent Q/K/V geometry')
        if min(*q, *k, *v) <= 0 or q[1] % k[1]:
            raise UnsupportedCall('Empty dimensions or invalid grouped query heads')
        return dict(batch=q[0], query_heads=q[1], kv_heads=k[1], query_length=q[2],
                    kv_length=k[2], qk_dim=q[3], value_dim=v[3])
