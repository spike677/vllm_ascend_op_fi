"""Conservative discovery: a signature is not proof of unique cache identity."""
from dataclasses import asdict, dataclass
import hashlib
import json
from .transform import TensorLayout, dimensions


class TargetError(RuntimeError):
    pass


@dataclass(frozen=True)
class MatmulMetadata:
    self_layout: TensorLayout
    mat2_layout: TensorLayout
    output_layout: TensorLayout
    api: str = "aclnnMatmul"

    def __post_init__(self):
        if self.api != "aclnnMatmul":
            raise ValueError("UNSUPPORTED_API")
        dimensions(self.self_layout, self.mat2_layout, self.output_layout)

    @property
    def signature(self):
        # CPU contract identity only. Runtime selection must use the native
        # candidate signature verbatim; do not replace its ABI encoding.
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "cpu-contract:" + hashlib.sha256(payload.encode()).hexdigest()


def select_stable_target(candidates, previous_candidates=None, policy='unique'):
    """Require one selected native candidate stable across repeated workloads.

    The native registry must aggregate multiple cache identities into
    candidate_count. A controller must not collapse them into one signature.
    """
    def narrow(items):
        if not items or policy == 'unique':
            return items
        if policy != 'max_k_min_m_unique':
            raise TargetError('UNKNOWN_TARGET_POLICY')
        if any(type(c.get(k)) is not int or c[k] <= 0 for c in items for k in ('M', 'K', 'N')):
            raise TargetError('INVALID_TARGET_GEOMETRY')
        # Explicit mechanism-oriented rule: greatest reduction pressure, then
        # smallest self working set. Equal-ranked targets remain ambiguous.
        max_k = max(c['K'] for c in items)
        selected = [c for c in items if c['K'] == max_k]
        min_m = min(c['M'] for c in selected)
        return [c for c in selected if c['M'] == min_m]
    candidates = narrow(candidates)
    previous_candidates = narrow(previous_candidates)
    if not candidates:
        raise TargetError("TARGET_NOT_FOUND")
    if len(candidates) != 1 or candidates[0].get("candidate_count") != 1:
        raise TargetError("AMBIGUOUS_TARGET")
    candidate = candidates[0]
    if not isinstance(candidate.get("signature"), str) or not candidate["signature"]:
        raise TargetError("INVALID_TARGET_SIGNATURE")
    if previous_candidates is None:
        raise TargetError("UNSTABLE_TARGET")
    if len(previous_candidates) != 1 or previous_candidates[0].get("candidate_count") != 1:
        raise TargetError("AMBIGUOUS_TARGET")
    keys = ("signature", "candidate_count", "M", "K", "N", "rank", "device")
    if any(candidate.get(k) != previous_candidates[0].get(k) for k in keys):
        raise TargetError("UNSTABLE_TARGET")
    prior_runs = previous_candidates[0].get("observations")
    current_runs = candidate.get("observations")
    if (type(prior_runs) is not int or type(current_runs) is not int
            or prior_runs < 1 or current_runs < 2 or current_runs <= prior_runs):
        raise TargetError("UNSTABLE_TARGET: new real Run observations required")
    return candidate["signature"]
