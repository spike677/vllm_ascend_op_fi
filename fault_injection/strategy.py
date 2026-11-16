"""Bridge model-discovered call sites to the existing native strategy runtime.

This is an operator-provider interface, not a second policy implementation.
Providers supply the actual native arguments and a genuine launch observer.
Model users select a mechanism; they never construct layer names here. FA is
deliberately distinct from FIA and from the denominator-replay extension.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from native_execution_strategy.models import InvocationContext
from native_execution_strategy.session import SessionResult, StrategySession


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _tensor_metadata(tensor: Any) -> dict:
    shape = list(tensor.shape)
    _require(all(type(n) is int and n > 0 for n in shape), "Concrete positive native tensor dimensions required")
    return {"shape": shape, "dtype": str(tensor.dtype).removeprefix("torch.")}


@dataclass(frozen=True)
class NativeFAInvocation:
    """Facts captured by the FA provider at its original native call boundary.

`kwargs` are the arguments actually passed to npu_fusion_attention_v3, not HF
AttentionInterface kwargs. In particular its boolean mask uses native FA
semantics. `sequence_lengths` comes from the provider's validated request
mapping; neither leading shape nor a module name establishes that mapping.
"""

    kwargs: Mapping[str, Any]
    sequence_lengths: Sequence[int]
    device: str
    operator: str = "FlashAttentionScore"


def attention_context(call: Any, adapter: Any, native: NativeFAInvocation) -> InvocationContext:
    """Read real Q/K/V geometry and delegate all capability checks to adapter.

No hidden baseline invocation, padding removal, K/V repetition, rescaling,
mask conversion or model-name dispatch takes place in this metadata bridge.
"""
    _require(isinstance(native, NativeFAInvocation), "Actual NativeFAInvocation required")
    _require(native.operator == "FlashAttentionScore" and adapter.name == native.operator,
             "FA bridge cannot certify FIA or another operator")
    options = dict(native.kwargs)
    required = {"head_num", "input_layout", "atten_mask", "scale", "keep_prob",
                "pre_tockens", "next_tockens", "inner_precise", "sparse_mode"}
    optional = {"pse", "padding_mask", "prefix", "actual_seq_qlen", "actual_seq_kvlen", "sink"}
    _require(required <= options.keys() and not options.keys() - required - optional,
             "Unrecognized or incomplete actual FA arguments")
    _require(options["input_layout"] == "BNSD", "TND requires its own request-mapping adapter")
    _require(isinstance(native.device, str) and bool(native.device), "Observed device identity required")
    _require(all(options.get(k) is None for k in optional),
             "Optional native inputs require a separately validated adapter")
    _require(options["atten_mask"] is not None, "Explicit native FA mask required")
    _require(call.dropout == 0 and options["keep_prob"] == 1,
             "Dropout is outside the certified inference adapter")
    _require(call.scaling == options["scale"], "HF and actual native scale differ")
    inputs = {name: _tensor_metadata(getattr(call, name)) for name in ("query", "key", "value")}
    inputs["atten_mask"] = _tensor_metadata(options["atten_mask"])
    q, k, v = (inputs[name]["shape"] for name in ("query", "key", "value"))
    _require(all(len(s) == 4 for s in (q, k, v)), "BNSD requires rank-four Q/K/V")
    _require(type(options["head_num"]) is int and options["head_num"] == q[1], "Actual head_num disagrees with Q")
    lengths = tuple(native.sequence_lengths)
    _require(all(type(n) is int and n > 0 for n in lengths), "Invalid explicit sequence lengths")
    _require(len(lengths) == q[0] == k[0] == v[0], "Request mapping and native B axes differ")
    _require(all(n == q[2] == k[2] == v[2] for n in lengths),
             "Padded, cached, cross or variable-length attention needs its own native adapter")
    actual = {
        "operator": native.operator, "request_count": len(lengths),
        "token_count": sum(lengths), "seq_lens": list(lengths), "layout": "BNSD",
        "inputs": inputs, "device": native.device,
        "attributes": {"scale": options["scale"], "keep_prob": options["keep_prob"],
                       "pre_tokens": options["pre_tockens"], "next_tokens": options["next_tockens"],
                       "heads": options["head_num"], "inner_precise": options["inner_precise"],
                       "sparse_mode": options["sparse_mode"], "optional_inputs_absent": True},
    }
    return adapter.inspect(actual)


class NativeStrategyBridge:
    """Preserve the existing version, legal-strategy and actual-launch checks.

Only a provider with a loaded StrategySession can create this bridge. The
caller must explicitly run a native observation before controlled execution;
an unobserved context is never auto-warmed up or admitted by model name.
"""

    def __init__(self, session: StrategySession):
        _require(isinstance(session, StrategySession), "Existing StrategySession required")
        self.session = session
        self.events: list[dict] = []

    def execute_attention(self, call: Any, native: NativeFAInvocation,
                          invoke_original: Callable[[], Any], observer: Callable,
                          invocation_id: str, evidence_dir: str | Path, *,
                          mode: str = "native") -> SessionResult:
        context = attention_context(call, self.session.adapter, native)
        result = self.session.execute(context, invoke_original, observer,
                                      invocation_id, evidence_dir, mode=mode)
        self.events.append({
            "module_path": call.path, "invocation_id": invocation_id, "mode": mode,
            "actual_context": context.as_dict(), "assessment": result.assessment.as_dict(),
            "observation_error": result.observation_error,
            "passthrough_reason": result.passthrough_reason,
        })
        return result

    def report(self) -> dict:
        return {"mechanism": "native_execution_strategy", "operator": self.session.adapter.name,
                "events": list(self.events), "model_validation_extended": False,
                "limits": ["Model registration is not operator certification.",
                           "FA certificates do not authorize FIA or denominator replay.",
                           "Launch observation failures retain unknown assessment."]}
