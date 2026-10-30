"""One genuine invocation per admitted session run, with separate observation.

The adapter's observe_current is a lookup of an existing native observation in
this API. It must not execute the operation. A missing baseline rejects a
controlled run before invocation; run_native explicitly creates that baseline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .models import (Assessment, InvocationContext, StrategyObservation,
                     assess_strategy_switch, require)
from .policies import Policy, PolicyDecision


@dataclass(frozen=True)
class SessionResult:
    value: Any
    observation: StrategyObservation | None
    decision: PolicyDecision | None
    assessment: Assessment
    observation_error: str | None = None
    passthrough_reason: str | None = None


class StrategySession:
    """The native call owns the value; observers supply evidence, not replacements.

No clone, retry, hidden baseline invocation or numerical inference occurs here.
Original operation and native-scope errors propagate. Observation/interpretation
errors retain the original returned object and leave assessment unknown.
"""
    def __init__(self, adapter: Any, policy: Policy):
        require(isinstance(policy, Policy), 'StrategySession requires a Policy')
        self.adapter = adapter
        self.policy = policy

    def run_native(self, context: InvocationContext, invoke_original: Callable[[], Any],
                   observer: Callable[[Any, Any, Path], dict], invocation_id: str,
                   evidence_dir: str | Path) -> SessionResult:
        """One explicit native observation, with no before/after switch claim."""
        return self.execute(context, invoke_original, observer, invocation_id, evidence_dir, mode='native')

    def run_controlled(self, context: InvocationContext, invoke_original: Callable[[], Any],
                       observer: Callable[[Any, Any, Path], dict], invocation_id: str,
                       evidence_dir: str | Path) -> SessionResult:
        """Evaluate each call against a real, previously observed native baseline."""
        return self.execute(context, invoke_original, observer, invocation_id, evidence_dir, mode='controlled')

    def _unsupported(self, context: InvocationContext, *, controlled: bool) -> str | None:
        require(isinstance(context, InvocationContext), 'Actual InvocationContext is required')
        # This is solely a real-context capability decision. Registry/version,
        # fingerprint and certificate failures must already have rejected the
        # adapter and must never be converted to this passthrough path.
        policy = getattr(self.adapter, 'on_unsupported', 'reject')
        require(policy in ('reject', 'passthrough'), 'Invalid adapter on_unsupported policy')
        reason = self.adapter.unsupported_reason(context, controlled=controlled)
        require(reason is None or (isinstance(reason, str) and bool(reason)), 'Invalid unsupported-context reason')
        if reason is not None:
            require(policy == 'passthrough', 'Unsupported invocation context: ' + reason)
        return reason

    def execute(self, context: InvocationContext, invoke_original: Callable[[], Any],
                observer: Callable[[Any, Any, Path], dict], invocation_id: str,
                evidence_dir: str | Path, *, mode: str = 'native') -> SessionResult:
        """Shared original-call site for native/controlled/off statistics alignment.

Harnesses comparing complete stacks should call execute directly from one
common caller line, rather than the convenience run_* wrappers.
"""
        require(mode in ('native', 'controlled'), 'Session mode must be native or controlled')
        require(isinstance(context, InvocationContext), 'Actual InvocationContext is required')
        require(callable(invoke_original), 'Original invocation must be callable')
        evidence_dir = Path(evidence_dir)
        reason = self._unsupported(context, controlled=mode == 'controlled')
        before, decision = None, None
        if reason is None:
            require(callable(observer), 'Observer must be callable')
            if mode == 'controlled':
                before = self.adapter.observe_current(context)
                require(isinstance(before, StrategyObservation), 'Adapter has no genuine native baseline observation')
                require(before.context.as_dict() == context.as_dict(), 'Native baseline belongs to a different context')

                # Lazy enumeration preserves native/off calls when candidate
                # scope is not needed by an unmatched or disabled policy.
                def candidates():
                    yield from self.adapter.legal_alternatives(context, before.strategy)

                decision = self.policy.evaluate(context, before.strategy, candidates(), self.adapter.validate_candidate)
        with self.adapter.native_scope(decision, invocation_id, evidence_dir, context=context,
                                       current=None if before is None else before.strategy) as plan:
            # The single original-call statement for every admitted mode.
            value = invoke_original()

        if reason is not None:
            # Unsupported passthrough never acquires a baseline or fabricates
            # native interpretation, numerical equality or model impact.
            return SessionResult(value, None, None, Assessment(), passthrough_reason=reason)

        # Observe after scope restoration so diagnostic reads do not inherit the
        # injection environment. Observers must be read-only with respect to value.
        after = None
        stage = 'observer'
        try:
            record = observer(value, plan, evidence_dir)
            stage = 'parse_native_evidence'
            require(type(record) is dict, 'Observer must return the original evidence record')
            after = self.adapter.parse_native_evidence(context, record)
            require(isinstance(after, StrategyObservation), 'Adapter did not return native evidence')
            require(after.context.as_dict() == context.as_dict(), 'Observed context differs from the actual invocation')
            stage = 'compare_native_strategy'
            assessment = Assessment() if before is None else assess_strategy_switch(before, after)
            return SessionResult(value, after, decision, assessment)
        except Exception as exc:
            # An evidence failure cannot justify manufacturing a different output,
            # re-running the operation, or calling absent measurements False.
            return SessionResult(value, after if isinstance(after, StrategyObservation) else None,
                                 decision, Assessment(), f'{stage}: {type(exc).__name__}: {exc}')
