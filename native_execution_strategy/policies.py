"""Context policies choose adapter-provided alternatives, never invent tiling."""
from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Callable, Iterable

from .config import PolicyConfig, Predicate
from .models import InvocationContext, NativeStrategy, require


def context_value(context: InvocationContext, field: str):
    if field.startswith('matrix_dims.'):
        return context.matrix_dims.get(field.split('.', 1)[1])
    if field.startswith('attributes.'):
        return context.attributes.get(field.split('.', 1)[1])
    if field.startswith('seq_lens.'):
        if context.seq_lens is None:
            return None
        suffix = field.split('.', 1)[1]
        if suffix == 'count':
            return len(context.seq_lens)
        if suffix == 'sum':
            return sum(context.seq_lens)
        if not context.seq_lens:
            return None
        return min(context.seq_lens) if suffix == 'min' else max(context.seq_lens)
    value = getattr(context, field)
    return list(value) if field == 'seq_lens' and value is not None else value


def evaluate_predicate(predicate: Predicate, context: InvocationContext) -> bool | None:
    value = context_value(context, predicate.field)
    if predicate.op == 'exists':
        return (value is not None) is predicate.value
    if value is None:
        return None
    if predicate.op in ('lt', 'lte', 'gt', 'gte'):
        if type(value) not in (int, float):
            return None
        return {'lt': operator.lt, 'lte': operator.le, 'gt': operator.gt, 'gte': operator.ge}[predicate.op](value, predicate.value)
    # bool is not silently equal to numeric 0/1 in semantic predicates.
    def equal(a, b):
        if type(a) is bool or type(b) is bool:
            return type(a) is type(b) and a == b
        return a == b
    if predicate.op in ('eq', 'ne'):
        result = equal(value, predicate.value)
        return result if predicate.op == 'eq' else not result
    found = any(equal(value, v) for v in predicate.value)
    return found if predicate.op == 'in' else not found


def predicates_match(predicates: Iterable[Predicate], context: InvocationContext) -> bool | None:
    results = [evaluate_predicate(p, context) for p in predicates]
    if False in results:
        return False
    return None if None in results else True


@dataclass(frozen=True)
class PolicyDecision:
    matched: bool
    action: str
    requested_strategy: NativeStrategy | None
    reason: str
    mode: str
    current_semantic_sha256: str
    legal_alternative_count: int = 0
    adaptive_rule_index: int | None = None

    @property
    def requires_execution_change(self) -> bool:
        """Intent only. Does not establish that native execution switched."""
        return self.action == 'request_strategy'

    def as_dict(self) -> dict:
        return {'matched': self.matched, 'action': self.action,
                'requested_strategy': None if self.requested_strategy is None else self.requested_strategy.as_dict(),
                'reason': self.reason, 'mode': self.mode, 'current_semantic_sha256': self.current_semantic_sha256,
                'legal_alternative_count': self.legal_alternative_count, 'adaptive_rule_index': self.adaptive_rule_index,
                'native_switch_verified': None}


class Policy:
    """Stateless selection: every eligible invocation is evaluated independently.

Adaptive rules are ordered context rules. If rules exist but none matches,
only an explicit preferred_strategy_ids fallback can request an alternative.
There is no hidden batch threshold, first-call special case or hit limit.
"""
    def __init__(self, config: PolicyConfig):
        require(isinstance(config, PolicyConfig), 'PolicyConfig is required')
        self.config = config

    def evaluate(self, context: InvocationContext, current: NativeStrategy,
                 legal_alternatives: Iterable[NativeStrategy],
                 validate_candidate: Callable[[InvocationContext, NativeStrategy], bool]) -> PolicyDecision:
        require(isinstance(context, InvocationContext) and isinstance(current, NativeStrategy), 'Invalid policy input')
        def unchanged(reason, matched=False, count=0, rule=None):
            return PolicyDecision(matched, 'pass_through', None, reason, self.config.mode,
                                  current.semantic_sha256, count, rule)
        if not self.config.enabled:
            return unchanged('disabled')
        match = predicates_match(self.config.predicates, context)
        if match is not True:
            return unchanged('context_unknown' if match is None else 'context_not_matched')
        preferred, rule_index = self.config.preferred_strategy_ids, None
        if self.config.mode == 'adaptive':
            for i, rule in enumerate(self.config.adaptive_rules):
                if predicates_match(rule.predicates, context) is True:
                    preferred, rule_index = rule.preferred_strategy_ids, i
                    break
            if self.config.adaptive_rules and rule_index is None and not preferred:
                return unchanged('no_adaptive_rule_match', matched=True)
        candidates = {}
        for candidate in legal_alternatives:
            require(isinstance(candidate, NativeStrategy), 'Adapter returned a non-strategy candidate')
            if candidate.semantic_sha256 == current.semantic_sha256:
                continue
            legal = validate_candidate(context, candidate)
            require(type(legal) is bool, 'Adapter candidate validation must return bool')
            if legal:
                candidates[candidate.semantic_sha256] = candidate
        if not candidates:
            return unchanged('no_legal_alternative', matched=True, rule=rule_index)
        # A configured preference list is an allowlist, not permission to
        # silently substitute an unrelated candidate when it is unavailable.
        eligible = [v for v in candidates.values() if not preferred or v.strategy_id in preferred]
        if not eligible:
            return unchanged('preferred_alternative_unavailable', matched=True, count=len(candidates), rule=rule_index)
        selected = min(eligible, key=lambda v: (preferred.index(v.strategy_id) if preferred else 0, v.strategy_id, v.semantic_sha256))
        return PolicyDecision(True, 'request_strategy', selected, 'legal_alternative_selected', self.config.mode,
                              current.semantic_sha256, len(candidates), rule_index)
