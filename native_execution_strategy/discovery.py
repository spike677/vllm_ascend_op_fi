"""Bounded native strategy scans, with no monotonicity or change assumption.

This module admits observations and accounts for resources. The supplied real
harness must enforce process/device timeouts and memory limits during each call;
a Python timer cannot interrupt a blocked native operation safely.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Iterable, Sized

from .config import DiscoveryBudget
from .models import (InvocationContext, ResourceEstimate, StrategyObservation,
                     canonical_sha256, require)


@dataclass(frozen=True)
class DiscoveryCase:
    case_id: str
    context: InvocationContext

    def __post_init__(self) -> None:
        require(isinstance(self.case_id, str) and bool(self.case_id), 'Discovery case_id is required')
        require(isinstance(self.context, InvocationContext), 'Discovery needs explicit actual context')


@dataclass(frozen=True)
class DiscoveryResult:
    status: str
    records: tuple[dict, ...]
    transitions: tuple[dict, ...]
    stop_reason: str | None
    elapsed_seconds: float
    accounted_device_seconds: float
    actual_resource_budget_verified: bool
    error: str | None = None

    def as_dict(self) -> dict:
        signatures = {r['semantic_sha256'] for r in self.records}
        contexts: dict[str, list[dict]] = {}
        for row in self.records:
            contexts.setdefault(canonical_sha256(row['context']), []).append(row)
        repeats = [{'context_sha256': key, 'observations': len(rows),
                    'strategy_stable': len({r['semantic_sha256'] for r in rows}) == 1}
                   for key, rows in contexts.items() if len(rows) > 1]
        return {'status': self.status, 'records': list(self.records), 'transitions': list(self.transitions),
                'stop_reason': self.stop_reason, 'elapsed_seconds': self.elapsed_seconds,
                'accounted_device_seconds': self.accounted_device_seconds,
                'actual_resource_budget_verified': self.actual_resource_budget_verified,
                'error': self.error, 'unique_observed_strategies': len(signatures),
                'strategy_changes_observed': len(signatures) > 1 if self.records else None,
                'same_context_repeat_observations': repeats,
                'monotonicity_assumed': False, 'no_change_is_valid': True,
                'numerics_observed': None, 'model_impact': None,
                'same_context_does_not_prove_equal_tensor_values': True,
                'in_call_resource_enforcement': 'EXTERNAL_HARNESS_REQUIRED'}


def discover(cases: Iterable[DiscoveryCase],
             observe: Callable[[DiscoveryCase], StrategyObservation],
             budget: DiscoveryBudget,
             estimate: Callable[[DiscoveryCase], ResourceEstimate],
             *, clock: Callable[[], float] = time.monotonic) -> DiscoveryResult:
    """Preserve requested order (including decreases/repeats); never binary search.

With device/memory budgets configured, missing actual accounting stops the scan
after retaining that observation. Exceeding a budget is a failure, not a reason
to discard the observation. Estimates are admission checks, not measurements.
"""
    require(isinstance(budget, DiscoveryBudget), 'DiscoveryBudget is required')
    start = clock()
    iterator = iter(cases)
    known_count = len(cases) if isinstance(cases, Sized) else None
    records, transitions, ids = [], [], set()
    committed, measured, stop, error, fingerprints = 0.0, True, None, None, None
    status = 'COMPLETE'
    while True:
        if known_count is not None and len(records) == known_count:
            break
        elapsed = clock() - start
        require(elapsed >= 0, 'Discovery clock moved backwards')
        if len(records) >= budget.max_cases:
            stop, status = 'max_cases', 'BUDGET_STOPPED'
            break
        if elapsed >= budget.max_elapsed_seconds:
            stop, status = 'max_elapsed_seconds', 'BUDGET_STOPPED'
            break
        try:
            case = next(iterator)
        except StopIteration:
            break
        try:
            require(isinstance(case, DiscoveryCase), 'Invalid discovery case')
            require(case.case_id not in ids, 'Duplicate discovery case_id')
            ids.add(case.case_id)
            costs = estimate(case)
            require(isinstance(costs, ResourceEstimate), 'Discovery requires explicit resource estimates')
            if budget.max_device_seconds is not None and committed + costs.device_seconds > budget.max_device_seconds:
                stop, status = 'estimated_device_budget', 'BUDGET_STOPPED'
                break
            if budget.max_memory_bytes is not None and costs.peak_memory_bytes > budget.max_memory_bytes:
                stop, status = 'estimated_memory_budget', 'BUDGET_STOPPED'
                break
            if clock() - start >= budget.max_elapsed_seconds:
                stop, status = 'max_elapsed_seconds_before_call', 'BUDGET_STOPPED'
                break
            observation = observe(case)
            require(isinstance(observation, StrategyObservation), 'Observer must return native evidence')
            require(observation.context.as_dict() == case.context.as_dict(), 'Observed context differs from requested actual context')
            if fingerprints is None:
                fingerprints = observation.fingerprints
            require(observation.fingerprints == fingerprints, 'Source/build/certificate changed during native scan')
            usage = observation.resources
            committed += max(costs.device_seconds, usage.device_seconds or 0.0)
            row = {'case_id': case.case_id, 'context': case.context.as_dict(),
                   'strategy': observation.strategy.as_dict(), 'semantic_sha256': observation.strategy.semantic_sha256,
                   'fingerprints': observation.fingerprints.as_dict(), 'raw_sha256': observation.raw_sha256,
                   'evidence': [{'path': e.path, 'sha256': e.sha256} for e in observation.evidence],
                   'estimated_resources': {'device_seconds': costs.device_seconds, 'peak_memory_bytes': costs.peak_memory_bytes},
                   'measured_resources': {'device_seconds': usage.device_seconds, 'peak_memory_bytes': usage.peak_memory_bytes}}
            if records:
                transitions.append({'before_case': records[-1]['case_id'], 'after_case': case.case_id,
                                    'strategy_changed': records[-1]['semantic_sha256'] != row['semantic_sha256']})
            records.append(row)
            if ((budget.max_device_seconds is not None and usage.device_seconds is None) or
                    (budget.max_memory_bytes is not None and usage.peak_memory_bytes is None)):
                measured, stop, status = False, 'missing_actual_resource_measurement', 'RESOURCE_ACCOUNTING_UNKNOWN'
                break
            if ((budget.max_device_seconds is not None and committed > budget.max_device_seconds) or
                    (budget.max_memory_bytes is not None and usage.peak_memory_bytes > budget.max_memory_bytes) or
                    clock() - start > budget.max_elapsed_seconds):
                measured = False
                stop, status = 'actual_resource_budget_exceeded', 'RESOURCE_BUDGET_EXCEEDED'
                break
        except Exception as exc:
            status, stop, error = 'OBSERVATION_FAILED', 'invalid_or_failed_native_observation', f'{type(exc).__name__}: {exc}'
            break
    elapsed = clock() - start
    require(elapsed >= 0, 'Discovery clock moved backwards')
    if status == 'COMPLETE':
        status = ('COMPLETE_STRATEGY_CHANGES_OBSERVED' if len({r['semantic_sha256'] for r in records}) > 1 else
                  'COMPLETE_NO_STRATEGY_CHANGE' if records else 'COMPLETE_EMPTY_SCAN')
    return DiscoveryResult(status, tuple(records), tuple(transitions), stop, elapsed, committed, measured, error)
