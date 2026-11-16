"""CPU protocol fixtures only; no framework, device or real kernel execution."""
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import unittest
import traceback

from native_execution_strategy.config import PolicyConfig, Predicate
from native_execution_strategy.models import (BuildFingerprints, EvidenceRef, InvocationContext,
                                             NativeStrategy, StrategyObservation)
from native_execution_strategy.policies import Policy
from native_execution_strategy.session import StrategySession


CTX = InvocationContext('fixture', request_count=3)
BASE = NativeStrategy('base', {'tile': 128})
ALT = NativeStrategy('alternative', {'tile': 64})
FP = BuildFingerprints('1' * 64, '2' * 64, '3' * 64)


def observed(context, strategy):
    return StrategyObservation(context, strategy, FP, (EvidenceRef('cpu_fixture_not_real.json', '4' * 64),))


class FixtureAdapter:
    """Explicit mock used to test callback counts and baseline ownership."""
    def __init__(self):
        self.baseline = None
        self.events = []
        self.decisions = []
        self.current = None
        self.active = False
        self.reject_candidates = False
        self.ignore_requested = False
        self.parse_error = False
        self.on_unsupported = 'reject'
        self.unsupported = None

    def unsupported_reason(self, context, *, controlled):
        return self.unsupported

    def observe_current(self, context):
        self.events.append('baseline_lookup')
        if self.baseline is None:
            raise ValueError('No previously captured native baseline')
        return self.baseline

    def legal_alternatives(self, context, current):
        if self.reject_candidates:
            raise AssertionError('Disabled/unmatched run evaluated candidates')
        self.events.append('candidates')
        return (ALT,)

    def validate_candidate(self, context, candidate):
        return candidate == ALT

    @contextmanager
    def native_scope(self, decision, invocation_id, evidence_dir, *, context, current=None):
        self.events.append('scope_enter')
        self.decisions.append(decision)
        self.current = current
        self.active = True
        requested = decision is not None and decision.requires_execution_change
        try:
            yield {'requested': requested, 'context': context, 'id': invocation_id}
        finally:
            self.active = False
            self.events.append('scope_exit')

    def parse_native_evidence(self, context, record):
        self.events.append('parse')
        if self.parse_error:
            raise ValueError('Raw launch binding failed')
        strategy = ALT if record['requested'] and not self.ignore_requested else BASE
        result = observed(context, strategy)
        # Same ownership rule as the FA adapter: only native records seed cache.
        if not record['requested']:
            self.baseline = result
        return result


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.adapter = FixtureAdapter()
        self.policy = Policy(PolicyConfig())
        self.session = StrategySession(self.adapter, self.policy)
        self.value = object()
        self.native_calls = 0
        self.observer_calls = 0

    def invoke(self):
        self.native_calls += 1
        self.assertTrue(self.adapter.active)
        self.adapter.events.append('original')
        return self.value

    def observe(self, value, plan, evidence_dir):
        self.observer_calls += 1
        self.assertIs(value, self.value)
        self.assertFalse(self.adapter.active)
        self.assertIsInstance(evidence_dir, Path)
        self.adapter.events.append('observe')
        return plan

    def native(self):
        return self.session.run_native(CTX, self.invoke, self.observe, 'native', 'unused_fixture_directory')

    def controlled(self, observer=None):
        return self.session.run_controlled(CTX, self.invoke, observer or self.observe, 'controlled', 'unused_fixture_directory')

    def test_original_identity_exactly_one_call_and_observer_after_scope(self):
        r = self.native()
        self.assertIs(r.value, self.value)
        self.assertEqual(self.native_calls, 1)
        self.assertEqual(self.observer_calls, 1)
        self.assertEqual(self.adapter.events, ['scope_enter', 'original', 'scope_exit', 'observe', 'parse'])
        self.assertIsNone(r.decision)
        self.assertIsNone(r.assessment.strategy_switched)
        self.assertIsNone(r.assessment.numerics_observed)
        self.assertIsNone(r.assessment.model_impact)

    def test_single_entry_native_controlled_share_exact_original_callsite(self):
        stacks = []
        results = []
        def original():
            stacks.append([(f.filename, f.name, f.lineno) for f in traceback.extract_stack()
                           if Path(f.filename).name == 'session.py'])
            return self.invoke()
        for mode in ('native', 'controlled'):
            results.append(self.session.execute(CTX, original, self.observe, mode, 'unused_fixture_directory', mode=mode))
        self.assertEqual(stacks[0], stacks[1])
        self.assertEqual([name for _, name, _ in stacks[0]], ['execute'])
        self.assertEqual(self.native_calls, 2)
        self.assertIs(results[0].value, self.value)
        self.assertIs(results[1].value, self.value)
        self.assertIsNone(results[0].assessment.strategy_switched)
        self.assertTrue(results[1].assessment.strategy_switched)

    def test_single_entry_off_and_bad_mode_preserve_count_and_unknowns(self):
        self.session.execute(CTX, self.invoke, self.observe, 'baseline', 'unused_fixture_directory')
        self.policy.config = replace(self.policy.config, enabled=False)
        self.adapter.reject_candidates = True
        result = self.session.execute(CTX, self.invoke, self.observe, 'off', 'unused_fixture_directory', mode='controlled')
        self.assertFalse(result.assessment.strategy_switched)
        self.assertIsNone(result.assessment.numerics_observed)
        self.assertIs(result.value, self.value)
        with self.assertRaisesRegex(ValueError, 'Session mode'):
            self.session.execute(CTX, self.invoke, self.observe, 'invalid', 'unused_fixture_directory', mode='inferred')
        self.assertEqual(self.native_calls, 2)

    def test_two_matches_choose_again_against_native_not_controlled_baseline(self):
        self.native()
        first, second = self.controlled(), self.controlled()
        self.assertEqual(self.native_calls, 3)
        self.assertTrue(first.assessment.strategy_switched)
        self.assertTrue(second.assessment.strategy_switched)
        self.assertEqual(first.decision.requested_strategy, ALT)
        self.assertEqual(second.decision.requested_strategy, ALT)
        self.assertEqual(self.adapter.baseline.strategy, BASE)
        self.assertIs(first.value, second.value)

    def test_off_executes_native_without_candidate_evaluation_or_false_success(self):
        self.native()
        self.controlled()
        self.policy.config = replace(self.policy.config, enabled=False)
        self.adapter.reject_candidates = True
        result = self.controlled()
        self.assertEqual(self.native_calls, 3)
        self.assertEqual(result.decision.reason, 'disabled')
        self.assertFalse(result.assessment.strategy_switched)
        self.assertIsNone(result.assessment.numerics_observed)
        self.assertIsNone(result.assessment.model_impact)

    def test_unmatched_is_true_native_passthrough(self):
        self.native()
        self.policy.config = PolicyConfig(predicates=(Predicate('request_count', 'gt', 5),))
        self.adapter.reject_candidates = True
        result = self.controlled()
        self.assertEqual(self.native_calls, 2)
        self.assertEqual(result.decision.reason, 'context_not_matched')
        self.assertFalse(result.assessment.strategy_switched)

    def test_missing_baseline_rejects_without_hidden_native_call(self):
        with self.assertRaisesRegex(ValueError, 'baseline'):
            self.controlled()
        self.assertEqual(self.native_calls, 0)
        self.assertEqual(self.observer_calls, 0)
        self.assertEqual(self.adapter.events, ['baseline_lookup'])

    def test_unsupported_passthrough_keeps_identity_without_baseline_or_observer(self):
        self.adapter.on_unsupported = 'passthrough'
        self.adapter.unsupported = 'unknown_TND_request_count'
        unknown = InvocationContext('fixture', layout='TND')
        for method in (self.session.run_native, self.session.run_controlled):
            result = method(unknown, self.invoke, self.observe, 'unsupported', 'unused_fixture_directory')
            self.assertIs(result.value, self.value)
            self.assertEqual(result.passthrough_reason, 'unknown_TND_request_count')
            self.assertIsNone(result.observation)
            self.assertIsNone(result.observation_error)
            self.assertEqual(result.assessment.as_dict(),
                             {'strategy_switched': None, 'numerics_observed': None, 'model_impact': None})
        self.assertEqual(self.native_calls, 2)
        self.assertEqual(self.observer_calls, 0)
        self.assertNotIn('baseline_lookup', self.adapter.events)
        self.assertNotIn('parse', self.adapter.events)

    def test_unsupported_reject_has_zero_original_calls(self):
        self.adapter.unsupported = 'unvalidated_shape'
        for method in (self.session.run_native, self.session.run_controlled):
            with self.assertRaisesRegex(ValueError, 'Unsupported invocation'):
                method(CTX, self.invoke, self.observe, 'unsupported', 'unused_fixture_directory')
        self.assertEqual(self.native_calls, 0)
        self.assertEqual(self.observer_calls, 0)
        self.assertEqual(self.adapter.events, [])

    def test_requested_change_is_not_observed_change(self):
        self.native()
        self.adapter.ignore_requested = True
        result = self.controlled()
        self.assertTrue(result.decision.requires_execution_change)
        self.assertFalse(result.assessment.strategy_switched)

    def test_observer_failure_keeps_value_and_unknown_assessment_no_retry(self):
        self.native()
        def failed_observer(value, plan, evidence_dir):
            self.assertIs(value, self.value)
            raise RuntimeError('Evidence incomplete')
        result = self.controlled(failed_observer)
        self.assertIs(result.value, self.value)
        self.assertEqual(self.native_calls, 2)
        self.assertIsNone(result.observation)
        self.assertIsNone(result.assessment.strategy_switched)
        self.assertIn('observer: RuntimeError', result.observation_error)
        self.assertFalse(self.adapter.active)

    def test_parser_failure_keeps_original_return_and_does_not_claim_false(self):
        self.native()
        self.adapter.parse_error = True
        result = self.controlled()
        self.assertIs(result.value, self.value)
        self.assertEqual(self.native_calls, 2)
        self.assertIsNone(result.assessment.strategy_switched)
        self.assertIn('parse_native_evidence', result.observation_error)

    def test_original_exception_identity_propagates_and_scope_restores(self):
        error = RuntimeError('Original native failure')
        def fail():
            self.native_calls += 1
            raise error
        try:
            self.session.run_native(CTX, fail, self.observe, 'failure', 'unused_fixture_directory')
        except RuntimeError as exc:
            self.assertIs(exc, error)
        else:
            self.fail('Original failure was hidden')
        self.assertEqual(self.native_calls, 1)
        self.assertEqual(self.observer_calls, 0)
        self.assertFalse(self.adapter.active)


if __name__ == '__main__':
    unittest.main()
