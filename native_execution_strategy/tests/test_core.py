"""CPU-only contract regression. Fixtures are not native execution evidence."""
import json
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from native_execution_strategy.config import (
    AdapterSelector, AdaptiveRule, DiscoveryBudget, PolicyConfig, Predicate, from_dict, loads,
)
from native_execution_strategy.discovery import DiscoveryCase, discover
from native_execution_strategy.models import (
    Assessment, BuildFingerprints, EvidenceRef, InvocationContext, NativeStrategy,
    ResourceEstimate, ResourceUsage, StrategyObservation, assess_strategy_switch,
)
from native_execution_strategy.policies import Policy, evaluate_predicate
from native_execution_strategy.registry import AdapterRegistration, AdapterRegistry
from native_execution_strategy.certificates import (RUNTIME, load_certificate, read_bound_file,
                                                     resolve_catalog, local_path)
from native_execution_strategy.source_fingerprints import PYTHON_SOURCE_FILES, python_source_fingerprints


FP = BuildFingerprints('1' * 64, '2' * 64, '3' * 64)
EVIDENCE = (EvidenceRef('cpu_fixture_not_native.json', '3' * 64),)
BASE = NativeStrategy('base', {'M': 128, 'R': 8})
M = NativeStrategy('query_tile', {'M': 64, 'R': 8})
R = NativeStrategy('key_group', {'M': 128, 'R': 4})


def context(n=None):
    return InvocationContext('FlashAttentionScore', request_count=n, dtype='float16', layout='BNSD')


def observation(ctx, strategy=BASE, resources=None, fingerprints=FP, raw='a' * 64):
    return StrategyObservation(ctx, strategy, fingerprints, EVIDENCE, raw,
                               resources or ResourceUsage(1, 64))


class ModelsTests(unittest.TestCase):
    def test_semantics_are_explicit_and_unknown_remains_unknown(self):
        c = InvocationContext('Matmul', matrix_dims={'M': 8, 'N': 128}, attributes={'shape': [8, 128]})
        self.assertIsNone(c.request_count)
        self.assertIsNone(c.token_count)
        self.assertIsNone(c.seq_lens)
        self.assertEqual(c.matrix_dims['M'], 8)

    def test_known_zero_and_empty_are_not_unknown(self):
        c = InvocationContext('empty', request_count=0, token_count=0, seq_lens=())
        self.assertEqual(c.as_dict()['seq_lens'], [])
        self.assertEqual(evaluate_predicate(Predicate('seq_lens.sum', 'eq', 0), c), True)
        self.assertIsNone(evaluate_predicate(Predicate('seq_lens.max', 'gt', 0), c))

    def test_count_bool_negative_nan_rejected(self):
        for bad in (True, -1, 1.5):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                InvocationContext('op', request_count=bad)
        with self.assertRaises(ValueError):
            InvocationContext('op', attributes={'scale': float('nan')})

    def test_constructor_copies_external_json(self):
        source = {'M': 128, 'nested': [1, 2]}
        s = NativeStrategy('base', source)
        source['nested'].append(3)
        self.assertEqual(s.parameters['nested'], [1, 2])

    def test_three_assessments_independent(self):
        self.assertEqual(Assessment(True, False, None).as_dict(),
                         {'strategy_switched': True, 'numerics_observed': False, 'model_impact': None})
        with self.assertRaises(ValueError):
            Assessment(1)
        actual = assess_strategy_switch(observation(context(3)), observation(context(3), M))
        self.assertTrue(actual.strategy_switched)
        self.assertIsNone(actual.numerics_observed)
        self.assertIsNone(actual.model_impact)

    def test_raw_padding_does_not_imply_semantic_switch(self):
        assessment = assess_strategy_switch(observation(context(1), raw='a' * 64),
                                             observation(context(1), raw='b' * 64))
        self.assertFalse(assessment.strategy_switched)
        with self.assertRaises(ValueError):
            assess_strategy_switch(observation(context(1)), observation(context(8), M))


class ConfigTests(unittest.TestCase):
    def public(self):
        return {'mechanism': 'native_execution_strategy',
                'adapter': {'operator': 'FlashAttentionScore', 'version': 'validated', 'route': 'S1S2'},
                'selection': {'target': 'query_tile', 'policy': 'legal_alternative'},
                'trigger': {'mode': 'every_match', 'condition': 'adaptive',
                            'predicates': [{'field': 'request_count', 'op': 'gte', 'value': 3}]},
                'discovery': {'workload_axis': 'request_count', 'max_cases': 8,
                              'resource_budget': {'max_elapsed_seconds': 20, 'max_memory_bytes': 4096}},
                'observation': {'verify_native_launch': True, 'compare_complete_target': True,
                                'msprobe': {'task': 'statistics', 'level': 'mix', 'summary_mode': 'statistics'}}}

    def test_public_schema_normalizes_without_batch_special_case(self):
        c = from_dict(self.public())
        self.assertEqual(c.adapter.name, 'FlashAttentionScore')
        self.assertEqual(c.adapter.route, 'S1S2')
        self.assertEqual(c.selection_target, 'query_tile')
        self.assertEqual(c.policy.mode, 'adaptive')
        self.assertEqual(c.policy.predicates[0].value, 3)
        self.assertEqual(c.discovery.max_memory_bytes, 4096)

    def test_legacy_internal_json_supported(self):
        c = loads('{"schema_version":1,"adapter":{"name":"op"},"policy":{"mode":"every_match"}}')
        self.assertEqual(c.adapter.version, 'validated')

    def test_unsupported_context_policy_is_explicit(self):
        d = self.public()
        self.assertEqual(from_dict(d).adapter.on_unsupported, 'reject')
        d['adapter']['on_unsupported'] = 'passthrough'
        self.assertEqual(from_dict(d).adapter.on_unsupported, 'passthrough')
        d['adapter']['on_unsupported'] = 'guess'
        with self.assertRaises(ValueError):
            from_dict(d)

    def test_duplicates_unknown_fields_and_mixed_schema_rejected(self):
        for text in ('{"adapter":{"name":"x","name":"y"}}',
                     '{"adapter":{"name":"x"},"unexpected":1}',
                     '{"adapter":{"name":"x"},"policy":{"mode":"once"}}',
                     '{"adapter":{"name":"x"},"schema_version":true}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                loads(text)
        d = self.public()
        d['policy'] = {}
        with self.assertRaises(ValueError):
            from_dict(d)

    def test_nonfinite_and_payload_collection_rejected(self):
        with self.assertRaises(ValueError):
            loads('{"adapter":{"name":"x"},"discovery":{"max_elapsed_seconds":NaN}}')
        d = self.public()
        d['observation']['msprobe']['summary_mode'] = 'md5'
        with self.assertRaises(ValueError):
            from_dict(d)
        d = self.public()
        d['observation']['msprobe']['tensor_list'] = ['.']
        with self.assertRaises(ValueError):
            from_dict(d)

    def test_predicates_are_whitelisted_not_executable(self):
        for field in ('shape.0', '__dict__', 'attributes.__class__.__mro__', 'request_count[0]'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                Predicate(field, 'eq', 8)
        self.assertIsNone(evaluate_predicate(Predicate('request_count', 'gt', 1), context()))
        self.assertTrue(evaluate_predicate(Predicate('request_count', 'exists', False), context()))

    def test_optional_yaml_safe_duplicate_and_alias_rejection(self):
        try:
            import yaml
        except ImportError:
            with self.assertRaisesRegex(ValueError, 'optional'):
                loads('adapter: {operator: op}', format='yaml')
            return
        self.assertEqual(loads('adapter: {operator: op}', format='yaml').adapter.name, 'op')
        for text in ('adapter: {operator: op, operator: other}', 'adapter: &x {operator: op}\npolicy: *x'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                loads(text, format='yaml')
        with self.assertRaises(yaml.YAMLError):
            loads('!!python/object/apply:os.system [echo unsafe]', format='yaml')


class PolicyTests(unittest.TestCase):
    def test_every_match_has_no_hit_cap(self):
        p = Policy(PolicyConfig(preferred_strategy_ids=('query_tile',)))
        decisions = [p.evaluate(context(3), BASE, [M, R], lambda c, s: True) for _ in range(4)]
        self.assertTrue(all(d.requested_strategy == M for d in decisions))
        self.assertTrue(all(d.as_dict()['native_switch_verified'] is None for d in decisions))

    def test_unknown_unmatched_disabled_do_not_validate_candidates(self):
        def forbidden(*args):
            self.fail('Unmatched invocation reached candidate validator')
        for c, cfg, reason in ((context(), PolicyConfig(predicates=(Predicate('request_count', 'gte', 3),)), 'context_unknown'),
                               (context(2), PolicyConfig(predicates=(Predicate('request_count', 'gte', 3),)), 'context_not_matched'),
                               (context(8), PolicyConfig(enabled=False), 'disabled')):
            with self.subTest(reason=reason):
                d = Policy(cfg).evaluate(c, BASE, [M], forbidden)
                self.assertEqual(d.reason, reason)
                self.assertEqual(d.action, 'pass_through')

    def test_only_legal_different_strategy_requested(self):
        p = Policy(PolicyConfig())
        d = p.evaluate(context(1), BASE, [BASE, M, R], lambda c, s: s == R)
        self.assertEqual(d.requested_strategy, R)
        d = p.evaluate(context(1), BASE, [BASE, M], lambda c, s: False)
        self.assertEqual(d.reason, 'no_legal_alternative')
        self.assertIsNone(d.requested_strategy)

    def test_adaptive_context_rule_and_no_implicit_fallback(self):
        cfg = PolicyConfig(mode='adaptive', adaptive_rules=(
            AdaptiveRule((Predicate('request_count', 'eq', 3),), ('query_tile',)),
            AdaptiveRule((Predicate('request_count', 'eq', 5),), ('key_group',))))
        p = Policy(cfg)
        self.assertEqual(p.evaluate(context(3), BASE, [R, M], lambda c, s: True).requested_strategy, M)
        self.assertEqual(p.evaluate(context(5), BASE, [M, R], lambda c, s: True).requested_strategy, R)
        self.assertEqual(p.evaluate(context(8), BASE, [M, R], lambda c, s: True).reason, 'no_adaptive_rule_match')

    def test_missing_preferred_strategy_does_not_substitute(self):
        p = Policy(PolicyConfig(preferred_strategy_ids=('query_tile',)))
        self.assertEqual(p.evaluate(context(8), BASE, [R], lambda c, s: True).reason, 'preferred_alternative_unavailable')


class RegistryTests(unittest.TestCase):
    def register(self, registry, version='v1', state='validated', fingerprints=FP):
        evidence = (EvidenceRef('cpu_certificate_fixture.json', fingerprints.certificate_sha256),) if state == 'validated' else ()
        r = AdapterRegistration('FA', version, fingerprints, state, evidence, 'S1S2')
        registry.register(r, SimpleNamespace(name='FA', version=version))

    def test_validated_alias_pins_exact_fingerprints(self):
        r = AdapterRegistry()
        self.register(r)
        x = r.resolve(AdapterSelector('FA', route='S1S2'))
        self.assertEqual(x.version, 'v1')
        self.assertEqual(x.fingerprints, FP)
        self.assertEqual(x.as_dict()['requested_version'], 'validated')

    def test_source_build_pin_can_resolve_unique_certificate_but_not_ambiguity(self):
        r = AdapterRegistry()
        self.register(r)
        pin = BuildFingerprints(FP.source_sha256, FP.build_sha256)
        self.assertEqual(r.resolve(AdapterSelector('FA', expected_fingerprints=pin)).fingerprints, FP)
        other = BuildFingerprints(FP.source_sha256, FP.build_sha256, '4' * 64)
        self.register(r, version='v2', fingerprints=other)
        with self.assertRaises(ValueError):
            r.resolve(AdapterSelector('FA', expected_fingerprints=pin))
        self.assertEqual(r.resolve(AdapterSelector('FA', expected_fingerprints=FP)).version, 'v1')
        with self.assertRaises(ValueError):
            r.resolve(AdapterSelector('FA', expected_fingerprints=BuildFingerprints(FP.source_sha256, FP.build_sha256, '9' * 64)))

    def test_experimental_cannot_inherit_validation(self):
        r = AdapterRegistry()
        self.register(r, version='candidate-20261030', state='experimental')
        with self.assertRaises(ValueError):
            r.resolve(AdapterSelector('FA'))
        with self.assertRaises(ValueError):
            r.resolve(AdapterSelector('FA', on_unsupported='passthrough'))
        self.assertEqual(r.resolve(AdapterSelector('FA', 'candidate-20261030')).version, 'candidate-20261030')
        with self.assertRaises(ValueError):
            AdapterRegistration('FA', 'v1', BuildFingerprints('1' * 64, '2' * 64), 'validated', EVIDENCE)

    def test_route_and_exact_build_must_match(self):
        r = AdapterRegistry()
        self.register(r)
        for selector in (AdapterSelector('FA', route='other'),
                         AdapterSelector('FA', expected_fingerprints=BuildFingerprints('1' * 64, '4' * 64, '3' * 64))):
            with self.assertRaises(ValueError):
                r.resolve(selector)
        with self.assertRaises(ValueError):
            r.resolve(AdapterSelector('FA', 'unknown-version', on_unsupported='passthrough'))

    def test_ambiguous_alias_and_replacement_rejected(self):
        r = AdapterRegistry()
        self.register(r)
        with self.assertRaises(ValueError):
            self.register(r)
        other = BuildFingerprints('4' * 64, '5' * 64, '3' * 64)
        self.register(r, version='v2', fingerprints=other)
        with self.assertRaises(ValueError):
            r.resolve(AdapterSelector('FA'))
        self.assertEqual(r.resolve(AdapterSelector('FA', expected_fingerprints=other)).version, 'v2')


class DiscoveryTests(unittest.TestCase):
    def cases(self, counts):
        return [DiscoveryCase(f'case{i}', context(n)) for i, n in enumerate(counts)]

    def scan(self, cases, observe, budget=None, estimate=None):
        return discover(cases, observe, budget or DiscoveryBudget(),
                        estimate or (lambda c: ResourceEstimate(1, 64)), clock=lambda: 0)

    def test_nonmonotonic_A_B_A_and_repeats_preserve_order(self):
        cases = self.cases([1, 4, 2, 8, 1])
        result = self.scan(cases, lambda c: observation(c.context, M if c.context.request_count in (4, 8) else BASE))
        d = result.as_dict()
        self.assertEqual(d['status'], 'COMPLETE_STRATEGY_CHANGES_OBSERVED')
        self.assertEqual([r['context']['request_count'] for r in d['records']], [1, 4, 2, 8, 1])
        self.assertEqual([r['strategy_changed'] for r in d['transitions']], [True] * 4)
        self.assertIsNone(d['numerics_observed'])
        self.assertIsNone(d['model_impact'])

    def test_no_change_is_valid(self):
        r = self.scan(self.cases([1, 2, 4]), lambda c: observation(c.context))
        self.assertEqual(r.status, 'COMPLETE_NO_STRATEGY_CHANGE')
        self.assertFalse(r.as_dict()['strategy_changes_observed'])

    def test_max_cases_prevents_extra_native_call(self):
        called = []
        def observe(c):
            called.append(c.case_id)
            return observation(c.context)
        r = self.scan(self.cases([1, 2, 4]), observe, DiscoveryBudget(max_cases=2))
        self.assertEqual(len(called), 2)
        self.assertEqual(r.stop_reason, 'max_cases')

    def test_preflight_resource_budget_prevents_call(self):
        def forbidden(c):
            self.fail('Over-budget call admitted')
        for budget in (DiscoveryBudget(max_memory_bytes=32), DiscoveryBudget(max_device_seconds=0.5)):
            r = self.scan(self.cases([1]), forbidden, budget)
            self.assertEqual(r.status, 'BUDGET_STOPPED')
            self.assertEqual(len(r.records), 0)

    def test_actual_excess_is_retained_and_scan_stops(self):
        r = self.scan(self.cases([1, 2]), lambda c: observation(c.context, resources=ResourceUsage(3, 64)),
                      DiscoveryBudget(max_device_seconds=2))
        self.assertEqual(r.status, 'RESOURCE_BUDGET_EXCEEDED')
        self.assertEqual(len(r.records), 1)
        self.assertFalse(r.actual_resource_budget_verified)

    def test_missing_accounting_is_unknown_not_pass(self):
        r = self.scan(self.cases([1, 2]), lambda c: observation(c.context, resources=ResourceUsage()),
                      DiscoveryBudget(max_memory_bytes=128))
        self.assertEqual(r.status, 'RESOURCE_ACCOUNTING_UNKNOWN')
        self.assertEqual(len(r.records), 1)

    def test_native_context_mismatch_and_duplicate_ids_rejected(self):
        r = self.scan(self.cases([1]), lambda c: observation(context(8)))
        self.assertEqual(r.status, 'OBSERVATION_FAILED')
        cases = [DiscoveryCase('same', context(1))] * 2
        r = self.scan(cases, lambda c: observation(c.context))
        self.assertEqual(r.status, 'OBSERVATION_FAILED')
        self.assertEqual(len(r.records), 1)

    def test_source_change_rejected(self):
        r = self.scan(self.cases([1, 2]), lambda c: observation(c.context, fingerprints=FP if c.context.request_count == 1 else
                                                              BuildFingerprints('4' * 64, '2' * 64, '3' * 64)))
        self.assertEqual(r.status, 'OBSERVATION_FAILED')

    def test_elapsed_budget_checked_after_native_call(self):
        clock = [0.0]
        def observe(c):
            clock[0] = 3.0
            return observation(c.context)
        r = discover(self.cases([1, 2]), observe, DiscoveryBudget(max_elapsed_seconds=2),
                     lambda c: ResourceEstimate(), clock=lambda: clock[0])
        self.assertEqual(r.status, 'RESOURCE_BUDGET_EXCEEDED')
        self.assertEqual(len(r.records), 1)


class CertificateTests(unittest.TestCase):
    """Synthetic reports exercise bindings only, not actual FA validation."""
    def setUp(self):
        from native_execution_strategy.adapters.fa_schema import (POD_SCHEMA_SHA256, decode, execution_plan, input_contract)
        scratch_root = Path(__file__).resolve().parent
        self.tmp = tempfile.TemporaryDirectory(dir=scratch_root, prefix='core_test_')
        self.assertTrue(Path(self.tmp.name).resolve().is_relative_to(scratch_root))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        import struct
        from native_execution_strategy.adapters.fa_schema import FIELDS, FORMATS
        raw = bytearray(1464)
        for name, value in {'inputParams.bSize': 1, 'coreParams.s1BaseSize': 128,
                            'coreParams.nRatio': 8, 'coreParams.s2BaseSize': 128}.items():
            field = FIELDS[name]
            struct.pack_into('<' + FORMATS[field['type']], raw, field['offset'], value)
        raw = bytes(raw)
        pod = decode(raw)
        self.contexts = [{'request_count': 1, 'input_contract': input_contract(pod), 'strategies': [
            {'strategy_id': 'base', 'parameters': {'query_tile': 128, 'key_group_ratio': 8, 'key_tile': 128,
                                                 'execution_plan': execution_plan(pod, raw=raw)}}]}]
        self.report = {'status': 'PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1',
                       'source_sha256': '1' * 64, 'build_sha256': '2' * 64,
                       'python_source_files': python_source_fingerprints(),
                       'validated_contexts': copy.deepcopy(self.contexts)}
        report_sha = self.write('report.json', self.report)
        self.cert = {'schema': 'nes.fa.validation_certificate.v1', 'id': 'cpu_fixture_not_native_validation',
                     'adapter': {'name': 'FlashAttentionScore', 'version': 'test-v1', 'route': 'S1S2'},
                     'validation_state': 'validated',
                     'fingerprints': {'source_sha256': '1' * 64, 'build_sha256': '2' * 64},
                     'runtime': dict(RUNTIME), 'projection_schema_sha256': POD_SCHEMA_SHA256,
                     'python_source_files': python_source_fingerprints(),
                     'validation_evidence': [{'path': 'report.json', 'sha256': report_sha}],
                     'contexts': copy.deepcopy(self.contexts)}
        self.write('certificate.json', self.cert)

    def write(self, name, data):
        raw = json.dumps(data, sort_keys=True, allow_nan=False).encode()
        (self.root / name).write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def save_report(self):
        self.cert['validation_evidence'][0]['sha256'] = self.write('report.json', self.report)
        self.write('certificate.json', self.cert)

    def load(self, **kwargs):
        return load_certificate(self.root / 'certificate.json', evidence_root=self.root, **kwargs)

    def test_certificate_sha_external_bytes_and_fingerprints_bound(self):
        c = self.load()
        self.assertEqual(c.fingerprints.certificate_sha256, hashlib.sha256((self.root / 'certificate.json').read_bytes()).hexdigest())
        self.assertEqual(len(c.evidence), 2)
        self.assertEqual(c.path, (self.root / 'certificate.json').resolve())
        (self.root / 'report.json').write_bytes(b'{}')
        with self.assertRaises(ValueError):
            self.load()

    def test_validated_certificate_rejects_wrong_or_missing_python_code_pins(self):
        original = copy.deepcopy(self.cert['python_source_files'])
        self.cert['python_source_files']['adapters/flash_attention.py'] = '9' * 64
        self.write('certificate.json', self.cert)
        with self.assertRaisesRegex(ValueError, 'Python source fingerprints'):
            self.load()
        self.cert['python_source_files'] = original
        del self.cert['python_source_files']['source_fingerprints.py']
        self.write('certificate.json', self.cert)
        with self.assertRaisesRegex(ValueError, 'Python source fingerprints'):
            self.load()
        del self.cert['python_source_files']
        self.write('certificate.json', self.cert)
        with self.assertRaisesRegex(ValueError, 'Python source fingerprints'):
            self.load()

    def test_external_report_must_bind_identical_python_closure(self):
        self.report['python_source_files']['models.py'] = '8' * 64
        self.save_report()
        with self.assertRaisesRegex(ValueError, 'report Python source fingerprints'):
            self.load()

    def test_dependency_closure_is_explicit_and_includes_its_own_source(self):
        pins = python_source_fingerprints()
        self.assertEqual(set(pins), set(PYTHON_SOURCE_FILES))
        self.assertEqual(len(pins), 15)
        self.assertIn('source_fingerprints.py', pins)
        self.assertNotIn('adapters/matmul.py', pins)

    def test_old_build_or_nonpass_cannot_validate_new_build(self):
        for key, value in (('build_sha256', '9' * 64), ('source_sha256', '8' * 64), ('status', 'CPU_COMPILE_PASSED')):
            with self.subTest(key=key):
                original = self.report[key]
                self.report[key] = value
                self.save_report()
                with self.assertRaises(ValueError):
                    self.load()
                self.report[key] = original

    def test_every_certified_context_and_strategy_needs_exact_coverage(self):
        self.report['validated_contexts'][0]['strategies'][0]['strategy_id'] = 'other'
        self.save_report()
        with self.assertRaises(ValueError):
            self.load()

    def test_input_contract_must_be_complete_and_match_coverage(self):
        self.cert['contexts'][0]['input_contract']['inputParams.preTokens'] = 777
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()
        del self.cert['contexts'][0]['input_contract']['inputParams.preTokens']
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()

    def test_full_execution_plan_not_a_label_or_incomplete_mapping(self):
        params = self.cert['contexts'][0]['strategies'][0]['parameters']
        params['execution_plan'] = {'M': 128, 'R': 8}
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()
        params['execution_plan'] = None
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()

    def test_recipe_must_match_decoded_plan_fields(self):
        self.cert['contexts'][0]['strategies'][0]['parameters']['query_tile'] = 64
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()

    def test_experimental_requires_flag_and_never_satisfies_validated(self):
        self.cert['validation_state'] = 'experimental'
        self.cert['validation_evidence'] = []
        del self.cert['python_source_files']  # Bootstrap may omit pins but stays explicitly experimental.
        self.cert['contexts'][0]['strategies'][0]['parameters']['execution_plan'] = None
        sha = self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()
        self.assertEqual(self.load(allow_experimental=True).data['validation_state'], 'experimental')
        self.write('catalog.json', {'schema': 'nes.adapter_catalog.v1', 'certificates': [{'path': 'certificate.json', 'sha256': sha}]})
        with self.assertRaises(ValueError):
            resolve_catalog('catalog.json', evidence_root=self.root, allow_experimental=True)
        self.assertEqual(resolve_catalog('catalog.json', evidence_root=self.root, version='test-v1',
                                         allow_experimental=True).data['validation_state'], 'experimental')

    def test_catalog_pins_actual_certificate_sha_and_unique_version(self):
        sha = hashlib.sha256((self.root / 'certificate.json').read_bytes()).hexdigest()
        catalog = {'schema': 'nes.adapter_catalog.v1', 'certificates': [{'path': 'certificate.json', 'sha256': sha}]}
        self.write('catalog.json', catalog)
        self.assertEqual(resolve_catalog('catalog.json', evidence_root=self.root).fingerprints.certificate_sha256, sha)
        self.cert['id'] = 'changed'
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            resolve_catalog('catalog.json', evidence_root=self.root)

    def test_path_traversal_absolute_and_symlink_escape_rejected(self):
        for path in ('../outside.json', '/etc/passwd', 'C:/secret.json', 'sub/../report.json', 'sub\\report.json'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                local_path(self.root, path)
        scratch_root = Path(__file__).resolve().parent
        external = tempfile.TemporaryDirectory(dir=scratch_root, prefix='outside_fixture_')
        self.assertTrue(Path(external.name).resolve().is_relative_to(scratch_root))
        self.addCleanup(external.cleanup)
        p = Path(external.name) / 'other.json'
        p.write_text('{}')
        try:
            (self.root / 'escape.json').symlink_to(p)
        except OSError:
            return  # Windows may not grant symlink creation; traversal cases still run.
        with self.assertRaises(ValueError):
            read_bound_file(self.root, 'escape.json', hashlib.sha256(b'{}').hexdigest())

    def test_certificate_cannot_self_reference_or_omit_schema_binding(self):
        self.cert['validation_evidence'] = [{'path': 'certificate.json', 'sha256': '0' * 64}]
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()
        del self.cert['projection_schema_sha256']
        self.write('certificate.json', self.cert)
        with self.assertRaises(ValueError):
            self.load()


if __name__ == '__main__':
    unittest.main()
