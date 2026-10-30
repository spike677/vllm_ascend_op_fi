"""CPU integration of the real certificate/core/FA scope; no native execution.

The current strategy in these tests is an explicitly labelled fixture recipe,
not an observation. Production replay requires parsed actual step0 evidence.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    sys.dont_write_bytecode = True
    package_root = args.package_root.resolve(strict=True)
    sys.path.insert(0, str(package_root.parent))
    from native_execution_strategy.adapters.flash_attention import FlashAttentionAdapter, EFFECTIVE_SCALE
    from native_execution_strategy.config import from_dict
    from native_execution_strategy.models import NativeStrategy
    from native_execution_strategy.policies import Policy
    from native_execution_strategy.registry import AdapterRegistry, AdapterRegistration
    certificate_path = package_root / 'data/fa_s1s2_experimental.json'
    certificate = json.loads(certificate_path.read_bytes())
    assert certificate['validation_state'] == 'experimental'
    checks = []
    parent = Path(__file__).resolve().parent / 'outputs'
    parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='adapter_cpu_', dir=parent) as directory:
        temp = Path(directory).resolve()
        assert temp.is_relative_to(parent.resolve())
        for target in ('query_tile', 'key_group', 'joint'):
            adapter = FlashAttentionAdapter(certificate_path, evidence_root=package_root,
                                           selection_target=target, allow_experimental=True, on_unsupported='reject')
            cfg = from_dict({'schema_version': 1,
                'adapter': {'name': 'FlashAttentionScore', 'version': certificate['adapter']['version'], 'route': 'S1S2'},
                'selection': {'target': target, 'policy': 'legal_alternative'},
                'trigger': {'mode': 'every_match', 'condition': 'adaptive',
                            'predicates': [{'field': 'operator', 'op': 'eq', 'value': 'FlashAttentionScore'}]}})
            registry = AdapterRegistry()
            registry.register(AdapterRegistration(adapter.name, adapter.version, adapter.fingerprints,
                adapter.certificate['validation_state'], adapter.loaded.evidence, route='S1S2'), adapter)
            assert registry.resolve(cfg.adapter).registration.validation_state == 'experimental'
            policy, off = Policy(cfg.policy), Policy(replace(cfg.policy, enabled=False))
            for batch in (1, 8):
                actual = {'operator': 'FlashAttentionScore', 'request_count': batch, 'token_count': batch * 2048,
                    'seq_lens': [2048] * batch, 'layout': 'BNSD', 'device': 'Ascend910B4',
                    'inputs': {name: {'shape': [batch, 16 if name == 'query' else 8, 2048, 128], 'dtype': 'float16'}
                               for name in ('query', 'key', 'value')},
                    'attributes': {'scale': EFFECTIVE_SCALE, 'keep_prob': 1, 'pre_tokens': 2147483647,
                        'next_tokens': 0, 'heads': 16, 'inner_precise': 0, 'sparse_mode': 2, 'optional_inputs_absent': True}}
                actual['inputs']['atten_mask'] = {'shape': [2048, 2048], 'dtype': 'bool'}
                context = adapter.inspect(actual)
                recipes = next(row['strategies'] for row in certificate['contexts'] if row['request_count'] == batch)
                recipe = next(row for row in recipes if row['parameters']['query_tile'] == 128 and row['parameters']['key_group_ratio'] == 8)
                fixture_current = NativeStrategy(**recipe)
                decisions = [policy.evaluate(context, fixture_current, adapter.legal_alternatives(context, fixture_current),
                                             adapter.validate_candidate) for _ in range(2)]
                assert all(d.requires_execution_change for d in decisions)
                assert decisions[0].requested_strategy == decisions[1].requested_strategy
                disabled = off.evaluate(context, fixture_current, adapter.legal_alternatives(context, fixture_current), adapter.validate_candidate)
                assert not disabled.requires_execution_change
                for step, decision in enumerate([None, *decisions, disabled]):
                    destination = temp / f'{target}_b{batch}_step{step}'
                    with adapter.native_scope(decision, f'fixture_{target}_{batch}_{step}', destination,
                                              context=context, current=fixture_current if step else None) as plan:
                        assert plan == json.loads(os.environ['NES_FA_PLAN_JSON'])
                        assert plan['mode'] == ('controlled' if step in (1, 2) else 'native')
                        assert plan['scope']['request_count'] == batch
                        assert plan['certificate']['validation_state'] == 'experimental'
                        # No torch, operator call, host event or numeric result is fabricated here.
                    assert 'NES_FA_PLAN_JSON' not in os.environ and 'NES_FA_LOG_PATH' not in os.environ
                checks.append({'target': target, 'batch': batch, 'same_policy_two_matching_decisions': True,
                               'off_native_mode_and_environment_restored': True, 'native_calls': 0})
            for batch in (2, 4):
                observed = dict(actual, request_count=batch, token_count=batch * 2048, seq_lens=[2048] * batch)
                observed['inputs'] = {name: dict(meta, shape=[batch, *meta['shape'][1:]]) if name != 'atten_mask' else meta
                                      for name, meta in actual['inputs'].items()}
                context = adapter.inspect(observed)
                try:
                    adapter.legal_alternatives(context, fixture_current)
                    raise AssertionError('Discovery-only context gained controlled support')
                except ValueError:
                    pass
        report = {'schema': 'nes.harness.adapter_cpu_review.v1', 'status': 'PASS_REAL_CERTIFICATE_POLICY_SCOPE_CPU_INTEGRATION',
            'certificate_sha256': hashlib.sha256(certificate_path.read_bytes()).hexdigest(),
            'certificate_validation_state': 'experimental', 'source_fingerprints': adapter.fingerprints.as_dict(),
            'checks': checks, 'discovery_B2_B4_remains_uncontrolled': True,
            'current_strategy_source': 'Explicit CPU fixture recipe; not claimed as an actual native observation',
            'native_calls': 0, 'framework_imports': False, 'containers_started': 0,
            'implementation_sha256': {p.relative_to(package_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in sorted(package_root.rglob('*.py'))},
            'test_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'controlled_context_checks': len(checks),
                      'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'native_calls': 0}))


if __name__ == '__main__':
    main()
