"""Freeze explicit pilot/full-model requests and limits; no framework or device imports."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    inputs = json.loads((root / 'inputs/manifest.json').read_bytes())
    vendor = json.loads((root / 'native_vendor_stage_a1/vendor_manifest.json').read_bytes())
    model = json.loads((root / 'model_manifest.json').read_bytes())
    certificate = root / 'runtime/native_execution_strategy/data/fa_cann91_s1s2_v1.json'
    assert sha(certificate) == 'be58b4e77098a8911ab97ad8711144a5a5999dc3742080726c594ad88e8fcc5b'
    assert sha(root / 'runtime_manifest.json') == 'ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b'
    setup = root / 'setup'
    assert not setup.exists()
    setup.mkdir()
    cases = []
    for name, batch, modes in [('pilot_b1', 1, ['calibration']),
        ('main_b1', 1, ['calibration', 'native', 'controlled', 'controlled', 'off']),
        ('main_b8', 8, ['calibration', 'native', 'controlled', 'controlled', 'off'])]:
        case = {'schema': 'nes.model_coverage.case.v1', 'case_id': name, 'phase': 'model_coverage',
            'operator_variant': 'private_vendor', 'batch_size': batch, 'mode_sequence': modes,
            'request_ids': inputs['request_sets']['b' + str(batch)], 'physical_device': 'Ascend910B4',
            'model_config_sha256': model['files']['config.json']['sha256'],
            'input_manifest_file': '/audit/inputs/manifest.json', 'input_manifest_sha256': sha(root / 'inputs/manifest.json'),
            'runtime_manifest_file': '/audit/runtime_manifest.json', 'runtime_manifest_sha256': sha(root / 'runtime_manifest.json'),
            'model_manifest_file': '/audit/model_manifest.json', 'model_manifest_sha256': sha(root / 'model_manifest.json'),
            'tokenizer_manifest_file': '/audit/tokenizer_manifest.json', 'tokenizer_manifest_sha256': sha(root / 'tokenizer_manifest.json'),
            'source_fingerprints_file': '/audit/source_fingerprints.json', 'source_fingerprints_sha256': sha(root / 'source_fingerprints.json'),
            'approved_pod_schema_file': '/audit/runtime/native_execution_strategy/adapters/fa_cann91_schema.json',
            'approved_pod_schema_sha256': sha(root / 'runtime/native_execution_strategy/adapters/fa_cann91_schema.json'),
            'approved_vendor_manifest_file': '/audit/native_vendor_stage_a1/vendor_manifest.json',
            'approved_vendor_manifest_sha256': sha(root / 'native_vendor_stage_a1/vendor_manifest.json'),
            'approved_custom_opp_path': '/audit/' + vendor['vendor_root'],
            'capture_msprobe': True, 'statistics_variant': 'task_local_fp32_high_v1',
            'approved_collector_manifest_file': '/audit/collector_high_a1/stage_manifest.json',
            'approved_collector_manifest_sha256': '743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af',
            'execution_config': {'mechanism': 'native_execution_strategy',
                'adapter': {'operator': 'FlashAttentionScore', 'version': 'validated', 'route': 'S1S2', 'on_unsupported': 'reject'},
                'selection': {'target': 'joint', 'policy': 'legal_alternative'},
                'trigger': {'mode': 'every_match', 'condition': 'adaptive'},
                'observation': {'verify_native_launch': True, 'compare_complete_target': True,
                    'msprobe': {'task': 'statistics', 'level': 'mix', 'summary_mode': 'statistics'}}},
            'expected_model_layers': 28, 'phases_per_run': ['prefill', 'decode'],
            'prefill_length': 2048, 'decode_length': 1, 'teacher_forced_decode': True,
            'controller_limits': {'card': 5, 'case_seconds': 900, 'controller_seconds': 1800, 'max_log_mib': 2048,
                                  'host_memory_gib': 16, 'cpus': 8},
            'expected_sdpa_calls': 28 * 2 * len(modes), 'not_an_acceptance': True}
        path = setup / (name + '.json')
        path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        cases.append({'case_id': name, 'file': path.name, 'sha256': sha(path),
                      'sdpa_calls': case['expected_sdpa_calls'], 'pilot': name.startswith('pilot')})
    manifest = {'schema': 'nes.model_coverage.plan.v1', 'cases': cases,
        'formal_sdpa_calls': 560, 'formal_controlled_prefill_calls_expected': 112,
        'formal_controlled_decode_passthrough_calls_expected': 112,
        'formal_controlled_layer_request_pairs_expected': 504,
        'unknown_eligible_call_blocks_coverage_acceptance': True,
        'scope': 'Current fixed complete Qwen3 model, B1/B8, eight explicit requests. Not arbitrary models/shapes/requests.'}
    (setup / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'CPU_CASES_FROZEN', 'manifest_sha256': sha(setup / 'manifest.json'), 'npu_calls': 0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    build(args.root.resolve(strict=True))
