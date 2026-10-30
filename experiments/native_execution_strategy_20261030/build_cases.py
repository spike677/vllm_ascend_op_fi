"""Freeze the 20-case / 56-original-call plan; no framework/device imports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--runtime-manifest', type=Path, required=True)
    parser.add_argument('--runtime-directory', type=Path, required=True)
    parser.add_argument('--collector-manifest', type=Path, required=True)
    parser.add_argument('--pod-schema', type=Path, required=True)
    parser.add_argument('--pod-schema-container-path', required=True)
    parser.add_argument('--adapter-name', required=True)
    parser.add_argument('--adapter-version', required=True)
    parser.add_argument('--adapter-certificate', type=Path)
    parser.add_argument('--adapter-certificate-container-path')
    parser.add_argument('--vendor-manifest', type=Path)
    parser.add_argument('--vendor-root')
    parser.add_argument('--phase', choices=('discovery', 'controlled'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Case directory must be fresh')
    inputs = json.loads(args.input_manifest.read_bytes())
    require(inputs['schema'] == 'nes.captured_inputs.v1', 'New task input provenance required')
    require(sha(args.collector_manifest) == '743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af',
            'Unapproved task-local high collector')
    runtime = json.loads(args.runtime_manifest.read_bytes())
    require(runtime['schema'] == 'nes.runtime_snapshot.v1', 'Expected frozen runtime inventory')
    for name, row in runtime['files'].items():
        path = args.runtime_directory / name
        require(path.is_file() and path.stat().st_size == row['bytes'] and sha(path) == row['sha256'],
                'Runtime artifact changed: ' + name)
    sys.path.insert(0, str(args.runtime_directory.resolve()))
    from native_execution_strategy.config import from_dict
    if args.phase == 'controlled':
        require(args.vendor_manifest is not None and args.vendor_root and args.adapter_certificate is not None
                and args.adapter_certificate_container_path, 'Controlled cases need actual new vendor and certificate')
        require(args.adapter_version != 'validated', 'Bootstrap must use a concrete explicit experimental adapter version')
        vendor = json.loads(args.vendor_manifest.read_bytes())
        require(vendor['schema'] == 'nes.fa.private_vendor.v1', 'Old vendor cannot validate the new strategy bridge')
    targets = (None,) if args.phase == 'discovery' else ('query_tile', 'key_group', 'joint')
    batches = (1, 2, 4, 8) if args.phase == 'discovery' else (1, 8)
    modes = ['native'] if args.phase == 'discovery' else ['native', 'controlled', 'controlled', 'off']
    rows, planned = [], []
    for target in targets:
        for batch in batches:
            for repeat in (0, 1):
                case_id = f"{args.phase}_{target + '_' if target else ''}b{batch}_r{repeat}"
                execution = {'schema_version': 1,
                    'adapter': {'name': args.adapter_name, 'version': args.adapter_version,
                                'route': 'S1S2', 'on_unsupported': 'reject'},
                    'selection': {'target': target or 'query_tile', 'policy': 'legal_alternative'},
                    'trigger': {'mode': 'every_match', 'condition': 'adaptive',
                                'predicates': [{'field': 'operator', 'op': 'eq', 'value': 'FlashAttentionScore'}]},
                    'discovery': {'max_cases': 32, 'max_elapsed_seconds': 1800.0}}
                from_dict(execution)
                case = {'schema': 'nes.harness.case.v1', 'case_id': case_id, 'phase': args.phase,
                    'operator_variant': 'stock' if args.phase == 'discovery' else 'private_vendor',
                    'batch_size': batch, 'request_count': batch, 'token_count': batch * 2048,
                    'seq_lens': [2048] * batch, 'repeat_index': repeat,
                    'mode_sequence': modes, 'native_calls_per_process': len(modes),
                    'execution_config': execution, 'selection_target': target,
                    'runtime_manifest_file': '/audit/runtime_manifest.json',
                    'runtime_manifest_sha256': sha(args.runtime_manifest),
                    'input_manifest_file': '/audit/inputs/manifest.json',
                    'input_manifest_sha256': sha(args.input_manifest),
                    'approved_pod_schema_file': args.pod_schema_container_path,
                    'approved_pod_schema_sha256': sha(args.pod_schema),
                    'capture_msprobe': True, 'statistics_variant': 'task_local_fp32_high_v1',
                    'approved_collector_manifest_file': '/audit/collector_high_a1/stage_manifest.json',
                    'approved_collector_manifest_sha256': sha(args.collector_manifest),
                    'capture_private_target_output': True, 'reference_query_indices': [0, 63, 64, 127, 128, 511, 512, 1023, 1024, 2047],
                    'native_log_required': True, 'observation_only': args.phase == 'discovery',
                    'controlled_validation_completed': False,
                    'initial_assessment': {'strategy_switched': None, 'numerics_observed': None, 'model_impact': None},
                    'execution_kind': 'original_operator_execution_not_model_forward',
                    'repeat_scope': 'Two independent processes; each controlled process shares one policy instance across two matching calls.',
                    'resource_budget': {'case_seconds': 300, 'controller_seconds': 1800, 'cases_per_controller': 8}}
                if args.phase == 'controlled':
                    case.update(approved_vendor_manifest_file='/audit/native_vendor_stage_a1/vendor_manifest.json',
                        approved_vendor_manifest_sha256=sha(args.vendor_manifest), approved_custom_opp_path=args.vendor_root,
                        adapter_certificate_file=args.adapter_certificate_container_path,
                        adapter_evidence_root='/audit/runtime/native_execution_strategy',
                        adapter_certificate_sha256=sha(args.adapter_certificate), allow_experimental=True)
                planned.append(case)
    require(len(planned) == (8 if args.phase == 'discovery' else 12), 'Wrong fixed matrix count')
    args.output.mkdir(parents=True, exist_ok=False)
    for case in planned:
        path = args.output / (case['case_id'] + '.json')
        with path.open('x', encoding='utf-8') as stream:
            json.dump(case, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        rows.append({'case_id': case['case_id'], 'file': path.name, 'sha256': sha(path),
                     'batch_size': case['batch_size'], 'repeat_index': case['repeat_index'],
                     'native_calls': case['native_calls_per_process'], 'selection_target': case['selection_target']})
    manifest = {'schema': 'nes.harness.matrix.v1', 'phase': args.phase, 'cases': rows,
                'original_native_calls': sum(row['native_calls'] for row in rows),
                'combined_preregistered_plan': {'discovery_cases': 8, 'controlled_cases': 12, 'total_cases': 20, 'native_calls': 56},
                'new_B2_B4_controlled_support': False, 'model_impact': None,
                'not_established': 'Requested legal candidate is intent; actual strategy and numerical outcomes require independent evidence.'}
    with (args.output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'FROZEN_CASES_NOT_EXECUTED', 'cases': len(rows),
                      'native_calls': manifest['original_native_calls'], 'manifest_sha256': sha(args.output / 'manifest.json')}))


if __name__ == '__main__':
    main()
