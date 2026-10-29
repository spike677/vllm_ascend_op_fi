"""Freeze a 2 batch x 4 requested profile x 2 independent repeat replay matrix."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-manifest', type=Path, required=True)
    p.add_argument('--vendor-manifest', type=Path)
    p.add_argument('--vendor-root', help='Approved absolute container path, set before process start')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--statistics', choices=('none', 'low', 'high'), default='high')
    p.add_argument('--phase', choices=('primary', 'automatic', 'stock_bridge'), default='primary')
    p.add_argument('--collector-manifest', type=Path)
    a = p.parse_args()
    assert not a.output.exists()
    inputs = json.loads(a.input_manifest.read_bytes())
    assert inputs['schema'] == 'fault28.causal.inputs.v1'
    if a.phase != 'stock_bridge':
        assert a.vendor_manifest is not None and a.vendor_root
        vendor_sha = sha(a.vendor_manifest)
    else:
        vendor_sha = None
    if a.statistics == 'high':
        assert a.collector_manifest is not None
        collector_sha = sha(a.collector_manifest)
        assert collector_sha == '743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af'
    a.output.mkdir(parents=True)
    rows = []
    profiles = [('base', (128, 8)), ('m_only', (64, 8)), ('r_only', (128, 4)), ('joint', (64, 4))]
    if a.phase == 'automatic':
        profiles = [('batch_m', (64, 8)), ('batch_r', (128, 4)), ('batch_joint', (64, 4))]
    elif a.phase == 'stock_bridge':
        profiles = [('base', (128, 8))]
    for profile, requested in profiles:
        for batch in (1, 8):
            effective_intent = (128, 8) if a.phase == 'automatic' and batch == 1 else requested
            for repeat in range(2):
                name = f'{profile}_b{batch}_r{repeat}'
                case = {'schema': 'fault28.causal.case.v1', 'case_id': name, 'profile': profile,
                    'operator_variant': 'stock' if a.phase == 'stock_bridge' else 'private_vendor',
                    'stock_bridge': a.phase == 'stock_bridge',
                    'requested_m': effective_intent[0], 'requested_n_ratio': effective_intent[1],
                    'profile_legality_and_execution': 'REQUIRES_SOURCE_AND_NATIVE_HOST_EVIDENCE',
                    'batch_size': batch, 'repeat_index': repeat, 'native_calls_per_process': 1,
                    'input_manifest_file': '/audit/inputs/manifest.json', 'input_manifest_sha256': sha(a.input_manifest),
                    'approved_vendor_manifest_file': '/audit/vendor_stage_a1/vendor_manifest.json',
                    'approved_vendor_manifest_sha256': vendor_sha, 'approved_custom_opp_path': a.vendor_root,
                    'capture_msprobe': a.statistics != 'none', 'statistics_variant': 'task_local_fp32_high_v1' if a.statistics == 'high' else 'official_low',
                    'capture_private_target_output': True, 'private_last_query_comparison': True,
                    'reference_query_indices': [0, 63, 64, 127, 128, 511, 512, 1023, 1024, 2047],
                    'phase': a.phase,
                    'repeat_scope': 'Separate process, one native call per case; each factor combination has two repeats'}
                if a.phase == 'stock_bridge':
                    case['case_id'] = name = f'stock_bridge_b{batch}_r{repeat}'
                    for key in ('approved_vendor_manifest_file', 'approved_vendor_manifest_sha256', 'approved_custom_opp_path'):
                        case.pop(key)
                if a.statistics == 'high':
                    case.update(approved_collector_manifest_file='/audit/collector_high_a1/stage_manifest.json',
                                approved_collector_manifest_sha256=collector_sha)
                path = a.output / (name + '.json')
                path.write_text(json.dumps(case, indent=2) + '\n', encoding='utf-8')
                rows.append({'case_id': name, 'file': path.name, 'sha256': sha(path), 'profile': profile,
                             'batch_size': batch, 'repeat_index': repeat})
    (a.output / 'manifest.json').write_text(json.dumps({'schema': 'fault28.causal.matrix.v1',
        'phase': a.phase, 'cases': rows, 'primary_endpoint': 'Complete target batch0 output all16heads x2048queries x128; exact FP16 values. Preregistered query boundaries also have CPU FP64 references.',
        'not_established': 'Requested profile labels are not actual tiling evidence; require typed host/raw and launch validation.'}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'CASES_FROZEN_NOT_EXECUTED', 'count': len(rows), 'manifest_sha256': sha(a.output / 'manifest.json')}))


if __name__ == '__main__':
    main()
