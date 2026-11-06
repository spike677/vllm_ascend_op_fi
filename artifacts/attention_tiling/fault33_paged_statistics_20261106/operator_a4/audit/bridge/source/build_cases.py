"""Seal a bounded stock/OFF bridge, B/F statistics pair and OFF restoration."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vendor-manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads(args.vendor_manifest.read_bytes())
    if manifest['schema'] != 'fault33.pa.private_vendor.v1':
        raise ValueError('Wrong vendor schema')
    args.output.mkdir(parents=True, exist_ok=False)
    common = dict(schema='fault33.pa.replay.case.v1', batch=8, q_heads=16, kv_heads=8, head_dim=128,
                  kv_length=16384, dtype='float16', block_size=128, inner_precise=0,
                  repeats=2, seed=330011, projection_seed=330012, deterministic_algorithms=True,
                  capture_msprobe=True, statistics_variant='task_local_fp32_high_v1',
                  approved_collector_manifest_file='/audit/collector_high_a1/stage_manifest.json',
                  approved_collector_manifest_sha256='743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af')
    rows = []
    for name, mode, variant, capture in [('stock_bridge', 'off', 'stock', False),
                                        ('baseline', 'off', 'private_vendor', True),
                                        ('fault', 'inner1024', 'private_vendor', True),
                                        ('restore', 'off', 'private_vendor', False)]:
        case = dict(common, case_id=name, mode=mode, operator_variant=variant, capture_msprobe=capture)
        if variant == 'private_vendor':
            case.update(approved_vendor_manifest_file='/audit/vendor_stage_a4/vendor_manifest.json',
                        approved_vendor_manifest_sha256=sha(args.vendor_manifest),
                        approved_custom_opp_path='/audit/' + manifest['vendor_root'])
        if not capture:
            case.update(statistics_variant='official_low')
            for key in ('approved_collector_manifest_file', 'approved_collector_manifest_sha256'):
                case.pop(key)
        if name == 'restore':
            case.update(repeats=4, mode_sequence=['off', 'inner1024', 'inner1024', 'off'])
        path = args.output / (name + '.json')
        path.write_text(json.dumps(case, indent=2) + '\n')
        rows.append({'case_id': name, 'file': path.name, 'sha256': sha(path), 'native_calls': case['repeats']})
    (args.output / 'matrix_manifest.json').write_text(json.dumps({
        'schema': 'fault33.pa.fixed_geometry.matrix.v1', 'cases': rows,
        'seed_contract': 'All cases use identical request-keyed CPU generator and projection seed.',
        'acceptance_order': 'stock/OFF complete outputs and visible plan bridge first; only then fault and restoration.',
        'formal_statistics_groups': ['baseline', 'fault'],
        'registered_geometry': {k: common[k] for k in ('batch', 'q_heads', 'kv_heads', 'head_dim', 'kv_length', 'block_size')}
    }, indent=2) + '\n')
    print(json.dumps({'status': 'SEALED_NOT_EXECUTED', 'cases': len(rows),
                      'original_api_calls': sum(row['native_calls'] for row in rows)}))


if __name__ == '__main__':
    main()
