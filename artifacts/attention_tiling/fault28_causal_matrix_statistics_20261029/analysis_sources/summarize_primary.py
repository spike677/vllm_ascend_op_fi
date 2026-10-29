"""Compact descriptive view of an accepted primary report; CPU only.

Adds the predeclared joint-minus-base contrast from SHA-bound private endpoints.
It does not change the original report, infer a hardware reduction tree, or read
msprobe/control logs. Private tensor values are never returned in this summary.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact(c):
    queries = [i for i, n in enumerate(c['nonzero_by_query']) if n]
    return {k: c[k] for k in ('nonzero_elements', 'max_abs', 'exact_int64_le_sha256')} | {
        'bit_different_elements': c.get('bit_different_elements'),
        'changed_queries_any_head': len(queries),
        'earliest_changed_query': min(queries, default=None),
        'latest_changed_query': max(queries, default=None),
        'last_query_changed_elements': c['nonzero_by_query'][2047],
        'last_query_max_abs': c['max_abs_units_by_query'][2047] * 2.0 ** -24,
    }


def units(bits):
    b = bits.astype(np.int64)
    e, f = (b >> 10) & 31, b & 1023
    require(bool(np.all(e != 31)), 'Nonfinite endpoint')
    magnitude = np.where(e == 0, f, (1024 + f) << np.maximum(e - 1, 0))
    return np.where(b & 32768, -magnitude, magnitude)


def describe(a, z):
    diff = units(z) - units(a)
    nonzero = diff != 0
    by_query = nonzero.sum(axis=(0, 2))
    max_query = np.abs(diff).max(axis=(0, 2))
    return compact({
        'nonzero_elements': int(nonzero.sum()),
        'max_abs': int(np.abs(diff).max()) * 2.0 ** -24,
        'bit_different_elements': int((a != z).sum()),
        'nonzero_by_query': by_query.tolist(),
        'max_abs_units_by_query': max_query.tolist(),
        'exact_int64_le_sha256': sha(diff.astype('<i8', copy=False).tobytes()),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--results-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite a derived report')
    raw = args.report.read_bytes()
    d = json.loads(raw)
    require(d['status'] == 'ANALYZED_FULL_SEQUENCE_FACTORIAL', 'Primary evidence gate not met')
    compact_contrasts = {k: compact(v) for k, v in d['full_sequence_numerical_contrasts']['contrasts'].items()}
    sources, endpoints = {}, {}
    for c in d['calls']:
        if c['repeat_index'] != 0:
            continue
        p = args.results_dir / c['case_id'] / 'result.json'
        result_raw = p.read_bytes()
        require(sha(result_raw) == d['evidence_sha256'][str(p)], 'Result is not bound to accepted report')
        r = json.loads(result_raw)
        meta = r['private_target_output']
        require(meta['shape'] == [16, 2048, 128] and meta['bytes'] == 8388608 and
                meta['dtype'] == 'torch.float16' and meta['byteorder'] == 'little', 'Endpoint geometry mismatch')
        endpoint = (p.parent / meta['file']).read_bytes()
        require(len(endpoint) == meta['bytes'] and sha(endpoint) == meta['sha256'] == c['target_output_sha256'],
                'Private endpoint does not match accepted output SHA')
        endpoints[c['batch_size'], c['profile']] = np.frombuffer(endpoint, dtype='<u2').reshape(16, 2048, 128)
        sources[c['case_id']] = {'result_sha256': sha(result_raw), 'private_output_sha256': sha(endpoint)}
    joint = {}
    for b in (1, 8):
        # Cross-check this independent exact decoder against the existing report.
        for before, after, name in [('base', 'm_only', 'M_at_ratio8'), ('base', 'r_only', 'ratio_at_M128')]:
            require(describe(endpoints[b, before], endpoints[b, after]) == compact_contrasts[f'B{b}.{name}'],
                    'Independent exact decoder disagrees with existing primary contrast')
        joint[str(b)] = describe(endpoints[b, 'base'], endpoints[b, 'joint'])
    references = {x['case_id']: x['by_query'] for x in d['fixed_query_fp64_reference']['cases']}
    error_directions = {}
    for b in (1, 8):
        for p in ('m_only', 'r_only', 'joint'):
            baseline, changed = references[f'base_b{b}_r0'], references[f'{p}_b{b}_r0']
            report = {}
            for field in ('rms_error', 'max_abs_error'):
                categories = {'lower': [], 'equal': [], 'higher': []}
                for a, z in zip(baseline, changed):
                    require(a['query'] == z['query'], 'FP64 reference query mismatch')
                    categories['lower' if z[field] < a[field] else 'higher' if z[field] > a[field] else 'equal'].append(a['query'])
                report[field] = categories
            error_directions[f'B{b}.{p}_minus_base'] = report
    plans = {k: {field: v[field] for field in ('target_queries_compared', 'changed_query_counts',
             'last_query_changed_head_counts', 'bmm_changed_fields', 'legal_key_set_unchanged',
             'query_split_between_cores', 'floating_point_instruction_order') if field in v}
             for k, v in d['all_query_source_plan_comparisons'].items()}
    result = {
        'schema': 'fault28.causal.primary_descriptive_summary.v1',
        'status': 'DESCRIPTIVE_SUMMARY_OF_ACCEPTED_PRIMARY_REPORT',
        'primary_report_sha256': sha(raw), 'script_sha256': sha(Path(__file__).read_bytes()),
        'primary_output_shape': [16, 2048, 128], 'exact_unit': '2^-24',
        'existing_contrasts': compact_contrasts, 'joint_minus_base': joint,
        'fp64_reference_error_directions': error_directions,
        'reference_queries': d['fixed_query_fp64_reference']['queries'],
        'fp64_error_direction_is_not_acceptance': True,
        'source_plan_comparisons': plans,
        'auxiliary_pair_comparisons': d['auxiliary_pair_comparisons'],
        'auxiliary_repeat_bit_stability': d['auxiliary_repeat_bit_stability'],
        'inputparams_alignment_policy': d['alignment_padding_policy'],
        'output_sources': sources,
        'private_tensor_values_in_report': False,
        'msprobe_used_to_assign_cause': False,
        'unique_reduction_instruction_cause': 'NOT_IDENTIFIED',
        'historical_fia_defect_reproduced': False,
        'scope': 'Repeatable intervention contrasts for fixed captured operator inputs; not model quality or a hardware instruction trace.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=True, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'joint_minus_base': joint, 'output_sha256': sha(args.output.read_bytes())}))


if __name__ == '__main__':
    main()
