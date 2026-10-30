"""Combine source/native-call audits; no tensors, runtime imports or devices."""
import argparse
import hashlib
import json
from pathlib import Path


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audit-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    files = ['discovery_a4_review_r3.json', 'controlled_query_tile_a1_review_r3.json',
             'controlled_key_group_a1_review.json', 'controlled_joint_a1_review.json']
    reports = [json.loads((args.audit_dir / name).read_bytes()) for name in files]
    require(all(x['status'] == 'PASS_NATIVE_SOURCE_AND_POD_REVIEW' for x in reports), 'Native audit incomplete')
    require(len({x['audit_source_sha256'] for x in reports}) == 1, 'Audit implementation differs')
    require(len({x['capture_sources']['manifest_sha256'] for x in reports}) == 1, 'Frozen capture generation differs')
    require(all(x['source_binding'] == reports[0]['source_binding'] for x in reports), 'Pinned source/build/schema differs')
    rows = [row for report in reports for row in report['cases']]
    require(len(rows) == 56, 'Expected 56 actual original calls')
    stock = {row['case_id']: row for row in reports[0]['cases']}
    require(set(stock) == {f'discovery_b{b}_r{r}' for b in (1, 2, 4, 8) for r in (0, 1)}, 'Stock inventory differs')
    stock_summary = []
    for b in (1, 2, 4, 8):
        a, z = stock[f'discovery_b{b}_r0'], stock[f'discovery_b{b}_r1']
        require(a['execution_plan_sha256'] == z['execution_plan_sha256']
                and a['input_contract_sha256'] == z['input_contract_sha256'], 'Stock repeat semantic plan differs')
        stock_summary.append({'actual_batch': b, 'actual_factors': a['actual_factors'],
            'actual_profile': a['actual_profile'], 'key': a['key'], 'block_dim': a['block_dim'],
            'execution_plan_sha256': a['execution_plan_sha256'],
            'input_contract_sha256': a['input_contract_sha256'], 'repeat_semantics_exact': True,
            'repeat_complete_raw_equal': a['raw_sha256'] == z['raw_sha256'],
            'section_sha256': {k: v['sha256'] for k, v in a['unexpanded_sdk_sections'].items()},
            'controlled_support_claimed': False})
    grouped = {}
    for row in rows:
        if row['case_id'].startswith('controlled_'):
            grouped.setdefault(row['case_id'], []).append(row)
    require(set(grouped) == {f'controlled_{target}_b{b}_r{r}'
        for target in ('query_tile', 'key_group', 'joint') for b in (1, 8) for r in (0, 1)}, 'Controlled case inventory differs')
    chains, repeat_plans = [], {}
    for target in ('query_tile', 'key_group', 'joint'):
        for b in (1, 8):
            for repeat in (0, 1):
                name = f'controlled_{target}_b{b}_r{repeat}'
                calls = sorted(grouped[name], key=lambda x: x['step'])
                require(len(calls) == 4 and [c['step'] for c in calls] == [0, 1, 2, 3], 'Four-call sequence missing')
                require([c['mode'] for c in calls] == ['native', 'controlled', 'controlled', 'off'], 'Mode sequence differs')
                require([c['selection_applied'] for c in calls] == [False, True, True, False], 'All-match selector events differ')
                require(all(c['host_callback_checked'] and c['host_raw_byte_exact'] for c in calls), 'Missing actual host/launch binding')
                require(all(c['sequence_invariants'] == calls[0]['sequence_invariants'] for c in calls)
                        and all(calls[0]['sequence_invariants'].values()), 'Per-sequence invariants failed')
                base = stock[f'discovery_b{b}_r{repeat}']
                for index in (0, 3):
                    require(calls[index]['execution_plan_sha256'] == base['execution_plan_sha256']
                            and calls[index]['input_contract_sha256'] == base['input_contract_sha256'],
                            'Private native/off differs semantically from same-B stock')
                require(calls[1]['actual_factors'] == calls[2]['actual_factors']
                        and calls[1]['actual_factors'] != calls[0]['actual_factors'], 'Actual selector projection did not repeat/change')
                key = (target, b)
                signature = [(c['execution_plan_sha256'], c['input_contract_sha256']) for c in calls]
                if key in repeat_plans:
                    require(repeat_plans[key] == signature, 'Independent-process plan repeat differs')
                repeat_plans[key] = signature
                chains.append({'case_id': name, 'actual_batch': b, 'requested_target_label': target,
                    'actual_factors_sequence': [c['actual_factors'] for c in calls],
                    'native_factors_before_generation': [c['host_native_factors'] for c in calls],
                    'actual_strategy_changed_calls': 2, 'selection_applied_sequence': [c['selection_applied'] for c in calls],
                    'input_contract_34_same_all_steps': True, 'execution_plan_169_controlled_repeat_equal': True,
                    'private_native_and_off_169_plan_and_34_input_equal_same_batch_stock': True,
                    'raw_launch_bytes_equal_host_each_call': True})
    report = {'schema': 'nes.fa.independent_native_matrix_review.v1',
        'status': 'PASS_INDEPENDENT_NATIVE_STRATEGY_AUDIT', 'audit_merge_source_sha256': sha(__file__),
        'reports': [{'path': name, 'sha256': sha(args.audit_dir / name)} for name in files],
        'capture_source_manifest_sha256': reports[0]['capture_sources']['manifest_sha256'],
        'capture_source_count': reports[0]['capture_sources']['file_count'],
        'source_binding': reports[0]['source_binding'], 'actual_original_calls': 56,
        'stock_calls': 8, 'private_calls': 48, 'private_cases': 12,
        'actual_controlled_strategy_changes': 24, 'private_native_calls': 12, 'off_recovery_calls': 12,
        'stock_discovery': stock_summary, 'controlled_sequences': chains,
        'all_private_native_off_match_same_batch_stock_169_plan_and_34_input': True,
        'controlled_two_matches_and_independent_process_repeats_stable': True,
        'observed_stock_selector_projection_same_across_batches': len({json.dumps(x['actual_factors'], sort_keys=True) for x in stock_summary}) == 1,
        'observed_stock_full_execution_plan_changes_across_batches': len({x['execution_plan_sha256'] for x in stock_summary}) > 1,
        'numerical_effect_claimed': False, 'model_effect_claimed': False, 'new_certificate_issued': False,
        'limitations': ['This is actual native strategy/provenance evidence; numeric and model effects require separate evidence.',
            'B2/B4 remain native observation only, not controlled support.',
            'The 192 bytes in three public SDK sections are bound as whole sections; their internal fields are not independently decoded.',
            'Only exact source-audited five alignment members are excluded semantically; complete raw bytes remain bound.',
            'A stock call has no private host callback; its log framing is independently reconstructed, with private native/off bridges separately matching all 169 plan items and 34 input fields.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'status': report['status'], 'sha256': sha(args.output), 'actual_calls': 56, 'output': str(args.output)}))


if __name__ == '__main__':
    main()
