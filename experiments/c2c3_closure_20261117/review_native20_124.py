"""Read-only review of saved actual QMM group receipts; no runtime admission.

Statistics are deliberately separate inputs to the independent raw comparator.
This audit never uses runtime records to choose a statistics first difference.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from fault_injection.coverage_audit import audit_saved_coverage, index_artifacts


def load(path):
    return json.loads(path.read_bytes())


def file_record(path):
    return dict(path=str(path), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def inventory(snapshot):
    if not snapshot['complete'] or snapshot['overflow_calls']:
        raise ValueError('INCOMPLETE_NATIVE_INVENTORY')
    if sum(row['observations'] for row in snapshot['targets']) != snapshot['eligible_run_calls']:
        raise ValueError('NATIVE_INVENTORY_TOTAL_MISMATCH')
    return {row['identity_sha256']: row['observations'] for row in snapshot['targets']
            if row['native']['operator'] == 'aclnnQuantMatmulWeightNz'}


def review(folder, *, c2=None):
    path = folder/'runtime/serving_receipt.json'
    data = load(path)
    qualification = data['qualification']
    coverage = qualification['coverage']
    files = sorted((folder/'group').glob('*.json')) + sorted((folder/'native_proofs').glob('*.json'))
    audit = audit_saved_coverage(coverage, artifact_index=index_artifacts(files), c2_report=c2)
    errors, phases, references = [], [], {}
    if not audit['report_consistent'] or audit['missing_artifact_sha256']:
        errors.append('COVERAGE_LEDGER_OR_ACTUAL_ARTIFACT_BYTES_FAILED')
    if qualification['failure'] or not qualification['recovery_verified']:
        errors.append('RECOVERY_NOT_CONFIRMED')
    expected_sites = expected_delta = expected_public_modules = None
    healthy_output = data['native_phases'][0]['model_output_sha256']
    input_hash = data['native_phases'][0]['input_sha256']
    for phase in data['native_phases']:
        name, events = phase['phase'], phase['native_events']
        fault = name in ('fault', 'fault_repeat')
        before, after = inventory(phase['native_before']), inventory(phase['native_after'])
        delta = {key: value-before.get(key, 0) for key, value in after.items()
                 if value != before.get(key, 0)}
        # native_identity identifies a native cache registration; the separately
        # recorded independent_native_sha256 hashes the entire moving inventory.
        observed = Counter(event['native_identity'] for event in events)
        sites = [tuple(event['site']) for event in events]
        public_modules = [row['module_object_identity'] for row in phase['projections']]
        if expected_sites is None:
            expected_sites = sites
            expected_delta, expected_public_modules = delta, public_modules
        # The final disabled OFF intentionally has no intervention callback.
        # The independent native inventory and public observer both stay live.
        callback_ok = (not events if name == 'off' else
                       sites == expected_sites and len(sites) == len(set(sites))
                       and dict(observed) == delta)
        if (not callback_ok or delta != expected_delta
                or public_modules != expected_public_modules
                or sum(delta.values()) != phase['independent_native_qmm_calls']):
            errors.append(name+':COMPLETE_NATIVE_DENOMINATOR_OR_TARGET_SET_CHANGED')
        if any(not row['public_chain_verified'] or row['errors'] for row in phase['projections']):
            errors.append(name+':PUBLIC_PRODUCER_LINEAGE_FAILED')
        if len(phase['projections']) != sum(delta.values()):
            errors.append(name+':PUBLIC_NATIVE_CALL_COUNT_MISMATCH')
        if phase['input_sha256'] != input_hash:
            errors.append(name+':MODEL_INPUT_CHANGED')
        effects = Counter()
        for event in events:
            site, receipt = tuple(event['site']), event.get('receipt')
            typed = event['current_typed_call']
            if (event['original_calls'] != 1 or not event['public_lineage_verified']
                    or not typed['executor_bound'] or not typed['formal_params_current']
                    or typed['cached_layout_used']):
                errors.append(name+':ACTUAL_CONTINUATION_OR_TYPED_FORMALS_UNPROVEN')
            if receipt is None:
                if fault:
                    errors.append(name+':MISSING_FAULT_RECEIPT')
                continue
            if (receipt['native_status'] != 0 or receipt['original_calls'] != 1
                    or not receipt['resources_released']
                    or receipt['input_sha256_before'] != receipt['input_sha256_after']):
                errors.append(name+':NATIVE_EXECUTION_OR_ORIGINAL_STORAGE_FAILED')
            if name == 'reference':
                references[site] = receipt
            if fault:
                if (receipt['replacements'] != 1 or receipt['changed_slots'] != 1
                        or receipt['replaced_formals'] != ['x1']
                        or not receipt['temporary_readback_verified']
                        or event['native_effect_state'] != 'FAULT_TRANSFORM_READ_BACK'):
                    errors.append(name+':FAULT_TRANSFORMATION_NOT_VERIFIED')
                effects['actual_interventions'] += 1
                effects['output_changed_vs_healthy'] += receipt['output_sha256'] != references[site]['output_sha256']
                effects['propagated_input_differs_from_healthy'] += not receipt['group_v3']['input_equal_healthy_reference']
            elif name in ('lossless', 'qualification_off', 'baseline', 'off_observed', 'off'):
                if (receipt['input_sha256_before'] != references[site]['input_sha256_before']
                        or receipt['output_sha256'] != references[site]['output_sha256']):
                    errors.append(name+':PER_TARGET_HEALTHY_OR_OFF_NOT_EXACT')
                if name == 'lossless' and not receipt['temporary_readback_verified']:
                    errors.append(name+':LOSSLESS_READBACK_MISSING')
        if not fault and phase['model_output_sha256'] != healthy_output:
            errors.append(name+':COMPLETE_HEALTHY_LOGITS_NOT_EXACT')
        phases.append(dict(phase=name, request_ids=phase['request_ids'],
            targets=len(events), independent_native_calls=sum(delta.values()), effects=dict(effects),
            model_output_sha256=phase['model_output_sha256'], model_output_finite=phase['model_output_finite']))
    result = dict(original_receipt=file_record(path), scope=coverage['scope'],
        coverage_audit=audit, phases=phases, checks_passed=not errors,
        errors=sorted(set(errors)), qualifications_granted=False,
        statistical_localization='SEPARATE_RAW_BF_COMPARATOR_ONLY')
    return result, coverage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    c2_result, c2 = review(args.attempt_case/'c2')
    c3_result, _ = review(args.attempt_case/'c3', c2=c2)
    result = dict(c2=c2_result, c3=c3_result, qualifications_granted=False)
    (args.out/'review.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({name: dict(checks_passed=row['checks_passed'], errors=row['errors'],
        counts=row['coverage_audit']['recomputed'],
        missing_artifacts=len(row['coverage_audit']['missing_artifact_sha256']))
        for name, row in result.items() if name in ('c2', 'c3')}))


if __name__ == '__main__':
    main()
