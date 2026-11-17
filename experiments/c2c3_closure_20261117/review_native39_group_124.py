"""CPU-only independent review of recorded Native39 groups and raw statistics.

Creates fresh sidecars, never edits receipts or grants runtime qualification.
Statistics comparisons receive only triad/config/collector files, never targets.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from fault_injection.coverage_audit import audit_saved_coverage, index_artifacts
from fault_injection.statistics_audit import compare_raw_statistics, compact_summary


def load(path):
    return json.loads(Path(path).read_bytes())


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest.hexdigest())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return file_record(path)


def target_key(row):
    """Cache registration alone is reused across real modules; include site."""
    if len(row['native_delta']) != 1 or row['native_delta'][0]['count'] != 1:
        raise ValueError('ONE_NATIVE_RUN_PER_REVIEWED_PUBLIC_CALL_REQUIRED')
    return tuple(row['site']), tuple(row['native_delta'][0]['key'])


def fault_effect_errors(effects, native_count):
    # A valid continuation/replacement is not itself a numerical fault effect.
    # Every planned target needs independent output-byte evidence of change.
    if native_count <= 0 or effects.get('changed_outputs', 0) != native_count:
        return ['PER_TARGET_NUMERICAL_EFFECT_INCOMPLETE']
    return []


def review_stage(folder, driver, identity, out, artifact_index, c2=None):
    path = folder / 'runtime/serving_receipt.json'
    data = load(path)
    q, phases = data['qualification'], data['native_phases']
    coverage = q['coverage']
    audit = audit_saved_coverage(coverage, artifact_index=artifact_index, c2_report=c2)
    audit_ref = write(out / 'coverage-audit.json', audit)
    errors, summaries, references, fault_outputs = [], [], {}, None
    healthy, input_hash = phases[0]['model_output_sha256'], phases[0]['input_sha256']
    expected = ['discovery_1', 'discovery_2', 'reference', 'lossless_self', 'lossless_mat2', 'lossless_dual', 'fault']
    if coverage['trigger'] == 'every_match':
        expected.append('fault_repeat')
    expected += ['off_observed', 'off']
    if [p['phase'] for p in phases] != expected:
        errors.append('INCOMPLETE_OR_REORDERED_PHASE_MATRIX')
    if not audit['report_consistent'] or audit['missing_artifact_sha256']:
        errors.append('SAVED_COVERAGE_OR_REFERENCED_BYTES_FAILED')
    if q['failure'] or not q['recovery_verified']:
        errors.append('GROUP_RECOVERY_NOT_VERIFIED')
    if len(data['requests']) != len(phases) or len(driver['phases']) != len(phases):
        raise ValueError('REQUEST_PHASE_MAPPING_INCOMPLETE')
    discovered, fault_models = None, []
    for i, phase in enumerate(phases):
        name, rows = phase['phase'], phase['calls']
        is_fault = name in ('fault', 'fault_repeat')
        native = [row for row in rows if row['native_delta']]
        keys = [target_key(row) for row in native]
        if len(keys) != len(set(keys)):
            errors.append(name + ':DUPLICATE_SITE_CACHE_LIFETIME')
        if discovered is None:
            discovered = set(keys)
        if set(keys) != discovered:
            errors.append(name + ':TARGET_UNIVERSE_CHANGED')
        if (not phase['independent_denominator_complete'] or
                sum(d['count'] for row in rows for d in row['native_delta']) != phase['native_denominator'] or
                len(native) != phase['native_denominator']):
            errors.append(name + ':INDEPENDENT_NATIVE_DENOMINATOR_MISMATCH')
        if any(row['original_calls'] != 1 for row in rows):
            errors.append(name + ':ORIGINAL_CONTINUATION_COUNT')
        if any(row.get('native') is not None for row in rows if not row['native_delta']):
            errors.append(name + ':NATIVE_INTERVENTION_ON_NON_TARGET')
        snapshot = data['requests'][i]['snapshot']
        mapping = driver['phases'][i]['mapping']
        if not (phase['request_id'] == snapshot['internal_request_id'] == mapping['internal_request_id']
                and driver['phases'][i]['phase'] == name and snapshot['num_requests'] == 1
                and snapshot['num_computed_tokens'] == 0 and snapshot['token_ids'] == data['requests'][0]['snapshot']['token_ids']
                and snapshot['positions'] == list(range(len(snapshot['token_ids'])))):
            errors.append(name + ':ACTUAL_FRESH_REQUEST_MAPPING_CHANGED')
        if phase['input_sha256'] != input_hash:
            errors.append(name + ':ACTUAL_MODEL_INPUT_CHANGED')
        if name == 'reference':
            references = {target_key(row): row['native']['last_observation'] for row in native}
        effects, current_outputs = Counter(), {}
        for row in native:
            state = row.get('native')
            if state is None:
                if name not in ('discovery_1', 'discovery_2', 'off'):
                    errors.append(name + ':MISSING_NATIVE_OBSERVATION')
                continue
            obs = state['last_observation']; proof = obs['proof']; key = target_key(row)
            if not all(obs[k] == 0 for k in ('pre_sync_status', 'copy_status', 'run_status', 'post_sync_status', 'free_status')):
                errors.append(name + ':NATIVE_EXECUTION_OR_SYNCHRONIZATION_FAILED')
            if state['inflight'] or state['leases']:
                errors.append(name + ':NATIVE_RESOURCES_PENDING')
            if obs['original_storage_sha256_before'] != obs['original_storage_sha256_after']:
                errors.append(name + ':ORIGINAL_STORAGE_CHANGED')
            if (proof['api'] != 'aclnnMatmul' or proof['slots'] != [0, 1, 2] or not proof['formal_order_proven']
                    or state['runtime_identity'] != 'MATCHED_BUILD' or state['exact_cache_api_version'] != 2):
                errors.append(name + ':ACTUAL_FORMALS_OR_RUNTIME_UNPROVEN')
            if (state['owner'] != row['actual_native_owner'] or state['generation'] != row['actual_native_generation']
                    or obs['owner'] != state['owner'] or obs['generation'] != state['generation']):
                errors.append(name + ':ACTUAL_NATIVE_OWNER_OR_GENERATION_MISMATCH')
            if json.loads(obs['actual_request_ids_json']) != [phase['request_id']]:
                errors.append(name + ':NATIVE_REQUEST_ID_MISMATCH')
            if name.startswith('lossless') or name == 'off_observed':
                if (obs['output_storage_sha256'] != references[key]['output_storage_sha256'] or
                        obs['original_storage_sha256_before'] != references[key]['original_storage_sha256_before']):
                    errors.append(name + ':TARGET_HEALTHY_OR_OFF_NOT_EXACT')
            if is_fault:
                fields = ('input_device_readback_verified', 'self_constant_positive_verified',
                          'per_column_sign_counts_verified', 'theoretical_output_exact_zero', 'amplitude_equal')
                if not all(obs.get(k) is True for k in fields):
                    errors.append(name + ':FINITE_CONSTRUCTION_PROOF_MISSING')
                if not all(obs[k]['finite'] == obs[k]['total'] > 0 for k in ('temporary_self', 'temporary_mat2')):
                    errors.append(name + ':SUBSTITUTED_INPUT_NOT_FINITE')
                if proof.get('input_mask') != 3 or any((t['original_address'] != t['run_address']) != (j < 2)
                                                      for j, t in enumerate(proof['tensors'])):
                    errors.append(name + ':REPLACED_FORMAL_MISMATCH')
                effects['changed_outputs'] += obs['output_storage_sha256'] != references[key]['output_storage_sha256']
                effects['original_input_nonfinite_from_propagation'] += any(obs[k]['finite'] != obs[k]['total'] for k in ('original_self', 'original_mat2'))
                effects['nonfinite_outputs'] += obs['output']['finite'] != obs['output']['total']
                current_outputs[key] = obs['output_storage_sha256']
            elif name.startswith('lossless'):
                if not obs['input_device_readback_verified'] or obs['changed'] != [0, 0]:
                    errors.append(name + ':LOSSLESS_READBACK_FAILED')
            elif name == 'off_observed':
                if obs['changed'] != [0, 0] or any(t['original_address'] != t['run_address'] for t in proof['tensors']):
                    errors.append(name + ':OFF_SUBSTITUTION_PRESENT')
        if is_fault:
            errors.extend(name + ':' + reason for reason in fault_effect_errors(effects, len(native)))
            if fault_outputs is not None and current_outputs != fault_outputs:
                errors.append(name + ':PER_TARGET_FAULT_REPEAT_CHANGED')
            fault_outputs = current_outputs
            fault_models.append(phase['model_output_sha256'])
        elif phase['model_output_sha256'] != healthy:
            errors.append(name + ':COMPLETE_HEALTHY_LOGITS_NOT_EXACT')
        summaries.append(dict(phase=name, public_calls=len(rows), native_eligible_calls=len(native),
            non_target_public_calls=len(rows)-len(native), distinct_cache_keys=len({key[1] for key in keys}),
            actual_request_id=phase['request_id'], input_sha256=phase['input_sha256'], effects=dict(effects),
            model_output_sha256=phase['model_output_sha256'], model_output_finite=phase['model_output_finite']))
    if len(set(fault_models)) != 1:
        errors.append('FULL_FAULT_REPEAT_NOT_EXACT')
    for stage in ('finish', 'unregister'):
        state = driver[stage]['provider_state']
        if state['enabled'] or state['inflight'] or state['leases'] or driver[stage]['recovery_required']:
            errors.append(stage + ':FINAL_PROVIDER_RESOURCES_PENDING')
        if driver[stage]['ticket']['worker'] != identity:
            errors.append(stage + ':WORKER_IDENTITY_CHANGED')
    if driver['unbind']['status'] != 'SERVING_DEPLOYMENT_REMOVED':
        errors.append('FINAL_SERVING_BINDING_PRESENT')
    result = dict(receipt=file_record(path), scope=coverage['scope'], counts=audit['recomputed'],
        coverage_audit=audit_ref, reference_count=audit['referenced_artifacts'],
        missing_artifacts=audit['missing_artifact_sha256'], phases=summaries,
        checks_passed=not errors, errors=sorted(set(errors)), qualifications_granted=False)
    write(out / 'summary.json', result)
    return result, coverage


def statistics(folder, out):
    results = {}
    for label, first, second in [('baseline_self', 2, 2), ('baseline_fault', 2, 6)]:
        capture = folder / 'statistics'
        required = [capture / f'observation/step{step}/rank0' / name
                    for step in (first, second) for name in ('dump.json', 'stack.json', 'construct.json')]
        missing = sorted({str(path) for path in required if not path.is_file()})
        if missing:
            summary = dict(status='MISSING_RAW_CAPTURE', missing_files=missing,
                           e5_statistics_comparable=False, first_exceedance=None,
                           qualifications_granted=False)
            write(out / label / 'summary.json', summary)
            results[label] = summary
            continue
        report = compare_raw_statistics(
            baseline=capture / f'observation/step{first}/rank0', fault=capture / f'observation/step{second}/rank0',
            baseline_config=capture / 'config.json', fault_config=capture / 'config.json',
            baseline_collector=capture / 'collector_identity.json', fault_collector=capture / 'collector_identity.json')
        full = write(out / label / 'full.json', report)
        summary = compact_summary(report)
        summary['full_report'] = full
        write(out / label / 'summary.json', summary)
        results[label] = summary
    return results


def identities(args, data, stages):
    manifest = load(args.case_manifest)[0]
    mounts = {row[1]: Path(row[0]) for row in manifest['mounts']}
    installed, native = mounts['/installed'], mounts['/native']
    original = args.attempt_case.parent / 'controller.json'
    control = next(row for row in load(original)['cases'] if row['id'] == args.attempt_case.name)
    pairs = {'demoncat/runtime/vllm_ascend/worker_extension/serving39_coverage.py':
             'src/demoncat/runtime/vllm_ascend/worker_extension/serving39_coverage.py',
             'demoncat_fault39/coverage_adapter.py': 'fault39/src/demoncat_fault39/coverage_adapter.py',
             'fault_injection/coverage_session.py': 'fault_injection/coverage_session.py'}
    errors, imported = [], []
    for attempt in data['attempts']:
        for row in attempt['setup']['imported_sources']:
            rel = str(Path(row['path']).relative_to('/installed'))
            actual = file_record(installed / rel)
            source = file_record(args.source / pairs[rel])
            exact = actual['sha256'] == source['sha256'] == row['sha256']
            imported.append(dict(recorded=row, source=source, installed=actual, exact=exact))
            if not exact:
                errors.append('IMPORTED_SOURCE_BYTES_MISMATCH:' + rel)
    receipt_path = native.parent / 'receipt.json'; receipt = load(receipt_path)
    artifacts = {rel: dict(expected=expected, actual=file_record(native.parent / rel))
                 for rel, expected in receipt['artifacts'].items()}
    if any(row['expected'] != row['actual']['sha256'] for row in artifacts.values()):
        errors.append('NATIVE_BUILD_ARTIFACT_BYTES_MISMATCH')
    adapter = file_record(installed / 'demoncat_fault39/coverage_adapter.py')['sha256']
    observer_root = Path('demoncat/runtime/vllm_ascend/worker_extension')
    observer_files = [observer_root / name for name in
                      ('serving39_coverage.py', 'collector_dispatch.py', 'gemm_lineage.py')]
    observer = hashlib.sha256(b''.join((installed / rel).read_bytes() for rel in observer_files)).hexdigest()
    observer_bindings = []
    for rel in observer_files:
        actual, source = file_record(installed / rel), file_record(args.source / 'src' / rel)
        observer_bindings.append(dict(actual=actual, source=source, exact=actual['sha256'] == source['sha256']))
        if actual['sha256'] != source['sha256']:
            errors.append('OBSERVER_SOURCE_BYTES_MISMATCH:' + str(rel))
    profile = data['attempts'][0]['setup']['binding']['build_profile_sha256']
    for name, stage in stages.items():
        if (stage['scope']['implementation_sha256'] != sha([adapter, observer, profile]) or
                stage['scope']['environment_sha256'] != sha([profile, data['identity']['worker']])):
            errors.append(name + ':SOURCE_BUILD_WORKER_SCOPE_BINDING_MISMATCH')
    frozen = []
    for row in control['frozen_scripts']:
        actual = file_record(row['frozen'])
        frozen.append(dict(recorded=row, actual=actual))
        if actual['sha256'] != row['sha256']:
            errors.append('FROZEN_PROBE_BYTES_MISMATCH')
    if control['exit_code'] != 0:
        errors.append('CONTROLLER_NONZERO_EXIT')
    return dict(controller=file_record(original), case_manifest=file_record(args.case_manifest),
        image_id=control['image_id'], source_root=str(args.source), installed_root=str(installed),
        imported_sources=imported, observer_bindings=observer_bindings, observer_sha256=observer,
        frozen_scripts=frozen, native_build_receipt=file_record(receipt_path),
        native_build_artifacts=artifacts, build_profile_sha256=profile, checks_passed=not errors, errors=errors)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-case', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--case-manifest', type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=False)
    data = load(args.attempt_case / 'result.json')
    index = index_artifacts(sorted((args.attempt_case / 'c2/native').glob('*.json')) +
                            sorted((args.attempt_case / 'c3/native').glob('*.json')))
    stage_results, c2 = {}, None
    for i, name in enumerate(('c2', 'c3')):
        row, coverage = review_stage(args.attempt_case / name, data['attempts'][i], data['identity']['worker'],
                                    args.out / name, index, c2=c2)
        if name == 'c2':
            c2 = coverage
        stage_results[name] = row
    health = all(a['healthy_after']['outputs'] == data['healthy_before']['outputs'] and
                 a['healthy_after']['prompt_token_ids'] == data['healthy_before']['prompt_token_ids']
                 for a in data['attempts'])
    stats = {name: statistics(args.attempt_case / name, args.out / 'statistics' / name) for name in ('c2', 'c3')}
    bindings = identities(args, data, stage_results)
    write(args.out / 'identities.json', bindings)
    result = dict(schema='demoncat.independent_native39_group_review.v1', attempt=file_record(args.attempt_case / 'result.json'),
        identity=data['identity'], stages=stage_results, healthy_generation_restored=health, statistics=stats,
        source_imports=[a['setup']['imported_sources'] for a in data['attempts']], identities=bindings,
        checks_passed=health and bindings['checks_passed'] and all(r['checks_passed'] for r in stage_results.values()),
        qualifications_granted=False)
    write(args.out / 'summary.json', result)
    print(json.dumps(dict(checks_passed=result['checks_passed'], stages={k:dict(counts=v['counts'], errors=v['errors'])
          for k,v in stage_results.items()}, statistics={k:dict(status=v['baseline_fault']['status'],
          first_exceedance=v['baseline_fault'].get('first_exceedance')) for k,v in stats.items()})))


if __name__ == '__main__':
    main()
