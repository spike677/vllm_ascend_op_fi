"""Read-only CPU audit of the recorded MODEL_LOCAL Vision17 group.

No framework imports, NPU API, model construction or container lifecycle call.
Canonical exports contain only exact recorded objects and remain on 124.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ATTEMPT = 'natural-vision17-moe8-rope22-card2-a1'
RESULT = ATTEMPT + '/natural-vision17-group/probe/result.json'
RESULT_SHA = '98390141cbdf84ea1565693e5cd06e76d725b7fdd4acd451659a9a10c540aff8'
SOURCE_SHA = 'b50acc508d8fa15e51bae75dd4e219541262f4ec785abf5ef5595ece229e00c0'
READER_SHA = '6a6747921760a819e45e43ea0c289cad12620147f4377a089757884edaa0c55a'
PHASES = ['native_baseline', 'baseline', 'baseline_repeat', 'lossless',
          'once:0', 'off', 'every_match:0', 'every_match:1', 'off']
FAULTS = (4, 6, 7)


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def record(path):
    path = Path(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'FILE_CHANGED_DURING_READ')
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest())


def load(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return record(path)


def module(path, name, expected=None):
    if expected:
        require(record(path)['sha256'] == expected, 'AUDIT_DEPENDENCY_HASH_MISMATCH')
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def check_phase(row, index, reference, reference_event, expected_paths):
    """Checks recorded actual-call census separately from admission decisions."""
    require(row['phase'] == PHASES[index], 'PHASE_ORDER_CHANGED')
    census = row['independent']
    require(census['complete'] is True and census['dropped_calls'] == 0 and
            census['observer_hooks'] == 0 and census['inflight'] == 0 and
            census['tensor_mutations'] == 0, 'INDEPENDENT_OBSERVER_INCOMPLETE')
    require(len(census['forwards']) == 1 and census['forwards'][0]['status'] == 'returned',
            'COMPLETE_MODEL_FORWARD_REQUIRED')
    inventory = [x for x in census['module_inventory']
                 if x['role'] in ('conv3d', 'conv3d_structural_candidate')]
    require({x['path'] for x in inventory} == expected_paths and len(inventory) == len(expected_paths),
            'STRUCTURAL_DENOMINATOR_CHANGED')
    actual = [x for x in census['events'] if x.get('kind') == 'operator'
              and x.get('operator') == 'aten.conv3d.default']
    require(census['operator_counts']['aten.conv3d.default'] == len(actual) == 1 and
            {x['path'] for x in actual} == expected_paths and len(row['calls']) == 1,
            'ACTUAL_CALL_DENOMINATOR_CHANGED')
    native = actual[0]
    require(native['original_returned'] is True and native['weight_owner_paths'] == [native['path']],
            'NATIVE_RETURN_OR_WEIGHT_OWNER_UNPROVEN')
    require(row['input'] == reference['input'], 'MODEL_INPUT_CHANGED')
    output = row['output']
    require(output['finite'] is True and output['shape'] == reference['output']['shape'] and
            output['dtype'] == reference['output']['dtype'], 'OUTPUT_SHAPE_DTYPE_OR_FINITE_CHANGED')
    injecting = index in FAULTS
    require((output == reference['output']) is (not injecting), 'MODEL_OUTPUT_EFFECT_OR_RESTORE_MISMATCH')
    # This observer is outside the adapter. It sees original call arguments,
    # then the returned result, not the inner func(**native) argument change.
    require(native['conv_geometry'] == dict(stride=[2, 16, 16],
            padding=[0, 0, 0], dilation=[1, 1, 1], groups=1),
            'PRE_INTERVENTION_GEOMETRY_MISMATCH')
    for key in ('input', 'output'):
        require(all(x['device'] == 'npu:0' for x in native[key]), 'ACTUAL_NPU_CALL_REQUIRED')
    cleanup = row['cleanup']
    require(cleanup['cleanup_confirmed'] is True and cleanup['hook_owned'] is False and
            cleanup['leases'] == cleanup['inflight'] == 0, 'PHASE_CLEANUP_PENDING')
    events = (row.get('mechanism') or {}).get('events', [])
    if index in (0, 5, 8):
        require(not events, 'UNWRAPPED_PHASE_HAS_MECHANISM_EVENT')
    else:
        require(len(events) == 1 and not row['mechanism']['errors'] and
                row['mechanism']['simulation'] is False and cleanup['ownership_verified'] is True and
                cleanup['enabled'] is False, 'ADAPTER_PHASE_NOT_CLOSED')
        event = events[0]
        require(event['path'] == native['path'] and event['ordinal'] == 0 and
                event['operator'] == native['operator'] and event['native_calls'] == 1 and
                event['original_execution_verified'] is True and event['return_readback_exact'] is True and
                event['inputs_unchanged'] is True, 'NATIVE_EVENT_NOT_BOUND')
        require(event['complete_input_weight_bias_identity'] == reference_event['complete_input_weight_bias_identity'],
                'ACTUAL_INPUT_WEIGHT_OR_BIAS_CHANGED')
        require(event['padding'] == ([0, 7, 7] if injecting else [0, 0, 0]) and
                event['baseline_padding'] == [0, 0, 0] and event['applied'] is injecting and
                event['output_changed'] is injecting and
                (event['complete_output_identity'] == reference_event['complete_output_identity']) is (not injecting),
                'PER_TARGET_OUTPUT_EFFECT_UNPROVEN')
        require(event['consumers'] and event['consumers'][0]['returned'] is True,
                'PROVIDER_CONSUMER_RECORD_MISSING')
    return dict(index=index, phase=row['phase'], structural_targets=len(inventory), actual_calls=len(actual),
                fault_applied=injecting, output=output, input_sha256=sha(row['input']),
                independent_pre_intervention_padding=native['conv_geometry']['padding'], cleanup=cleanup,
                provider_applied_padding=None if not events else events[0]['padding'],
                event_output=None if not events else events[0]['complete_output_identity'],
                provider_consumer_records=len(events[0]['consumers']) if events else 0,
                independent_consumer_lineage_complete=False)


def export_references(data, out, auditor):
    preliminary = {n: auditor.audit_saved_coverage(data[n], c2_report=data['c2'] if n == 'c3' else None)
                   for n in ('c2', 'c3')}
    needed = set().union(*(set(x['missing_artifact_sha256']) for x in preliminary.values()))
    directory = out / 'recorded-canonical-objects'
    directory.mkdir()
    origins, paths = {}, []
    def export(value, origin):
        digest = sha(value)
        if digest not in needed:
            return
        if digest not in origins:
            path = directory / (digest + '.json')
            with path.open('xb') as stream:
                stream.write(canonical(value))
            paths.append(path)
            origins[digest] = []
        origins[digest].append(origin)
    phases = data['all_phases']
    export(dict(baseline=phases[1], repeat=phases[2], lossless=phases[3]),
           {'baseline': '/all_phases/1', 'repeat': '/all_phases/2', 'lossless': '/all_phases/3'})
    for name in ('c2', 'c3'):
        export(data[name]['phase_rows'], '/' + name + '/phase_rows')
    for i, row in enumerate(phases):
        pointer = '/all_phases/' + str(i)
        export(row, pointer)
        for j, event in enumerate((row.get('mechanism') or {}).get('events', [])):
            event_pointer = pointer + '/mechanism/events/' + str(j)
            export(event, event_pointer)
            export(dict(event=event, model_output=row['output']),
                   {'event': event_pointer, 'model_output': pointer + '/output'})
        for call in row['calls']:
            export(dict(call=call['key'], output=row['output']),
                   {'call': pointer + '/calls', 'output': pointer + '/output'})
    index = auditor.index_artifacts(paths)
    require(not needed - set(index), 'EXACT_REFERENCE_BYTES_MISSING')
    reports = {n: auditor.audit_saved_coverage(data[n], artifact_index=index,
               c2_report=data['c2'] if n == 'c3' else None) for n in ('c2', 'c3')}
    for n, report in reports.items():
        require(report['report_consistent'] is True and not report['errors'] and
                not report['missing_artifact_sha256'], 'COVERAGE_AUDIT_FAILED:' + n)
        write(out / (n + '-coverage-audit.json'), report)
    write(out / 'canonical-export-manifest.json', dict(origins=origins, references=len(needed),
          semantics='Exact original canonical objects; no invented events or tensors.'))
    return {n: {k: v for k, v in report.items() if k not in ('target_ids', 'scope')}
            for n, report in reports.items()}


def check_coverage(data):
    scopes = [data[n]['scope'] for n in ('c2', 'c3')]
    require(scopes[0] == scopes[1] and scopes[0]['evidence_kind'] == 'natural_pretrained_model',
            'MODEL_LOCAL_SCOPE_CHANGED')
    for name, count in (('c2', 1), ('c3', 2)):
        report = data[name]
        require(report['contract_satisfied'] is True and not report['rejection_reasons'] and
                report['real_worker_evidence'] is False and report['real_pretrained_model_evidence'] is True and
                report['public_qualification_granted'] is False and report['scheduler_requests_claimed'] is False and
                report['invocation_scope'] == 'MODEL_LOCAL', 'COVERAGE_SCOPE_OVERCLAIM')
        require(report['planned_targets'] == report['planned_targets_hit'] == 1 and
                report['observed_calls'] == count + 1 and report['hits'] == report['observed_eligible_calls'] == count,
                'COVERAGE_DENOMINATOR_MISMATCH')
        calls = report['calls']
        require(len(calls) == count + 1 and len({x['target_id'] for x in calls}) == 1,
                'COVERAGE_TARGET_SET_CHANGED')
        for i, call in enumerate(calls):
            fault = i < count
            require(call['decision']['admitted'] is fault and call['receipt']['native_applied'] is fault and
                    call['receipt']['outcome'] == ('HIT' if fault else 'PASSTHROUGH') and
                    call['receipt']['original_calls'] == 1 and call['propagation_added'] is False,
                    'CALL_OUTCOME_NOT_BOUND')
            target = call['decision']['ticket']['call']['target']
            require(target['scope_kind'] == 'MODEL_LOCAL' and target['device_type'] == 'npu',
                    'TARGET_SCOPE_NOT_MODEL_LOCAL_NPU')
        recovery = report['recovery']
        require(recovery['complete'] is True and recovery['state'] == 'DISABLED_CONFIRMED' and
                recovery['inflight'] == 0 and not any(recovery[x] for x in
                ('held_leases', 'pending_drains', 'pending_leases', 'errors')), 'GROUP_RECOVERY_PENDING')
        require(len(recovery['targets']) == len(recovery['workers']) == 1, 'RECOVERY_TARGET_OR_LIFETIME_MISSING')
        worker = recovery['workers'][0]
        require(all(worker[x] == 0 for x in ('active_leases', 'device_bytes', 'host_bytes', 'inflight_calls', 'owned_hooks'))
                and worker['baseline_output_sha256'] == worker['off_output_sha256'], 'HEALTH_RECOVERY_NOT_EXACT')
    require(data['c2']['owner'] == data['c3']['owner'] and
            data['c3']['generation'] == data['c2']['generation'] + 1, 'LIVE_GENERATION_CHAIN_BROKEN')


def bindings(root, data):
    source = root / 'source-a36'
    manifest = load(source / 'source_manifest.json')
    require(record(source / 'source_manifest.json')['sha256'] == SOURCE_SHA, 'SOURCE_MANIFEST_CHANGED')
    reader = module(root / 'agent7_audit_cpu_builds_v5.py', 'agent7_image_reader', READER_SHA)
    inventory = reader.check_files(source, manifest['files'])
    require(not inventory['mismatches'] and len(manifest['files']) == 625, 'SOURCE_FILES_CHANGED')
    controller_path = root / ATTEMPT / 'controller.json'
    controller = load(controller_path)
    case = next(x for x in controller['cases'] if x['id'] == 'natural-vision17-group')
    command = case['command']
    require(case['exit_code'] == 0 and case['source_manifests'] == [dict(manifest_sha256=SOURCE_SHA,
            base_commit=manifest['base_commit'], files=625)] and
            f'type=bind,src={source},dst=/source,readonly' in command and '--statistics' not in command[-1],
            'ACTUAL_EXECUTION_SOURCE_OR_SCOPE_MISMATCH')
    image = reader.image_metadata(case['image_id'])
    model_sources = {}
    for path, expected in data['model_sources'].items():
        actual = record(source / path[len('/source/'):]) if path.startswith('/source/') else reader.image_file(image, path)
        require(actual['sha256'] == expected, 'MODEL_IMPLEMENTATION_CHANGED:' + path)
        model_sources[path] = expected
    weights = {}
    for name, expected in data['weights'].items():
        actual = record(root / 'models/Qwen3-VL-2B-Instruct' / name)
        require(all(actual[k] == expected[k] for k in ('sha256', 'bytes')), 'MODEL_WEIGHT_CHANGED')
        weights[name] = expected
    require(sha(weights) == data['model_weights_sha256'] and sha(model_sources) == data['model_source_sha256'],
            'MODEL_BINDING_DIGEST_MISMATCH')
    require(record(source / 'experiments/c2c3_closure_20261117/natural_tensor_group_probe.py')['sha256'] ==
            data['probe_sha256'], 'ACTUAL_PROBE_CHANGED')
    files = ['tensor_natural.py', 'rmsnorm_return.py', 'index22.py', 'tensor_observer.py', 'tensor_callable.py',
             '_scoped_dispatch.py', 'module_faults.py', 'route8.py', 'moe_census.py', 'moe_restore.py',
             'conv3d_natural.py', 'serving_tensor.py', 'serving_moe.py']
    provider_sources = {}
    for name in files:
        paths = [x for x in manifest['files'] if Path(x['path']).name == name]
        require(len(paths) == 1, 'AMBIGUOUS_PROVIDER_SOURCE:' + name)
        provider_sources[name] = record(source / paths[0]['path'])['sha256']
    require(sha(provider_sources) == data['c2']['scope']['implementation_sha256'], 'PROVIDER_SOURCE_DIGEST_MISMATCH')
    input_image = record('/home/jw/Wan2.2/examples/i2v_input.JPG')
    config = record(root / 'models/Qwen3-VL-2B-Instruct/config.json')
    require(input_image['sha256'] == data['provenance']['image_sha256'] and
            config['sha256'] == data['provenance']['config_sha256'], 'MODEL_INPUT_OR_CONFIG_CHANGED')
    probe_text = (source / 'experiments/c2c3_closure_20261117/natural_tensor_group_probe.py').read_text()
    authority = source / 'fault_injection/coverage_session.py'
    authority_text = authority.read_text()
    require('proof = provider.group.c2_proof()' in probe_text and
            "provider.run('every_match', forwards=args.c3_forwards, c2_proof=proof)" in probe_text and
            '_ISSUED_C2_PROOFS' in authority_text and '_issued_c2_proof' in authority_text,
            'REVIEWED_LIVE_PROOF_CHAIN_SOURCE_MISSING')
    return dict(controller=record(controller_path), exit_code=case['exit_code'], image_id=case['image_id'],
                source_manifest=record(source / 'source_manifest.json'), source_commit=manifest['base_commit'],
                source_files_verified=len(manifest['files']), model_sources=model_sources, weights=weights,
                provider_sources=provider_sources, provider_digest=sha(provider_sources), probe_sha256=data['probe_sha256'],
                proof_authority=record(authority), model_config_sha256=config['sha256'], image_input_sha256=input_image['sha256'],
                live_proof_basis='Frozen direct object handoff and issued-proof gate; offline audit cannot reissue a live capability.')


def audit(root, out):
    receipt = record(root / RESULT)
    require(receipt['sha256'] == RESULT_SHA, 'ORIGINAL_RECEIPT_CHANGED')
    data = load(root / RESULT)
    require(data['status'] == 'NATURAL_PRETRAINED_MODEL_GROUP_CONTRACT_COMPLETE' and
            data['mechanism'] == '17.spatial_padding' and data['standard_vllm_worker'] is False and
            data['statistics_status'] == 'NOT_COLLECTED_SEPARATE_E5_REQUIRED' and data['provider_error'] is None,
            'ORIGINAL_COMPLETION_OR_SCOPE_MISMATCH')
    phases = data['all_phases']
    require(len(phases) == 9, 'EXACT_NINE_PHASES_REQUIRED')
    candidates = data['discovery']['candidate_set']
    require(candidates['eligible'] == candidates['discovered'] == 1 and candidates['discovery_complete'] is True,
            'DISCOVERY_DENOMINATOR_MISMATCH')
    expected = {x['path'] for x in phases[0]['independent']['module_inventory']
                if x['role'] in ('conv3d', 'conv3d_structural_candidate')}
    require(len(expected) == 1, 'EXACT_STRUCTURAL_TARGET_REQUIRED')
    reference_event = phases[1]['mechanism']['events'][0]
    reviewed = [check_phase(row, i, phases[0], reference_event, expected) for i, row in enumerate(phases)]
    require(phases[4]['output'] == phases[6]['output'] == phases[7]['output'], 'FAULT_REPEAT_NOT_EXACT')
    check_coverage(data)
    identity = bindings(root, data)
    auditor_path = root / 'source-a36/fault_injection/coverage_audit.py'
    auditor = module(auditor_path, 'agent7_saved_coverage')
    audits = export_references(data, out, auditor)
    summary = dict(schema='demoncat.agent7.vision17_recorded_group_review.v1', original_receipt=receipt,
        auditor=record(Path(__file__)), frozen_coverage_auditor=record(auditor_path), source_bindings=identity,
        record_checks_passed=True, scope='MODEL_LOCAL_NATURAL_PRETRAINED_IMAGE_PREFILL',
        model=data['provenance']['model_id'], revision=data['provenance']['public_revision'],
        runtime=data['runtime'], phases=reviewed, coverage=audits, structural_target_count=len(expected),
        c2_recorded_hits=1, c3_recorded_hits=2, full_output_shape=phases[0]['output']['shape'],
        provider_consumer_first=reference_event['consumers'][0],
        independent_consumer_lineage_complete=False,
        independent_final_native_arguments_verified=False,
        native_argument_limit='Outer independent observer records original padding [0,0,0]; changed [0,7,7] is provider-recorded at the reviewed inner native call, not an independent post-adapter launch trace.',
        consumer_limit='393 provider storage-id lineage records are not 393 independently proven consumers; freed storage identities can be reused. The independent census does not record the first add inputs.',
        full_output_comparison_basis='Complete CPU-copy hashes recorded during actual execution; this audit does not reload private tensor payloads.',
        resource_scope='Owned fault hooks, in-flight calls and explicit leases; not all model allocator memory or observer CPU copies.',
        recommendation='Recorded MODEL_LOCAL C2/C3 coverage and OFF contracts consistent; final acceptance belongs to root and consumer-lineage limitation remains.',
        standard_worker_qualification=False, public_qualification_granted=False, e5=False,
        statistics_status=data['statistics_status'], historical_failed_card5_attempt_not_upgraded=True)
    write(out / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    value = audit(args.root, args.out)
    print(json.dumps({k: value[k] for k in ('record_checks_passed', 'scope', 'c2_recorded_hits', 'c3_recorded_hits',
                                          'independent_consumer_lineage_complete', 'public_qualification_granted', 'e5')}))


if __name__ == '__main__':
    main()
