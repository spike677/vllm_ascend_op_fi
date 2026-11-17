"""Independent CPU-only review of the named natural MoE standard-Worker group.

Reads immutable receipts, exact canonical references, source and weight hashes.
Does not execute framework/model code or start a container/NPU operation.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from experiments.c2c3_closure_20261117.review_vision17_group_124 import (
    canonical, load, module, record, require, sha, write, READER_SHA, SOURCE_SHA)


ATTEMPT = 'natural-vision17-moe8-rope22-card2-a1'
RESULT_SHA = '0bc9d2c82134f41a5ffdb61d980fdc9d4c99d23621e00354f098870322ad4a5d'
PHASES = ['native_baseline', 'baseline', 'baseline_repeat', 'lossless',
          'c2_fault', 'c2_off', 'c3_fault_0', 'c3_fault_1', 'c3_off']
FAULTS = (4, 6, 7)


def probe_digest(value):
    # Frozen natural_rms_worker_probe.digest uses the default JSON separators.
    # Scope environment/workload hashes must not be rewritten as compact JSON.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def check_native(native, injecting):
    require(native['original_calls'] == 1 and native['original_returned'] is True,
            'ACTUAL_ORIGINAL_CALL_COUNT_OR_RETURN')
    original, consumed = native['arguments'], native['consumed_arguments']
    require(original['permuted_tokens'] == consumed['permuted_tokens'], 'EXPERT_HIDDEN_REPLACED')
    require(original['padded_mode'] is consumed['padded_mode'] is False and
            original['restore_shape'] is consumed['restore_shape'] is None, 'UNREVIEWED_NATIVE_OPTIONS')
    for key in ('sorted_indices', 'probs'):
        for field in ('shape', 'dtype', 'device', 'stride', 'offset'):
            require(original[key][field] == consumed[key][field], 'REMAP_METADATA_CHANGED')
        same_object = original[key]['object_id'] == consumed[key]['object_id']
        same_storage = original[key]['storage_id'] == consumed[key]['storage_id']
        require(same_object is (not injecting) and same_storage is (not injecting),
                'CONSUMED_I_P_IDENTITY_NOT_MATCHING_PHASE')
    require(all(consumed[k]['device'] == 'npu:0' for k in ('permuted_tokens', 'sorted_indices', 'probs')),
            'ACTUAL_NPU_NATIVE_INPUT_REQUIRED')


def check_producer(native, trace, request_id):
    proof = native['producer_proof']
    rows = trace['python_boundaries']
    selected = [rows[i] for i in proof['source_sequences']]
    require([x['role'] for x in selected] == ['topk_producer', 'expert_pipeline', 'routing_producer',
            'expert_mlp', 'native_combine'] and
            all(x['model_path'] == native['model_path'] and x['request_id'] == request_id and
                x['returned'] is True for x in selected), 'COMPLETE_ACTUAL_PRODUCER_CHAIN_REQUIRED')
    topk, pipeline, routing, mlp, combine = selected
    h, i, p = [native['arguments'][k] for k in ('permuted_tokens', 'sorted_indices', 'probs')]
    cm = combine['input']['combine_metadata']
    routed = routing['output']
    route_in = routing['input']['token_dispatch_input']
    expert_in = pipeline['input']['fused_experts_input']
    require(topk['output'][0]['object_id'] == p['object_id'] == cm['topk_weights']['object_id'] ==
            routed['combine_metadata']['topk_weights']['object_id'] == route_in['topk_weights']['object_id'] ==
            expert_in['topk_weights']['object_id'], 'PROBABILITY_PRODUCER_OBJECT_MISMATCH')
    require(topk['output'][1]['object_id'] == route_in['topk_ids']['object_id'] == expert_in['topk_ids']['object_id'],
            'TOPK_EXPERT_ID_OBJECT_MISMATCH')
    require(h['object_id'] == mlp['output']['object_id'] == combine['input']['hidden_states']['object_id'] and
            mlp['input']['mlp_compute_input']['hidden_states']['object_id'] == routed['hidden_states']['object_id'],
            'ACTUAL_EXPERT_MLP_PRODUCER_MISMATCH')
    absolute = [x for x in trace['native_operators'] if x['combine_sequence'] == combine['sequence']
                and x['operator'] == 'aten.abs.default']
    require(len(absolute) == 1 and absolute[0]['original_calls'] == 1 and
            absolute[0]['input'][0]['object_id'] == cm['expanded_row_idx']['object_id'] ==
            routed['combine_metadata']['expanded_row_idx']['object_id'] and
            absolute[0]['output'][0]['object_id'] == i['object_id'], 'ABS_INDEX_PRODUCER_MISMATCH')
    require(route_in['routing']['apply_router_weight_on_input'] is False and
            expert_in['routing']['apply_router_weight_on_input'] is False and
            topk['input']['top_k'] == p['shape'][1] == 6, 'ROUTING_SEMANTICS_UNADAPTED')
    require(native['output'][0]['object_id'] == combine['output']['object_id'] ==
            pipeline['output']['routed_out']['object_id'], 'ACTUAL_NATIVE_ROUTED_RETURN_MISMATCH')


def check_phase(request, index, baseline, expected_paths):
    row = request['worker']; injecting = index in FAULTS
    require(row['phase'] == request['phase'] == PHASES[index], 'REQUEST_PHASE_MISMATCH')
    request_id = row['request_id']; mapping = request['mapping']; scheduler = row['scheduler']
    require(mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR' and
            mapping['internal_request_id'] == scheduler['internal_request_id'] == request_id and
            scheduler['num_requests'] == 1 and scheduler['num_computed_tokens'] == 0 and
            scheduler['num_prompt_tokens'] == scheduler['num_scheduled_tokens'] == 10 and
            scheduler['token_ids'] == baseline['scheduler']['token_ids'] and scheduler['positions'] == list(range(10)),
            'ACTUAL_ENQUEUED_PREFILL_MAPPING_MISMATCH')
    require(row['input'] == baseline['input'] and row['logits_boundary_returned'] is True,
            'MODEL_INPUT_OR_REAL_LOGITS_BOUNDARY_MISMATCH')
    for key in ('hidden', 'logits'):
        require(row['output'][key]['finite'] is True and
                row['output'][key]['shape'] == baseline['output'][key]['shape'] and
                row['output'][key]['dtype'] == baseline['output'][key]['dtype'] and
                (row['output'][key] == baseline['output'][key]) is (not injecting),
                'COMPLETE_MODEL_EFFECT_OR_RESTORE_MISMATCH')
    independent = row['independent']; trace = independent['moe_trace']
    require(independent['complete'] is True and independent['dropped_calls'] == 0 and
            independent['observer_hooks'] == independent['inflight'] == independent['tensor_mutations'] == 0 and
            trace['complete'] is True and not trace['errors'] and trace['simulation'] is False and
            trace['model_methods_replaced'] is False and trace['operator_reexecution'] is False,
            'INDEPENDENT_CENSUS_INCOMPLETE')
    require(set(trace['structural_moe_paths']) == expected_paths and
            len(trace['structural_moe_paths']) == trace['observed_unpermute_calls'] == len(row['calls']) == 26,
            'STRUCTURAL_OR_ACTUAL_DENOMINATOR_MISMATCH')
    natives = [x for x in trace['native_operators'] if x['operator'] == 'npu.npu_moe_token_unpermute.default']
    require(len(natives) == 26 and {x['model_path'] for x in natives} == expected_paths,
            'ACTUAL_NATIVE_TARGET_SET_MISMATCH')
    events = (row.get('mechanism') or {}).get('events', [])
    require(len(events) == (0 if index in (0, 5, 8) else 26), 'ADAPTER_EVENT_DENOMINATOR_MISMATCH')
    event_map = {x['path']: x for x in events}
    require(len(event_map) == len(events), 'DUPLICATE_TARGET_EVENT')
    for native in natives:
        check_native(native, injecting)
        check_producer(native, trace, request_id)
        if not events:
            continue
        event = event_map[native['model_path']]
        require(all(event[k] is True for k in ('original_execution_verified', 'paired_group_readback_exact',
                'input_storage_unchanged', 'consumer_returned', 'complete_permutation_value_verified',
                'complete_native_output_oracle_exact', 'return_readback_exact')) and
                event['native_calls'] == event['observed_original_native_calls'] == 1 and
                event['applied'] is injecting and event['output_changed'] is injecting and
                event['actual_producer_join'] == native['producer_proof'], 'PER_TARGET_EFFECT_ORACLE_MISMATCH')
        require(event['source_groups'] == ([1, 2, 3, 4, 5, 6, 7, 8, 9, 9] if injecting else list(range(10))) and
                event['token_groups'] == 10 and event['topk'] == 6, 'PAIRED_GROUP_SEMANTICS_MISMATCH')
    cleanup = row['cleanup']
    require(cleanup['cleanup_confirmed'] is True and cleanup['hook_owned'] is False and
            cleanup['inflight'] == cleanup['leases'] == 0, 'PHASE_ADAPTER_CLEANUP_PENDING')
    return dict(phase=row['phase'], request_id=request_id, actual_targets=len(natives),
                independently_joined_producers=len(natives), actual_native_calls=len(natives),
                applied_targets=26 if injecting else 0, output=row['output'], output_tokens=request['output_tokens'],
                adapter_cleanup=cleanup, true_standard_worker_request=True)


def canonical_references(data, out, auditor):
    preliminary = {n: auditor.audit_saved_coverage(data[n], c2_report=data['c2'] if n == 'c3' else None)
                   for n in ('c2', 'c3')}
    needed = set().union(*(set(x['missing_artifact_sha256']) for x in preliminary.values()))
    target = out / 'recorded-canonical-objects'; target.mkdir()
    paths = []; origins = {}
    def export(value, origin):
        digest = sha(value)
        if digest not in needed:
            return
        if digest not in origins:
            path = target / (digest + '.json')
            with path.open('xb') as stream:
                stream.write(canonical(value))
            paths.append(path); origins[digest] = []
        origins[digest].append(origin)
    rows = [x['worker'] for x in data['requests']]
    export(dict(baseline=rows[1], repeat=rows[2], lossless=rows[3]),
           {'baseline': '/requests/1/worker', 'repeat': '/requests/2/worker', 'lossless': '/requests/3/worker'})
    for n in ('c2', 'c3'):
        export(data[n]['phase_rows'], '/' + n + '/phase_rows')
    for index, row in enumerate(rows):
        pointer = '/requests/' + str(index) + '/worker'
        export(row, pointer)
        for ordinal, event in enumerate((row.get('mechanism') or {}).get('events', [])):
            event_pointer = pointer + '/mechanism/events/' + str(ordinal)
            export(event, event_pointer)
            export(dict(event=event, model_output=row['output']),
                   {'event': event_pointer, 'model_output': pointer + '/output'})
        for call in row['calls']:
            export(dict(call=call['key'], output=row['output']), {'calls': pointer + '/calls', 'output': pointer + '/output'})
    indexed = auditor.index_artifacts(paths)
    require(not needed - set(indexed), 'EXACT_ORIGINAL_REFERENCE_MISSING')
    summaries = {}
    for n in ('c2', 'c3'):
        result = auditor.audit_saved_coverage(data[n], artifact_index=indexed,
                  c2_report=data['c2'] if n == 'c3' else None)
        require(result['report_consistent'] is True and result['referenced_artifact_bytes_verified'] is True and
                not result['errors'], 'COVERAGE_REFERENCE_AUDIT_FAILED')
        write(out / (n + '-coverage-audit.json'), result)
        summaries[n] = {k: v for k, v in result.items() if k not in ('target_ids', 'scope')}
    write(out / 'canonical-export-manifest.json', dict(origins=origins, unique_references=len(indexed)))
    return summaries


def audit(root, out):
    path = root / ATTEMPT / 'natural-moe8-group/probe/result.json'
    receipt = record(path); require(receipt['sha256'] == RESULT_SHA, 'ORIGINAL_RECEIPT_CHANGED')
    data = load(path)
    require(data['status'] == 'STANDARD_WORKER_MOE_GROUP_CONTRACT_COMPLETE' and
            data['standard_vllm_worker'] is True and data['public_qualification_granted'] is False and
            data['statistics_status'] == 'SEPARATE_STANDARD_WORKER_CAPTURE_REQUIRED', 'ORIGINAL_SCOPE_CHANGED')
    requests = data['requests']; require(len(requests) == 9, 'NINE_ACTUAL_REQUESTS_REQUIRED')
    require(len({x['mapping']['internal_request_id'] for x in requests}) == 9, 'REQUEST_REUSE')
    baseline = requests[0]['worker']
    provenance = data['setup']['provenance']
    expected = set(provenance['structural_moe_paths'])
    require(len(expected) == 26 and data['qualification']['candidate_set']['eligible'] == 26,
            'MODEL_STRUCTURAL_TARGET_COUNT_CHANGED')
    phases = [check_phase(x, i, baseline, expected) for i, x in enumerate(requests)]
    require(requests[4]['worker']['output'] == requests[6]['worker']['output'] == requests[7]['worker']['output'],
            'FAULT_REPETITION_NOT_EXACT')
    require(data['c2']['phase_rows'] == [x['worker'] for x in requests[4:6]] and
            data['c3']['phase_rows'] == [x['worker'] for x in requests[6:9]], 'GROUP_TO_REQUEST_JOIN_CHANGED')
    for n, hits, calls in (('c2', 26, 52), ('c3', 52, 78)):
        q = data[n]
        require(q['scope']['evidence_kind'] == 'natural_worker' and q['real_worker_evidence'] is True and
                q['real_pretrained_model_evidence'] is False and q['invocation_scope'] == 'INITIALIZED_WORKER' and
                q['contract_satisfied'] is True and not q['rejection_reasons'] and
                q['planned_targets'] == q['planned_targets_hit'] == 26 and
                q['hits'] == q['observed_eligible_calls'] == hits and q['observed_calls'] == calls,
                'WORKER_COVERAGE_SCOPE_OR_COUNTS_CHANGED')
        recovery = q['recovery']
        require(recovery['complete'] is True and recovery['state'] == 'DISABLED_CONFIRMED' and
                recovery['inflight'] == 0 and not any(recovery[k] for k in
                ('held_leases', 'pending_drains', 'pending_leases', 'errors')), 'RECOVERY_PENDING')
        require(len(recovery['targets']) == 26 and len(recovery['workers']) == 1 and
                all(x['drained'] is True and x['released'] is True and x['off_exact'] is True for x in recovery['targets']),
                'TARGET_RECOVERY_MISSING')
        worker = recovery['workers'][0]
        require(all(worker[k] == 0 for k in ('active_leases', 'device_bytes', 'host_bytes', 'inflight_calls', 'owned_hooks'))
                and worker['baseline_output_sha256'] == worker['off_output_sha256'], 'SAME_WORKER_HEALTH_MISMATCH')
    require(data['c2']['scope'] == data['c3']['scope'] and data['c2']['owner'] == data['c3']['owner'] and
            data['c2']['generation'] + 1 == data['c3']['generation'], 'LIVE_SCOPE_GENERATION_CHANGED')
    require(data['cleanup']['owned_serving_hooks'] == 0 and data['cleanup']['active_adapter'] is False and
            data['cleanup']['independent_observer'] is False and data['healthy_before'] == data['healthy_after'],
            'POST_CLEANUP_WORKER_GENERATION_UNHEALTHY')
    source = root / 'source-a36'; manifest_path = source / 'source_manifest.json'
    require(record(manifest_path)['sha256'] == SOURCE_SHA, 'SOURCE_MANIFEST_CHANGED')
    manifest = load(manifest_path)
    reader = module(root / 'agent7_audit_cpu_builds_v5.py', 'moe_image_reader', READER_SHA)
    source_check = reader.check_files(source, manifest['files'])
    require(not source_check['mismatches'], 'SOURCE_FILE_CHANGED')
    controller_path = root / ATTEMPT / 'controller.json'; controller = load(controller_path)
    case = next(x for x in controller['cases'] if x['id'] == 'natural-moe8-group')
    require(case['exit_code'] == 0 and f'type=bind,src={source},dst=/source,readonly' in case['command'] and
            '--statistics' not in case['command'][-1], 'ACTUAL_COMMAND_SCOPE_MISMATCH')
    image = reader.image_metadata(case['image_id']); source_count = 0
    for absolute, expected_sha in provenance['model_sources'].items():
        require(reader.image_file(image, absolute)['sha256'] == expected_sha, 'MODEL_SOURCE_CHANGED')
        source_count += 1
    for descriptor in provenance['moe_sources'].values():
        require(reader.image_file(image, descriptor['path'])['sha256'] == descriptor['sha256'], 'MOE_SOURCE_CHANGED')
    for name, identity in provenance['weights'].items():
        actual = record(Path('/home/haoran/models/DeepSeek-V2-Lite-W8A8') / name)
        require(all(actual[k] == identity[k] for k in ('sha256', 'bytes')), 'WEIGHT_SHARD_CHANGED')
    require(sha(provenance['weights']) == data['c2']['scope']['model_weights_sha256'] and
            sha(provenance['model_sources']) == data['c2']['scope']['model_source_sha256'] and
            probe_digest(data['setup']['environment']) == data['c2']['scope']['environment_sha256'], 'SCOPE_IDENTITY_DIGEST_CHANGED')
    require(probe_digest(dict(weights=provenance['weights'], tokens=baseline['scheduler']['token_ids'],
            model_config=data['actual_model_config'])) == data['c2']['scope']['workload_sha256'],
            'ACTUAL_WORKLOAD_DIGEST_CHANGED')
    for filename, field in (('config.json', 'sha256'), ('quant_model_description.json', 'quantization_sha256')):
        require(record(Path('/home/haoran/models/DeepSeek-V2-Lite-W8A8') / filename)['sha256'] ==
                data['actual_model_config'][field], 'MODEL_CONFIG_CHANGED')
    require(data['actual_model_config']['routing_modified'] is False, 'MODEL_ROUTING_CHANGED')
    provider_names = ['tensor_natural.py','rmsnorm_return.py','index22.py','tensor_observer.py','tensor_callable.py',
       '_scoped_dispatch.py','module_faults.py','route8.py','moe_census.py','moe_restore.py','conv3d_natural.py',
       'serving_tensor.py','serving_moe.py']
    provider_sources = {}
    for name in provider_names:
        matches = [x for x in manifest['files'] if Path(x['path']).name == name]
        require(len(matches) == 1, 'PROVIDER_SOURCE_AMBIGUOUS')
        provider_sources[name] = record(source / matches[0]['path'])['sha256']
    require(sha(provider_sources) == data['c2']['scope']['implementation_sha256'], 'IMPLEMENTATION_DIGEST_CHANGED')
    auditor_path = source / 'fault_injection/coverage_audit.py'
    auditor = module(auditor_path, 'moe_saved_coverage')
    coverage = canonical_references(data, out, auditor)
    selected_sources = [record(source / p) for p in (
        'experiments/c2c3_closure_20261117/natural_moe_worker_probe.py',
        'experiments/c2c3_closure_20261117/natural_rms_worker_probe.py',
        'fault_injection/coverage_session.py', 'fault_injection/moe_restore.py', 'fault_injection/route8.py')]
    result = dict(schema='demoncat.agent7.moe8_worker_group_review.v1', original_receipt=receipt,
        auditor=record(Path(__file__)), record_checks_passed=True, public_qualification_granted=False,
        recommendation='Limited actual standard-Worker TP1 eager MoE C2/C3 candidate for root review; no E5.',
        scope='INITIALIZED_STANDARD_WORKER_TP1_FRESH_PREFILL', mechanism='8.restore_offset',
        model='DeepSeek-V2-Lite-W8A8', source_manifest=record(manifest_path), source_files_verified=len(manifest['files']),
        image_id=case['image_id'], controller=record(controller_path), exit_code=case['exit_code'],
        model_source_files_verified=source_count, moe_source_files_verified=len(provenance['moe_sources']),
        weight_shards_verified=len(provenance['weights']), weights_bytes=sum(x['bytes'] for x in provenance['weights'].values()),
        model_weights_sha256=data['c2']['scope']['model_weights_sha256'], model_source_sha256=data['c2']['scope']['model_source_sha256'],
        provider_sources=provider_sources, selected_sources=selected_sources, phases=phases, coverage=coverage,
        all_eligible_targets=26, actual_requests=9, actual_unpermute_calls=26*9, c2_hits=26, c3_hits=52,
        cleanup=data['cleanup'], healthy_before=data['healthy_before'], healthy_after=data['healthy_after'],
        e5=False, statistics_status=data['statistics_status'],
        complete_value_basis='Live CPU readbacks and exact rounded group oracle recorded by the reviewed adapter; private tensor payloads were not independently replayed offline.',
        resource_scope='Owned serving hooks/adapters/observer, explicit leases and in-flight calls; not zero model allocator memory.',
        limitations=['Only fixed 10-token, TP1, serial eager request scope; no multi-request batch, TP2, decode or cross-model claim.',
                     'Fault remaps paired token contribution groups [1..9,9]; it is not an elementwise index offset or an out-of-bounds write.',
                     'Complete producer/object/consumer identities independently rejoined from actual native records; raw full tensor values remain private.',
                     'No statistics/mix capture in this run; no independent msprobe first difference can be granted.'])
    write(out / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False); result = audit(args.root, args.out)
    print(json.dumps({k: result[k] for k in ('record_checks_passed','all_eligible_targets','actual_requests',
                                          'c2_hits','c3_hits','e5','public_qualification_granted')}))


if __name__ == '__main__':
    main()
