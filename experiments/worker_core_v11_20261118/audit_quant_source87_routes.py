"""Stdlib-only source87 metadata audit; never grants routed-row qualification."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import struct

INVENTORY_SHA = 'e67e9cbb6e2c9027bcf801d28df26be759b64d5f6d43888693bca261be776b21'
RESULT_SHA = '2ac9936b3188ce2ce6b2377a7f4a8920d87df8c4c54251d3c77d6cb0e9fc3130'
RUNTIME_SHA = 'd302ccfc6c66a336476bd93eddadfd679c27afa93549b1186ef72d4453712bd9'
ORDER = ('topk', 'routing', 'cumsum', 'gmm1', 'gmm2', 'abs', 'unpermute')
LAYOUTS = dict(topk_ids=('torch.int32', [8, 6], 48, 'i'), raw_indices=('torch.int32', [48], 48, 'i'),
    expert_counts=('torch.int64', [64], 64, 'q'), cumulative_counts=('torch.int64', [64], 64, 'q'),
    absolute_indices=('torch.int32', [48], 48, 'i'))


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def arithmetic(record):
    """Recalculate integer relations independently of the runtime validator."""
    require(set(record['snapshots']) == set(LAYOUTS), 'FIVE_SNAPSHOTS_REQUIRED')
    values = {}
    for name, (dtype, shape, size, code) in LAYOUTS.items():
        snap = record['snapshots'][name]
        data = snap['values']
        require(snap['dtype'] == dtype and snap['shape'] == shape and len(data) == size
            and all(type(value) is int for value in data), 'INTEGER_LAYOUT:'+name)
        raw = struct.pack('<'+code*size, *data)
        require(len(raw) == snap['bytes'] and sha(raw) == snap['sha256'], 'INTEGER_BYTES:'+name)
        require(snap['metadata_before'] == snap['metadata_after'] and snap['epoch_before'] == snap['epoch_after']
            and snap['parent_sequence'] == record['parent_sequence'] and snap['forward_id'] == record['forward_id'],
            'CURRENT_SNAPSHOT_IDENTITY:'+name)
        values[name] = data
    ids, permutation = values['topk_ids'], values['raw_indices']
    require(all(0 <= value < 64 for value in ids) and all(len(set(ids[i:i+6])) == 6 for i in range(0, 48, 6)), 'EXPERT_RANGE_OR_UNIQUENESS')
    require(sorted(permutation) == list(range(48)) and values['absolute_indices'] == permutation, 'NONNEGATIVE_INVERSE_PERMUTATION')
    counts = [ids.count(expert) for expert in range(64)]
    require(values['expert_counts'] == counts, 'EXPERT_COUNT_MISMATCH')
    cumulative = [sum(counts[:i+1]) for i in range(64)]
    require(values['cumulative_counts'] == cumulative and cumulative[-1] == 48, 'CUMULATIVE_COUNT_MISMATCH')
    inverse = [None]*48
    for source, destination in enumerate(permutation):
        inverse[destination] = source
    require([ids[source] for source in inverse] == sorted(ids), 'EXPERT_SORT_GROUP_MISMATCH')
    slots = [dict(destination=i, source_token=source//6, route_slot=source%6, expert=ids[source])
             for i, source in enumerate(inverse)]
    proof = record['token_route_proof']
    require(proof is not None and proof['source_to_sorted'] == permutation and proof['sorted_slots'] == slots
        and proof['expert_counts'] == counts and proof['cumulative_counts'] == cumulative,
        'REPORTED_ARITHMETIC_PROOF_MISMATCH')
    require(proof['evidence_sha256'] == digest({k:v for k,v in record.items() if k != 'token_route_proof'}), 'ARITHMETIC_EVIDENCE_DIGEST')
    require(not any(proof[key] for key in ('native_registration_proven', 'routed_output_row_proof', 'native_qualified', 'c2', 'c3')),
            'ARITHMETIC_IS_NOT_ADMISSION_AUTHORITY')
    return {name:snap['sha256'] for name,snap in record['snapshots'].items()}


def audit(inventory_path, result_path, source_root):
    packed = Path(inventory_path).read_bytes()
    raw = gzip.decompress(packed) if packed.startswith(b'\x1f\x8b') else packed
    result_raw = Path(result_path).read_bytes()
    require(sha(raw) == INVENTORY_SHA and sha(result_raw) == RESULT_SHA, 'EXACT_SOURCE87_ORIGINAL_BYTES_REQUIRED')
    data, result = json.loads(raw), json.loads(result_raw)
    runtime = source_root/'src/demoncat/runtime/vllm_ascend/worker_extension/quant_token_routes.py'
    require(sha(runtime.read_bytes()) == RUNTIME_SHA, 'OBSERVED_INTEGER_OBSERVER_SOURCE_CHANGED')
    constants = {node.targets[0].id:ast.literal_eval(node.value) for node in ast.parse(runtime.read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('SCHEMAS', 'REVIEWED')}
    source_checks = []
    for item in result['setup']['sources']:
        relative = item['path'].removeprefix('/installed/')
        local = source_root/('src/'+relative if relative.startswith('demoncat/') else relative)
        require(sha(local.read_bytes()) == item['sha256'], 'OBSERVED_RUNTIME_SOURCE_CHANGED:'+relative)
        source_checks.append(dict(path=relative, sha256=item['sha256']))
    require(len(data['native_forwards']) == len(data['forwards']) == 2 and data['structural_candidate_count'] == 135
        and len(data['structural_candidates']) == 135 and data['actual_projection_calls'] == data['actual_native_calls'] == 270
        and data['actual_native_prepare_calls'] == data['actual_native_execute_calls'] == 270
        and data['actual_public_api_calls'] == 540 and data['native_public_binding_complete'] is True
        and data['request_row_proven_calls'] == 20 and data['complete_healthy_replays'] is True
        and data['failure'] is None and data['group_owner_installed'] is False, 'FULL_OBSERVED_DENOMINATOR_REQUIRED')
    require(result['healthy_before'] == result['healthy_after'] and result['integer_route_capture'] is True
        and result['statistics'] is False and not any(result[k] for k in ('c1', 'c2', 'c3')), 'ZERO_FAULT_HEALTHY_OBSERVATION_REQUIRED')
    cleanup = {k:v for k,v in result['remove'].items() if k.startswith('owned_') or k in ('installed', 'observer_cleanup_pending', 'group_owner_installed', 'native_faults')}
    require(all(value == 0 or value is False for value in cleanup.values()), 'ALL_OBSERVER_RESOURCES_MUST_BE_ZERO')
    forwards, comparisons = [], []
    for index, forward in enumerate(data['native_forwards']):
        graph = forward['row_provenance']; boundary = graph['source_boundaries']; census = boundary['integer_routes']
        events = {row['sequence']:row for row in graph['events']}
        boundaries = {row['sequence']:row for row in boundary['boundaries']}
        records = census['records']
        require(len(records) == census['total_parents'] == 26 and census['overflow'] == 0, 'COMPLETE_26_ROUTE_PARENTS_REQUIRED')
        require(graph['actual_calls'] == graph['retained_calls'] == len(events) == 12853 and graph['missing_calls'] == 0
            and len(graph['unknown_edges']) == 12783 and not graph['errors'] and graph['complete'] is True
            and not boundary['errors'] and boundary['complete'] is True, 'COMPLETE_ROW_OBSERVATION_REQUIRED')
        require(len(forward['candidates']) == len(data['forwards'][index]['projections']) == 135
            and forward['complete_native_denominator'] is True, 'COMPLETE_NATIVE_FORWARD_REQUIRED')
        statuses = Counter(row['status'] for row in forward['candidates'])
        require(statuses == {'NATIVE_CALL_AND_REQUEST_ROWS_OBSERVED_NOT_LOSSLESS_QUALIFIED':10, 'UNVERIFIED':125}
            and all(row['qualified'] is False for row in forward['candidates']), 'NO_CANDIDATE_QUALIFICATION_UPGRADE')
        for module, (expected_sha, functions) in constants['REVIEWED'].items():
            actual_source = boundary['sources'][module]
            require(actual_source['sha256'] == expected_sha and all(actual_source['functions'][name]['role'] == role
                for name, role in functions.items()), 'PINNED_ACTUAL_PYTHON_ROUTE_SOURCES')
        helper_index = {(item['parent_sequence'], item['snapshot']):item for item in census['helper_calls']}
        require(len(helper_index) == len(census['helper_calls']) == 130, 'EXACT_130_INTEGER_HELPERS_REQUIRED')
        layers = {}; stage_calls = Counter(); registration = {}; snapshot_bytes = 0
        for record_index, record in enumerate(records):
            require(record['first_failure'] is None and record['complete'] is True and record['joins_complete'] is True
                and record['forward_id'] == forward['forward_id'] and record['readback_bytes'] == 1600
                and record['native_registration_proven'] is False and record['routed_output_row_proof'] is False,
                'COMPLETE_INTEGER_RECORD_REQUIRED')
            hashes = arithmetic(record)
            stages = record['stages']
            require(tuple(stages) == ORDER and [stages[k]['sequence'] for k in ORDER] == sorted(set(stages[k]['sequence'] for k in ORDER)), 'REAL_STAGE_ORDER')
            parent = events[record['parent_sequence']]
            link = parent['integer_route_evidence']
            require(parent['operator'] == 'vllm.moe_forward_shared.default' and parent['original_calls'] == 1
                and link['record_index'] == record_index and link['sha256'] == digest(record)
                and link['integer_map_valid'] is True and link['routed_output_row_proof'] is False, 'ACTUAL_PARENT_RECORD_JOIN')
            paths = set()
            for role, stage in stages.items():
                event = events[stage['sequence']]; src = stage['source']; registered = stage['registration']
                require(stage['original_calls'] == event['original_calls'] == 1 and stage['completed'] is True
                    and registered['actual_schema'] == event['schema'] == constants['SCHEMAS'][role]
                    and stage['outputs'] == event['outputs'], 'ACTUAL_STAGE_EVENT:'+role)
                source = boundaries[src['boundary_sequence']]
                require(source['status'] == 'OBSERVED' and all(source[k] == src[k] for k in ('role','function','model_path','source_sha256')),
                        'SOURCE_FRAME_JOIN:'+role)
                actual_lines = [line for line in registered['actual_dispatch_dump'].splitlines() if line.startswith('PrivateUse1')]
                require(len(actual_lines) == 1 and actual_lines[0].startswith('PrivateUse1:')
                    and registered['torch_version'] == '2.9.0+cpu' and registered['native_registration_proven'] is False,
                    'CURRENT_DUMP_NOT_NATIVE_OWNER_PROOF:'+role)
                registration[role] = dict(schema=registered['actual_schema'], privateuse1_origin=actual_lines[0].split(' :: ')[0],
                    native_registration_proven=False)
                paths.add(src['model_path']); stage_calls[role] += 1
            require(len(paths) == 1 and next(iter(paths)) not in layers, 'EXACT_UNIQUE_LAYER_PARENT')
            layers[next(iter(paths))] = hashes
            options = stages['routing']['formals']
            expected = dict(active_num=48, expert_capacity=-1, expert_num=64, drop_pad_mode=0,
                expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=1,
                active_expert_range=[0,64], row_idx_type=0, scale=None, offset=None)
            require(all(type(options[k]) is type(value) and options[k] == value for k,value in expected.items()), 'ACTUAL_ROUTE_OPTIONS')
            require(stages['gmm1']['formals']['bias'] is stages['gmm1']['formals']['offset'] is None
                and all(stages['gmm2']['formals'][key] == value for key,value in
                    dict(split_item=2, group_type=0, group_list_type=1, act_type=0, output_dtype='torch.bfloat16',
                         bias=None, offset=None, antiquant_scale=None, antiquant_offset=None, activation_input=None,
                         activation_quant_scale=None, activation_quant_offset=None, tuning_config=None).items())
                and stages['unpermute']['formals']['padded_mode'] is False
                and stages['unpermute']['formals']['restore_shape'] is None,
                'ACTUAL_GMM_AND_UNPERMUTE_OPTIONS')
            def tensor(role, name):
                return stages[role]['formals'][name]['tensor']
            require(tensor('routing','x') == record['parent_input_before_declaration']
                and tensor('routing','expert_idx') == stages['topk']['outputs'][1]
                and tensor('cumsum','self') == stages['routing']['outputs'][2]
                and tensor('gmm1','x') == stages['routing']['outputs'][0]
                and tensor('gmm1','x_scale') == stages['routing']['outputs'][3]
                and tensor('gmm1','group_list') == stages['cumsum']['outputs'][0]
                and stages['gmm2']['formals']['x'] == [dict(tensor=stages['gmm1']['outputs'][0])]
                and stages['gmm2']['formals']['per_token_scale'] == [dict(tensor=stages['gmm1']['outputs'][1])]
                and tensor('gmm2','group_list') == stages['routing']['outputs'][2]
                and tensor('abs','self') == stages['routing']['outputs'][1]
                and tensor('unpermute','permuted_tokens') == stages['gmm2']['outputs'][0]
                and tensor('unpermute','sorted_indices') == stages['abs']['outputs'][0]
                and tensor('unpermute','probs') == stages['topk']['outputs'][0], 'ACTUAL_RETAINED_OBJECT_DESCRIPTOR_CHAIN')
            producers = dict(topk_ids=('topk',1), raw_indices=('routing',1), expert_counts=('routing',2),
                cumulative_counts=('cumsum',0), absolute_indices=('abs',0))
            for name, snap in record['snapshots'].items():
                role, output_index = producers[name]; helper = helper_index[(record['parent_sequence'],name)]
                require(snap['sequence'] == stages[role]['sequence'] and snap['metadata_before'] == stages[role]['outputs'][output_index]
                    and helper['source'] == snap['metadata_before'] and helper['model_sequence'] == snap['sequence']
                    and helper['forward_id'] == forward['forward_id'] and helper['bytes'] == snap['bytes']
                    and helper['classification'] == 'OBSERVER_INTEGER_READBACK_NOT_MODEL_INVOCATION'
                    and helper['status'] == 'COMPLETE' and helper['cleanup'] == 'REMOVED'
                    and helper['total_dispatches'] == 3 and helper['overflow'] == 0, 'EXACT_INTEGER_HELPER:'+name)
                operations = helper['operations']
                require([op['operator'] for op in operations] == ['aten._to_copy.default','aten.resolve_conj.default','aten.resolve_neg.default']
                    and all(op['original_calls'] == 1 and op['status'] == 'RETURNED' for op in operations)
                    and operations[0]['inputs'] == [snap['metadata_before']]
                    and all(meta['device'] == 'cpu' and meta['dtype'] == snap['dtype']
                            for op in operations for meta in op['outputs']), 'ACTUAL_INTEGER_ONLY_HELPER_CALLS')
                snapshot_bytes += snap['bytes']
        require(census['readback_bytes'] == census['retained_integer_payload_bytes'] == snapshot_bytes == 41600
            and census['helper_operator_calls'] == census['helper_total_dispatches'] == 390
            and census['helper_dispatch_overflow'] == 0 and census['peak_owned_cpu_copy_bytes'] == 512
            and not any(census[k] for k in ('live_parent_refs','live_tensor_refs','retained_helper_modes','helper_active')),
            'BOUNDED_HELPER_ACCOUNTING_AND_CLEANUP')
        require(set(layers) == {'model.layers.'+str(n)+'.mlp.experts' for n in range(1,27)}, 'COMPLETE_MOE_LAYER_SET')
        comparisons.append(layers)
        paths = [row['path'] for row in data['forwards'][index]['projections'] if row['output_row_lineage']['verified']]
        require(len(paths) == 10, 'UNCHANGED_TEN_OBSERVED_PROJECTION_ROWS')
        shared = [events[row['parent_sequence']]['shared_moe_row_diagnostic'] for row in records]
        reasons = Counter(row['reason'] or 'CURRENT_SHARED_INPUT_OBSERVED' for row in shared)
        require(sum(row['resumed'] is True for row in shared) == 1
            and reasons == {'CURRENT_SHARED_INPUT_OBSERVED':1, 'MOE_CURRENT_REQUEST_INPUT_ROWS_UNPROVEN':25},
            'INTEGER_MAP_NOT_CURRENT_REQUEST_INPUT_PROOF')
        forwards.append(dict(index=index, parents=26, integer_snapshots=130, arithmetic_checks_passed=26,
            per_parent_bytes=1600, integer_bytes=snapshot_bytes, helper_calls=130, helper_dispatches=390,
            stage_calls=dict(stage_calls), missing_or_refused_records=0, graph_calls=12853, unknown_edges=12783,
            projection_candidates=135, row_proven_candidates=10, unverified_candidates=125,
            shared_parent_input_rows_resumed=1, remaining_parent_input_reasons=dict(reasons),
            first_unproven_projection=next(row['path'] for row in data['forwards'][index]['projections'] if not row['output_row_lineage']['verified']),
            observed_post_first_route_edges=[{key:events[sequence][key] for key in ('sequence','operator','row_preserving','reason')}
                for sequence in (933,934,935,936,937,938,939)],
            row_proven_projection_paths=paths, registration_observations=registration,
            readback_seconds=census['readback_ns']/1e9, observer_seconds=census['observer_ns']/1e9,
            source_profile_seconds=census['source_profile_ns']/1e9, timing_categories_overlap=True,
            max_owned_cpu_copy_bytes=512, resources_zero=True))
    require(comparisons[0] == comparisons[1], 'INTEGER_SNAPSHOTS_DIFFER_BETWEEN_HEALTHY_REPLAYS')
    return dict(schema='demoncat.quant.source87_integer_diagnostic.v1',
        status='INTEGER_RELATIONS_COMPLETE_NATIVE_ROUTED_ROWS_UNPROVEN',
        inputs=dict(inventory_sha256=INVENTORY_SHA, inventory_bytes=len(raw), compressed_sha256=sha(packed),
            result_sha256=RESULT_SHA, result_bytes=len(result_raw)), runtime_source_checks=source_checks,
        native_build=data['native_build'], observed_forwards=forwards, integer_snapshot_pairs_equal=130,
        actual_projection_calls=270, actual_native_parents=270, actual_native_prepares=270, actual_native_executes=270,
        integer_parent_records=52, integer_snapshots=260, integer_payload_bytes=83200,
        helper_dispatches_separate_from_model=780, healthy_before_equals_after=True,
        cleanup=cleanup, float_activation_values_exported=False,
        native_registration_proven=False, routed_output_row_proof=False, native_qualified_targets=0,
        c1=False, c2=False, c3=False, qualification_granted=False,
        remaining_gates=data['remaining_gates'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    value = audit(args.inventory, args.result, args.source_root)
    with args.out.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=value['status'], integer_parents=52, sha256=sha(args.out.read_bytes()), qualification_granted=False)))


if __name__ == '__main__':
    main()
