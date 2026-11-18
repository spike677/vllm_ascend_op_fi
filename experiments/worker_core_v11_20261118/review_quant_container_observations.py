"""Stdlib-only reconciliation of the bounded four-container originals.

This reader never imports an observer, calls a Native API, or grants row/C2/C3
authority. The optional local DSO bytes and the recorded build identity are
reported separately. No original values or tensor addresses enter its summary.
"""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tarfile

try:
    from .review_quant_routed_frames import EpochLedger, key, literal_constants, review_native_denominator
except ImportError:
    from review_quant_routed_frames import EpochLedger, key, literal_constants, review_native_denominator


ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'quant-four-container-one-decode/'
MEMBERS = ('controller.json', 'reviewed_cases.json', 'reviewed_stale_allocation_scope.json',
    PREFIX + 'result.json', PREFIX + 'container-trace/observation.json',
    PREFIX + 'container-trace/cleanup-observation-001.json',
    PREFIX + 'shared-scale/serving.json', PREFIX + 'shared-scale/inventory/shared_scale_inventory.json')
STAGES = {1: ('routing', 'route_dispatch'), 2: ('gmm1', 'rchain_gmm1'),
          3: ('gmm2', 'rchain_gmm2'), 4: ('unpermute', 'route_combine')}
FUNCTIONS = ('aclCreateTensorList', 'aclDestroyTensorList', 'aclCreateIntArray', 'aclDestroyIntArray')
IMAGE = 'sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0'
PINS = dict(nn='6dc3b4fcfcf57b10be0018e89f4edbc079f02b4bb0aaadb109df350939fd4c99',
    custom_real='de1633c2f1f41e505277697ee82f3ffba940f730bed9986c5aeff4c8514b713f',
    torch_npu='dd07e4afb4e31a5123d0b97a4a319acaa96158705a22d704bfc0a91f662a6041',
    custom_extension='e63806486e0311c2df8ce9b120ae98f55e8c8eb91ccaf204ef0cf7d6b469ed67')
EXISTING_ROW_SITES = {
    'model.layers.0.self_attn.kv_a_proj_with_mqa', 'model.layers.0.self_attn.q_proj', 'model.layers.0.self_attn.o_proj',
    'model.layers.0.mlp.gate_up_proj', 'model.layers.0.mlp.down_proj',
    'model.layers.1.self_attn.kv_a_proj_with_mqa', 'model.layers.1.self_attn.q_proj', 'model.layers.1.self_attn.o_proj',
    'model.layers.1.mlp.shared_experts.gate_up_proj', 'model.layers.1.mlp.shared_experts.down_proj',
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def parsed(raw):
    def reject(value):
        raise ValueError('NONFINITE_JSON:' + value)
    return json.loads(raw, parse_constant=reject)


def load_archive(path, expected):
    require(re.fullmatch('[a-f0-9]{64}', expected), 'EXPLICIT_ORIGINAL_PACKAGE_SHA_REQUIRED')
    raw = Path(path).read_bytes()
    require(len(raw) <= 64 * 1024 * 1024 and digest(raw) == expected, 'ORIGINAL_PACKAGE_BYTES_CHANGED')
    values, hashes = {}, {}
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(MEMBERS) and {m.name for m in members} == set(MEMBERS),
                'EXACT_EIGHT_ORIGINAL_MEMBERS_REQUIRED')
        require(sum(m.size for m in members) <= 128 * 1024 * 1024, 'BOUNDED_ORIGINAL_PACKAGE_REQUIRED')
        for member in members:
            require(member.isfile() and not PurePosixPath(member.name).is_absolute()
                and '..' not in PurePosixPath(member.name).parts and member.name not in values,
                    'REGULAR_UNIQUE_ORIGINAL_MEMBERS_REQUIRED')
            data = archive.extractfile(member).read()
            require(len(data) == member.size, 'COMPLETE_ORIGINAL_MEMBER_REQUIRED')
            values[member.name] = parsed(data)
            hashes[member.name] = dict(sha256=digest(data), bytes=len(data))
    return values, hashes


def build_evidence(directory, expected, source_root):
    directory, source_root = Path(directory), Path(source_root)
    raw = (directory / 'build.json').read_bytes()
    require(digest(raw) == expected, 'NATIVE_BUILD_MANIFEST_SHA_CHANGED')
    build = parsed(raw)
    receipt_raw = (directory / 'build-receipt.json').read_bytes(); receipt = parsed(receipt_raw)
    require(digest(receipt_raw) == build['build_receipt_sha256'], 'NATIVE_BUILD_RECEIPT_SHA_CHANGED')
    require(build['schema'] == 'demoncat.quant.container_deployment.v1'
        and build['image_id'] == receipt['image_id'] == IMAGE and not build['qualified']
        and receipt['status'] == 'SDK_AND_TYPED_CPU_CONTRACT_PASS_NOT_NPU'
        and receipt['returncode'] == 0 and receipt['cleanup']['confirmed'] is True,
            'COMPLETE_REAL_SDK_BUILD_RECEIPT_REQUIRED')
    require(build['sources'] == receipt['sources'] and build['source_commit'] == receipt['source_commit'],
            'NATIVE_BUILD_SOURCE_IDENTITY_CHANGED')
    for name, expected_sha in build['sources'].items():
        path = (source_root / name).resolve(strict=True)
        require(path.is_relative_to(source_root.resolve()) and digest(path.read_bytes()) == expected_sha,
                'ACTUAL_NATIVE_COMPILE_SOURCE_BYTES:' + name)
    profile_raw = (directory / 'profile.json').read_bytes(); profile = parsed(profile_raw)
    require(digest(profile_raw) == build['files']['profile.json'], 'NATIVE_PROFILE_FILE_CHANGED')
    unsigned = dict(profile); unsigned.pop('profile_sha256')
    require(digest(canonical(unsigned)) == profile['profile_sha256'] == build['profile_sha256']
        and profile['sources'] == build['sources'] and profile['functions'] == list(FUNCTIONS)
        and profile['image'] == IMAGE, 'COMPLETE_EXACT_PROFILE_REQUIRED')
    require(all(profile['images'][name]['file_sha256'] == value for name, value in PINS.items()),
            'REVIEWED_REAL_FUNCTION_AND_CALLER_IMAGES_CHANGED')
    metadata = {}
    for name in ('cpu-contract.json', 'sdk-load.json', 'driver-identity.json',
                 'absolute-needed-contract.json', 'image-inspect.json'):
        data = (directory / name).read_bytes()
        require(digest(data) == receipt['artifacts'][name], 'NATIVE_STAGE_ORIGINAL_SHA_CHANGED:' + name)
        metadata[name] = parsed(data)
    cpu, load, drivers = (metadata[name] for name in ('cpu-contract.json', 'sdk-load.json', 'driver-identity.json'))
    require(cpu['status'] == 'PASS_ALL_19_SDK_TYPED_CPU_CONTRACTS_NOT_NPU'
        and len(cpu['cases']) == len(set(cpu['expected_cases'])) == 19
        and [row['case'] for row in cpu['cases']] == cpu['expected_cases']
        and all(row['status'] == 'PASS' and row['returncode'] == 0 for row in cpu['cases'])
        and cpu['sources'] == build['sources'], 'COMPLETE_19_SDK_TYPED_CASE_RECEIPTS_REQUIRED')
    checked_logs = 0
    for row in cpu['cases']:
        log = directory / row['log']
        if log.exists():
            require(digest(log.read_bytes()) == row['log_sha256'], 'AVAILABLE_CPU_CASE_LOG_BYTES_CHANGED')
            checked_logs += 1
    if (directory / 'cpu_contract').exists():
        require(digest((directory / 'cpu_contract').read_bytes()) == cpu['executable_sha256'],
                'AVAILABLE_CPU_CONTRACT_EXECUTABLE_CHANGED')
    for name in ('cpu-contract.json', 'sdk-load.json'):
        require(receipt['stage_receipts'][name]['sha256'] == receipt['artifacts'][name]
            and receipt['stage_receipts'][name]['status'] == metadata[name]['status'], 'STAGE_RECEIPT_JOIN_CHANGED')
    require(load['status'] == 'SDK_DSO_LOADED_MARKERS_ONLY_NOT_RUNTIME_ACCEPTANCE'
        and load['profile_sha256'] == build['profile_sha256']
        and load['driver_identity_after'] == drivers
        and load['driver_identity_sha256'] == receipt['driver_identity_sha256'] == receipt['artifacts']['driver-identity.json']
        and load['creator_calls'] == load['install_calls'] == load['explicit_driver_api_calls'] == 0
        and load['loader_constructor_driver_calls'] == 'NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION',
            'ACTUAL_MARKER_LOAD_SCOPE_AND_DRIVER_IDENTITY_REQUIRED')
    by_path = {row['path']: row for row in drivers['files'].values()}
    require(len(by_path) == 3 and {row['path'] for row in load['loaded_driver_mappings']} == set(by_path),
            'THREE_EXACT_DRIVER_MAPPING_ORIGINALS_REQUIRED')
    for row in load['loaded_driver_mappings']:
        original = by_path[row['path']]
        require(row['mapped_inode'] == original['inode'] and row['sha256'] == original['sha256'],
                'DRIVER_MAPPING_ORIGINAL_IDENTITY_CHANGED')
    elf = metadata['absolute-needed-contract.json']
    require(elf['status'] == 'PASS_REAL_ELF_ABSOLUTE_NEEDED_SAME_SONAME'
        and elf['original_sha256_before'] == elf['original_sha256_after']
        and metadata['image-inspect.json']['Id'] == IMAGE, 'ACTUAL_UNCHANGED_ORIGINAL_ELF_CONTRACT_REQUIRED')
    available = []
    for name, expected_sha in build['files'].items():
        path = directory / name
        if path.exists():
            require(digest(path.read_bytes()) == expected_sha, 'AVAILABLE_NATIVE_BINARY_BYTES_CHANGED:' + name)
            available.append(name)
    return build, profile, dict(build_sha256=expected, compile_source_commit=build['source_commit'],
        compile_sources_verified=len(build['sources']), cpu_case_receipts=19,
        cpu_case_log_bytes_rehashed=checked_logs,
        profile_sha256=build['profile_sha256'], actual_dso_bytes_rehashed=build['library'] in available,
        actual_dso_sha256=build['files'][build['library']],
        limitation='Recorded build/loaded-profile evidence; absent DSO bytes are not independently rehashed.',
        loader_constructor_driver_calls='NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION')


def executable_offset(offset, image):
    return type(offset) is int and any(row['vaddr'] <= offset < row['vaddr'] + row['size'] for row in image['segments'])


def review_calls(observation, profile):
    events, state = observation['events'], observation['state']
    require([row['sequence'] for row in events] == list(range(1, len(events) + 1))
        and len(events) == state['retained_events'] == state['total_events'] == state['total_calls'] * 2,
            'LOSSLESS_NATIVE_ENTRY_RETURN_DENOMINATOR_REQUIRED')
    require(state['installed'] == state['stopped'] == state['resolver_ready'] == 1
        and not any(state[k] for k in ('active_calls', 'scope_token', 'live_registrations', 'registration_count',
            'first_error', 'abi_exceptions', 'getter_capture_failures', 'overflow', 'foreign_scope_thread_calls')),
            'NATIVE_OBSERVER_ERROR_OR_PENDING_LIFETIME')
    entered, completed, live, handle_history, funcs = {}, {}, {}, Counter(), {}
    counts, stages, nulls, members, payload_offset = Counter(), Counter(), 0, 0, 0
    previous_time = 0; outside = 0; integer_lengths = Counter(); tensor_layouts = Counter()
    for event in events:
        op, phase, call = event['operation'], event['phase'], event['call']
        require(op in (1, 2, 3, 4) and phase in (1, 2) and event['depth'] == 1
            and event['thread'] == state['owner_thread'] and event['pid'] > 0
            and event['error'] == event['status'] == event['reserved'] == 0,
                'FAILED_FOREIGN_OR_UNKNOWN_NATIVE_CALL')
        require(event['monotonic_ns'] >= previous_time, 'NATIVE_MONOTONIC_SEQUENCE_CHANGED')
        previous_time = event['monotonic_ns']
        require(event['function'] == event['function_base'] + event['function_offset']
            and event['caller'] == event['caller_base'] + event['caller_offset']
            and executable_offset(event['function_offset'], profile['images']['nn'])
            and event['caller_kind'] in (1, 2)
            and executable_offset(event['caller_offset'], profile['images'][
                'torch_npu' if event['caller_kind'] == 1 else 'custom_extension']), 'ACTUAL_PC_PROFILE_JOIN_FAILED')
        function = (event['function'], event['function_base'], event['function_offset'])
        require(funcs.setdefault(op, function) == function, 'REAL_FUNCTION_CHANGED_DURING_OBSERVATION')
        if phase == 1:
            require(call == len(entered) + 1 and call not in entered and event['original_calls'] == 0,
                    'NATIVE_ENTRY_IDENTITY_CHANGED')
            entered[call] = event; counts[op] += 1; stages[(event['stage'], op)] += 1
            outside += int(event['scope_token'] == 0)
            require(event['count'] <= 128, 'CREATOR_MEMBER_BUDGET_EXCEEDED')
            if op in (1, 3):
                require(event['handle'] == event['registration'] == 0, 'CREATOR_MUST_PRECEDE_ACTUAL_HANDLE')
                content = event.get('tensor_descriptors', []) if op == 1 else event.get('integer_values', [])
                size = 304 if op == 1 else 8
                require(len(content) == event['count'] and event['payload_bytes'] == len(content) * size,
                        'COMPLETE_ACTUAL_CREATOR_PAYLOAD_REQUIRED')
                if content:
                    require(event['payload_offset'] == payload_offset, 'RAW_PAYLOAD_GAP_OR_OVERLAP')
                    payload_offset += event['payload_bytes']
                if op == 1:
                    members += len(content)
                    for tensor in content:
                        require(tensor['descriptor'] > 0 and tensor['address'] > 0
                            and tensor['complete'] == 1 and tensor['called_mask'] == 127
                            and tensor['null_mask'] == tensor['truncated_mask'] == 0
                            and tensor['statuses'] == [0] * 7
                            and all(0 < count <= 8 for count in tensor['counts']), 'PUBLIC_SDK_DESCRIPTOR_CAPTURE_INCOMPLETE')
                        tensor_layouts[(tensor['format'], *tensor['counts'])] += 1
                else:
                    require(all(type(x) is int and -(1 << 63) <= x < (1 << 63) for x in content),
                            'ACTUAL_HOST_INT64_PAYLOAD_REQUIRED')
                    integer_lengths[(event['stage'], len(content))] += 1
            else:
                require(event['payload_bytes'] == event['count'] == 0, 'DESTROY_MUST_NOT_READ_SAVED_PAYLOAD')
        else:
            require(call in entered and call not in completed and event['original_calls'] == 1
                and event['payload_bytes'] == 0, 'ORIGINAL_FUNCTION_MUST_RETURN_ONCE')
            start = entered[call]; completed[call] = event
            require(all(event[k] == start[k] for k in ('operation', 'count', 'pid', 'thread', 'depth',
                'scope_token', 'forward', 'operator_sequence', 'boundary_sequence', 'stage', 'caller',
                'caller_kind', 'caller_base', 'caller_offset', 'function', 'function_base', 'function_offset')),
                    'ACTUAL_ENTRY_RETURN_IDENTITY_CHANGED')
            if op in (1, 3):
                require(event['handle'] > 0 and event['registration'] == call and event['handle'] not in live,
                        'FRESH_CURRENT_REGISTRATION_REQUIRED')
                live[event['handle']] = event; handle_history[event['handle']] += 1
            elif event['handle']:
                created = live.pop(event['handle'], None)
                require(created is not None and created['registration'] == start['registration'] == event['registration']
                    and created['operation'] + 1 == op and start['handle'] == event['handle']
                    and created['scope_token'] == event['scope_token'], 'DESTROY_CURRENT_LIFETIME_JOIN_FAILED')
            else:
                require(start['handle'] == start['registration'] == event['registration'] == 0,
                        'NULL_DESTROY_IS_NOT_A_LIFETIME')
                nulls += 1
    require(not live and len(entered) == len(completed) == state['total_calls'], 'MISSING_REAL_DESTROY_OR_RETURN')
    require([counts[n] for n in (1, 2, 3, 4)] == state['entries'] == state['returns'] == state['original_calls']
        and outside == state['outside_scope_calls'] and payload_offset == state['metadata_bytes'],
            'RAW_NATIVE_COUNTER_RECOUNT_MISMATCH')
    arrays = state['arrays']
    require(arrays['capture_calls'] == members and arrays['getter_calls'] == arrays['acquired_arrays']
        == arrays['released_arrays'] == 3 * members and not any(arrays[k] for k in (
            'active_captures', 'active_getters', 'live_arrays', 'getter_failures', 'getter_exceptions',
            'capture_exceptions', 'ownership_errors')), 'REAL_GETTER_ARRAYS_NOT_ALL_ACCOUNTED')
    require(state['total_calls'] <= 4096 and len(events) <= 8192 and payload_offset <= 8 * 1024 * 1024,
            'ACTUAL_OBSERVER_BUDGET_EXCEEDED')
    return dict(actual_calls=len(entered), actual_events=len(events), original_calls_by_api=dict(zip(FUNCTIONS, state['original_calls'])),
        created_nonnull_lifetimes=counts[1] + counts[3], retired_nonnull_lifetimes=counts[2] + counts[4] - nulls,
        null_destroy_calls=nulls, reused_handle_registrations=sum(n - 1 for n in handle_history.values()),
        outside_source_windows=outside, source_window_calls=len(entered) - outside,
        copied_descriptor_members=members, copied_payload_bytes=payload_offset,
        sdk_arrays_acquired=arrays['acquired_arrays'], sdk_arrays_released=arrays['released_arrays'],
        stage_call_counts=[dict(stage=stage, operation=op, count=count) for (stage, op), count in sorted(stages.items())],
        integer_lengths=[dict(stage=stage, elements=size, count=count) for (stage, size), count in sorted(integer_lengths.items())],
        descriptor_layout_counts=[dict(format=k[0], ranks=list(k[1:]), count=n) for k, n in sorted(tensor_layouts.items())])


def tensor_packets(value):
    if isinstance(value, list):
        return [item for child in value for item in tensor_packets(child)]
    if isinstance(value, dict):
        if set(value) == {'tensor'}:
            return [value['tensor']]
        return [item for child in value.values() for item in tensor_packets(child)]
    return []


def descriptor_matches_public(descriptor, tensor):
    """Current scope metadata comparison, explicitly not a GWS-slot assignment."""
    types = {2: ('torch.int8', 1), 27: ('torch.bfloat16', 2), 0: ('torch.float32', 4), 9: ('torch.int64', 8)}
    dtype, width = types.get(descriptor['dtype'], (None, 0))
    view, stride, storage = descriptor['counts']
    return (descriptor['address'] == tensor['storage_address'] and descriptor['offset'] == tensor['offset']
        and descriptor['shape'][:view] == tensor['shape'] and descriptor['strides'][:stride] == tensor['stride']
        and dtype == tensor['dtype'] and tensor['device'] == 'npu:0'
        and math.prod(descriptor['storage'][:storage]) * width == tensor['storage_bytes'])


def review_windows(observation, graph, public_forward, source_root, expected_layers=26):
    boundary = graph['source_boundaries']; ledger = EpochLedger(graph['events'])
    events = {row['sequence']: row for row in graph['events']}
    frames = {row['sequence']: row for row in boundary['boundaries']}
    require(len(frames) == len(boundary['boundaries']) and not boundary['errors'] and not graph['errors']
        and graph['actual_calls'] == graph['retained_calls'] == len(events) and graph['missing_calls'] == 0,
            'COMPLETE_INDEPENDENT_SOURCE_AND_DISPATCH_RECORDS_REQUIRED')
    worker = Path(source_root) / 'src/demoncat/runtime/vllm_ascend/worker_extension'
    constants = literal_constants(worker / 'quant_token_routes.py'); schemas = constants['SCHEMAS']
    reviewed = {}
    for filename in ('quant_token_routes.py', 'quant_route_frames.py'):
        for module, (source_sha, functions) in literal_constants(worker / filename)['REVIEWED'].items():
            require(boundary['sources'][module]['sha256'] == source_sha, 'ACTUAL_PRODUCTION_SOURCE_PIN_CHANGED')
            for function, role in functions.items():
                reviewed.setdefault(role, set()).add((source_sha, function))
    scopes = observation['public_scopes']; groups = {}; stages = Counter()
    require([s['native_scope']['token'] for s in scopes] == list(range(1, len(scopes) + 1)), 'PUBLIC_SCOPE_TOKEN_ORDER_CHANGED')
    for scope in scopes:
        native, proof = scope['native_scope'], scope['evidence']
        stage, role = STAGES[native['stage']]; stages[stage] += 1
        require(native['version'] == 1 and native['size'] == 56 and native['thread'] == observation['state']['owner_thread'],
                'CURRENT_NATIVE_SCOPE_ABI_AND_THREAD_REQUIRED')
        event, frame = events[native['operator_sequence']], frames[native['boundary_sequence']]
        require(scope['status'] == 'RETURNED' and scope['original_calls'] == event['original_calls'] == 1
            and scope['original_error'] is None and scope['leave_attempts'] == [dict(native_status=0,
                observed_error=None, observed_original_calls=1)], 'SOURCE_OPERATOR_OR_SCOPE_RETURN_FAILED')
        require(proof['forward_id'] == graph['forward_id'] == public_forward['forward_id']
            and proof['request_ids'] == public_forward['request_ids'] and native['forward'] == 1,
                'CURRENT_REQUEST_FORWARD_SCOPE_CHANGED')
        require(proof['schema'] == schemas[stage] == event['schema']
            and event['schema_sha256'] == digest(canonical(schemas[stage]))
            and proof['operator'] == event['operator'] and proof['stage'] == stage
            and native['operator_sequence'] == proof['operator_sequence']
            and native['boundary_sequence'] == proof['boundary_sequence'], 'ACTUAL_CURRENT_SCHEMA_AND_OPERATOR_JOIN')
        require(frame['role'] == role and frame['status'] == 'OBSERVED'
            and frame['function'] == proof['function'] and frame['source_sha256'] == proof['source_sha256']
            and (proof['source_sha256'], proof['function']) in reviewed[role]
            and native['operator_sequence'] in frame['operator_sequences'], 'CURRENT_PINNED_FRAME_JOIN')
        require(scope['after_input_metadata'] == proof['input_objects']
            and [row['tensor'] for row in proof['input_objects']] == event['inputs']
            and not event['mutable'] and event['status'] == 'OBSERVED', 'CURRENT_ARGUMENT_OBJECT_OR_MUTATION_CHANGED')
        require(Counter(canonical(item) for item in tensor_packets(proof['formals']))
            == Counter(canonical(item) for item in event['inputs'])
            and set(proof['formals']) == set(proof['argument_origins'])
            and set(proof['argument_origins'].values()) <= {'POSITIONAL', 'KEYWORD', 'CURRENT_SCHEMA_DEFAULT'},
                'ACTUAL_FORMAL_TENSOR_AND_ARGUMENT_ORIGIN_JOIN')
        options = {
            'routing': dict(active_num=48, expert_capacity=-1, expert_num=64, drop_pad_mode=0,
                expert_tokens_num_type=1, expert_tokens_num_flag=True, quant_mode=1,
                active_expert_range=[0, 64], row_idx_type=0, scale=None, offset=None),
            'gmm1': dict(bias=None, offset=None),
            'gmm2': dict(split_item=2, group_type=0, group_list_type=1, act_type=0,
                output_dtype='torch.bfloat16', bias=None, offset=None, antiquant_scale=None,
                antiquant_offset=None, activation_input=None, activation_quant_scale=None,
                activation_quant_offset=None, tuning_config=None),
            'unpermute': dict(padded_mode=False, restore_shape=None),
        }[stage]
        require(all(type(proof['formals'][name]) is type(value) and proof['formals'][name] == value
                    for name, value in options.items()), 'THIS_BOUNDED_PUBLIC_FORMAL_SCOPE_CHANGED')
        for item in proof['input_objects']:
            require(item['write_epoch'] == ledger.at(event['sequence'], key(item['tensor']))
                == ledger.at(event['sequence'], key(item['tensor']), after=True), 'CURRENT_ARGUMENT_EPOCH_CHANGED')
        ancestor = frame
        while ancestor['role'] != 'rchain_model':
            parent = ancestor['parent_source_sequence']
            require(type(parent) is int and 0 < parent < ancestor['sequence'], 'CURRENT_MOE_MODEL_ANCESTRY_REQUIRED')
            ancestor = frames[parent]
        row = groups.setdefault(ancestor['sequence'], {})
        require(stage not in row, 'DUPLICATE_STAGE_IN_CURRENT_MOE_MODEL')
        row[stage] = scope
        for raw in observation['events']:
            if raw['scope_token'] == native['token']:
                require(all(raw[k] == native[k] for k in ('forward', 'operator_sequence', 'boundary_sequence', 'stage', 'thread'))
                    and raw['caller_kind'] == (2 if stage == 'routing' else 1), 'RAW_CALL_CURRENT_SOURCE_WINDOW_JOIN')
    require(stages == Counter({stage: expected_layers for stage, _ in STAGES.values()})
        and len(groups) == expected_layers, 'COMPLETE_26_BY_FOUR_NATURAL_WINDOWS_REQUIRED')
    source_state = observation['source_scopes']
    require(source_state['calls'] == len(scopes) and not source_state['errors'] and not source_state['active']
        and not source_state['native_qualified'] and not source_state['request_rows_proven'],
            'CURRENT_SOURCE_WINDOW_OBSERVER_ERRORS_OR_LIVE_SCOPE')
    routing_ranges = 0; gmm2_layouts = Counter(); unpermute_nonsemantic = 0; member_matches = Counter()
    for group in groups.values():
        require(set(group) == {stage for stage, _ in STAGES.values()}
            and [group[stage]['native_scope']['operator_sequence'] for stage, _ in STAGES.values()]
            == sorted(group[stage]['native_scope']['operator_sequence'] for stage, _ in STAGES.values()),
                'COMPLETE_CURRENT_ROUTING_GMM1_GMM2_UNPERMUTE_ORDER')
        route = group['routing']; token = route['native_scope']['token']
        calls = [row for row in observation['events'] if row['scope_token'] == token and row['phase'] == 1]
        integer = [row for row in calls if row['operation'] == 3]
        require(len(integer) == 1 and integer[0].get('integer_values') == route['evidence']['formals']['active_expert_range'],
                'CURRENT_NATIVE_ROUTE_RANGE_VS_PUBLIC_ACTUAL_ARGUMENT')
        routing_ranges += 1
        current_gmm2 = group['gmm2']; token = current_gmm2['native_scope']['token']
        formal = current_gmm2['evidence']['formals']
        current = {name: tensor_packets(formal[name]) for name in ('x', 'weight', 'scale', 'per_token_scale')}
        current['returned_output'] = events[current_gmm2['native_scope']['operator_sequence']]['outputs']
        require(all(len(items) == 1 for items in current.values()), 'CURRENT_SINGLE_MEMBER_GMM2_OBJECTS_REQUIRED')
        seen = set()
        for raw in observation['events']:
            if raw['scope_token'] == token and raw['operation'] == 1 and raw['phase'] == 1:
                gmm2_layouts[raw['count']] += 1
                if raw['count']:
                    require(raw['count'] == 1, 'CURRENT_GMM2_MEMBER_COUNT_CHANGED')
                    member = raw['tensor_descriptors'][0]
                    matches = [name for name, items in current.items() if descriptor_matches_public(member, items[0])]
                    require(len(matches) == 1 and matches[0] not in seen, 'CURRENT_CREATOR_PUBLIC_MEMBER_JOIN_AMBIGUOUS')
                    seen.add(matches[0]); member_matches[matches[0]] += 1
        require(seen == set(current), 'CURRENT_CREATOR_PUBLIC_GMM2_MEMBERS_INCOMPLETE')
        token = group['unpermute']['native_scope']['token']
        # This is intentionally not an assertion of a GWS parameter role.
        unpermute_nonsemantic += sum(row['scope_token'] == token and row['operation'] == 3 and row['phase'] == 1
                                    and row['count'] > 0 for row in observation['events'])
    require(not graph['native_qualified'] and not boundary['native_qualified']
        and boundary['integer_routes'] is None and boundary['routed_call_frames'] is None,
            'THIS_ATTEMPT_MUST_NOT_REUSE_OLD_INTEGER_OR_FRAME_PROOF')
    return dict(source_windows=len(scopes), complete_model_groups=len(groups), stages=dict(stages),
        current_inputs_and_write_epochs_checked=len(scopes), actual_routing_range_arguments_compared=routing_ranges,
        gmm2_tensor_list_counts={str(k): v for k, v in sorted(gmm2_layouts.items())},
        gmm2_current_descriptor_matches=dict(member_matches), descriptor_matches_assign_gws_roles=False,
        unpermute_integer_arrays_without_gws_parameter_role=unpermute_nonsemantic,
        current_integer_route_capture=False, routed_rows_granted=False)


def review_row_scope(inventory):
    forward, native = inventory['forwards'][0], inventory['native_forwards'][0]
    graph = native['row_provenance']; nodes = {row['node_id']: row for row in graph['nodes']}
    ledger = EpochLedger(graph['events']); projections = {row['module_object_identity']: row for row in inventory['structural_candidates']}
    verified = []
    for call in inventory['public_calls']:
        if call['operator'] != 'npu_dynamic_quant':
            continue
        row = call['request_row_lineage']
        if row['verified'] is not True:
            require(row.get('native_qualified') is not True, 'UNKNOWN_ROW_MUST_REMAIN_UNQUALIFIED')
            continue
        node = nodes[row['node_id']]; prefix = row['observation_prefix']; tensor = row['tensor']
        require(row['actual_rows'] == forward['row_mapping'] and row['forward_id'] == prefix['forward_id'] == forward['forward_id']
            and node['rows_sha256'] == digest(json.dumps(row['actual_rows'], sort_keys=True, allow_nan=False).encode())
            and node['row_axis'] == 0
            and node['metadata'] == tensor and row['observed_write_epoch'] == node['write_epoch']
            and not row['native_qualified'], 'OLD_REQUEST_ROW_PROOF_OBJECT_IDENTITY_CHANGED')
        require(prefix['missing_calls'] == prefix['errors'] == prefix['source_errors'] == 0
            and prefix['actual_calls'] == prefix['retained_calls']
            and ledger.at(prefix['actual_calls'], key(tensor), after=True) == row['observed_write_epoch'],
                'OLD_ROW_PROOF_PREFIX_NOT_CURRENT')
        require(call['float_input']['storage_addr'] == tensor['storage_address']
            and call['float_input']['shape'] == tensor['shape'] and call['float_input']['stride'] == tensor['stride'],
                'ROW_PROOF_ACTUAL_PROJECTION_INPUT_CHANGED')
        verified.append(projections[call['module_object_identity']]['path'])
    require(len(verified) == len(set(verified)) == inventory['request_row_proven_calls'] == 10
        and set(verified) == EXISTING_ROW_SITES,
            'EXISTING_TEN_PROJECTION_ROW_SCOPE_CHANGED')
    require(len(projections) == 135 and all(row['actual_calls'] == 1 and not row['native_qualified']
                                         for row in inventory['structural_candidates']), 'FULL_135_STRUCTURAL_DENOMINATOR_REQUIRED')
    return dict(total_projection_candidates=135, existing_row_proven=10, unproven_rows=125,
                existing_verified_paths=verified, new_row_authority=False)


def review(values, hashes, runtime_source, native_directory, native_sha, native_source):
    result = values[PREFIX + 'result.json']; inventory = values[PREFIX + 'shared-scale/inventory/shared_scale_inventory.json']
    serving = values[PREFIX + 'shared-scale/serving.json']; observation = values[PREFIX + 'container-trace/observation.json']
    cleanup = values[PREFIX + 'container-trace/cleanup-observation-001.json']
    controller = values['controller.json']; cases = values['reviewed_cases.json']
    require(len(cases) == len(controller['cases']) == len(result['phases']) == 1, 'ONE_BOUNDED_DECODE_ATTEMPT_REQUIRED')
    require(controller['case_manifest_sha256'] == hashes['reviewed_cases.json']['sha256']
        and controller['stale_allocation_scope_file_sha256'] == hashes['reviewed_stale_allocation_scope.json']['sha256'],
            'CONTROLLER_INPUT_ORIGINAL_HASH_JOIN')
    case, controlled = cases[0], controller['cases'][0]
    require(controlled['exit_code'] == controlled['container_state']['ExitCode'] == 0
        and not controlled['container_state']['OOMKilled'] and controlled['cleanup']['status'] == 'OWNED_CONTAINER_REMOVED'
        and controlled['review'] == case['review'], 'ACTUAL_CONTROLLER_CASE_NOT_COMPLETED')
    require(case['review']['expected_hits'] == 0 and case['review']['physical_cards'] == [5]
        and case['review']['trace_build_sha256'] == native_sha == observation['build_sha256'], 'EXACT_ZERO_F_CASE_BUILD_REQUIRED')
    require(len(inventory['native_forwards']) == len(inventory['forwards']) == 1
        and inventory['actual_projection_calls'] == inventory['actual_native_calls']
        == inventory['actual_native_prepare_calls'] == inventory['actual_native_execute_calls'] == 135
        and inventory['actual_public_api_calls'] == inventory['retained_public_api_calls'] == 270
        and inventory['missing_public_call_details'] == inventory['unowned_public_calls'] == 0
        and inventory['failure'] is None, 'COMPLETE_CURRENT_NATIVE_PUBLIC_INVENTORY_REQUIRED')
    native_denominator = review_native_denominator(inventory['native_forwards'][0], inventory['forwards'][0], inventory['public_calls'])
    source = Path(runtime_source)
    loaded = result['setup']['loaded_sources']
    require(len(loaded) == len({row['module'] for row in loaded}) == 17, 'COMPLETE_LOADED_SOURCE_SET_REQUIRED')
    for row in loaded:
        name = row['module']; require(re.fullmatch(r'[A-Za-z_][A-Za-z_0-9.]+', name), 'MODULE_NAME')
        path = source / ('src/' if name.startswith('demoncat.') else '') / (name.replace('.', '/') + '.py')
        require(digest(path.read_bytes()) == row['sha256'], 'ACTUAL_WORKER_SOURCE_BYTES:' + name)
    frozen = controlled['frozen_scripts']
    require(len(frozen) == 1 and frozen[0]['destination'] == '/probe.py'
        and frozen[0]['sha256'] == digest((source / 'experiments/worker_core_v11_20261118/probe_quant_containers.py').read_bytes()),
            'ACTUAL_FROZEN_PROBE_BYTES_CHANGED')
    build, profile, provenance = build_evidence(native_directory, native_sha, native_source)
    require(observation['profile_sha256'] == build['profile_sha256']
        and observation['mapping']['sha256'] == build['files'][build['library']]
        and observation['mapping']['path'] == '/container-trace/' + build['library'], 'ACTUAL_TRACE_MAPPING_BUILD_JOIN')
    call_summary = review_calls(observation, profile)
    windows = review_windows(observation, inventory['native_forwards'][0]['row_provenance'], inventory['forwards'][0], source)
    row_scope = review_row_scope(inventory)
    require(result['container_observation']['sha256'] == hashes[PREFIX + 'container-trace/observation.json']['sha256']
        and result['container_observation']['bytes'] == hashes[PREFIX + 'container-trace/observation.json']['bytes']
        and result['cleanup']['original']['sha256'] == hashes[PREFIX + 'container-trace/cleanup-observation-001.json']['sha256']
        and result['cleanup']['original']['bytes'] == hashes[PREFIX + 'container-trace/cleanup-observation-001.json']['bytes'],
            'IMMUTABLE_BEFORE_RELEASE_ORIGINALS_REQUIRED')
    require(observation['events'] == cleanup['events'] and observation['public_scopes'] == cleanup['public_scopes'],
            'CLEANUP_MUST_RETAIN_EXACT_CALL_AND_SCOPE_ORIGINALS')
    released = result['cleanup']['released']
    require(not result['cleanup']['errors'] and released['installed'] == 0 and released['stopped'] == released['retired'] == 1
        and not any(released[k] for k in ('retained_events', 'metadata_bytes', 'live_registrations', 'active_calls',
            'scope_token', 'record_capacity_bytes', 'payload_capacity_bytes', 'registration_count'))
        and released['arrays'] == observation['state']['arrays'], 'EXACT_OWNER_BUFFER_AND_GETTER_ARRAY_RELEASE_REQUIRED')
    require(all(released[k] == observation['state'][k] for k in ('owner_hi', 'owner_lo', 'generation', 'owner_thread',
                'total_calls', 'total_events', 'entries', 'returns', 'original_calls')), 'SAME_OWNER_RELEASE_RECEIPT_REQUIRED')
    require(result['identity']['worker'] == inventory['worker_identity'] == inventory['lifetime']['worker']
        and all(row['pid'] == result['identity']['worker']['pid'] for row in observation['events'])
        and not result['identity']['quarantined'], 'SAME_ACTUAL_WORKER_LIFETIME_REQUIRED')
    for key_name, value in serving.items():
        if key_name.startswith('owned_'):
            require(value == 0, 'OWNED_OBSERVER_RESOURCE_REMAINED:' + key_name)
    require(serving['installed'] is False and not serving['observer_cleanup_pending'] and serving['error'] is None
        and result['cleanup']['inventory'] == serving and result['healthy_before'] == result['healthy_after']
        and len(result['healthy_before']) == 8 and result['native_faults'] == serving['native_faults'] == 0,
            'SAME_WORKER_HEALTH_AND_CLEANUP_REQUIRED')
    require(not any(inventory[k] for k in ('qualified_native_targets', 'c1', 'c2', 'c3', 'group_owner_installed'))
        and not any(result[k] for k in ('c1', 'c2', 'c3', 'request_rows_proven'))
        and not observation['native_qualified'] and not observation['request_rows_proven']
        and not serving['complete_healthy_replays'] and not serving['recovered'], 'NO_NEW_ROW_OR_GROUP_QUALIFICATION')
    elapsed = (datetime.fromisoformat(controlled['ended_utc']) - datetime.fromisoformat(controlled['started_utc'])).total_seconds()
    return dict(schema='demoncat.quant.container_independent_review.v1',
        status='ORIGINALS_RECONCILED_CREATOR_OBSERVABILITY_ONLY_NOT_ROUTED_ROW_PROOF',
        originals=hashes, runtime_source_commit=case['review']['source_commit'], native_build=provenance,
        loaded_worker_sources_verified=len(loaded), native_projection_denominator=native_denominator,
        container_calls=call_summary, source_windows=windows, request_row_scope=row_scope,
        case_elapsed_seconds=elapsed, trace_elapsed_seconds=observation['state']['elapsed_ns'] / 1e9,
        scope_observer_elapsed_ns=observation['source_scopes']['elapsed_ns'],
        same_worker_health=True, owned_scoped_resources_released=True, resident_dso_not_unloaded=True,
        limitations=['No current GWS slot, API version, executor or stream observations in this diagnostic.',
            'Creator member snapshots cannot establish later GWS use or descriptor liveness.',
            'GMM1 windows have no four-container calls; no Native implementation inferred from that absence.',
            'Unpermute integer payload has no GWS parameter role; it is not interpreted as a route index.',
            'No integer route snapshots were collected in this attempt; old snapshots cannot authorize this Worker.'],
        native_qualified=False, routed_request_rows_proven=False, c1=False, c2=False, c3=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True); parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--native-build-root', type=Path, required=True); parser.add_argument('--native-build-sha256', required=True)
    parser.add_argument('--runtime-source-root', type=Path, default=ROOT)
    parser.add_argument('--native-source-root', type=Path, default=ROOT); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(); values, hashes = load_archive(args.archive, args.archive_sha256)
    summary = review(values, hashes, args.runtime_source_root, args.native_build_root,
                     args.native_build_sha256, args.native_source_root)
    summary['archive_sha256'] = args.archive_sha256
    raw = (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.out.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(status=summary['status'], summary_sha256=digest(raw),
                         native_qualified=False, routed_request_rows_proven=False)))


if __name__ == '__main__':
    main()
