"""Independent, stdlib-only WeightNz ORIGINALS reader; never a live admission.

No observer/Native binding is imported. Reported assessments and PASS strings
are not acceptance oracles. SDK compile bytes and runtime bytes are separate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tarfile

try:
    from . import review_quant_container_observations as four
except ImportError:
    import review_quant_container_observations as four

ROOT = Path(__file__).resolve().parents[2]
PREFIX = 'quant-weightnz-current-slots-one-decode/'
MEMBERS = tuple(name.replace(four.PREFIX, PREFIX) for name in four.MEMBERS) + (
    PREFIX + 'container-trace/startup-before-llm.json',)
PAIR = ('aclnnGroupedMatmulWeightNzGetWorkspaceSize', 'aclnnGroupedMatmulWeightNz')
FORMALS = ('x', 'weight', 'bias', 'scale', 'offset', 'antiquantScale', 'antiquantOffset',
    'perTokenScale', 'groupList', 'activationInput', 'activationQuantScale', 'activationQuantOffset',
    'splitItem', 'groupType', 'groupListType', 'actType', 'tuningConfig', 'quantGroupSize',
    'out', 'activationFeatureOut', 'dynQuantScaleOut', 'workspaceSize', 'executor')
SLOTS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 16, 18, 19, 20)
SCALARS = (12, 13, 14, 15, 17)
FUNCTIONS = {
    'prepare_function': ('opapi', PAIR[0], 0x1ba8bd0),
    'execute_function': ('opapi', PAIR[1], 0x1b8a730),
    'device_function': ('device_runtime', 'aclrtGetDevice', 42824),
    'list_size_function': ('nn', 'aclGetTensorListSize', 1089552),
    'int_size_function': ('nn', 'aclGetIntArraySize', 1089336),
}
CALLERS = {1: 0x175bc10, 2: 0x1747cb4}
PINS = dict(four.PINS, opapi='b3890625d3e0cc11ef32460083698ecaa8f702b5edd0169ea617e2e6bb76270c',
    device_runtime='bdfa66949fc798ff21c0124a025b4227e119678cc819febed9fe1c39f2a87306')
EXTENSION = dict(version=1, typed_pair=list(PAIR), max_calls=128, max_pairs=32, max_bytes=1024*1024)
G_FIELDS = ('sequence call prepare_registration owner_hi owner_lo generation scope_token forward '
    'operator_sequence boundary_sequence stage pid thread function function_base function_offset '
    'caller caller_base caller_offset original_calls container_events monotonic_ns error').split()
TENSOR_BYTES = 304
SLOT_BYTES = struct.calcsize('@II10QiI')
EVENT_BYTES = 16 + 8*len(G_FIELDS) + 16 + 23*8 + 5*8 + 15*SLOT_BYTES + TENSOR_BYTES + 4*8
NATIVE_CASES = ('original_four', 'normal', 'size_mismatch', 'size_status', 'getter_throw',
    'unknown_container', 'wrong_caller', 'changed_owner', 'device_error', 'wrong_device', 'foreign_thread',
    'workspace_short', 'failed_prepare', 'failed_prepare_null', 'prepare_throw', 'no_execute',
    'execute_error', 'execute_retry', 'execute_throw', 'call_budget', 'pair_budget', 'byte_budget',
    'executor_swapped', 'executor_reused', 'scope_changed', 'null_optional', 'release_order', 'active_scope_guard')
OLD_CASES = ('normal', 'payload', 'scope', 'foreign_thread', 'getter_status', 'getter_throw',
    'destroy_retry', 'unknown_destroy', 'null_destroy', 'failed_create', 'reused_address',
    'duplicate_live', 'call_budget', 'byte_budget', 'member_budget', 'owner', 'scope_retry', 'read_bounds', 'empty')
NONE_GRANTED = dict(native_qualified=False, routed_request_rows_proven=False, c1=False, c2=False, c3=False,
    e5=False, device_drain_proven=False)
digest, canonical, require = four.digest, four.canonical, four.require


def reviewer_sources():
    """Pin the four actual loaded review modules, never the runtime snapshot."""
    routes = sys.modules[four.review_native_denominator.__module__]
    arithmetic = sys.modules[routes.arithmetic.__module__]
    paths = [Path(__file__).resolve(), Path(four.__file__).resolve(),
             Path(routes.__file__).resolve(), Path(arithmetic.__file__).resolve()]
    require(len(set(paths)) == 4 and all(path.parent == paths[0].parent for path in paths),
            'ADJACENT_REVIEW_SOURCE_CLOSURE_REQUIRED')
    return {path.name: digest(path.read_bytes()) for path in paths}


def parsed(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'DUPLICATE_JSON_KEY:' + key)
            result[key] = value
        return result
    def bad(value):
        raise ValueError('NONFINITE_JSON:' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def integer(value, low=0, high=(1 << 64)-1):
    return type(value) is int and low <= value <= high


def integer_packet(value):
    if isinstance(value, dict):
        return all(integer_packet(v) for v in value.values())
    if isinstance(value, list):
        return all(integer_packet(v) for v in value)
    return integer(value, -(1 << 63))


def same(left, right, names, reason):
    require(all(type(left[name]) is type(right[name]) and left[name] == right[name] for name in names), reason)


def load_archive(path, expected):
    require(re.fullmatch('[a-f0-9]{64}', expected) is not None, 'EXPLICIT_PACKAGE_SHA_REQUIRED')
    raw = Path(path).read_bytes()
    require(len(raw) <= 128*1024*1024 and digest(raw) == expected, 'ORIGINAL_PACKAGE_CHANGED')
    values, hashes = {}, {}
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        # A failed run can lack late originals; absence is reported, never filled.
        require(len(members) <= len(MEMBERS) and sum(m.size for m in members) <= 256*1024*1024,
                'BOUNDED_ORIGINAL_PACKAGE_REQUIRED')
        for member in members:
            require(member.isfile() and member.name in MEMBERS and member.name not in values
                and not PurePosixPath(member.name).is_absolute() and '..' not in PurePosixPath(member.name).parts,
                    'REGULAR_UNIQUE_ALLOWED_ORIGINAL_REQUIRED')
            data = archive.extractfile(member).read()
            require(len(data) == member.size, 'COMPLETE_ORIGINAL_MEMBER_REQUIRED')
            values[member.name] = parsed(data)
            hashes[member.name] = dict(sha256=digest(data), bytes=len(data))
    return values, hashes


def source_file(root, name):
    root = Path(root).resolve(); path = (root / name).resolve(strict=True)
    require(path.is_relative_to(root), 'SOURCE_PATH_ESCAPES_ROOT')
    return path


def build_evidence(directory, expected, source_root):
    directory = Path(directory)
    def read(name, sha=None):
        path = source_file(directory, name); raw = path.read_bytes()
        if sha is not None:
            require(digest(raw) == sha, 'BUILD_ORIGINAL_CHANGED:' + name)
        return parsed(raw)
    build = read('build.json', expected)
    receipt = read('build-receipt.json', build['build_receipt_sha256'])
    require(build['schema'] == 'demoncat.quant.container_deployment.v1'
        and build['image_id'] == receipt['image_id'] == four.IMAGE
        and all(build[key] is False for key in ('qualified', 'c1', 'c2', 'c3'))
        and build['actual_npu_calls'] == receipt['actual_npu_calls'] == 0
        and receipt['status'] == 'SDK_AND_GMM2_TYPED_CPU_CONTRACT_PASS_NOT_NPU'
        and receipt['returncode'] == 0 and receipt['cleanup']['confirmed'] is True,
            'CPU_ONLY_COMPLETE_TYPED_BUILD_REQUIRED')
    require(build['sources'] == receipt['sources'] and build['source_commit'] == receipt['source_commit'],
            'COMPILE_SOURCE_IDENTITY_CHANGED')
    required = {
        'fault_injection/native_adapters/quant_container_trace.h',
        'fault_injection/native_adapters/quant_container_trace.cpp',
        'fault_injection/native_adapters/test_quant_container_trace.cpp',
        'fault39/native/runtime_identity.h', 'fault39/native/metadata_array_owner.h',
        'fault39/native/metadata_array_state.h', 'fault39/tools/version_profile.py',
        'experiments/worker_core_v11_20261118/build_quant_container_trace.py',
        'experiments/c2c3_closure_20261117/providers_quant_build_group.py',
        'fault_injection/native_adapters/quant_gmm2_trace.h',
        'fault_injection/native_adapters/quant_gmm2_trace.inc',
        'fault_injection/native_adapters/test_quant_gmm2_trace.cpp',
        'experiments/worker_core_v11_20261118/build_quant_gmm2_trace.py'}
    require(set(build['sources']) == required, 'EXACT_THIRTEEN_NATIVE_SOURCE_DEPENDENCIES_REQUIRED')
    for name, sha in build['sources'].items():
        require(digest(source_file(source_root, name).read_bytes()) == sha, 'COMPILE_SOURCE_BYTES:' + name)
    profile = read('profile.json', build['files']['profile.json'])
    unsigned = dict(profile); unsigned.pop('profile_sha256')
    require(digest(canonical(unsigned)) == profile['profile_sha256'] == build['profile_sha256']
        and profile['sources'] == build['sources'] and profile['image'] == four.IMAGE
        and profile['functions'] == list(four.FUNCTIONS + PAIR)
        and profile['gmm2_extension'] == build['gmm2_extension'] == EXTENSION,
            'EXACT_TYPED_BUILD_PROFILE_REQUIRED')
    require(all(profile['images'][name]['file_sha256'] == sha for name, sha in PINS.items()),
            'FIXED_ORIGINAL_FUNCTION_OR_CALLER_BYTES_CHANGED')
    require(profile['sdk_headers']['include/aclnnop/aclnn_grouped_matmul_weight_nz.h']
        == 'f41308d5814aa82ba92ff77aa8e258b15ca3ae72d2c0c61fadc2cdd95de6d7fd'
        and profile['sdk_headers']['include/acl/acl_rt.h']
        == '8cce153ef9eb5ed49b0f5bfb5608de706e8d466c836ad2eb50745ec2e8629bc9', 'ACTUAL_SDK_HEADERS_CHANGED')
    metadata = {}
    for name in ('cpu-contract.json', 'gmm2-cpu-contract.json', 'sdk-load.json', 'gmm2-startup.json',
                 'driver-identity.json', 'absolute-needed-contract.json', 'image-inspect.json'):
        metadata[name] = read(name, receipt['artifacts'][name])
    logs, executables = 0, []
    for name, cases, prefix, executable in (
        ('cpu-contract', OLD_CASES, 'PASS_CPU_SDK_TYPED_FUNCTION_DOUBLE_NOT_NPU ', 'cpu_contract'),
        ('gmm2-cpu-contract', NATIVE_CASES, 'PASS_CPU_SDK_GMM2_TYPED_FUNCTION_DOUBLE_NOT_NPU ', 'gmm2_cpu_contract')):
        cpu = metadata[name + '.json']
        require(cpu['expected_cases'] == list(cases) and [r['case'] for r in cpu['cases']] == list(cases)
            and all(r['returncode'] == 0 and r['status'] == 'PASS' for r in cpu['cases'])
            and cpu['sources'] == build['sources'], 'COMPLETE_TYPED_CPU_CASE_RECORDS:' + name)
        for row in cpu['cases']:
            path = directory / row['log']
            if path.exists():
                raw = source_file(directory, row['log']).read_bytes()
                require(digest(raw) == row['log_sha256'] and raw == (prefix + row['case'] + '\n').encode(),
                        'AVAILABLE_CPU_LOG_CHANGED')
                logs += 1
        if (directory / executable).exists():
            require(digest((directory / executable).read_bytes()) == cpu['executable_sha256'], 'CPU_EXECUTABLE_CHANGED')
            executables.append(executable)
    for name in ('cpu-contract.json', 'gmm2-cpu-contract.json', 'sdk-load.json', 'gmm2-startup.json'):
        require(receipt['stage_receipts'][name] == dict(status=metadata[name]['status'], sha256=receipt['artifacts'][name]),
                'BUILD_STAGE_HASH_JOIN')
    load, drivers, startup = (metadata[n] for n in ('sdk-load.json', 'driver-identity.json', 'gmm2-startup.json'))
    require(load['profile_sha256'] == profile['profile_sha256']
        and load['driver_identity_after'] == drivers
        and load['driver_identity_sha256'] == receipt['driver_identity_sha256'] == receipt['artifacts']['driver-identity.json']
        and startup['status'] == 'GMM2_ORIGINALS_RESOLVED_WITHOUT_DEVICE_OR_OPERATOR_CALLS'
        and all(startup[n] == 0 for n in ('operator_calls', 'device_query_calls', 'install_calls', 'actual_npu_calls'))
        and startup['loader_constructor_driver_calls'] == 'NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION',
            'CPU_STARTUP_SCOPE_OR_DRIVER_IDENTITY_CHANGED')
    by_path = {r['path']: r for r in drivers['files'].values()}
    require(len(by_path) == 3 and {r['path'] for r in load['loaded_driver_mappings']} == set(by_path)
        and all(r['mapped_inode'] == by_path[r['path']]['inode'] and r['sha256'] == by_path[r['path']]['sha256']
                for r in load['loaded_driver_mappings']), 'EXACT_READONLY_DRIVER_MAPPINGS_REQUIRED')
    elf = metadata['absolute-needed-contract.json']
    require(elf['original_sha256_before'] == elf['original_sha256_after']
        and elf['status'] == 'PASS_REAL_ELF_ABSOLUTE_NEEDED_SAME_SONAME'
        and metadata['image-inspect.json']['Id'] == four.IMAGE, 'UNCHANGED_ORIGINAL_CPU_ELF_REQUIRED')
    available = []
    for name, sha in build['files'].items():
        if (directory / name).exists():
            require(digest(source_file(directory, name).read_bytes()) == sha, 'AVAILABLE_BUILD_FILE_CHANGED:' + name)
            available.append(name)
    return build, profile, dict(build_sha256=expected, compile_source_commit=build['source_commit'],
        source_files_rehashed=len(required), cpu_case_records=47, cpu_case_logs_rehashed=logs,
        cpu_executables_rehashed=executables, actual_dso_bytes_rehashed=build['library'] in available,
        actual_dso_sha256=build['files'][build['library']], scope='SDK_CPU_BUILD_NOT_NPU_ACCEPTANCE')


def descriptor(value):
    require(integer(value['descriptor'], 1) and integer(value['address'], 1)
        and value['complete'] == 1 and value['called_mask'] == 127
        and value['null_mask'] == value['truncated_mask'] == 0 and value['statuses'] == [0]*7
        and len(value['counts']) == 3 and all(integer(n, 1, 8) for n in value['counts'])
        and all(len(value[k]) == 8 and all(integer(n, -(1 << 63), (1 << 63)-1) for n in value[k])
                for k in ('shape', 'strides', 'storage')), 'COMPLETE_TYPED_DESCRIPTOR_REQUIRED')


def review_containers(observation, profile, group_descriptors=()):
    """Replay actual four-API lifetimes, including extra GWS metadata captures."""
    events, state = observation['events'], observation['state']
    require(integer_packet(state), 'FOUR_API_STATE_NOT_NATIVE_INTEGER_FIELDS')
    require(len(events) == state['retained_events'] == state['total_events'] == 2*state['total_calls'],
            'COMPLETE_FOUR_API_EVENT_DENOMINATOR')
    require(state['installed'] == state['stopped'] == state['resolver_ready'] == 1 and not any(state[k] for k in (
        'active_calls', 'scope_token', 'live_registrations', 'registration_count', 'first_error', 'abi_exceptions',
        'getter_capture_failures', 'overflow', 'foreign_scope_thread_calls')), 'FOUR_API_NOT_COMPLETE_IDLE')
    live, entries, returns, histories, functions = {}, {}, {}, Counter(), {}
    counts, members, offset, outside, previous = Counter(), 0, 0, 0, 0
    for sequence, row in enumerate(events, 1):
        require(integer_packet(row), 'FOUR_API_EVENT_NOT_NATIVE_INTEGER_FIELDS')
        op, phase, call = row['operation'], row['phase'], row['call']
        require(row['sequence'] == sequence and integer(call, 1) and type(op) is type(phase) is int
            and op in (1, 2, 3, 4) and phase in (1, 2) and row['depth'] == 1
            and row['thread'] == state['owner_thread'] and integer(row['pid'], 1)
            and row['status'] == row['error'] == row['reserved'] == 0, 'FOUR_API_EVENT_FAILED_OR_MALFORMED')
        require(row['monotonic_ns'] >= previous, 'FOUR_API_CLOCK_ORDER'); previous = row['monotonic_ns']
        require(row['function'] == row['function_base'] + row['function_offset']
            and row['caller'] == row['caller_base'] + row['caller_offset']
            and four.executable_offset(row['function_offset'], profile['images']['nn'])
            and row['caller_kind'] in (1, 2) and four.executable_offset(row['caller_offset'],
                profile['images']['torch_npu' if row['caller_kind'] == 1 else 'custom_extension']), 'FOUR_API_FUNCTION_OWNER')
        function = (row['function'], row['function_base'], row['function_offset'])
        require(functions.setdefault(op, function) == function, 'FOUR_API_FUNCTION_CHANGED')
        if phase == 1:
            require(call == len(entries)+1 and row['original_calls'] == 0, 'FOUR_API_ENTRY_ORDER')
            entries[call] = row; counts[op] += 1; outside += int(row['scope_token'] == 0)
            require(integer(row['count'], 0, 128), 'CONTAINER_MEMBER_BOUND')
            if op in (1, 3):
                require(row['handle'] == row['registration'] == 0, 'CREATOR_HANDLE_ALREADY_ASSIGNED')
                values = row.get('tensor_descriptors', []) if op == 1 else row.get('integer_values', [])
                require(len(values) == row['count'] and row['payload_bytes'] == len(values)*(304 if op == 1 else 8),
                        'CREATOR_PAYLOAD_INCOMPLETE')
                if values:
                    require(row['payload_offset'] == offset, 'CREATOR_PAYLOAD_GAP_OR_OVERLAP'); offset += row['payload_bytes']
                if op == 1:
                    for item in values:
                        descriptor(item)
                    members += len(values)
                else:
                    require(all(integer(n, -(1 << 63), (1 << 63)-1) for n in values), 'HOST_INTEGER_PAYLOAD')
            else:
                require(row['count'] == row['payload_bytes'] == 0, 'DESTROY_MUST_NOT_READ_MEMBERS')
                if row['handle']:
                    created = live.get(row['handle'])
                    require(created is not None and created['registration'] == row['registration']
                        and created['operation']+1 == op, 'DESTROY_OF_NONCURRENT_REGISTRATION')
        else:
            require(call in entries and call not in returns and row['original_calls'] == 1
                and row['payload_bytes'] == 0, 'FOUR_API_EXACT_ORIGINAL_RETURN')
            entry = entries[call]; returns[call] = row
            same(row, entry, ('operation', 'count', 'pid', 'thread', 'depth', 'scope_token', 'forward',
                'operator_sequence', 'boundary_sequence', 'stage', 'caller', 'caller_kind', 'caller_base',
                'caller_offset', 'function', 'function_base', 'function_offset'), 'FOUR_API_RETURN_IDENTITY')
            if op in (1, 3):
                require(integer(row['handle'], 1) and row['registration'] == call and row['handle'] not in live,
                        'FRESH_CREATOR_REGISTRATION')
                live[row['handle']] = row; histories[row['handle']] += 1
            elif row['handle']:
                created = live.pop(row['handle'], None)
                require(created is not None and entry['handle'] == row['handle'] and created['registration']
                    == entry['registration'] == row['registration'] and created['scope_token'] == row['scope_token'],
                        'DESTROY_LIFETIME_CHANGED')
            else:
                require(entry['handle'] == entry['registration'] == row['registration'] == 0, 'NULL_DESTROY_HAS_REGISTRATION')
    require(not live and len(entries) == len(returns) == state['total_calls']
        and [counts[n] for n in (1, 2, 3, 4)] == state['entries'] == state['returns'] == state['original_calls']
        and outside == state['outside_scope_calls'] and offset == state['metadata_bytes'], 'FOUR_API_COUNTER_RECOUNT')
    for value in group_descriptors:
        descriptor(value)
    captures = members + len(group_descriptors); arrays = state['arrays']
    require(arrays['capture_calls'] == captures and arrays['getter_calls'] == arrays['acquired_arrays']
        == arrays['released_arrays'] == captures*3 and not any(arrays[k] for k in ('active_captures', 'active_getters',
        'live_arrays', 'getter_failures', 'getter_exceptions', 'capture_exceptions', 'ownership_errors')),
            'CREATOR_PLUS_GWS_GETTER_ARRAY_ACCOUNTING')
    require(len(entries) <= 4096 and len(events) <= 8192 and offset <= 8*1024*1024, 'FOUR_API_BUDGET')
    return dict(actual_calls=len(entries), events=len(events), creator_descriptor_members=members,
        current_group_list_captures=len(group_descriptors), metadata_array_captures=captures,
        arrays_acquired=arrays['acquired_arrays'], arrays_released=arrays['released_arrays'],
        repeated_handle_registrations=sum(n-1 for n in histories.values()), pending_lifetimes=0)


def review_startup(startup, observation, result, inventory, build, profile, native_sha):
    g = observation['gmm2']; process = startup['identity']; worker = result['identity']['worker']
    require(startup == g['startup'] == result['pre_llm_startup'] == result['setup']['pre_llm_startup'],
            'SAME_PRE_LLM_STARTUP_ORIGINAL')
    require(startup['schema'] == 'demoncat.quant.gmm2_startup.v1'
        and startup['native_build_sha256'] == native_sha and startup['native_profile_sha256'] == build['profile_sha256']
        and startup['native_compile_source_commit'] == build['source_commit'], 'STARTUP_COMPILE_BUILD_JOIN')
    require(integer(process['pid'], 1) and integer(process['thread'], 1)
        and re.fullmatch(r'proc:[a-fA-F0-9-]{36}:' + str(process['pid']) + r':[0-9]+', process['process_start_identity'])
        and worker['pid'] == process['pid'] and worker['process_start_identity'] == process['process_start_identity']
        and integer(worker['logical_device']) and worker['logical_device'] == worker['configured_logical_device']
        == worker['observed_logical_device'] and worker['rank'] == 0
        and worker['worker_start_identity'] and worker['model_instance_id'], 'SAME_REAL_PROCESS_WORKER_DEVICE')
    require(result['identity'] == result['worker_before_requests'] == result['setup']['worker_binding'] == g['worker_identity']
        and worker == inventory['worker_identity'] == inventory['lifetime']['worker']
        and not result['identity']['quarantined'], 'SAME_WORKER_BEFORE_AND_AFTER_SETUP')
    require(all(startup[k] == 0 for k in ('operator_calls', 'explicit_device_query_calls', 'install_calls'))
        and startup['loader_constructor_driver_calls'] == 'NOT_COUNTED_MAY_PROBE_DEVICE_INFORMATION',
            'PRE_LLM_STARTUP_MUST_NOT_EXECUTE_OPERATOR')
    for phase in ('state_before', 'state_after'):
        for journal in ('containers', 'gmm2'):
            state = startup[phase][journal]
            require(not any(state[k] for k in ('installed', 'retired', 'total_calls', 'first_error', 'abi_exceptions',
                'active_calls', 'record_capacity_bytes')), 'STARTUP_JOURNAL_NOT_FRESH')
    require(startup['state_after']['gmm2']['resolver_ready'] == 1, 'STARTUP_RESOLVER_NOT_READY')
    owners = startup['original_functions']
    require(set(owners) == set(FUNCTIONS), 'EXACT_FIVE_ORIGINAL_FUNCTIONS_REQUIRED')
    for name, (image_name, symbol, offset) in FUNCTIONS.items():
        owner = owners[name]; image = profile['images'][image_name]
        require(owner['symbol'] == symbol and owner['offset'] == offset
            and integer(owner['base'], 1) and owner['address'] == owner['base'] + offset
            and owner['canonical_path'] == image['path'] and owner['file_bytes'] == image['file_bytes']
            and owner['file_sha256'] == image['file_sha256'] == PINS[image_name]
            and four.executable_offset(offset, image)
            and startup['state_after']['gmm2'][name] == g['state'][name] == owner['address'], 'STARTUP_EXACT_FUNCTION:' + name)
    require(owners['prepare_function']['base'] == owners['execute_function']['base']
        and owners['list_size_function']['base'] == owners['int_size_function']['base'], 'STARTUP_SPLIT_IMAGE_MAPPING')
    return dict(same_process=True, same_worker=True, resolved_original_functions=5, no_explicit_startup_operator_calls=True)


def review_typed(observation, profile, expected_pairs=26):
    g, raw = observation['gmm2'], observation['events']; events, state = g['events'], g['state']
    startup = g['startup']; process = startup['identity']; worker = g['worker_identity']['worker']
    require(integer_packet(state), 'TYPED_STATE_NOT_NATIVE_INTEGER_FIELDS')
    require(g['schema'] == 'demoncat.quant.gmm2_trace.v1' and g['formal_names'] == list(FORMALS), 'TYPED_ABI_SCHEMA')
    require(not g['failures'] and state['installed'] == state['stopped'] == state['resolver_ready'] == 1
        and not any(state[k] for k in ('active_calls', 'pending_executors', 'untracked_executors', 'overflow',
        'first_error', 'abi_exceptions', 'outside_scope_calls', 'foreign_thread_calls', 'getter_failures', 'failed_with_executor')),
            'TYPED_NOT_COMPLETE_IDLE')
    require(len(events) == state['retained_events'] == state['total_events'] == expected_pairs*4
        and state['total_calls'] == expected_pairs*2 and all(state[k] == expected_pairs for k in (
        'prepares', 'prepare_returns', 'executes', 'execute_returns', 'original_prepares', 'original_executes',
        'paired_execute_returns')), 'TYPED_EVENT_DENOMINATOR')
    require(integer(state['record_capacity_bytes'], len(events)*EVENT_BYTES, 1024*1024), 'BOUNDED_TYPED_JOURNAL')
    same(state, observation['state'], ('owner_hi', 'owner_lo', 'generation'), 'JOURNAL_OWNER_CHANGED')
    require(state['thread'] == process['thread'] == observation['state']['owner_thread'], 'JOURNAL_THREAD_CHANGED')
    scopes = {s['native_scope']['token']: s for s in observation['public_scopes']}
    require(len(scopes) == len(observation['public_scopes']), 'DUPLICATE_PUBLIC_SCOPE')
    wanted = {n for n, s in scopes.items() if s['native_scope']['stage'] == 3}
    require(len(wanted) == expected_pairs, 'GMM2_WINDOW_DENOMINATOR')
    by_sequence = {r['sequence']: r for r in raw}; destroys = {}
    for row in raw:
        if row['phase'] == 1 and row['operation'] in (2, 4) and row['handle']:
            destroys.setdefault(row['registration'], []).append(row)
    bases = {r['caller_base'] for r in raw if r['caller_kind'] == 1}
    require(len(bases) == 1, 'SINGLE_PINNED_TORCH_CALLER_MAPPING')
    require({r['function_base'] for r in raw} == {startup['original_functions']['list_size_function']['base']},
            'CREATOR_AND_SIZE_GETTERS_DIFFERENT_IMAGE_MAPPING')
    calls, returned, prepare, executions, pending, used_scopes = {}, {}, {}, {}, {}, Counter()
    groups, joined_slots, null_slots, quant_groups, streams, scalar_values = [], 0, 0, Counter(), Counter(), Counter()
    previous = 0
    for sequence, e in enumerate(events, 1):
        require(integer_packet(e), 'TYPED_EVENT_NOT_NATIVE_INTEGER_FIELDS')
        require(e['version'] == 1 and e['size'] == EVENT_BYTES and e['sequence'] == sequence
            and all(integer(e[k]) for k in G_FIELDS) and e['operation'] in (1, 2) and e['phase'] in (1, 2)
            and e['reserved'] == e['status'] == e['error'] == e['device_status'] == 0
            and type(e['device']) is int and e['device'] == worker['logical_device'], 'TYPED_EVENT_FAILED_OR_MALFORMED')
        require(e['monotonic_ns'] >= previous, 'TYPED_CLOCK_ORDER'); previous = e['monotonic_ns']
        require(e['pid'] == process['pid'] and e['thread'] == process['thread'], 'TYPED_PROCESS_THREAD_CHANGED')
        same(e, state, ('owner_hi', 'owner_lo', 'generation'), 'TYPED_OWNER_CHANGED')
        original = startup['original_functions']['prepare_function' if e['operation'] == 1 else 'execute_function']
        require((e['function'], e['function_base'], e['function_offset']) ==
            (original['address'], original['base'], original['offset'])
            and e['caller_base'] in bases and e['caller_offset'] == CALLERS[e['operation']]
            and e['caller'] == e['caller_base'] + e['caller_offset']
            and four.executable_offset(e['caller_offset'], profile['images']['torch_npu']), 'TYPED_OWNER_OR_CALLER_CHANGED')
        require(e['scope_token'] in wanted, 'TYPED_CURRENT_SCOPE_MISSING')
        same(e, scopes[e['scope_token']]['native_scope'], ('forward', 'operator_sequence', 'boundary_sequence', 'stage', 'thread'),
             'TYPED_SCOPE_CHANGED')
        count = e['container_events']
        require(integer(count, 0, len(raw)) and (not count or raw[count-1]['monotonic_ns'] <= e['monotonic_ns'])
            and (count == len(raw) or e['monotonic_ns'] <= raw[count]['monotonic_ns']), 'CROSS_JOURNAL_TIME_JOIN')
        call = e['call']
        if e['phase'] == 1:
            require(call == len(calls)+1 and e['original_calls'] == 0, 'TYPED_ENTRY_ORDER')
            calls[call] = e
            if e['operation'] == 2:
                registration = e['prepare_registration']; previous_prepare = pending.get(e['executor'])
                require(previous_prepare is not None and previous_prepare['call'] == registration
                    and registration not in executions, 'CURRENT_EXECUTOR_REGISTRATION_MISSING_OR_REUSED')
                executions[registration] = e
                same(e, previous_prepare, ('scope_token', 'pid', 'thread', 'device', 'forward', 'operator_sequence',
                    'boundary_sequence', 'stage'), 'EXECUTOR_SCOPE_CHANGED')
                require(e['workspace_bytes'] >= previous_prepare['workspace_bytes']
                    and (previous_prepare['workspace_bytes'] == 0 or integer(e['workspace'], 1))
                    and integer(e['stream']), 'EXECUTOR_WORKSPACE_OR_STREAM')
                streams[e['stream']] += 1
            continue
        require(call in calls and call not in returned and e['original_calls'] == 1, 'TYPED_ORIGINAL_RETURN_COUNT')
        entry = calls[call]; returned[call] = e
        same(e, entry, ('call', 'operation', 'prepare_registration', 'owner_hi', 'owner_lo', 'generation', 'scope_token',
            'pid', 'thread', 'device', 'function', 'function_base', 'function_offset', 'caller', 'caller_base', 'caller_offset',
            'forward', 'operator_sequence', 'boundary_sequence', 'stage'), 'TYPED_RETURN_IDENTITY')
        if e['operation'] == 2:
            same(e, entry, ('workspace', 'workspace_bytes', 'executor', 'stream'), 'EXECUTE_RETURN_ARGUMENTS_CHANGED')
            require(e['executor'] in pending, 'EXECUTOR_ALREADY_RETIRED'); pending.pop(e['executor'])
            continue
        require(entry['prepare_registration'] == call and call not in prepare and integer(e['executor'], 1)
            and e['executor'] not in pending, 'FRESH_PREPARE_EXECUTOR_REQUIRED')
        prepare[call] = (entry, e); pending[e['executor']] = e; used_scopes[e['scope_token']] += 1
        require(entry['formals'] == e['formals'] and len(entry['formals']) == 23
            and all(integer(n) for n in entry['formals']) and entry['scalars'] == e['scalars']
            and len(entry['scalars']) == 5 and all(integer(n, -(1 << 63), (1 << 63)-1) for n in entry['scalars'])
            and all((entry['scalars'][i] & ((1 << 64)-1)) == entry['formals'][n] for i, n in enumerate(SCALARS))
            and all(entry['formals'][n] > 0 for n in (21, 22)), 'ACTUAL_23_TYPED_FORMALS_REQUIRED')
        quant_groups[entry['scalars'][4]] += 1
        scalar_values[tuple(entry['scalars'])] += 1
        require([s['formal_index'] for s in entry['containers']] == list(SLOTS)
            and [s['formal_index'] for s in e['containers']] == list(SLOTS), 'EXACT_NAMED_SLOT_ORDER')
        named = {}
        for slot, end in zip(entry['containers'], e['containers']):
            index = slot['formal_index']; kind = 3 if index == 16 else 1
            same(slot, end, tuple(k for k in slot if k != 'live_at_return'), 'SLOT_CHANGED_AT_RETURN')
            require(slot['handle'] == entry['formals'][index] and slot['kind'] == kind
                and slot['joined'] == end['joined'] == end['live_at_return'] == 1
                and slot['reserved'] == slot['size_status'] == 0 and slot['actual_count'] == slot['count'], 'SLOT_TYPED_JOIN_FAILED')
            if not slot['handle']:
                require(not any(slot[k] for k in ('registration', 'creator_entry', 'creator_return', 'count',
                    'actual_count', 'payload_offset', 'payload_bytes')), 'NULL_SLOT_INVENTED_LIFETIME')
                null_slots += 1; named[index] = []; continue
            first, last = by_sequence.get(slot['creator_entry']), by_sequence.get(slot['creator_return'])
            death = destroys.get(slot['registration'], [])
            require(first is not None and last is not None and len(death) == 1, 'MISSING_CREATOR_DESTROY_ORIGINAL')
            require(first['phase'] == 1 and last['phase'] == 2 and first['operation'] == last['operation'] == kind
                and first['call'] == last['call'] == last['registration'] == slot['registration']
                and last['handle'] == death[0]['handle'] == slot['handle']
                and first['sequence'] < last['sequence'] <= entry['container_events']
                <= e['container_events'] < death[0]['sequence'], 'CONTAINER_NOT_LIVE_AT_CURRENT_GWS')
            for source in (first, last):
                same(source, entry, ('scope_token', 'pid', 'thread', 'forward', 'operator_sequence', 'boundary_sequence', 'stage'),
                     'CREATOR_SCOPE_CHANGED')
            same(first, slot, ('count', 'payload_offset', 'payload_bytes'), 'CREATOR_PAYLOAD_CHANGED')
            content = first.get('tensor_descriptors', []) if kind == 1 else first.get('integer_values', [])
            require(len(content) == slot['count'] and slot['payload_bytes'] == len(content)*(304 if kind == 1 else 8),
                    'SLOT_PAYLOAD_INCOMPLETE')
            named[index] = content; joined_slots += 1
        require(entry['group_list'] == e['group_list'] and entry['group_list']['descriptor'] == entry['formals'][8],
                'CURRENT_GROUP_LIST_IDENTITY')
        if entry['formals'][8]:
            descriptor(entry['group_list']); groups.append(entry['group_list'])
        scope = scopes[entry['scope_token']]['evidence']; formals = scope['formals']
        require(entry['scalars'][:4] == [formals[n] for n in ('split_item', 'group_type', 'group_list_type', 'act_type')],
                'ACTUAL_SCALARS_PUBLIC_JOIN')
        for index, name in {0:'x', 1:'weight', 2:'bias', 3:'scale', 4:'offset', 5:'antiquant_scale',
            6:'antiquant_offset', 7:'per_token_scale', 9:'activation_input', 10:'activation_quant_scale',
            11:'activation_quant_offset'}.items():
            tensors = four.tensor_packets(formals[name])
            require(len(named[index]) == len(tensors) and all(four.descriptor_matches_public(d, t)
                for d, t in zip(named[index], tensors)), 'NAMED_GWS_PUBLIC_TENSOR:' + name)
        require(named[16] == (formals['tuning_config'] or []), 'ACTUAL_TUNING_CONFIG_JOIN')
        public_group = four.tensor_packets(formals['group_list'])
        require(len(public_group) == int(bool(entry['formals'][8])) and (not public_group
            or four.descriptor_matches_public(entry['group_list'], public_group[0])), 'GROUP_LIST_PUBLIC_TENSOR_JOIN')
        require(not named[19] and not named[20], 'THIS_PUBLIC_BRANCH_OPTIONAL_OUTPUTS_CHANGED')
    require(not pending and len(calls) == len(returned) == expected_pairs*2
        and len(prepare) == len(executions) == expected_pairs and set(prepare) == set(executions)
        and used_scopes == Counter({token: 1 for token in wanted}), 'ALL_CURRENT_PREPARE_EXECUTE_PAIRS_REQUIRED')
    return dict(actual_pairs=len(prepare), actual_calls=len(calls), actual_events=len(events),
        joined_nonnull_slots=joined_slots, explicit_null_slots=null_slots, group_list_captures=len(groups),
        signed_quant_group_sizes={str(k): v for k, v in sorted(quant_groups.items())}, distinct_streams=len(streams),
        actual_signed_scalar_tuples=[dict(values=list(k), calls=v) for k,v in sorted(scalar_values.items())],
        pending_executors=0, row_authority=False), groups


def review_named_outputs(observation, graph):
    dispatch = {r['sequence']: r for r in graph['events']}
    raw = {r['sequence']: r for r in observation['events']}; count = 0
    for event in observation['gmm2']['events']:
        if event['operation'] != 1 or event['phase'] != 1:
            continue
        slot = event['containers'][SLOTS.index(18)]
        require(slot['handle'] > 0, 'ACTUAL_GWS_OUTPUT_LIST_REQUIRED')
        members = raw[slot['creator_entry']]['tensor_descriptors']
        actual = dispatch[event['operator_sequence']]['outputs']
        require(len(members) == len(actual) == 1 and four.descriptor_matches_public(members[0], actual[0]),
                'NAMED_GWS_OUTPUT_ACTUAL_PUBLIC_RETURN_JOIN')
        count += 1
    return dict(actual_named_output_returns=count, no_numerical_or_row_claim=True)


def review_cleanup(result, observation, cleanup, serving):
    # Raw events, names, startup and lifetime bytes must be unchanged at cleanup.
    for name in ('events', 'public_scopes', 'mapping', 'build_sha256', 'profile_sha256'):
        require(observation[name] == cleanup[name], 'CLEANUP_ORIGINAL_CHANGED:' + name)
    same(observation['state'], cleanup['state'], tuple(k for k in observation['state'] if k != 'elapsed_ns'),
         'CONTAINER_STATE_CHANGED_BEFORE_RELEASE')
    for name in ('events', 'formal_names', 'startup', 'worker_identity', 'failures', 'original_execute_return_is_not_device_drain'):
        require(observation['gmm2'][name] == cleanup['gmm2'][name], 'CLEANUP_TYPED_ORIGINAL_CHANGED:' + name)
    same(observation['gmm2']['state'], cleanup['gmm2']['state'],
         tuple(k for k in observation['gmm2']['state'] if k != 'elapsed_ns'), 'TYPED_STATE_CHANGED_BEFORE_RELEASE')
    receipt = result['cleanup']; released = receipt['released']; c, g = released['containers'], released['gmm2']
    require(integer_packet(c) and integer_packet(g), 'RELEASE_STATES_MUST_BE_NATIVE_INTEGER_FIELDS')
    require(receipt['errors'] == receipt['failures'] == [] and receipt['first_failure'] is None,
            'OWNED_CLEANUP_OR_FIRST_FAILURE_REMAINS')
    require(released['resource_scope'] == 'OBSERVER_HOST_JOURNALS_ONLY_NO_DEVICE_BUFFERS'
        and released['device_drain_proven'] is False and released['preloaded_library_remains_resident'] is True,
            'RELEASE_SCOPE_MUST_NOT_CLAIM_DEVICE_DRAIN_OR_DSO_UNLOAD')
    for state, empty in ((c, ('retained_events', 'metadata_bytes', 'live_registrations', 'active_calls', 'scope_token',
        'record_capacity_bytes', 'payload_capacity_bytes', 'registration_count')), (g, ('retained_events', 'active_calls',
        'pending_executors', 'untracked_executors', 'record_capacity_bytes'))):
        require(state['installed'] == 0 and state['retired'] == state['stopped'] == 1
            and not any(state[k] for k in (*empty, 'first_error', 'abi_exceptions', 'overflow')),
                'OWNED_JOURNAL_RESOURCES_NOT_ZERO')
    same(c, observation['state'], ('owner_hi', 'owner_lo', 'generation', 'owner_thread', 'total_calls', 'total_events',
        'entries', 'returns', 'original_calls', 'arrays'), 'CONTAINER_RELEASE_OWNER_COUNTERS_CHANGED')
    same(g, observation['gmm2']['state'], ('owner_hi', 'owner_lo', 'generation', 'thread', 'total_calls', 'total_events',
        'prepares', 'prepare_returns', 'executes', 'execute_returns', 'original_prepares', 'original_executes',
        'paired_execute_returns', *FUNCTIONS), 'TYPED_RELEASE_OWNER_COUNTERS_CHANGED')
    require(receipt['inventory'] == serving and serving['installed'] is False
        and not serving['observer_cleanup_pending'] and serving['error'] is None
        and all(value == 0 for key, value in serving.items() if key.startswith('owned_')),
            'OWNED_INVENTORY_HOOKS_NOT_REMOVED')
    require(len(result['healthy_before']) == len(result['healthy_after']) == 8
        and result['healthy_before'] == result['healthy_after'] == result['phases'][0]['output_tokens']
        and all(len(tokens) == 2 and all(integer(token) for token in tokens) for tokens in result['healthy_before']),
            'SAME_WORKER_TWO_TOKEN_HEALTH_REQUIRED')
    return dict(originals_unchanged=True, host_journals_released=True, same_worker_healthy_tokens=True,
        captured_tokens_equal_healthy_before_and_after=True, token_requests=8, tokens_per_request=2,
        text_or_logits_comparison=False, resident_dso_not_unloaded=True, device_drain_proven=False)


def review_runtime_sources(result, case, controlled, source):
    source = Path(source)
    support_name = 'experiments/worker_core_v11_20261118/probe_quant_containers.py'
    support_sha = digest(source_file(source, support_name).read_bytes())
    loaded = result['setup']['loaded_sources']
    modules = ('fault_injection.quant_container_native', 'fault_injection.metadata_arrays',
        'demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_group_inventory',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_container_scopes',
        'demoncat.runtime.vllm_ascend.worker_extension.shared_scale_group_inventory',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_token_routes',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_route_frames',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_mla_metadata_rows',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_moe_rows',
        'demoncat.runtime.vllm_ascend.worker_extension.serving_shared_scale_inventory',
        'demoncat.runtime.vllm_ascend.worker_extension.shared_scale_inventory',
        'demoncat.runtime.vllm_ascend.worker_extension.quant_inventory',
        'fault_injection.quant_group_native', 'fault_injection.quant_native_isolation',
        'fault_injection.deployments.quant_group_inventory', 'probe_quant_containers',
        'fault_injection.quant_gmm2_native', 'demoncat.runtime.vllm_ascend.worker_extension.identity')
    require(len(loaded) == len(modules) and {r['module'] for r in loaded} == set(modules), 'EXACT_LOADED_RUNTIME_SOURCE_SET')
    for row in loaded:
        name = row['module']
        if name == 'probe_quant_containers':
            require(row['path'] == '/probe_quant_containers.py' and row['sha256'] == support_sha, 'READONLY_SUPPORT_BYTES')
            continue
        relative = ('src/' if name.startswith('demoncat.') else '') + name.replace('.', '/') + '.py'
        require(PurePosixPath(row['path']).is_relative_to('/installed')
            and digest(source_file(source, relative).read_bytes()) == row['sha256'], 'LOADED_RUNTIME_SOURCE_BYTES:' + name)
    require(result['support_source'] == dict(module='probe_quant_containers', path='/probe_quant_containers.py', sha256=support_sha)
        and result['requested_builds']['support_sha256'] == case['review']['support_source_sha256'] == support_sha,
            'SUPPORT_SOURCE_ORIGINAL_JOIN')
    mounts = [row for row in case['mounts'] if row[1] == '/probe_quant_containers.py']
    require(len(mounts) == 1 and len(mounts[0]) == 3 and mounts[0][2] == 'ro', 'ONE_READONLY_SUPPORT_MOUNT_REQUIRED')
    frozen = controlled['frozen_scripts']
    probes = [r for r in frozen if r['destination'] == '/probe.py']
    require(len(probes) == 1 and probes[0]['sha256'] == digest(source_file(source,
        'experiments/worker_core_v11_20261118/probe_quant_gmm2.py').read_bytes()), 'ACTUAL_FROZEN_PROBE_BYTES')
    require(case['env']['VLLM_ENABLE_V1_MULTIPROCESSING'] == '0' and case['env']['TASK_QUEUE_ENABLE'] == '0',
            'EXACT_SAME_PROCESS_SYNCHRONOUS_SCOPE')
    return dict(runtime_source_commit=case['review']['source_commit'], loaded_sources_rehashed=len(loaded),
        probe_sha256=probes[0]['sha256'], support_sha256=support_sha)


def review(values, hashes, runtime_source, native_directory, native_sha, native_source):
    """Diagnose failed/absent observations; no exception may turn into admission."""
    report = dict(schema='demoncat.quant.gmm2_independent_review.v1',
        status='PARTIAL_OR_REFUSED_ORIGINALS', originals=hashes, reviewer_sources=reviewer_sources(),
        reviewer_scope='POSTHOC_CPU_REVIEW_NOT_PART_OF_RUNTIME_SOURCE107', issues=[], sections={}, **NONE_GRANTED)
    def run(name, function):
        try:
            value = function(); report['sections'][name] = dict(consistent=True, facts=value); return value
        except (KeyError, TypeError, ValueError, IndexError, OSError) as error:
            report['issues'].append(dict(section=name, reason=str(error), exception=type(error).__name__))
            report['sections'][name] = dict(consistent=False); return None
    def original(name):
        require(name in values and name in hashes, 'ORIGINAL_MISSING:' + name)
        return values[name]
    build_tuple = None
    def built():
        nonlocal build_tuple
        build_tuple = build_evidence(native_directory, native_sha, native_source)
        return build_tuple[2]
    run('native_cpu_build_provenance', built)
    result = values.get(PREFIX + 'result.json', {})
    report['probe_reported_status_not_acceptance_oracle'] = result.get('status')
    observation = values.get(PREFIX + 'container-trace/observation.json', {})
    g = observation.get('gmm2', {})
    report['recorded_partial_facts_not_acceptance'] = dict(
        retained_container_events=len(observation.get('events', [])), retained_typed_events=len(g.get('events', [])),
        typed_first_error=g.get('state', {}).get('first_error'), pending_executors=g.get('state', {}).get('pending_executors'))
    case_context = None
    def package():
        nonlocal case_context
        require(set(values) == set(hashes) == set(MEMBERS), 'EXACT_NINE_ORIGINALS_REQUIRED')
        controller, cases = original('controller.json'), original('reviewed_cases.json')
        require(len(cases) == len(controller['cases']) == len(result['phases']) == 1, 'ONE_BOUNDED_ATTEMPT_REQUIRED')
        require(controller['case_manifest_sha256'] == hashes['reviewed_cases.json']['sha256']
            and controller['stale_allocation_scope_file_sha256'] == hashes['reviewed_stale_allocation_scope.json']['sha256'],
                'CONTROLLER_INPUT_HASH_JOIN')
        case, controlled = cases[0], controller['cases'][0]; case_context = case, controlled
        require(controlled['review'] == case['review'] and case['review']['expected_hits'] == 0
            and case['review']['physical_cards'] in ([2], [5])
            and case['review']['trace_build_sha256'] == result['requested_builds']['trace_sha256']
                == observation['build_sha256'] == native_sha, 'EXACT_ZERO_F_CASE_BUILD_SCOPE')
        for receipt, name in ((result['gmm2_observation'], 'observation.json'),
                              (result['cleanup']['original'], 'cleanup-observation-001.json'),
                              (result['startup_original'], 'startup-before-llm.json')):
            item = hashes[PREFIX + 'container-trace/' + name]
            require(receipt['sha256'] == item['sha256'] and receipt['bytes'] == item['bytes']
                and receipt['path'] == '/evidence/container-trace/' + name, 'EXACT_IMMUTABLE_ORIGINAL_REFERENCE:' + name)
        require(controlled['exit_code'] == controlled['container_state']['ExitCode'] == 0
            and not controlled['container_state']['OOMKilled'] and controlled['cleanup']['status'] == 'OWNED_CONTAINER_REMOVED',
                'CONTROLLER_FAILED_OR_CLEANUP_INCOMPLETE')
        return dict(original_count=len(values), physical_card=case['review']['physical_cards'][0], zero_fault=True)
    run('original_package_and_controller', package)
    if case_context:
        run('runtime_source_provenance', lambda: review_runtime_sources(result, *case_context, runtime_source))
    inventory = values.get(PREFIX + 'shared-scale/inventory/shared_scale_inventory.json', {})
    def census():
        require(len(inventory['native_forwards']) == len(inventory['forwards']) == 1
            and inventory['actual_projection_calls'] == inventory['actual_native_calls']
            == inventory['actual_native_prepare_calls'] == inventory['actual_native_execute_calls'] == 135
            and inventory['actual_public_api_calls'] == inventory['retained_public_api_calls'] == 270
            and inventory['missing_public_call_details'] == inventory['unowned_public_calls'] == 0
            and inventory['failure'] is None, 'FULL_RAW_135_NATIVE_PUBLIC_CENSUS_REQUIRED')
        phase = result['phases'][0]; mapping = phase['mapping']; requests = phase['request_ids']
        require(len(requests) == len(set(requests)) == len(mapping) == 8
            and set(requests) == {row['internal_request_id'] for row in mapping}
            and len({row['external_request_id'] for row in mapping}) == 8
            and inventory['forwards'][0]['request_ids'] == requests
            and len(phase['output_tokens']) == 8 and all(len(row) == 2 for row in phase['output_tokens']),
                'ACTUAL_B8_REQUEST_MAPPING')
        require(all(inventory[k] is False or type(inventory[k]) is int and inventory[k] == 0
            for k in ('qualified_native_targets', 'c1', 'c2', 'c3', 'group_owner_installed'))
            and all(result[k] is False for k in ('c1', 'c2', 'c3', 'request_rows_proven'))
            and result['native_faults'] == 0 and observation['native_qualified'] is False
            and observation['request_rows_proven'] is False, 'OBSERVATION_MUST_NOT_UPGRADE_AUTHORITY')
        return four.review_native_denominator(inventory['native_forwards'][0], inventory['forwards'][0], inventory['public_calls'])
    run('full_projection_census', census)
    run('unchanged_existing_row_scope', lambda: four.review_row_scope(inventory))
    if build_tuple:
        build, profile, _ = build_tuple
        def mapping():
            require(observation['profile_sha256'] == build['profile_sha256']
                and observation['mapping']['sha256'] == build['files'][build['library']]
                and observation['mapping']['path'] == '/container-trace/' + build['library'], 'CURRENT_OBSERVER_MAPPING')
            return dict(actual_loaded_profile_joined=True, local_dso_rehash=build_tuple[2]['actual_dso_bytes_rehashed'])
        run('current_loaded_observer', mapping)
        run('startup_worker_process_join', lambda: review_startup(original(PREFIX + 'container-trace/startup-before-llm.json'),
            observation, result, inventory, build, profile, native_sha))
        typed = run('typed_calls', lambda: review_typed(observation, profile))
        if typed is not None:
            report['sections']['typed_calls']['facts'] = typed[0]
            run('container_lifetimes_and_metadata', lambda: review_containers(observation, profile, typed[1]))
        else:
            report['sections']['container_lifetimes_and_metadata'] = dict(consistent=False, deferred='GROUP_CAPTURE_COUNT_UNRESOLVED')
    def windows():
        return four.review_windows(observation, inventory['native_forwards'][0]['row_provenance'],
                                   inventory['forwards'][0], runtime_source)
    run('independent_current_source_windows', windows)
    run('named_output_return_join', lambda: review_named_outputs(observation, inventory['native_forwards'][0]['row_provenance']))
    run('cleanup_and_same_worker_health', lambda: review_cleanup(result, observation,
        original(PREFIX + 'container-trace/cleanup-observation-001.json'), original(PREFIX + 'shared-scale/serving.json')))
    report['observation_consistent'] = not report['issues'] and all(v['consistent'] for v in report['sections'].values())
    if report['observation_consistent']:
        report['status'] = 'ORIGINALS_RECONCILED_TYPED_OBSERVATION_ONLY'
    report['limitations'] = ['Saved JSON cannot grant live qualification.',
        'Exact startup process and Worker links are source-bound recorded observations, not a new live proof.',
        'GWS metadata joins do not establish routed token/request semantics or descriptor immutability.',
        'Execute return is not device drain; original DSO mappings remain resident.',
        'No fault, B/F numerical comparison, statistics/mix E5 or new row qualification in this scope.',
        'SDK CPU build and runtime observation use separately verified source snapshots.']
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True); parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--native-build-root', type=Path, required=True); parser.add_argument('--native-build-sha256', required=True)
    parser.add_argument('--native-source-root', type=Path, required=True)
    parser.add_argument('--runtime-source-root', type=Path, required=True); parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        values, hashes = load_archive(args.archive, args.archive_sha256)
        summary = review(values, hashes, args.runtime_source_root, args.native_build_root,
                         args.native_build_sha256, args.native_source_root)
    except (ValueError, OSError, tarfile.TarError) as error:
        summary = dict(schema='demoncat.quant.gmm2_independent_review.v1', status='ORIGINAL_PACKAGE_REFUSED',
            observation_consistent=False, issues=[dict(section='archive', reason=str(error))], **NONE_GRANTED)
    summary['archive_sha256'] = args.archive_sha256
    summary.setdefault('reviewer_sources', reviewer_sources())
    raw = (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with args.out.open('xb') as stream:
        stream.write(raw)
    print(json.dumps(dict(status=summary['status'], summary_sha256=digest(raw), **NONE_GRANTED)))
    return 0 if summary.get('observation_consistent') else 2


if __name__ == '__main__':
    raise SystemExit(main())
