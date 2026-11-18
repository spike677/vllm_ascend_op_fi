"""Read-only offline review of the bounded legal-position RoPE Worker result.

Standard library only: this program never imports a provider, Torch, vLLM or a
device runtime. Aggregate success flags are not evidence. JSON cannot recreate
the live C2 capability; the old result format lacks its issuance/use receipt.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys


VARIANT = 'LEGAL_POSITIONS_INPUT_WRONG_ROW_NOT_UINT32_OVERFLOW'
MECHANISM = '22.wrong_row.fused_positions'
FROZEN_PROBE_SHA256 = '95e0e93275f5140de6607ffdd283c65370374bc5936716223997542e091e565f'
FROZEN_IMPLEMENTATION_SHA256 = 'b77ff882d7c07f163f70ab32dfe9a7d0e004e3114a2c28d70b0f771b0b604cbd'
PHASES = ('native_baseline', 'baseline', 'baseline_repeat', 'lossless',
          'c2_fault', 'c2_off', 'c3_fault_0', 'c3_fault_1', 'c3_off')
SOURCE_PINS = {
    'vllm_ascend.ops.rotary_embedding': '74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7',
    'vllm.model_executor.layers.rotary_embedding.base': '32e2145231012f98c8c613cc37dfe621996a3508af77ee7e0265baefde5c1139',
    'vllm_ascend.ops.triton.rope': '4c579d5d22b08d7657f1406fe072c2d17f6fa0a87b0421a520b61e6e1177f7f9',
    'triton.runtime.jit': '55c8b09db2780f1d5106fb1652bda7a5807c7b243b1e43a763265d4755b58dd3',
}
ENTRY_JOINS = ('positions_exact_object', 'cache_exact_object', 'query_actual_storage',
    'key_actual_storage', 'query_stride', 'key_stride', 'cache_stride',
    'full_original_rotary_dim', 'original_head_size', 'neox_style',
    'uses_positions_cache', 'no_preselected_cos_sin', 'token_count')


def sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def is_sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def probe_digest(value):
    # The root probe uses default JSON separators; provider seals use sha().
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def zeros(value, fields):
    return all(type(value[key]) is int and value[key] == 0 for key in fields)


def same_view(a, b):
    return all(a[key] == b[key] for key in ('storage_id', 'offset', 'shape', 'stride', 'dtype', 'device'))


def same_span(a, b):
    return (all(a[key] == b[key] for key in ('storage_id', 'offset', 'dtype', 'device'))
            and math.prod(a['shape']) == math.prod(b['shape']))


def contiguous(meta):
    stride = 1
    for size, actual in reversed(list(zip(meta['shape'], meta['stride']))):
        if size != 1 and actual != stride:
            return False
        stride *= size
    return len(meta['shape']) == len(meta['stride'])


def tensor_identity(value):
    return (isinstance(value, dict) and is_sha(value.get('sha256'))
            and isinstance(value.get('shape'), list) and bool(value['shape'])
            and all(type(n) is int and n >= 0 for n in value['shape'])
            and isinstance(value.get('dtype'), str))


def identity_pair(value):
    return isinstance(value, list) and len(value) == 2 and all(map(tensor_identity, value))


def worker_key(target):
    return [target[key] for key in ('process_id', 'process_start_identity', 'worker_id',
                                    'rank', 'device_type', 'logical_device')]


class Refusal(Exception):
    pass


class Review:
    def __init__(self, result, *, expected_targets=None, _cpu_contract=False):
        self.result, self.expected_targets, self.cpu = result, expected_targets, _cpu_contract
        self.issues, self.checks, self.completed = [], 0, set()
        self.rows, self.targets, self.paths = [], {}, []

    def need(self, condition, code):
        self.checks += 1
        if not condition:
            raise Refusal(code)

    def section(self, name, function):
        try:
            function()
            self.completed.add(name)
        except Refusal as error:
            self.issues.append(dict(section=name, code=str(error)))
        except (KeyError, TypeError, ValueError, IndexError, AttributeError, OverflowError) as error:
            self.issues.append(dict(section=name, code='MISSING_OR_MALFORMED_EVIDENCE',
                                    field=str(error)[:160]))

    def envelope(self):
        r = self.result
        self.need(r['mechanism'] == MECHANISM and r['raw_device_launch_payload_observed'] is False,
                  'WRONG_VARIANT_OR_DEVICE_PAYLOAD_CLAIM')
        self.need(not any(r.get(key) for key in ('error', 'cleanup_error', 'abort_error')), 'RUN_RECORDED_FAILURE')
        self.need(is_sha(r['probe_sha256']), 'PROBE_SOURCE_HASH_MISSING')
        provenance = r['setup']['provenance']
        self.need(provenance['variant'] == VARIANT and provenance['raw_device_launch_payload_observed'] is False,
                  'WRONG_MECHANISM_SEMANTICS')
        self.need(provenance['simulation'] is self.cpu and provenance['standard_worker'] is (not self.cpu),
                  'ACTUAL_STANDARD_WORKER_REQUIRED')
        identity = provenance['worker_identity']
        self.need(identity['status'] == 'WORKER_IDENTIFIED' and identity.get('quarantined') is False,
                  'WORKER_IDENTITY_NOT_HEALTHY')
        self.worker = identity['worker']
        self.need(all(self.worker[key] for key in ('pid', 'process_start_identity', 'worker_start_identity',
                                                  'model_instance_id')), 'WORKER_LIFETIME_INCOMPLETE')
        self.need(type(self.worker['pid']) is int and self.worker['pid'] > 0, 'ACTUAL_PROCESS_ID_REQUIRED')
        self.need(type(self.worker['rank']) is int and self.worker['rank'] == 0, 'SINGLE_RANK_SCOPE_REQUIRED')
        self.need([item['phase'] for item in r['requests']] == list(PHASES), 'EXACT_NINE_FRESH_PHASES_REQUIRED')
        self.rows = [item['worker'] for item in r['requests']]
        self.paths = [reader['model_path'] for reader in self.rows[0]['independent']['rope_trace']['readers']]
        self.need(bool(self.paths) and len(set(self.paths)) == len(self.paths), 'EMPTY_OR_DUPLICATE_NATURAL_READER_SET')
        if self.expected_targets is not None:
            self.need(len(self.paths) == self.expected_targets, 'EXPECTED_COUNT_DIFFERS_FROM_ACTUAL_READERS')
        self.need(set(provenance['structural_rope_sites']) == set(self.paths), 'STRUCTURAL_READER_DENOMINATOR_MISMATCH')
        self.need(isinstance(r['healthy_before'], list) and len(r['healthy_before']) == 1
                  and r['healthy_after'] == r['healthy_before'], 'POST_CLEANUP_GENERATION_NOT_RESTORED')
        if not self.cpu:
            self.need(r['standard_vllm_worker'] is True, 'STANDARD_VLLM_WORKER_REQUIRED')
            self.need(r['probe_sha256'] == FROZEN_PROBE_SHA256, 'FROZEN_PROBE_SOURCE_CHANGED')
            model = r['model_identity']
            self.need(model['route_configuration_modified'] is False and is_sha(model['config_sha256'])
                      and bool(model['weights']) and model['weights'] == provenance['weights'], 'MODEL_SOURCE_OR_WEIGHTS_CHANGED')
            self.need(all(is_sha(v['sha256']) and type(v['bytes']) is int and v['bytes'] > 0
                          for v in model['weights'].values()), 'FULL_WEIGHT_MANIFEST_REQUIRED')

    def requests(self):
        seen_internal, seen_external, seen_invocations = set(), set(), set()
        baseline = self.rows[0]
        for phase, item, row in zip(PHASES, self.result['requests'], self.rows):
            self.need(row['phase'] == phase and row['logits_boundary_returned'] is True, 'PHASE_OR_LOGITS_BOUNDARY_MISMATCH')
            request, invocation = row['request_id'], row['invocation']
            self.need(isinstance(request, str) and request and request not in seen_internal
                      and isinstance(invocation, str) and invocation and invocation not in seen_invocations,
                      'REUSED_OR_MISSING_ACTUAL_REQUEST')
            seen_internal.add(request); seen_invocations.add(invocation)
            mapping, scheduler = item['mapping'], row['scheduler']
            self.need(mapping['internal_request_id'] == request
                      and mapping['external_request_id'] not in seen_external
                      and isinstance(mapping['external_request_id'], str) and mapping['external_request_id']
                      and type(mapping['request_state_object_id']) is int and mapping['request_state_object_id'] > 0
                      and mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR', 'DRIVER_REQUEST_MAPPING_MISMATCH')
            seen_external.add(mapping['external_request_id'])
            if self.cpu:
                self.need(scheduler['cpu_contract'] is True and scheduler['request_id'] == request, 'CPU_SCHEDULER_FIXTURE_MISMATCH')
            else:
                self.need(scheduler['internal_request_id'] == request and scheduler['num_requests'] == 1
                          and scheduler['num_computed_tokens'] == 0
                          and scheduler['num_scheduled_tokens'] == scheduler['num_prompt_tokens'] == len(scheduler['positions'])
                          and scheduler['positions'] == list(range(len(scheduler['token_ids'])))
                          and sha(scheduler['token_ids']) == scheduler['token_ids_sha256']
                          and type(scheduler['request_object_id']) is int and scheduler['request_object_id'] > 0,
                          'SCHEDULER_PREFILL_MAPPING_OR_POSITION_MISMATCH')
                self.need(scheduler['token_ids'] == self.rows[0]['scheduler']['token_ids'], 'ACTUAL_REQUEST_TOKENS_CHANGED')
                # Compare complete captured device input bytes to the separately
                # recorded scheduler vectors; CPU mirror alone is insufficient.
                for name, values in (('input_ids', scheduler['token_ids']), ('positions', scheduler['positions'])):
                    captured = row['input'][1][name]
                    code = {'torch.int32': 'i', 'torch.int64': 'q'}[captured['dtype']]
                    digest = hashlib.sha256(struct.pack('<' + code * len(values), *values)).hexdigest()
                    self.need(captured['shape'] == [len(values)] and captured['sha256'] == digest,
                              'FULL_CAPTURED_INPUT_DIFFERS_FROM_SCHEDULER')
            self.need(row['input'] == baseline['input'], 'COMPLETE_ROOT_INPUT_CHANGED')
            for name in ('hidden', 'logits'):
                output = row['output'][name]
                self.need(tensor_identity(output) and output['finite'] is True, 'COMPLETE_FINITE_HIDDEN_LOGITS_REQUIRED')
                self.need(output['shape'] == baseline['output'][name]['shape']
                          and output['dtype'] == baseline['output'][name]['dtype'], 'FULL_OUTPUT_CONTRACT_CHANGED')
            if not self.cpu:
                self.need(row['output']['logits']['shape'] == self.result['setup']['provenance']['logits_boundary']['output_contract'],
                          'FULL_VOCAB_OUTPUT_MISSING')
                self.need(row['output']['hidden']['shape'][0] == len(scheduler['positions']), 'FULL_HIDDEN_TOKEN_ROWS_MISSING')
            if phase in PHASES[:4] or phase.endswith('off'):
                self.need(row['output'] == baseline['output'], 'HEALTHY_OR_OFF_COMPLETE_OUTPUT_CHANGED')
                self.need(item['output_tokens'] == self.result['healthy_before'], 'HEALTHY_OR_OFF_SAMPLED_TOKEN_CHANGED')
            else:
                self.need(row['output'] != baseline['output'], 'FAULT_FULL_MODEL_OUTPUT_DID_NOT_CHANGE')
            cleanup = row['cleanup']
            self.need(cleanup['cleanup_confirmed'] is True and cleanup['hook_owned'] is False
                      and type(cleanup['inflight']) is int and cleanup['inflight'] == 0
                      and type(cleanup['leases']) is int and cleanup['leases'] == 0
                      and cleanup.get('enabled', False) is False, 'PER_REQUEST_RESOURCES_NOT_ZERO')
        cleanup = self.result['cleanup']
        self.need(zeros(cleanup, ('owned_serving_hooks',)) and cleanup['active_adapter'] is False
                  and cleanup['independent_observer'] is False and cleanup['standard_worker'] is (not self.cpu)
                  and cleanup['requests'] == [row['request_id'] for row in self.rows], 'FINAL_CLEANUP_IDENTITY_OR_RESOURCES_INCOMPLETE')

    def trace(self, phase, row):
        fault = 'fault' in phase
        independent, baseline = row['independent'], self.rows[1]['independent']['rope_trace']
        trace = independent['rope_trace']
        readers, outer, launches = trace['readers'], trace['outer_operators'], trace['triton_launches']
        n, request = len(self.paths), row['request_id']
        self.need(independent['source'] == 'independent_workload_observer'
                  and zeros(independent, ('dropped_calls', 'observer_hooks', 'inflight'))
                  and len(independent['forwards']) == 1 and independent['forwards'][0]['status'] == 'returned',
                  'INDEPENDENT_ROOT_OBSERVATION_INCOMPLETE')
        self.need(not trace['errors'] and trace['simulation'] is self.cpu
                  and trace['raw_device_launch_payload_observed'] is False
                  and trace['native_join_evidence_kind'] == 'SOURCE_BOUND_TRITON_RUNTIME_CALL'
                  and trace['operator_reexecution'] is False and trace['mutations'] == 0,
                  'INDEPENDENT_TRITON_TRACE_MISSING_OR_FAILED')
        self.need([r['model_path'] for r in readers] == self.paths and len(readers) == len(outer) == len(launches) == n
                  and set(trace['structural_cache_owners']) == set(self.paths)
                  and trace['structural_calls_per_request'] == {request: {path: 1 for path in self.paths}}
                  and trace['readers_per_request'] == {request: n}
                  and trace['root_forwards'] == [dict(request_id=request, status='returned')], 'ALL_NATURAL_READERS_DENOMINATOR_CHANGED')
        self.need(len(row['calls']) == n and all(r['sequence'] == i for i, r in enumerate(readers)), 'NATURAL_READER_OCCURRENCE_MISSING')
        self.need(trace['sources'] == self.result['setup']['provenance']['rope_sources'], 'RUNTIME_SOURCE_SET_CHANGED')
        self.need(trace['actual_dispatch'] == baseline['actual_dispatch'], 'ACTUAL_CALLABLE_DISPATCH_CHANGED')
        if not self.cpu:
            self.need(all(trace['sources'][name]['sha256'] == value for name, value in SOURCE_PINS.items())
                      and trace['sources']['reader']['has_triton'] is True, 'UNREVIEWED_READER_OR_TRITON_SOURCE')
            guards = list(trace['actual_dispatch'].values())
            self.need(bool(guards) and all(guard['source_sha256'] == sha(guard['methods']) and guard['callable_identity']
                      for guard in guards) and all(any(guard['module_object_id'] == reader['module_object_id']
                      for guard in guards) for reader in readers), 'ACTUAL_READER_CALLABLE_GUARD_MISSING')
        events = (row['mechanism'] or {}).get('events', [])
        if phase not in ('native_baseline', 'c2_off', 'c3_off'):
            self.need(len(events) == n and not row['mechanism']['errors']
                      and row['mechanism']['variant'] == VARIANT
                      and row['mechanism']['integer_overflow_reproduced'] is False, 'MECHANISM_EVENT_DENOMINATOR_CHANGED')
        else:
            self.need(row['mechanism'] is None, 'UNEXPECTED_OFF_OR_NATIVE_ADAPTER')
        operators = {call['operator'] for call in row['calls']}
        self.need(len(operators) == 1, 'MIXED_UNREVIEWED_ROPE_OPERATOR')
        operator = next(iter(operators))
        self.need(independent['operator_counts'][operator] == n, 'INDEPENDENT_OPERATOR_DENOMINATOR_CHANGED')
        if not self.cpu:
            self.need(operator == 'vllm.npu_rotary_embedding.default', 'UNREVIEWED_OUTER_ROPE_OPERATOR')
            native_outer = [event for event in independent['events'] if event.get('kind') == 'operator'
                            and event.get('operator') == operator]
            self.need(len(native_outer) == n and all(event['original_returned'] is True for event in native_outer),
                      'INDEPENDENT_OPERATOR_EVENTS_MISSING')
            self.need(all(event['operator'] == operator for event in independent['events']
                          if event.get('kind') == 'operator' and 'rotary' in event.get('schema', '')),
                      'UNCLASSIFIED_NATURAL_ROTARY_OPERATOR')
            self.need(all(event['path'] == call['path'] and event['input'] == call['metadata']
                          for event, call in zip(native_outer, row['calls'])), 'INDEPENDENT_OPERATOR_CALLSITE_JOIN_CHANGED')
        for i, (reader, out, launch, call) in enumerate(zip(readers, outer, launches, row['calls'])):
            self.reader(phase, row, i, reader, out, launch, call, baseline['readers'][i], events[i] if events else None)
        self.need(all(native['original_calls'] == 1 and native['original_returned'] is True
                      and native['request_id'] == request and native['reader_sequence'] in range(n)
                      for native in trace['native_operators']), 'NESTED_OPERATOR_EXECUTION_INCOMPLETE')

    def reader(self, phase, row, index, reader, outer, triton, call, healthy, event):
        fault, request, path = 'fault' in phase, row['request_id'], self.paths[index]
        self.need(call['key'] == [path, index] and call['path'] == path and call['ordinal'] == index
                  and reader['request_id'] == request and reader['scheduler'] == row['scheduler'], 'READER_REQUEST_CALLSITE_JOIN_FAILED')
        expected = reader['scheduler_positions']
        self.need(isinstance(expected, list) and len(expected) > 1
                  and all(type(v) is int for v in expected) and expected[0] != expected[-1], 'LEGAL_POSITION_VECTOR_MISSING')
        if not self.cpu:
            self.need(expected == row['scheduler']['positions'], 'READER_SCHEDULER_POSITIONS_CHANGED')
        actual = expected[:-1] + [expected[0]] if fault else expected
        self.need(reader['actual_positions'] == actual and min(actual) >= 0 and max(actual) < reader['cache']['shape'][0]
                  and reader['independent_position_transform'] == ('last_from_first' if fault else 'identity'),
                  'ACTUAL_LEGAL_LAST_ROW_TRANSFORM_NOT_PROVEN')
        self.need(all(reader[key] is True for key in ('returned', 'actual_positions_unchanged',
                    'root_positions_unchanged', 'complete_cache_unchanged', 'candidate_rows_different')),
                  'OLD_INPUT_OR_CACHE_CHANGED')
        self.need(reader['module_object_id'] == healthy['module_object_id']
                  and reader['cache_lifetime_object_id'] == healthy['cache_lifetime_object_id']
                  and reader['cache'] == healthy['cache'] and reader['cache_sha256'] == healthy['cache_sha256']
                  and is_sha(reader['cache_sha256']) and reader['arguments'] == healthy['arguments'], 'READER_OR_CACHE_LIFETIME_CHANGED')
        candidate_rows = reader['candidate_rows']
        first, last = candidate_rows[str(expected[0])], candidate_rows[str(expected[-1])]
        self.need(tensor_identity(first) and tensor_identity(last) and first['sha256'] != last['sha256']
                  and candidate_rows == healthy['candidate_rows'], 'DISTINCT_LEGAL_CACHE_ROWS_UNPROVEN')
        for key in ('complete_output', 'non_target_output', 'non_target_qk_before'):
            self.need(identity_pair(reader[key]), 'COMPLETE_READER_HASHES_MISSING')
        self.need(tensor_identity(reader['query_before']) and tensor_identity(reader['key_before']), 'COMPLETE_QK_INPUT_HASHES_MISSING')
        self.need(reader['non_target_qk_before'] == healthy['non_target_qk_before']
                  and reader['non_target_output'] == healthy['non_target_output'], 'NON_TARGET_INPUT_OR_OUTPUT_CHANGED')
        self.need((reader['complete_output'] != healthy['complete_output']) if fault else
                  (reader['complete_output'] == healthy['complete_output']
                   and reader['query_before'] == healthy['query_before'] and reader['key_before'] == healthy['key_before']),
                  'COMPLETE_READER_EFFECT_OR_HEALTHY_REPEAT_MISMATCH')
        p, q, k, cache = call['metadata']
        arguments = reader['arguments']
        self.need(arguments['offsets'] is None and type(arguments['head_size']) is int and arguments['head_size'] > 0
                  and arguments['rotary_dim'] == arguments['head_size'] and type(arguments['is_neox_style']) is bool
                  and p['shape'] == reader['positions']['shape'] == [len(expected)]
                  and p['dtype'] == reader['positions']['dtype'] and p['dtype'] in ('torch.int32', 'torch.int64')
                  and p['device'] == q['device'] == k['device'] == cache['device'] == reader['positions']['device']
                  and q['dtype'] == k['dtype'] == cache['dtype']
                  and all(contiguous(meta) for meta in (p, reader['positions'], q, k)),
                  'REVIEWED_FULL_DIM_CONTIGUOUS_INPUT_CONTRACT_CHANGED')
        self.need(same_view(cache, reader['cache']) and cache['object_id'] == reader['cache_lifetime_object_id']
                  and len(reader['output']) == 2 and reader['output_aliases_actual_qk'] == [True, True]
                  and same_view(q, reader['output'][0]) and same_view(k, reader['output'][1]), 'ACTUAL_QK_OUTPUT_OR_CACHE_OBJECT_JOIN_FAILED')
        self.need(reader['positions']['storage_id'] != p['storage_id'] if fault else same_view(p, reader['positions']),
                  'REPLACEMENT_POSITIONS_NOT_SEPARATE_OR_HEALTHY_IDENTITY_CHANGED')
        for value, meta in zip(reader['complete_output'], reader['output']):
            self.need(value['shape'] == meta['shape'] and value['dtype'] == meta['dtype'], 'COMPLETE_OUTPUT_GEOMETRY_CHANGED')
        consumers = [consumer for consumer in reader['consumers'] if consumer.get('returned') is True]
        self.need(bool(consumers) and any(consumer['actual_query_and_key_views'] is True
                  and all(any(same_view(t, output) for t in consumer['input']) for output in reader['output'])
                  for consumer in consumers), 'ACTUAL_RETURNED_QK_CONSUMER_UNPROVEN')
        self.need(outer['sequence'] == index and outer['model_path'] == path and outer['request_id'] == request
                  and outer['reader_sequences'] == [index] and outer['original_calls'] == 1
                  and outer['original_returned'] is True and outer['operator'] == call['operator']
                  and outer['schema'] == call['schema']
                  and set(outer['reader_argument_join']) == {'positions', 'query', 'key', 'cos_sin_cache'}
                  and all(v is True for v in outer['reader_argument_join'].values())
                  and set(outer['reader_scalar_join']) == {'head_size', 'rotary_dim', 'is_neox_style'}
                  and all(v is True for v in outer['reader_scalar_join'].values()), 'ACTUAL_OUTER_READER_JOIN_FAILED')
        self.triton(triton, reader, q, k, request, path, index)
        healthy_triton = self.rows[1]['independent']['rope_trace']['triton_launches'][index]
        self.need(triton['kernel_name'] == healthy_triton['kernel_name']
                  and triton['kernel_hash'] == healthy_triton['kernel_hash']
                  and triton['launches'][0]['actual_grid'] == healthy_triton['launches'][0]['actual_grid'],
                  'ACTUAL_COMPILED_KERNEL_OR_GRID_CHANGED')
        if event is not None:
            self.need(event['path'] == path and event['ordinal'] == index and event['request_id'] == request
                      and event['native_calls'] == 1 and event['applied'] is fault
                      and event['semantics'] == VARIANT and event['target_token'] == len(expected)-1
                      and event['correct_source_row'] == expected[-1] and event['replacement_source_row'] == expected[0]
                      and event['positions_unchanged'] is True and event['cache_unchanged'] is True
                      and event['non_target_output_unchanged'] is True
                      and event['actual_qk_before'] == [reader['query_before'], reader['key_before']]
                      and event['actual_qk_after'] == reader['complete_output'], 'ADAPTER_AND_INDEPENDENT_READER_DISAGREE')

    def triton(self, triton, reader, q, k, request, path, index):
        self.need(triton['sequence'] == index and triton['reader_sequence'] == index
                  and triton['request_id'] == request and triton['model_path'] == path
                  and triton['simulation'] is self.cpu and triton['returned'] is True
                  and triton['actual_launch_call_count'] == 1 and len(triton['launches']) == 1
                  and bool(triton['kernel_name']) and bool(triton['kernel_hash']), 'UNIQUE_ACTUAL_TRITON_LAUNCH_MISSING')
        if not self.cpu:
            self.need(triton['kernel_source_sha256'] == SOURCE_PINS['vllm_ascend.ops.triton.rope']
                      and triton['jit_source_sha256'] == SOURCE_PINS['triton.runtime.jit'], 'TRITON_RUNTIME_SOURCE_CHANGED')
        launch = triton['launches'][0]
        for joins in (triton['entry_argument_join'], launch['actual_argument_join']):
            self.need(set(joins) == set(ENTRY_JOINS) and all(v is True for v in joins.values()), 'TRITON_ARGUMENT_JOIN_INCOMPLETE')
        self.need(launch['returned'] is True and launch['actual_non_constexpr_matches_bound'] is True
                  and launch['argument_evidence'] == 'PINNED_JIT_CALLSITE_LIVE_LOCALS'
                  and launch['raw_device_launch_payload_observed'] is False
                  and len(launch['actual_grid']) == 3 and all(type(v) is int and v > 0 for v in launch['actual_grid'])
                  and type(launch['actual_stream']) is int and launch['actual_stream'] >= 0, 'ACTUAL_LAUNCH_CALLSITE_MISSING')
        a, p, c = launch['arguments'], reader['positions'], reader['cache']
        self.need(a['pos_ptr'] == p and a['cos_sin_ptr'] == c
                  and same_span(a['q_ptr'], q) and same_span(a['k_ptr'], k)
                  and a['q_row_stride'] == a['q_ptr']['stride'][0]
                  and a['k_row_stride'] == a['k_ptr']['stride'][0]
                  and a['cos_sin_row_stride'] == c['stride'][0]
                  and a['num_tokens'] == len(reader['actual_positions'])
                  and type(a['n_qh']) is int and a['n_qh'] > 0 and type(a['n_kh']) is int and a['n_kh'] > 0
                  and math.prod(q['shape']) == a['num_tokens'] * a['n_qh'] * a['hd']
                  and math.prod(k['shape']) == a['num_tokens'] * a['n_kh'] * a['hd']
                  and a['rope_dim'] == reader['arguments']['rotary_dim']
                  and a['hd'] == reader['arguments']['head_size']
                  and a['IS_NEOX_STYLE'] is reader['arguments']['is_neox_style'] and a['USE_COS_SIN'] is True
                  and all(a[key] is None for key in ('cos_ptr', 'cos_row_stride', 'sin_ptr', 'sin_row_stride')),
                  'RECORDED_FINAL_LAUNCH_ARGUMENTS_DO_NOT_JOIN_READER')

    def candidates(self):
        candidates = self.result['qualification']['candidate_set']
        rows = candidates['candidates']
        n = len(self.paths)
        self.need(len(rows) == candidates['eligible'] == candidates['discovered'] == n,
                  'CANDIDATE_SET_DROPPED_NATURAL_READERS')
        baseline = self.rows[1]['independent']['rope_trace']
        for index, (candidate, reader, outer) in enumerate(zip(rows, baseline['readers'], baseline['outer_operators'])):
            target, path = candidate['identity'], self.paths[index]
            self.need(candidate['candidate_id'] == path + ':' + str(index)
                      and candidate['eligibility'] == dict(status='SUPPORTED', reasons=[]), 'CANDIDATE_READER_IDENTITY_CHANGED')
            self.need(target['process_id'] == self.worker['pid']
                      and target['process_start_identity'] == self.worker['process_start_identity']
                      and target['worker_id'] == self.worker['worker_start_identity']
                      and target['model_instance_id'] == self.worker['model_instance_id']
                      and target['rank'] == self.worker['rank'] and target['scope_kind'] == 'INITIALIZED_WORKER'
                      and target['device_type'] == ('cpu' if self.cpu else 'npu')
                      and target['module_object_identity'] == reader['module_object_id']
                      and target['module_roles'] == ['FUSED_ROPE_POSITIONS_READER']
                      and target['operator_api'] == outer['operator'] and target['provider_identity'] == MECHANISM,
                      'TARGET_WORKER_MODULE_DEVICE_IDENTITY_CHANGED')
            device = reader['positions']['device'].split(':')
            self.need(target['device_type'] == device[0] and target['logical_device'] == (None if self.cpu else int(device[1]))
                      and type(target['model_object_identity']) is int and target['model_object_identity'] > 0
                      and target['model_object_identity'] == rows[0]['identity']['model_object_identity'],
                      'ACTUAL_DEVICE_OR_MODEL_OBJECT_CHANGED')
            signature = dict(path=path, query=reader['query_before']['shape'], key=reader['key_before']['shape'],
                             dtype=reader['query_before']['dtype'], arguments=reader['arguments'])
            self.need(json.loads(target['tensor_signature']) == signature
                      and target['boundary']['call_token'] == path + ':' + str(index)
                      and target['boundary']['schema_sha256'] == sha(outer['schema'])
                      and target['boundary']['implementation_kind'] == 'torch_dispatch'
                      and target['boundary']['kind'] == 'operator_api', 'TARGET_BOUNDARY_OR_SIGNATURE_CHANGED')
            proof = dict(target['provider_proof'])
            self.need(proof == dict(cache_object=str(reader['cache_lifetime_object_id']),
                      cache_storage=str(reader['cache']['storage_id']), actual_site=path,
                      runtime_sources_sha256=sha(baseline['sources'])), 'TARGET_CACHE_OR_SOURCE_PROOF_CHANGED')
            target_id = sha(target)
            self.need(target_id not in self.targets, 'DUPLICATE_TARGET_IDENTITY')
            self.targets[target_id] = (index, target)
        self.need(self.result['qualification']['independent_discovery_sha256'] == sha(self.rows[:4]),
                  'HEALTHY_DISCOVERY_ARTIFACT_HASH_MISMATCH')

    def group(self, name, fault_phases):
        group, opened = self.result[name], self.result[name + '_open']
        n, forwards = len(self.paths), len(fault_phases)
        phases = fault_phases + [name + '_off']
        rows = [self.rows[PHASES.index(phase)] for phase in phases]
        expected_ids = set(self.targets)
        self.need(type(group['generation']) is int and group['selection'] == 'all_eligible'
                  and group['trigger'] == ('once' if name == 'c2' else 'every_match')
                  and opened['trigger'] == group['trigger'] and opened['generation'] == group['generation']
                  and opened['targets'] == n and group['generation'] == (1 if name == 'c2' else 2), 'GROUP_OWNER_GENERATION_OR_SCOPE_MISMATCH')
        self.need(isinstance(group['owner'], str) and group['owner'] and isinstance(group['session_id'], str)
                  and group['session_id'], 'GROUP_OWNER_OR_SESSION_ID_MISSING')
        scope = group['scope']
        self.need(scope['mechanism'] == MECHANISM and scope['evidence_kind'] == ('cpu_contract' if self.cpu else 'natural_worker')
                  and all(is_sha(scope[key]) for key in ('workload_sha256', 'environment_sha256',
                            'implementation_sha256', 'model_weights_sha256', 'model_source_sha256')), 'GROUP_PROVENANCE_INCOMPLETE')
        self.need(scope['model_weights_sha256'] == sha(self.result['setup']['provenance']['weights'])
                  and scope['model_source_sha256'] == sha(self.result['setup']['provenance']['model_sources']), 'GROUP_MODEL_PROVENANCE_MISMATCH')
        if not self.cpu:
            model = self.result['model_identity']
            workload = dict(weights=model['weights'], config_sha256=model['config_sha256'],
                            tokens=self.rows[0]['scheduler']['token_ids'])
            self.need(scope['implementation_sha256'] == FROZEN_IMPLEMENTATION_SHA256
                      and scope['environment_sha256'] == probe_digest(self.result['setup']['environment'])
                      and scope['workload_sha256'] == probe_digest(workload), 'FROZEN_IMPLEMENTATION_WORKLOAD_OR_ENVIRONMENT_CHANGED')
        self.need(all(target['boundary']['implementation_sha256'] == scope['implementation_sha256']
                      for _, target in self.targets.values()), 'TARGET_IMPLEMENTATION_SCOPE_MISMATCH')
        self.need(group['phase_rows'] == rows and group['execution_boundary'] == 'operator_api'
                  and group['mechanism_semantics'] == VARIANT and group['raw_device_launch_payload_observed'] is False,
                  'GROUP_PHASES_OR_VARIANT_MISMATCH')
        discovery, observation = group['discovery'], group['observation']
        self.need(discovery['candidates_sha256'] == sha(self.result['qualification']['candidate_set'])
                  and discovery['artifact_sha256'] == sha(self.rows[:4])
                  and discovery['observed_target_ids'] == sorted(expected_ids)
                  and discovery['source'] == 'independent_workload_observer'
                  and discovery['workload_sha256'] == scope['workload_sha256']
                  and discovery['implementation_sha256'] == scope['implementation_sha256'], 'INDEPENDENT_DISCOVERY_SEAL_MISMATCH')
        workers = {sha(worker_key(target)): worker_key(target) for _, target in self.targets.values()}
        self.need(len(workers) == 1 and {sha(w) for w in discovery['expected_workers']} == set(workers)
                  and discovery['observed_workers'] == discovery['expected_workers'], 'ACTUAL_WORKER_DENOMINATOR_MISMATCH')
        self.need(len(group['calls']) == n * (forwards + 1) and observation['observed_calls'] == len(group['calls'])
                  and observation['dropped_calls'] == 0 and observation['artifact_sha256'] == sha(rows), 'GROUP_OBSERVATION_DENOMINATOR_MISMATCH')
        self.need(group['budget']['max_calls'] == n * forwards and group['budget']['max_concurrency'] == n
                  and 0 < group['budget']['max_seconds'] <= 600, 'BOUNDED_GROUP_BUDGET_CHANGED')
        tickets, hit_counts, seen_calls = [], Counter(), set()
        for ordinal, call in enumerate(group['calls']):
            row = rows[ordinal // n]
            index = ordinal % n
            tid = call['target_id']
            target_index, target = self.targets[tid]
            decision, receipt = call['decision'], call['receipt']
            ticket, lease = decision['ticket'], decision['lease']
            original = ticket['call']
            fault = row['phase'] in fault_phases
            self.need(target_index == index and call['call_id'] == row['invocation'] + ':' + str(index)
                      and call['call_id'] not in seen_calls and call['forward_id'] == row['invocation']
                      and call['request_ids'] == [row['request_id']], 'GROUP_CALL_REQUEST_OR_TARGET_JOIN_FAILED')
            seen_calls.add(call['call_id'])
            self.need(call['owner'] == group['owner'] == lease['owner'] and call['generation'] == group['generation'] == lease['generation']
                      and call['worker'] == worker_key(target) and call['rank'] == target['rank']
                      and call['device'] == target['logical_device'] and call['model_instance'] == target['model_instance_id']
                      and call['operator'] == target['operator_api'], 'CALL_OWNER_GENERATION_OR_WORKER_CHANGED')
            expected_phase = 'OFF_RECOVERY' if not fault else ('PRE_FAULT_PROPAGATION' if index == 0 else 'POST_FIRST_ANOMALY')
            self.need(ticket['observer_id'] == discovery['observer_id'] and ticket['sequence'] == ordinal
                      and ticket['eligible'] is True and ticket['propagation_added'] is False
                      and original['target'] == target and original['call_id'] == call['call_id']
                      and original['forward_context'] == call['forward_id'] and original['request_ids'] == call['request_ids']
                      and original['phase'] == call['phase'] == expected_phase
                      and call['eligible'] is True and call['propagation_added'] is False, 'INDEPENDENT_TICKET_JOIN_FAILED')
            self.need(decision['session_id'] == group['session_id'] and decision['admitted'] is fault
                      and decision['reason'] == ('ADMITTED' if fault else 'OFF_RECOVERY')
                      and lease['target'] == target and lease['lease_id'] == 'rope-site-' + str(index)
                      and zeros(lease, ('device_bytes', 'host_bytes'))
                      and receipt['decision'] == decision and receipt['outcome'] == ('HIT' if fault else 'PASSTHROUGH')
                      and receipt['native_applied'] is fault and receipt['original_calls'] == 1, 'PER_CALL_ADMISSION_OR_COMPLETION_INVALID')
            trace = row['independent']['rope_trace']; reader = trace['readers'][index]
            execution = sha(dict(reader=reader, outer=trace['outer_operators'][index],
                triton=[value for value in trace['triton_launches'] if value['reader_sequence'] == reader['sequence']])) if fault else sha(reader)
            effect = sha(dict(actual=reader, healthy=self.rows[1]['independent']['rope_trace']['readers'][index],
                              model_output=row['output'])) if fault else None
            self.need(receipt['execution_sha256'] == execution and receipt['effect_sha256'] == effect,
                      'COMPLETE_INDEPENDENT_EXECUTION_OR_EFFECT_HASH_MISMATCH')
            tickets.append(ticket)
            if fault: hit_counts[tid] += 1
        self.need(observation['calls_sha256'] == sha(tickets) and hit_counts == Counter({tid: forwards for tid in expected_ids})
                  and not group['missing_outcomes'] and not group['propagation_added_calls']
                  and group['planned_targets'] == group['planned_targets_hit'] == n
                  and group['hits'] == group['observed_eligible_calls'] == n * forwards
                  and group['observed_calls'] == n * (forwards + 1), 'RECOMPUTED_ALL_TARGET_HITS_MISMATCH')
        self.recovery(group, rows[-1], expected_ids, workers)

    def recovery(self, group, off, expected_ids, workers):
        recovery = group['recovery']
        self.need(recovery['state'] == 'DISABLED_CONFIRMED' and recovery['owner'] == group['owner']
                  and recovery['generation'] == group['generation'] and zeros(recovery, ('inflight',))
                  and all(recovery[key] == [] for key in ('held_leases', 'pending_drains', 'pending_leases', 'errors')),
                  'GROUP_RESOURCES_OR_DRAINS_NOT_ZERO')
        observed = set()
        for row in recovery['targets']:
            lease = row['lease']; tid = sha(lease['target'])
            self.need(tid in expected_ids and tid not in observed and lease['owner'] == group['owner']
                      and lease['generation'] == group['generation']
                      and lease['lease_id'] == 'rope-site-' + str(self.targets[tid][0])
                      and zeros(lease, ('device_bytes', 'host_bytes'))
                      and row['drained'] is True and row['released'] is True and row['off_exact'] is True
                      and row['pending_reason'] is None and row['proof_sha256'] == sha(off)
                      and row['off_output_sha256'] == sha(self.rows[0]['output']), 'TARGET_OFF_OR_RELEASE_PROOF_MISMATCH')
            observed.add(tid)
        self.need(observed == expected_ids and len(recovery['workers']) == len(workers), 'RECOVERY_TARGET_OR_WORKER_MISSING')
        for row in recovery['workers']:
            self.need(sha(row['worker']) in workers and row['owner'] == group['owner']
                      and row['generation'] == group['generation'] and row['proof_sha256'] == sha(off)
                      and row['baseline_output_sha256'] == row['off_output_sha256'] == sha(self.rows[0]['output'])
                      and all(type(row[key]) is int and row[key] == 0 for key in
                          ('active_leases', 'device_bytes', 'host_bytes', 'inflight_calls', 'owned_hooks')),
                      'SAME_WORKER_FULL_OFF_OR_ZERO_RESOURCES_UNPROVEN')

    def cross_group(self):
        c2, c3 = self.result['c2'], self.result['c3']
        self.need(c2['owner'] == c3['owner'] and c2['generation'] < c3['generation']
                  and c2['session_id'] != c3['session_id'] and c2['scope'] == c3['scope']
                  and c2['discovery']['candidates_sha256'] == c3['discovery']['candidates_sha256']
                  and c2['discovery']['observed_target_ids'] == c3['discovery']['observed_target_ids'],
                  'C3_SCOPE_LIFETIME_OR_TARGET_SET_DIFFERS_FROM_C2')

    def run(self):
        self.section('envelope', self.envelope)
        if 'envelope' in self.completed:
            self.section('requests', self.requests)
            for phase, row in zip(PHASES, self.rows):
                self.section(phase + '.trace', lambda p=phase, r=row: self.trace(p, r))
            self.section('candidates', self.candidates)
            if 'candidates' in self.completed:
                self.section('c2', lambda: self.group('c2', ['c2_fault']))
                self.section('c3', lambda: self.group('c3', ['c3_fault_0', 'c3_fault_1']))
                self.section('cross_group', self.cross_group)
        c2_sections = {'envelope', 'requests', 'candidates', 'c2'} | {p + '.trace' for p in PHASES[:6]}
        c2_valid = c2_sections.issubset(self.completed)
        c3_valid = c2_valid and {'c3', 'cross_group'}.issubset(self.completed) and all(
            p + '.trace' in self.completed for p in PHASES[6:])
        # Frozen source46 does not export issuance/use receipts. Do not accept a
        # newly invented JSON proof, copied dataclass or aggregate success flag.
        self.issues.append(dict(section='c3.live_c2_capability', code='C3_PROOF_ISSUANCE_USE_RECEIPT_MISSING'))
        return dict(schema='demoncat.rope_result_independent_review.v1',
            status='EVIDENCE_INCOMPLETE' if c3_valid else 'EVIDENCE_REJECTED',
            variant=VARIANT, actual_natural_targets=len(self.paths), checked_conditions=self.checks,
            c2_execution_evidence_valid=c2_valid, c3_execution_evidence_valid=c3_valid,
            c2_review_passed=c2_valid and not self.cpu, c3_review_passed=False,
            live_c2_capability_verified=False, public_qualification_granted=False,
            source_bound_c3_inference=('CONSISTENT_WITH_ORIGINAL_LIVE_C2_GATE_IN_FROZEN_11607CDA'
                if c3_valid and not self.cpu else 'NOT_ESTABLISHED'),
            source_bound_inference_is_capability_identity_observation=False,
            raw_device_payload_qualified=False, integer_overflow_qualified=False,
            cpu_contract=self.cpu, issues=self.issues,
            limits=['Recorded complete tensor digests and source-bound JIT callsite evidence; no device ABI payload.',
                    'No raw statistics acceptance. No live capability is reconstructed from JSON.'])


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


def read_result(path, expected_sha256):
    if not is_sha(expected_sha256):
        raise ValueError('EXPLICIT_RAW_RESULT_SHA256_REQUIRED')
    raw = Path(path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError('RAW_RESULT_SHA256_MISMATCH')
    def reject_constant(value):
        raise ValueError('NONFINITE_JSON_VALUE:' + value)
    return json.loads(raw, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant), actual


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--expected-targets', type=int)
    args = parser.parse_args(argv)
    if args.expected_targets is not None and args.expected_targets <= 0:
        parser.error('--expected-targets must be positive; the denominator still comes from readers')
    try:
        result, digest = read_result(args.result, args.expected_sha256)
        report = Review(result, expected_targets=args.expected_targets).run()
        report['result_sha256'] = digest
    except (OSError, ValueError, TypeError, RecursionError) as error:
        report = dict(schema='demoncat.rope_result_independent_review.v1', status='EVIDENCE_REJECTED',
            c2_review_passed=False, c3_review_passed=False, public_qualification_granted=False,
            issues=[dict(section='input', code=str(error)[:180])])
    print(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0 if report.get('c2_review_passed') and report.get('c3_review_passed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
