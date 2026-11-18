"""Read-only stdlib audit of actual frame/object/epoch records; no qualification."""
import argparse
import ast
from bisect import bisect_right
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path

try:
    from .audit_quant_source87_routes import arithmetic, digest
except ImportError:  # Direct CLI execution needs only this adjacent stdlib file.
    from audit_quant_source87_routes import arithmetic, digest


FRAME_SOURCE_SHA = '97cb26e3f18e00b1ec8fde88c86bc41412d83c2b7b64e21ecab99bf32f55ce33'
REQUIRED = ('rchain_prepare', 'route_pipeline', 'route_dispatch', 'rchain_builder', 'route_mlp',
            'rchain_entry', 'rchain_quant', 'rchain_cumsum', 'rchain_gmm1', 'rchain_gmm2',
            'route_combine', 'rchain_finalize')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def key(meta):
    return (meta['device'], meta['storage_address'], meta['storage_bytes'], meta['storage_identity'])


def without_epoch(value):
    if isinstance(value, list):
        return [without_epoch(item) for item in value]
    if isinstance(value, dict):
        return {name:without_epoch(item) for name,item in value.items() if name != 'write_epoch'}
    return value


class EpochLedger:
    """One linear scan and sparse histories, not a full map copy per event."""
    def __init__(self, events):
        self.history = {}
        require([event['sequence'] for event in events] == list(range(1, len(events) + 1)),
                'COMPLETE_ORDERED_DISPATCH_EVENTS_REQUIRED')
        for event in events:
            seq, changed = event['sequence'], set()
            for item in event.get('invalidated_storages', []):
                identity = (item['device'], item['address'], item['bytes'], item['storage_identity'])
                history = self.history.setdefault(identity, [])
                require(identity not in changed and type(item['write_epoch']) is int
                    and item['write_epoch'] == len(history) + 1, 'RAW_WRITE_EPOCH_SEQUENCE_INVALID')
                changed.add(identity)
                history.append(seq)

    def at(self, sequence, identity, *, after=False):
        return bisect_right(self.history.get(identity, ()), sequence if after else sequence - 1)


def literal_constants(path):
    values = {}
    def value(node):
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, (ast.Tuple, ast.List)):
            return [value(item) for item in node.elts]
        if isinstance(node, ast.Dict):
            return {value(k):value(v) for k,v in zip(node.keys,node.values)}
        raise ValueError('NON_LITERAL_RUNTIME_CONSTANT')
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = value(node.value)
            except (ValueError, KeyError):
                pass
    return values


def review_native_denominator(forward, public, public_calls, *, expected_calls=135):
    """Recount existing quant pass-through originals, without granting a lease."""
    rows = forward['native_events']
    before, after = forward['before'], forward['after']
    require(len(rows) == expected_calls*5 and after['last_sequence']-before['last_sequence'] == len(rows)
        and not forward['unowned_native_events'] and len(forward['candidates']) == expected_calls,
            'COMPLETE_RAW_NATIVE_DENOMINATOR')
    require(forward['forward_id'] == public['forward_id'] == forward['row_provenance']['forward_id'],
            'CURRENT_NATIVE_PUBLIC_FORWARD')
    public_indices = []
    for number,candidate in enumerate(forward['candidates']):
        group = rows[number*5:(number+1)*5]
        require([row['kind'] for row in group] == [1,2,2,2,2] and candidate['native'] == group[0],
                'ONE_RAW_PARENT_FOUR_ACTUAL_NATIVE_EVENTS')
        parent, native = group[0]['parent'], [row['native'] for row in group[1:]]
        start = before['last_sequence']+number*5+1
        require(parent['token'] == start and [row['sequence'] for row in native] == list(range(start+1,start+5))
            and [row['kind'] for row in native] == [1,2,3,4]
            and all(row['registration'] == start+1 for row in native), 'RAW_NATIVE_PREPARE_EXECUTE_REGISTRATION')
        require(parent['thread'] > 0 and parent['device'] == 0 and parent['reserved'] == 0
            and all(row['thread'] == parent['thread'] and row['device'] == parent['device']
                and row['status'] == row['reserved'] == 0 for row in native)
            and [row['original_calls'] for row in native] == [0,1,0,1]
            and native[1]['executor'] > 0 and len({row['executor'] for row in native[1:]}) == 1
            and all(not any(row['binding'].values()) for row in group[1:]), 'RAW_PASS_THROUGH_NATIVE_CALLS')
        require(all(row['smooth_tensor'] == row['group_index_tensor'] == row['offset_tensor'] == 0
            and row['destination_dtype'] == 2 for row in native), 'RAW_QUANT_OPTIONAL_FORMALS')
        index = candidate['public_api_index']; require(type(index) is int and 0 <= index < len(public_calls), 'ACTUAL_PUBLIC_OWNER_INDEX')
        current = public_calls[index]; public_indices.append(index)
        require(current['operator'] == 'npu_dynamic_quant' and current['original_calls'] == 1
            and current['forward_id'] == public['forward_id'] and current['request_ids'] == public['request_ids']
            and current['module_object_identity'] == candidate['module_object_identity']
            and not current['errors'] and not current['independent_native']['errors']
            and current['independent_native']['native_events'] == group, 'ACTUAL_PUBLIC_NATIVE_PARENT_OWNER')
        for name,label,dtype in (('input','float_input',27),('quant','quantized_output',2),('scale','per_token_scale',0)):
            actual, view = current[label], parent[name]
            shape, stride = actual['shape'], actual['stride']
            require(view['descriptor'] > 0 and view['reserved'] == 0 and view['dtype'] == dtype
                and view['format'] == actual['actual_format'] == 2 and actual['device'] == 'npu:0'
                and view['dimensions'] == len(shape) == len(stride) and len(shape) in (1,2)
                and view['address'] == actual['storage_addr'] and view['bytes'] == actual['storage_bytes']
                and view['offset'] == actual['storage_offset'] == 0
                and view['rows'] == shape[0] and view['columns'] == (shape[1] if len(shape) == 2 else 0)
                and view['stride0'] == stride[0] and view['stride1'] == (stride[1] if len(stride) == 2 else 0)
                and stride == ([shape[1],1] if len(shape) == 2 else [1])
                and actual['dtype'] == {27:'torch.bfloat16',2:'torch.int8',0:'torch.float32'}[dtype]
                and view['bytes'] == math.prod(shape)*actual['element_size']
                and all(row[name] == view for row in native), 'ACTUAL_RAW_PUBLIC_NATIVE_VIEW:' + name)
        require(native[2]['stream'] > 0 and native[2]['stream'] == native[3]['stream']
            and native[2]['workspace_capacity'] >= native[1]['workspace_required']
            and native[2]['workspace'] == native[3]['workspace']
            and native[2]['workspace_capacity'] == native[3]['workspace_capacity']
            and bool(native[2]['workspace']) == bool(native[2]['workspace_capacity']), 'ACTUAL_NATIVE_WORKSPACE_AND_STREAM')
    require(len(set(public_indices)) == expected_calls and len(public['projections']) == expected_calls,
            'UNIQUE_COMPLETE_PUBLIC_PROJECTION_DENOMINATOR')
    declared = [index for row in public['projections'] for index in row['api_call_indices']]
    require(len(set(declared)) == len(declared) == expected_calls*2
        and set(public_indices).issubset(declared), 'COMPLETE_PUBLIC_QUANT_AND_QMM_CALL_INDICES')
    for projection in public['projections']:
        selected = [public_calls[index] for index in projection['api_call_indices']]
        require(Counter(item['operator'] for item in selected) == {'npu_dynamic_quant':1,'npu_quant_matmul':1}
            and all(item['original_calls'] == 1 and not item['errors'] and item['forward_id'] == public['forward_id']
                and item['module_object_identity'] == projection['module_object_identity'] for item in selected),
                'ACTUAL_PROJECTION_PUBLIC_CALL_OWNERS')
    for state in (before,after):
        require(not any(state[name] for name in ('abi_exceptions','orphan_descriptors','pending','executing','quarantined'))
            and not any(state['census'][name] for name in ('errors','overflow','preparing','pending','executing')),
                'NATIVE_OBSERVER_ERROR_OR_LIVE_RESOURCES')
    for name in ('model_parents','native_prepares','native_executes'):
        require(after['census'][name]-before['census'][name] == expected_calls, 'RAW_NATIVE_COUNTER_DELTA:' + name)
    return dict(raw_model_parents=expected_calls,raw_prepare_calls=expected_calls,raw_execute_calls=expected_calls,
                raw_public_calls=len(declared),actual_native_events=len(rows))


def review_record(record, graph, boundary, integer, reviewed, schemas, *, ledger=None):
    """Recompute conditions from raw fields, never from completion booleans."""
    events = {event['sequence']:event for event in graph['events']}
    originals = {row['sequence']:row for row in boundary['boundaries']}
    ledger = ledger or EpochLedger(graph['events'])
    issues, checks = [], []
    def test(name, function):
        try:
            function()
            checks.append(name)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            issues.append(dict(check=name, reason=str(error)))
    frames = {}
    for item in record['frames']:
        frames.setdefault(item['role'], []).append(item)
    def frame(role):
        require(len(frames.get(role, [])) == 1, 'ONE_CURRENT_FRAME_REQUIRED:' + role)
        return frames[role][0]
    def tensor(packet):
        require(isinstance(packet, dict) and isinstance(packet.get('tensor'), dict), 'ACTUAL_TENSOR_PACKET_REQUIRED')
        return packet['tensor']
    def fields(packet, expected):
        require(packet['payload_type'] == expected and type(packet['object_identity']) is int,
                'EXACT_TYPED_PAYLOAD_REQUIRED:' + expected)
        return packet['fields']
    def equal(left, right, reason):
        require(left == right, reason)
    def source_frames():
        model = frame('rchain_model')
        require(set(frames) == set(REQUIRED + ('rchain_model',)) and len(record['frames']) == len(REQUIRED) + 1,
                'EXACT_REQUIRED_FRAME_SET')
        require(model['boundary_sequence'] == record['model_boundary_sequence']
            and model['model_path'] == record['model_path']
            and model['input']['self']['object_identity'] == record['owner_object_identity'], 'ACTUAL_MODEL_FRAME_OWNER')
        order = [frame(role)['boundary_sequence'] for role in REQUIRED]
        require(order == sorted(set(order)) and model['boundary_sequence'] < order[0], 'CURRENT_FRAME_ENTRY_ORDER')
        for row in record['frames']:
            original = originals[row['boundary_sequence']]
            require(row['original_calls'] == 1 and row['status'] == 'RETURNED' and original['status'] == 'OBSERVED',
                    'COMPLETE_ACTUAL_SOURCE_FRAME_REQUIRED')
            require((row['source_sha256'], row['function']) in reviewed[row['role']], 'PINNED_FRAME_SOURCE_OR_FUNCTION')
            require(all(row[k] == original[k] for k in ('role','function','source_sha256','model_path'))
                and row['operator_sequences'] == original['operator_sequences'], 'SOURCE_FRAME_ORIGINAL_JOIN')
            sequence = row['boundary_sequence']
            while sequence != model['boundary_sequence']:
                parent = originals[sequence]['parent_source_sequence']
                require(type(parent) is int and 0 < parent < sequence, 'CURRENT_MODEL_FRAME_ANCESTRY')
                sequence = parent
        equal(record['forward_id'], graph['forward_id'], 'CURRENT_FORWARD_REQUIRED')
        require(record.get('first_failure') is None and record.get('failed_operator_sequence') is None,
                'OBSERVED_FRAME_FAILURE_REMAINS_A_REFUSAL')
        require(all(record[name] is False for name in ('native_registration_proven','routed_output_row_proof','native_qualified')),
                'OBSERVATION_CANNOT_GRANT_NATIVE_AUTHORITY')
    test('source_frames_and_owner', source_frames)

    def integer_join():
        arithmetic(integer)
        link = record['integer_route_evidence']
        parent = events[integer['parent_sequence']]
        require(integer['forward_id'] == graph['forward_id'] and link['sha256'] == digest(integer)
            and parent['integer_route_evidence'] == link and parent['original_calls'] == 1
            and parent['status'] == 'OBSERVED' and parent['sequence'] in frame('rchain_model')['operator_sequences']
            and record['request_scope_sha256'] == integer['request_scope_sha256'],
                'EXACT_CURRENT_INTEGER_PARENT_DIGEST')
        stage_roles = (('topk','route_selector'), ('routing','route_dispatch'), ('cumsum','rchain_cumsum'),
                       ('gmm1','rchain_gmm1'), ('gmm2','rchain_gmm2'), ('abs','route_combine'), ('unpermute','route_combine'))
        sequences = [integer['stages'][stage]['sequence'] for stage,_ in stage_roles]
        require(sequences == sorted(set(sequences)) and parent['sequence'] < sequences[0], 'ALL_NATIVE_STAGE_ORDER')
        for stage, role in stage_roles:
            raw = integer['stages'][stage]; event = events[raw['sequence']]
            require(raw['original_calls'] == event['original_calls'] == 1 and event['status'] == 'OBSERVED'
                and raw['registration']['actual_schema'] == event['schema'] == schemas[stage]
                and raw['outputs'] == event['outputs'], 'ACTUAL_STAGE_SCHEMA_AND_RETURN:' + stage)
            require(set(raw['argument_origins']) == set(raw['formals'])
                and all(value in ('POSITIONAL','KEYWORD','CURRENT_SCHEMA_DEFAULT') for value in raw['argument_origins'].values()),
                    'ACTUAL_STAGE_ARGUMENT_ORIGINS:' + stage)
            source = raw['source']; original = originals[source['boundary_sequence']]
            require(original['status'] == 'OBSERVED' and raw['sequence'] in original['operator_sequences']
                and all(source[name] == original[name] for name in ('role','function','model_path','source_sha256'))
                and (source['source_sha256'],source['function']) in reviewed[source['role']],
                    'ACTUAL_PINNED_STAGE_SOURCE_FRAME:' + stage)
            if role != 'route_selector':
                require(raw['sequence'] in frame(role)['operator_sequences'], 'ACTUAL_STAGE_FRAME:' + stage)
        for stage, role in (('gmm1','rchain_gmm1'), ('gmm2','rchain_gmm2'), ('unpermute','route_combine')):
            raw = integer['stages'][stage]; event = events[raw['sequence']]; current = frame(role)
            require(raw['original_calls'] == event['original_calls'] == 1 and event['status'] == 'OBSERVED' and event['schema'] == schemas[stage]
                and raw['registration']['actual_schema'] == event['schema'] and raw['outputs'] == event['outputs'],
                    'ACTUAL_NATIVE_STAGE_SCHEMA_AND_RETURN:' + stage)
            equal(record['actual_stage_frame_links'][stage], dict(operator_sequence=raw['sequence'],
                source_boundary_sequence=current['boundary_sequence']), 'ACTUAL_STAGE_FRAME_REFERENCE:' + stage)
            require(raw['sequence'] in current['operator_sequences'], 'NATIVE_CALL_OUTSIDE_CURRENT_FRAME:' + stage)
        return parent
    test('integer_arithmetic_parent_and_native_stage_refs', integer_join)

    def integer_snapshots():
        helpers = [item for item in boundary['integer_routes']['helper_calls']
                   if item['parent_sequence'] == integer['parent_sequence']]
        helper_index = {item['snapshot']:item for item in helpers}
        producers = dict(topk_ids=('topk',1), raw_indices=('routing',1), expert_counts=('routing',2),
                        cumulative_counts=('cumsum',0), absolute_indices=('abs',0))
        require(len(helpers) == len(helper_index) == 5 and set(helper_index) == set(producers),
                'ALL_FIVE_UNIQUE_INTEGER_HELPERS')
        for name,(stage,index) in producers.items():
            snapshot, raw, helper = integer['snapshots'][name], integer['stages'][stage], helper_index[name]
            equal(snapshot['metadata_before'], raw['outputs'][index], 'ACTUAL_INTEGER_PRODUCER:' + name)
            require(snapshot['sequence'] == raw['sequence']
                and snapshot['epoch_before'] == snapshot['epoch_after'] == ledger.at(raw['sequence'],key(snapshot['metadata_before']),after=True),
                    'CURRENT_INTEGER_SNAPSHOT_EPOCH:' + name)
            require(helper['forward_id'] == graph['forward_id'] and helper['model_sequence'] == raw['sequence']
                and helper['source'] == snapshot['metadata_before'] and helper['bytes'] == snapshot['bytes']
                and helper['classification'] == 'OBSERVER_INTEGER_READBACK_NOT_MODEL_INVOCATION'
                and helper['status'] == 'COMPLETE' and helper['cleanup'] == 'REMOVED' and helper['overflow'] == 0,
                    'EXACT_SEPARATE_INTEGER_HELPER:' + name)
            operations = helper['operations']
            require(helper['total_dispatches'] == len(operations) == 3
                and [item['operator'] for item in operations] == ['aten._to_copy.default','aten.resolve_conj.default','aten.resolve_neg.default']
                and all(item['original_calls'] == 1 and item['status'] == 'RETURNED' for item in operations)
                and operations[0]['inputs'] == [snapshot['metadata_before']]
                and all(meta['device'] == 'cpu' and meta['dtype'] == snapshot['dtype']
                    for item in operations for meta in item['outputs']), 'INTEGER_HELPER_RAW_DISPATCHES:' + name)
    test('raw_integer_snapshots_epochs_and_separate_helpers', integer_snapshots)

    def builder():
        build = frame('rchain_builder'); dispatch = frame('route_dispatch')
        d = fields(build['input']['token_dispatch_output'], 'MoETokenDispatchOutput')
        equal(build['input']['token_dispatch_output'], dispatch['output'], 'ACTUAL_DISPATCH_TO_BUILDER')
        fused = fields(build['input']['fused_experts_input'], 'MoEFusedExpertsInput')
        equal(build['input']['fused_experts_input'], frame('route_pipeline')['input']['fused_experts_input'], 'ACTUAL_FUSED_PAYLOAD')
        dispatched = fields(dispatch['input']['token_dispatch_input'], 'MoETokenDispatchInput')
        for name in ('hidden_states','topk_ids','topk_weights','routing','quant'):
            equal(dispatched[name], fused[name], 'ACTUAL_DISPATCH_INPUT:' + name)
        result = fields(build['output'], 'MoEMlpComputeInput')
        for name in ('hidden_states','group_list','dynamic_scale','topk_scales','group_list_type'):
            equal(result[name], d[name], 'BUILDER_FIELD_COPY:' + name)
        for name in ('weights','quant'):
            equal(result[name], fused[name], 'BUILDER_CURRENT_OWNER:' + name)
        require(result['fusion'] is True and build['input']['use_fusion_ops'] is True, 'ACTUAL_FUSION_BOOL')
        for role in ('route_mlp','rchain_entry'):
            equal(frame(role)['input']['mlp_compute_input'], build['output'], 'CURRENT_BUILT_PAYLOAD:' + role)
        quant = frame('rchain_quant')['input']
        for name in ('hidden_states','group_list','group_list_type','dynamic_scale'):
            equal(quant[name], result[name], 'ACTUAL_QUANT_ARGUMENT:' + name)
        weights = fields(result['weights'], 'MoEWeights')
        for name in ('w1','w2','w1_scale','w2_scale','w1_scale_bias','w2_scale_bias','w1_offset','w2_offset'):
            equal(quant[name], weights[name], 'ACTUAL_WEIGHT_ARGUMENT:' + name)
        require(quant['fusion'] is True and quant['dynamic_eplb'] is False and quant['use_mxfp_quant'] is False
            and quant['group_list_type'] == 1 and quant['dynamic_scale'] is not None
            and all(quant[name] is None for name in ('w1_scale_bias','w2_scale_bias','w1_offset','w2_offset')),
                'OBSERVED_W8A8_BRANCH_ARGUMENTS')
        prepared = fields(frame('rchain_prepare')['output'], 'MoEPrepareOutput')
        equal(prepared['hidden_states'], frame('rchain_prepare')['input']['hidden_states'], 'IDENTITY_PREPARE_RETURN')
        equal(prepared['hidden_states'], fused['hidden_states'], 'ACTUAL_PREPARE_TO_PIPELINE')
        cumulative = frame('rchain_cumsum')
        equal(cumulative['input']['group_list'], quant['group_list'], 'CURRENT_CUMULATIVE_SOURCE')
        require(type(cumulative['input']['src_list_type']) is int and cumulative['input']['src_list_type'] == 1
            and type(cumulative['input']['dst_list_type']) is int and cumulative['input']['dst_list_type'] == 0,
                'EXACT_CUMULATIVE_CONVERSION')
        raw = integer['stages']
        equal(tensor(cumulative['input']['group_list']), raw['cumsum']['formals']['self']['tensor'], 'RAW_CUMULATIVE_INPUT')
        equal([tensor(cumulative['output'])], raw['cumsum']['outputs'], 'RAW_CUMULATIVE_RETURN')
        equal(tensor(dispatched['hidden_states']), raw['routing']['formals']['x']['tensor'], 'RAW_DISPATCH_INPUT')
        equal(tensor(dispatched['topk_ids']), raw['routing']['formals']['expert_idx']['tensor'], 'RAW_EXPERT_INDEX_INPUT')
        for name, position in (('hidden_states',0),('group_list',2),('dynamic_scale',3)):
            equal(tensor(d[name]), raw['routing']['outputs'][position], 'RAW_DISPATCH_OUTPUT:' + name)
        combine = fields(d['combine_metadata'], 'MoEAllGatherCombineMetadata')
        equal(frame('route_combine')['input']['combine_metadata'], d['combine_metadata'], 'CURRENT_COMBINE_PAYLOAD')
        equal(tensor(combine['expanded_row_idx']), raw['routing']['outputs'][1], 'RAW_COMBINE_INVERSE_INDEX')
        equal(tensor(combine['topk_weights']), raw['topk']['outputs'][0], 'RAW_COMBINE_PROBABILITY')
        equal(tensor(combine['expanded_row_idx']), raw['abs']['formals']['self']['tensor'], 'RAW_ABSOLUTE_INDEX_INPUT')
        equal(raw['unpermute']['formals']['sorted_indices']['tensor'], raw['abs']['outputs'][0], 'RAW_UNPERMUTE_INDEX')
        equal(raw['unpermute']['formals']['probs']['tensor'], tensor(combine['topk_weights']), 'RAW_UNPERMUTE_PROBABILITY')
    test('typed_builder_mlp_payloads', builder)

    def native_objects():
        stages = integer['stages']
        for stage, role, names in (('gmm1','rchain_gmm1',('x','group_list','weight','weight_scale','x_scale')),
                                  ('gmm2','rchain_gmm2',('hidden_states','per_token_scale','group_list'))):
            raw, current = stages[stage], frame(role)
            for name in names:
                operand = current['input'][name]
                actual = (raw['formals']['x'][0] if name == 'hidden_states' else
                          raw['formals']['per_token_scale'][0] if name == 'per_token_scale' else raw['formals'][name])
                equal(tensor(operand), tensor(actual), 'RAW_CURRENT_NATIVE_INPUT:' + stage + ':' + name)
                equal(operand['write_epoch'], ledger.at(raw['sequence'],key(tensor(operand))), 'NATIVE_INPUT_WRITE_EPOCH')
            returns = current['output'] if stage == 'gmm1' else [current['output']]
            equal([tensor(item) for item in returns], raw['outputs'], 'RAW_NATIVE_RETURN:' + stage)
        one, two = frame('rchain_gmm1'), frame('rchain_gmm2')
        quant = frame('rchain_quant')['input']
        def single(value):
            return value[0] if isinstance(value,list) and len(value) == 1 else value
        for name, argument in (('x','hidden_states'),('x_scale','dynamic_scale'),('weight','w1'),('weight_scale','w1_scale')):
            equal(one['input'][name], single(quant[argument]), 'CURRENT_GMM1_QUANT_ARGUMENT:' + name)
        equal(one['input']['group_list'], frame('rchain_cumsum')['output'], 'CURRENT_GMM1_CUMULATIVE_GROUP')
        for name, argument in (('weight','w2'),('weight_scale','w2_scale'),('group_list','group_list')):
            equal(two['input'][name], quant[argument], 'CURRENT_GMM2_QUANT_ARGUMENT:' + name)
        for name, formal in (('weight','weight'),('weight_scale','scale')):
            equal(without_epoch(two['input'][name]), stages['gmm2']['formals'][formal], 'RAW_GMM2_ARGUMENT:' + name)
        require(one['input']['bias'] is two['input']['bias'] is None
            and one['input']['use_mxfp_quant'] is two['input']['use_mxfp_quant'] is False
            and two['input']['group_list_type'] == 1, 'CURRENT_DEVICE_WRAPPER_FLAGS')
        require(stages['gmm1']['formals']['bias'] is stages['gmm1']['formals']['offset'] is None
            and all(stages['gmm2']['formals'][name] == value for name,value in dict(split_item=2,group_type=0,
                group_list_type=1,act_type=0,bias=None,offset=None,antiquant_scale=None,antiquant_offset=None,
                activation_input=None,activation_quant_scale=None,activation_quant_offset=None,tuning_config=None).items())
            and stages['unpermute']['formals']['padded_mode'] is False
            and stages['unpermute']['formals']['restore_shape'] is None, 'ACTUAL_DEVICE_AND_UNPERMUTE_OPTIONS')
        equal(two['input']['hidden_states'], one['output'][0], 'ACTUAL_GMM1_TO_GMM2')
        equal(two['input']['per_token_scale'], one['output'][1], 'ACTUAL_GMM1_SCALE_TO_GMM2')
        for role in ('rchain_quant','rchain_entry','route_mlp'):
            equal(frame(role)['output'], two['output'], 'ACTUAL_MLP_RETURN:' + role)
        combine = frame('route_combine')
        equal(combine['input']['hidden_states'], two['output'], 'ACTUAL_GMM2_TO_COMBINE')
        equal(tensor(combine['input']['hidden_states']), stages['unpermute']['formals']['permuted_tokens']['tensor'],
              'RAW_CURRENT_UNPERMUTE_INPUT')
        equal(tensor(combine['output']), stages['unpermute']['outputs'][0], 'RAW_UNPERMUTE_RETURN')
        final = frame('rchain_finalize')
        equal(final['input']['hidden_states'], combine['output'], 'ACTUAL_COMBINE_TO_FINALIZE')
        equal(final['output'], combine['output'], 'IDENTITY_FINALIZE_RETURN')
        public = events[integer['parent_sequence']]['outputs']
        equal([tensor(item) for item in record['public_return']], public, 'RAW_PUBLIC_TUPLE_RETURN')
        equal(tensor(final['output']), public[1], 'FINALIZE_TO_EXACT_ROUTED_TUPLE_MEMBER')
    test('native_inputs_outputs_and_current_epochs', native_objects)

    def consumers():
        rows = record['consumers']; require(len(rows) == 2, 'TWO_POST_ROUTE_CONSUMERS_REQUIRED')
        parent = events[integer['parent_sequence']]
        held = {'shared':parent['outputs'][0], 'routed':parent['outputs'][1]}
        cursor = parent['sequence']
        for number, row in enumerate(rows):
            seq = row['sequence']; event = events[seq]
            require(seq > cursor and seq in frame('rchain_model')['operator_sequences'], 'CURRENT_MODEL_CONSUMER_ORDER')
            for at in range(cursor + 1, seq):
                alias = events[at]
                if alias['operator'] != 'aten.alias.default':
                    continue
                for name, meta in list(held.items()):
                    if alias['inputs'] == [meta]:
                        require(alias['original_calls'] == 1 and alias['status'] == 'OBSERVED'
                            and len(alias['outputs']) == 1, 'ACTUAL_ALIAS_SINGLE_RETURN')
                        result = alias['outputs'][0]
                        require(all(meta[k] == result[k] for k in meta if k != 'object_id'), 'ACTUAL_ALIAS_EXTENT_CHANGED')
                        held[name] = result
            args = row['formals']; target = tensor(args['self'])
            equal(target, held['routed'], 'CURRENT_ROUTED_ALIAS_CONSUMER')
            require(row['schema'] == event['schema'] and row['original_calls'] == event['original_calls'] == 1
                and event['status'] == 'OBSERVED',
                    'RAW_CONSUMER_SCHEMA_AND_CALL')
            require(row['scalar_owner_object_identity'] == record['owner_object_identity'], 'CURRENT_SCALAR_OWNER')
            require(row['status'] == 'RETURNED' and row['admission_granted'] is row['request_rows_proven'] is False
                and set(row['argument_origins']) == set(args)
                and all(origin in ('POSITIONAL','KEYWORD','CURRENT_SCHEMA_DEFAULT') for origin in row['argument_origins'].values()),
                    'CONSUMER_OBSERVATION_AND_ACTUAL_ARGUMENT_ORIGINS')
            if number == 0:
                require(event['operator'] in ('aten.mul_.Tensor','aten.mul_.Scalar')
                    and type(args['other']) in (int,float) and math.isfinite(args['other'])
                    and type(row['scalar_config']) in (int,float) and args['other'] == row['scalar_config'],
                        'ACTUAL_SCALAR_AND_CURRENT_CONFIG')
                equal(event['inputs'], [target], 'SCALAR_HAS_NO_UNOBSERVED_TENSOR_OPERAND')
            else:
                require(event['operator'] == 'aten.add_.Tensor' and args['alpha'] == 1, 'ACTUAL_ADD_OPERATOR_AND_ALPHA')
                equal(tensor(args['other']), held['shared'], 'ACTUAL_SHARED_ADDITION_OPERAND')
                equal(event['inputs'], [target,held['shared']], 'RAW_ADD_OPERANDS')
            equal(row['target_before']['tensor'], target, 'CURRENT_CONSUMER_BEFORE_TENSOR')
            equal(args['self'], row['target_before'], 'CURRENT_CONSUMER_FORMAL_EPOCH')
            equal(row['target_before']['write_epoch'], ledger.at(seq,key(target)), 'CONSUMER_BEFORE_EPOCH')
            equal(row['observed_invalidations'], event['invalidated_storages'], 'RAW_ALL_ALIAS_INVALIDATIONS')
            equal(row['target_after']['write_epoch'], ledger.at(seq,key(target),after=True), 'CONSUMER_AFTER_EPOCH')
            require(row['target_after']['write_epoch'] == row['target_before']['write_epoch'] + 1,
                    'EXACT_ONE_OBSERVED_WRITE_EPOCH')
            equal(event['outputs'], [row['target_after']['tensor']], 'RAW_INPLACE_RETURN')
            require(all(value == row['target_after']['tensor'][name] for name,value in target.items() if name != 'version'),
                    'INPLACE_GEOMETRY_OR_OBJECT_CHANGED')
            held['routed'] = row['target_after']['tensor']; cursor = seq
        view = record['final_view']; event = events[view['sequence']]
        require(view['sequence'] > cursor and event['operator'] == 'aten.view.default'
            and event['original_calls'] == view['original_calls'] == 1 and event['status'] == 'OBSERVED'
            and view['sequence'] in frame('rchain_model')['operator_sequences'], 'ACTUAL_FINAL_VIEW_CALL')
        equal(view['input'], held['routed'], 'CURRENT_FINAL_VIEW_INPUT_PACKET')
        equal(event['inputs'], [held['routed']], 'ACTUAL_FINAL_VIEW_INPUT')
        equal(event['outputs'], [view['output']], 'ACTUAL_FINAL_VIEW_OUTPUT')
        equal(tensor(frame('rchain_model')['output']), view['output'], 'ACTUAL_MODEL_RETURN_OBJECT')
        require(all(held['routed'][name] == view['output'][name] for name in
            ('storage_address','storage_identity','storage_bytes','offset','shape','stride','dtype','device'))
            and view['output']['shape'] == tensor(frame('rchain_model')['input']['hidden_states'])['shape'],
                'ACTUAL_FINAL_VIEW_EXTENT')
    test('raw_scalar_consumers_aliases_and_epoch_replay', consumers)

    # This result intentionally has no Native qualification field that can be true.
    return dict(model_path=record['model_path'], model_boundary_sequence=record['model_boundary_sequence'],
        checked_sections=checks, issues=issues, independently_complete=not issues,
        reported_first_failure=record.get('first_failure'),
        reported_input_rows_verified=record.get('input_request_rows',{}).get('verified',False),
        reported_input_rows_reason=record.get('input_request_rows',{}).get('reason'), native_row_authority=False)


def audit(inventory_path, inventory_sha, result_path, result_sha, source_root):
    packed = Path(inventory_path).read_bytes()
    raw = gzip.decompress(packed) if packed.startswith(b'\x1f\x8b') else packed
    result_raw = Path(result_path).read_bytes()
    require(sha(raw) == inventory_sha and sha(result_raw) == result_sha, 'EXPLICIT_ORIGINAL_SHA_MISMATCH')
    data, result = json.loads(raw), json.loads(result_raw)
    worker = source_root/'src/demoncat/runtime/vllm_ascend/worker_extension'
    require(sha((worker/'quant_route_frames.py').read_bytes()) == FRAME_SOURCE_SHA, 'FROZEN_FRAME_OBSERVER_PROFILE_REQUIRED')
    sources = []
    for item in result['setup']['sources']:
        relative = item['path'].removeprefix('/installed/')
        local = source_root/('src/'+relative if relative.startswith('demoncat/') else relative)
        require(local.resolve().is_relative_to(source_root.resolve()), 'SOURCE_PATH_OUTSIDE_FROZEN_ROOT')
        require(sha(local.read_bytes()) == item['sha256'], 'ACTUAL_INSTALLED_SOURCE_SHA:' + relative)
        sources.append(dict(path=relative, sha256=item['sha256']))
    require(len({item['path'] for item in sources}) == len(sources) == 14
        and any(item['sha256'] == FRAME_SOURCE_SHA for item in sources), 'EXACT_LOADED_OBSERVER_SOURCE_CLOSURE_REQUIRED')
    reviewed, reviewed_modules = {}, {}
    for filename in ('quant_route_frames.py','quant_token_routes.py'):
        for module, (source_sha, functions) in literal_constants(worker/filename)['REVIEWED'].items():
            reviewed_modules[module] = (source_sha,functions)
            for function, role in functions.items():
                reviewed.setdefault(role, set()).add((source_sha, function))
    schemas = literal_constants(worker/'quant_token_routes.py')['SCHEMAS']
    require(len(data['native_forwards']) == len(data['forwards']) == 2
        and data['structural_candidate_count'] == len(data['structural_candidates']) == 135
        and data['actual_projection_calls'] == data['actual_native_calls'] == data['actual_native_prepare_calls']
            == data['actual_native_execute_calls'] == 270 and data['actual_public_api_calls'] == 540
        and data['native_public_binding_complete'] is True and data['complete_healthy_replays'] is True
        and data['failure'] is None,
            'COMPLETE_TWO_FORWARD_135_TARGET_DENOMINATOR_REQUIRED')
    require(result['healthy_before'] == result['healthy_after'] and result['integer_route_capture'] is True
        and result['statistics'] is False and not any(result[k] for k in ('c1','c2','c3')) and data['group_owner_installed'] is False,
            'ZERO_FAULT_HEALTHY_CENSUS_REQUIRED')
    cleanup = {k:v for k,v in result['remove'].items() if k.startswith('owned_') or k in ('installed','observer_cleanup_pending','group_owner_installed','native_faults')}
    cleanup_fields = ('installed','observer_cleanup_pending','group_owner_installed','native_faults','owned_hooks',
        'owned_method_patches','owned_api_patches','owned_row_dispatch_modes','owned_integer_route_parent_refs',
        'owned_integer_route_tensor_refs','owned_source_profile_callbacks','owned_early_dispatch_modes','owned_nested_dispatch_modes',
        'owned_scheduler_input_refs','owned_live_metadata_records','owned_routed_frame_parents','owned_routed_frame_tensor_holds',
        'owned_routed_frame_payload_holds')
    require(set(cleanup_fields).issubset(cleanup) and all(type(value) in (int,bool) and value == 0 for value in cleanup.values()),
            'ALL_OWNED_OBSERVER_RESOURCES_NOT_RELEASED')
    require(len({row['row_provenance']['forward_id'] for row in data['native_forwards']}) == 2, 'TWO_DISTINCT_ACTUAL_FORWARDS')
    output, forward_summaries = [], []
    for index,forward in enumerate(data['native_forwards']):
        graph = forward['row_provenance']; boundary = graph['source_boundaries']; frames = boundary['routed_call_frames']
        integers = boundary['integer_routes']['records']
        native_counts = review_native_denominator(forward,data['forwards'][index],data['public_calls'])
        for module,(source_sha,functions) in reviewed_modules.items():
            current = boundary['sources'][module]
            require(current['sha256'] == source_sha and all(current['functions'][name]['role'] == role
                for name,role in functions.items()), 'ACTUAL_LOADED_PRODUCTION_SOURCE_BINDING:' + module)
        require(graph['missing_calls'] == 0 and not graph['errors'] and not boundary['errors']
            and graph['actual_calls'] == graph['retained_calls'] == len(graph['events'])
            and len(integers) == 26 and len(forward['candidates']) == len(data['forwards'][index]['projections']) == 135
            and forward['complete_native_denominator'] is True
            and all(item['qualified'] is False for item in forward['candidates']), 'FULL_RAW_DENOMINATOR_REQUIRED')
        require(len({item['sequence'] for item in boundary['boundaries']}) == len(boundary['boundaries']),
                'DUPLICATE_RAW_SOURCE_BOUNDARY_SEQUENCE')
        require(frames['total_parents'] == len(frames['records']) == 26 and frames['overflow'] == 0
            and frames['live_parents'] == frames['live_tensor_holds'] == frames['live_payload_holds'] == 0,
                'COMPLETE_BOUNDED_FRAME_DENOMINATOR_REQUIRED')
        require(frames['tensor_readbacks'] == frames['graph_row_bindings'] == 0
            and all(frames[name] is False for name in ('native_registration_proven','routed_output_row_proof','native_qualified','c1','c2','c3'))
            and 0 < frames['max_parents'] <= 32 and 0 < frames['max_frames_per_parent'] <= 48
            and 0 < frames['max_metadata_bytes_per_parent'] <= 512*1024
            and frames['metadata_bytes'] == sum(row['metadata_bytes'] for row in frames['records'])
            and all(0 < row['metadata_bytes'] <= frames['max_metadata_bytes_per_parent']
                and len(row['frames']) <= frames['max_frames_per_parent'] for row in frames['records']), 'BOUNDED_OBSERVATION_NOT_ROW_ADMISSION')
        require({row['model_path'] for row in frames['records']} == {'model.layers.'+str(i)+'.mlp' for i in range(1,27)}
            and len({row['model_boundary_sequence'] for row in frames['records']}) == 26
            and len({row['parent_sequence'] for row in integers}) == 26, 'EXACT_COMPLETE_MOE_MODEL_SET_REQUIRED')
        links = [row['integer_route_evidence']['record_index'] for row in frames['records']
                 if isinstance(row.get('integer_route_evidence'),dict)]
        require(len(links) == len(set(links)), 'INTEGER_PARENT_REUSED_FOR_MULTIPLE_MODEL_FRAMES')
        census = boundary['integer_routes']
        require(census['total_parents'] == len(integers) == 26 and census['overflow'] == 0
            and len(census['helper_calls']) == 130 and census['helper_dispatch_overflow'] == 0
            and census['readback_bytes'] == census['retained_integer_payload_bytes'] ==
                sum(snapshot['bytes'] for row in integers for snapshot in row['snapshots'].values()) == 41600
            and census['helper_operator_calls'] == census['helper_total_dispatches'] ==
                sum(item['total_dispatches'] for item in census['helper_calls']) == 390
            and not any(census[name] for name in ('live_parent_refs','live_tensor_refs','retained_helper_modes','helper_active')),
                'EXACT_SEPARATE_INTEGER_BUDGET_AND_CLEANUP')
        ledger = EpochLedger(graph['events'])
        for record in frames['records']:
            link = record.get('integer_route_evidence')
            integer = (integers[link['record_index']] if isinstance(link,dict) and type(link.get('record_index')) is int
                and 0 <= link['record_index'] < len(integers) else None)
            if integer is None:
                # Missing links remain explicit missing evidence. Never recover
                # them by storage address or choose a convenient other parent.
                output.append(dict(model_path=record['model_path'], independently_complete=False,
                    issues=[dict(check='current_integer_parent',reason='MISSING_EXACT_PARENT_REFERENCE')],
                    reported_first_failure=record.get('first_failure'), native_row_authority=False))
            else:
                output.append(review_record(record,graph,boundary,integer,reviewed,schemas,ledger=ledger))
        forward_summaries.append(dict(index=index,forward_id=graph['forward_id'],actual_graph_calls=len(graph['events']),
            unknown_edges=len(graph['unknown_edges']),projection_candidates=135,
            reported_row_proven=sum(row['output_row_lineage']['verified'] is True for row in data['forwards'][index]['projections']),
            reported_input_rows_verified=sum(row.get('input_request_rows',{}).get('verified') is True for row in frames['records']),
            raw_native_denominator=native_counts,
            actual_integer_parents=26,integer_snapshot_bytes=41600,separate_helper_dispatches=390,
            frame_callbacks=frames['callbacks'],frame_observer_ns=frames['observer_ns'],frame_metadata_bytes=frames['metadata_bytes'],
            per_parent_metadata_bytes_min=min(row['metadata_bytes'] for row in frames['records']),
            per_parent_metadata_bytes_max=max(row['metadata_bytes'] for row in frames['records']),
            frame_timing_scope=frames['timing_scope'],integer_observer_ns=census['observer_ns'],
            integer_readback_ns=census['readback_ns'],source_profile_ns=census['source_profile_ns'],timings_overlap=True))
    return dict(schema='demoncat.quant.routed_frames_independent_review.v1',
        status='RAW_FRAMES_AUDITED_NO_NATIVE_ROW_AUTHORITY',
        inputs=dict(inventory_sha256=sha(raw),result_sha256=sha(result_raw)), sources=sources,
        actual_projection_calls=270, candidates_per_forward=135, observed_model_parents=len(output),
        independently_complete_python_chains=sum(item['independently_complete'] for item in output),
        refused_or_incomplete_python_chains=sum(not item['independently_complete'] for item in output),
        records=output, forwards=forward_summaries, cleanup=cleanup, native_registration_proven=False, routed_output_row_proof=False,
        native_qualified=False, c1=False, c2=False, c3=False, qualification_granted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('inventory','result','source-root','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('inventory-sha256','result-sha256'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        value=audit(args.inventory,args.inventory_sha256,args.result,args.result_sha256,args.source_root)
        code=0
    except (ValueError,KeyError,TypeError,IndexError,OSError) as error:
        value=dict(status='RAW_ARTIFACT_AUDIT_FAILED',reason=str(error),qualification_granted=False)
        code=2
    with args.out.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=value['status'],summary_sha256=sha(args.out.read_bytes()),qualification_granted=False)))
    raise SystemExit(code)


if __name__=='__main__':main()
