"""Frozen, standard-library checks for the explicit public39 custom profile.

This file reads saved metadata only. Shapes, allocations, ordinals and object
identities must come from actual request receipts, never schema examples.
It cannot issue a live capability, reload an injector or execute a kernel.
"""
import hashlib
import math
from pathlib import PurePosixPath


PROFILE = 'reviewed-custom-gemm-v1'
OPERATOR = 'vllm.unquantized_gemm.default'
SCHEMA = 'vllm::unquantized_gemm(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor'
SOURCES = {
    'vllm_ascend.patch.worker.patch_unquantized_gemm': '0317441758008fa74133b2ee7d3e3b2588d632d6522acfb6a8c98ee21554e5bf',
    'vllm.model_executor.layers.utils': '1d908a5cf4ff519fdd987b7a8798070f57510d9031f8f9c2e013149cdb060cad',
    'vllm.utils.torch_utils': 'b8c0a763d52c0eabb7b34c14b648b6f5e43f67d806f2819755b0ea23f4250221'}
TORCH_VERSION = '2.9.0+cpu'
TORCH_NPU_VERSION = '2.9.0.post1+gitee7ba04'
META_MODULE = 'torch_npu.op_plugin.meta._meta_registrations'
META_SHA = '3e3997c2b99dcff7a4ee4d693bd7e2365f6cf7e906dee5e08585081c7ea13df4'
RMS = 'npu.npu_rms_norm.default'
SWI = 'npu.npu_swiglu.default'
CONSUMERS = {
    RMS: dict(schema='npu::npu_rms_norm(Tensor input, Tensor gamma, float epsilon=9.9999999999999995e-07) -> (Tensor, Tensor)',
        function='npu_rms_norm_meta', signature='(Tensor _0, Tensor _1, float _2) -> (Tensor _0, Tensor _1)'),
    SWI: dict(schema='npu::npu_swiglu(Tensor input, int dim=-1) -> Tensor',
        function='npu_swiglu_meta', signature='(Tensor _0, int _1) -> Tensor _0')}
EXTRA_MODULES = ['fault_injection/deployments/postwrite_npu_consumers.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/gemm_lineage.py']


def positive(value):
    return type(value) is int and value > 0


def geometry(meta, *, dtype, device, contiguous=False):
    """Validate actual saved storage geometry, including the FP32 RMS rstd."""
    size = {'torch.bfloat16': 2, 'torch.float32': 4}[dtype]
    shape, stride = meta['shape'], meta['stride']
    if not shape or len(shape) != len(stride) or not all(positive(n) for n in shape): return False
    if not all(type(s) is int and s >= 0 for s in stride): return False
    offset = meta['storage_offset']
    if type(offset) is not int or offset < 0 or meta['dtype'] != dtype or meta['device'] != device: return False
    if not all(positive(meta[k]) for k in ('storage_identity', 'data_ptr', 'storage_bytes', 'logical_bytes')): return False
    if meta['logical_bytes'] != math.prod(shape)*size or meta['data_ptr']-offset*size <= 0 or meta['data_ptr'] % size: return False
    if (offset+sum((n-1)*s for n,s in zip(shape,stride))+1)*size > meta['storage_bytes']: return False
    if contiguous:
        step = 1
        for n,s in zip(reversed(shape),reversed(stride)):
            if n != 1 and s != step: return False
            step *= n
    return True


def private_lines(dispatch):
    # Include inactive entries: one active line does not establish the original
    # registration when Library.impl(..., allow_override=True) displaced it.
    return [line for line in dispatch.splitlines() if line.startswith('PrivateUse1')]


class ProjectionProfile:
    def __init__(self, legacy):
        self.legacy = legacy

    def __getattr__(self, name):
        return getattr(self.legacy, name)

    def producer_geometry(self, proof):
        if proof['operator'] != OPERATOR: return self.legacy.producer_geometry(proof)
        operands = proof['operands']
        if set(operands) != {'x', 'weight'}: return None
        x, weight = operands['x']['shape'], operands['weight']['shape']
        return x[:-1]+[weight[0]] if x and len(weight) == 2 and x[-1] == weight[1] else None

    def producer(self, audit, proof, output, where):
        implementation = proof['registered_implementation']
        audit.check(proof['operator'] == OPERATOR and proof['schema'] == SCHEMA
            and proof['schema_sha256'] == hashlib.sha256(SCHEMA.encode()).hexdigest()
            and proof['boundary'] == 'reviewed_vllm_custom_gemm_return'
            and proof['original_return_storage'] is True and proof['native_matmul_claim'] is False
            and proof['returned_tensor'] == output and proof['expected_output_shape'] == self.producer_geometry(proof) == output['shape']
            and set(proof['operands']) == {'x', 'weight'}, 'CUSTOM_ORIGINAL_GEOMETRY_AND_NO_BIAS', where)
        audit.check(implementation['schema'] == SCHEMA and implementation['sources'] == SOURCES
            and implementation['public_operator'] == OPERATOR
            and implementation['lowering'] == 'torch.nn.functional.linear(x, weight, None)'
            and implementation['registered_dispatch_key'] == 'PrivateUse1'
            and implementation['source_and_loaded_code_verified'] is True and implementation['owned_weight_verified'] is True
            and positive(implementation['module_weight_object_identity']) and positive(implementation['registered_library_object_identity'])
            and 'native_self' not in implementation and 'native_mat2' not in implementation,
            'CUSTOM_THREE_SOURCES_LOADED_CODE_AND_WEIGHT_OBJECT', where)
        for operand in proof['operands'].values():
            audit.check(geometry(operand, dtype='torch.bfloat16', device=output['device'])
                and operand['storage_identity'] != output['storage_identity'], 'CUSTOM_REAL_OPERAND_STORAGE', where)
        audit.check(geometry(output, dtype='torch.bfloat16', device=output['device'], contiguous=True)
            and self.legacy.geometry_valid(output, exclusive=True, device=output['device'].split(':')[0]),
            'CUSTOM_EXCLUSIVE_ORIGINAL_RETURN_SPAN', where)
        registration(audit, implementation, where)

    def consumers(self, audit, observations, producer, indices, names, where, device):
        self.legacy.consumers(audit, observations, producer, indices, names, where, device,
            operators=self.legacy.CONSUMERS | set(CONSUMERS))
        for row in observations:
            if row['operator'] in CONSUMERS:
                self.consumer(audit, row, producer, indices, where)
            else:
                audit.check(row.get('reviewed_consumer') is None, 'NO_FOREIGN_REVIEWED_CONSUMER_PROOF', where)

    def consumer(self, audit, row, producer, indices, where):
        operator = row['operator']; contract = CONSUMERS[operator]; proof = row['reviewed_consumer']
        schema = contract['schema']; lines = proof['actual_dispatch_dump'].splitlines()
        expected = ('PrivateUse1: registered at build/CMakeFiles/torch_npu.dir/compiler_depend.ts:8413 :: '
            + contract['signature'] + ' [ boxed unboxed ]')
        audit.check(proof['operator'] == operator and proof['schema'] == schema
            and proof['schema_sha256'] == hashlib.sha256(schema.encode()).hexdigest()
            and proof['torch_version'] == TORCH_VERSION and proof['torch_npu_version'] == TORCH_NPU_VERSION
            and proof['meta_shape_source'] == dict(module=META_MODULE, sha256=META_SHA, function=contract['function'])
            and 'schema: '+schema in lines and 'alias analysis kind: FROM_SCHEMA' in lines
            and private_lines(proof['actual_dispatch_dump']) == [expected]
            and proof['schema_has_no_alias_or_write'] is True and proof['original_returned_once'] is True,
            'CURRENT_READ_ONLY_NPU_CONSUMER_REGISTRATION', where)
        value = proof['input']; device = producer['device']; outputs = proof['outputs']
        audit.check(row['operand_views'] == [value] and len(value['shape']) == 2
            and geometry(value, dtype='torch.bfloat16', device=device, contiguous=True)
            and self.legacy.reads_indices(value, producer, indices), 'ACTUAL_NPU_INPUT_READS_SELECTED_STORAGE', where)
        tensors = [value]
        if operator == RMS:
            gamma = proof['gamma']; epsilon = proof['scalars'].get('epsilon')
            audit.check(proof['scalars'] == {'epsilon': epsilon} and type(epsilon) is float and math.isfinite(epsilon) and epsilon > 0
                and gamma['shape'] == [value['shape'][-1]] and geometry(gamma, dtype='torch.bfloat16', device=device, contiguous=True)
                and len(outputs) == 2 and outputs[0]['shape'] == value['shape']
                and outputs[1]['shape'] == value['shape'][:-1]+[1]
                and geometry(outputs[0], dtype='torch.bfloat16', device=device, contiguous=True)
                and geometry(outputs[1], dtype='torch.float32', device=device, contiguous=True),
                'ACTUAL_RMS_GAMMA_AND_TWO_RETURN_GEOMETRIES', where)
            tensors.append(gamma)
        else:
            audit.check(proof['scalars'] == {'dim': -1} and type(proof['scalars']['dim']) is int
                and value['shape'][-1] % 2 == 0 and 'gamma' not in proof and len(outputs) == 1
                and outputs[0]['shape'] == [value['shape'][0], value['shape'][1]//2]
                and geometry(outputs[0], dtype='torch.bfloat16', device=device, contiguous=True),
                'ACTUAL_SWIGLU_LAST_EVEN_DIM_RETURN_GEOMETRY', where)
        tensors += outputs
        spans = []
        for tensor in tensors:
            size = 4 if tensor['dtype'] == 'torch.float32' else 2
            start = tensor['data_ptr']-tensor['storage_offset']*size
            spans.append((start,start+tensor['storage_bytes']))
        audit.check(proof['formats'] == [2]*len(tensors) and all(type(v) is int for v in proof['formats'])
            and len({t['storage_identity'] for t in tensors}) == len(tensors)
            and all(left[1] <= right[0] or right[1] <= left[0]
                for i,left in enumerate(spans) for right in spans[i+1:]),
            'ACTUAL_ND_READ_ONLY_NONALIASED_CONSUMER_STORAGE', where)

    def inventory(self, audit, item, where, device):
        keys = self.legacy.inventory(audit, item, where, device,
            producers=self.legacy.PRODUCERS | {OPERATOR}, consumers=self.legacy.CONSUMERS | set(CONSUMERS))
        for index,row in enumerate(item['calls']):
            loc = str(where)+':'+str(index)
            if row['operator'] == OPERATOR:
                audit.check(row['schema'] == SCHEMA, 'EVERY_CUSTOM_CALL_EXACT_SCHEMA', loc)
                if row.get('producer_proof') is not None:
                    self.producer(audit, row['producer_proof'], row['output_geometry'], loc)
                if row['eligible']:
                    audit.check(row.get('producer_proof') is not None, 'ELIGIBLE_CUSTOM_HAS_ORIGINAL_PRODUCER_PROOF', loc)
            if row['eligible']:
                self.consumers(audit, row['consumer_observations'], row['output_geometry'], [0], row['consumers'], loc, device)
        return keys


def registration(audit, implementation, where):
    """Current dispatcher ownership is checked from the runtime's raw receipt."""
    saved = implementation['current_registration']; dump = implementation['actual_dispatch_dump']
    lines = dump.splitlines(); path = saved['source_path']; module = 'vllm.utils.torch_utils'
    # Line 789 is from the inspected file with this exact SHA, not a template
    # for unknown installations or a line inferred from a schema example.
    origin = path+':789'; active = 'PrivateUse1: registered at '+origin+' :: (none) [ boxed ]'
    definition = 'debug: registered at '+origin
    audit.check(saved['guard_profile'] == 'TORCH_2_9_SINGLE_PRIVATEUSE1_NO_SHADOW_SOURCE_ORIGIN_V1'
        and saved['torch_version'] == TORCH_VERSION and saved['source_module'] == module
        and saved['source_sha256'] == SOURCES[module]
        and PurePosixPath(path).is_absolute() and '..' not in PurePosixPath(path).parts
        and path.endswith('/vllm/utils/torch_utils.py')
        and saved['construction_lines'] == [789,789] and saved['actual_registration_line'] == 789
        and type(saved['shadowed_privateuse1_registrations']) is int and saved['shadowed_privateuse1_registrations'] == 0
        and saved['current_privateuse1_registration'] == active and saved['definition_registration'] == definition
        and private_lines(dump) == [active] and [v for v in lines if v.startswith('debug:')] == [definition]
        and [v for v in lines if v.startswith('schema:')] == ['schema: '+SCHEMA]
        and 'alias analysis kind: FROM_SCHEMA' in lines,
        'CURRENT_UNSHADOWED_PRIVATEUSE1_DEFINITION_AND_SOURCE_OWNER', where)


def deployment(audit, document, host, records):
    binding = document['binding']
    setups = [row for row in records if row['method'].endswith('.setup')]
    identities = [row for row in records if row['method'] == 'demoncat_worker_identity' and 'responses' in row]
    audit.check(isinstance(binding,dict) and binding['worker'] == host['worker']
        and len(setups) == 1 and setups[0].get('responses') == [binding]
        and bool(identities) and all(row['responses'][0]['worker'] == host['worker'] for row in identities)
        and all(name in binding['loaded_sources'] for name in EXTRA_MODULES),
        'CUSTOM_ORIGINAL_SETUP_AND_CURRENT_WORKER_LOADED_DEPENDENCIES')


def completed_c1(audit, profile, details):
    """Cross-request object, ordinal and Native span joins, after common C1."""
    rows = details['rows']; first = details['inventories'][0]
    selected = next(row for row in first['calls'] if row['eligible'])
    audit.check(selected['operator'] == OPERATOR, 'CUSTOM_PROFILE_SELECTED_ACTUAL_CUSTOM_GEMM')
    reference = selected['producer_proof']; binding = reference['registered_implementation']
    selection = details['selection']
    audit.check(selection['producer_proof'] == reference
        and selection['consumer_observations'] == selected['consumer_observations'], 'CUSTOM_SELECTION_ORIGINAL_COMPLETE_LINEAGE')
    baseline_keys = [profile.key(row) for row in first['calls']]
    native_threads = []
    for row in rows[:5]:
        phase = row['phase']; inventory = row['complete_inventory']
        audit.check([profile.key(call) for call in inventory['calls']] == baseline_keys
            and inventory['modules'] == first['modules'], 'CUSTOM_EVERY_PHASE_FULL_ACTUAL_ORDINAL_DENOMINATOR', phase)
        actual = next(call for call in inventory['calls'] if profile.key(call) == profile.key(selected))
        proof = actual['producer_proof']; current = proof['registered_implementation']
        audit.check(current == binding and proof['operands']['weight'] == reference['operands']['weight'],
            'CUSTOM_SAME_CURRENT_LIBRARY_AND_OWNED_WEIGHT_OBJECT', phase)
        audit.check(any(item['operator'] in CONSUMERS for item in actual['consumer_observations']),
            'CUSTOM_SELECTED_REAL_RMS_OR_SWIGLU_CONSUMPTION', phase)
        if phase in ('observe','observe_repeat'):
            audit.check(row['report'].get('native_lease_receipt') is None, 'CUSTOM_HEALTHY_OBSERVATIONS_OPEN_NO_NATIVE_LEASE', phase)
        if phase in ('baseline', 'fault', 'off'):
            target = row['report']['actual_target']
            audit.check(target['producer_proof'] == proof and target['consumer_observations'] == actual['consumer_observations'],
                'CUSTOM_SELECTED_RAW_ORDINAL_LINEAGE_EQUALS_ACTUAL_TARGET', phase)
            receipt = native_lease(audit, row['report'], proof['returned_tensor'], details['native_owner'], row['native_generation'], phase)
            if receipt is not None: native_threads.append(receipt['thread'])
    audit.check(len(native_threads) == 2 and len(set(native_threads)) == 1, 'CUSTOM_SAME_NATIVE_OWNER_THREAD')
    return dict(profile=PROFILE, selected_key=selected['key'], selected_operator=OPERATOR,
        full_projection_denominator=len(first['calls']), qualification_granted=False)


def native_lease(audit, report, geometry, owner, generation, phase):
    """Require actual performed ABI return receipts, never an invented zero."""
    receipt = report['native_lease_receipt']
    if phase == 'off':
        audit.check(receipt is None, 'CUSTOM_OFF_OPENS_NO_NATIVE_LEASE')
        return None
    audit.check(receipt['schema'] == 'demoncat.postwrite.lease_receipt.v1'
        and receipt['owner'] == owner and receipt['generation'] == generation
        and positive(receipt['handle']) and positive(receipt['thread'])
        and receipt['geometry'] == geometry and type(receipt['element_offset']) is int and receipt['element_offset'] == 0
        and type(receipt['elements']) is int and receipt['elements'] == math.prod(geometry['shape'])
        and receipt['closed'] is True and receipt['first_failure'] is None,
        'ACTUAL_NATIVE_ORIGINAL_SPAN_OWNER_GENERATION_AND_CLOSE', phase)
    events = receipt['events']; opened = receipt['open']; applied = receipt['applications'][0]
    audit.check([event['operation'] for event in events] == ['open','apply','restore','release','live']
        and [event['sequence'] for event in events] == list(range(1,6))
        and receipt['open'] == events[0] and receipt['applications'] == [events[1]]
        and receipt['restore'] == events[2] and receipt['release'] == events[3]
        and receipt['live_readbacks'] == [events[4]], 'ACTUAL_NATIVE_CALL_ORDER_AND_UNMODIFIED_EVENT_PROJECTIONS', phase)
    for event in events:
        audit.check(event['handle'] == receipt['handle'] and event['owner'] == owner and event['generation'] == generation
            and event['returned'] is True and type(event['return_code']) is int and event['return_code'] == 0
            and event['error'] is None, 'ACTUAL_NATIVE_SUCCESSFUL_EXACT_TICKET_RETURNS', phase+':'+event['operation'])
    device, separator, index = geometry['device'].rpartition(':')
    audit.check(device == 'npu' and separator == ':' and index.isdecimal()
        and type(opened['device']) is int and opened['device'] == int(index)
        and opened['data_ptr'] == geometry['data_ptr'] and opened['bytes'] == geometry['logical_bytes'] == geometry['storage_bytes']
        and opened['bytes'] == receipt['elements']*2, 'ACTUAL_NATIVE_OPEN_USES_EXACT_ORIGINAL_DEVICE_SPAN', phase)
    fault = phase == 'fault'; intervention = report['intervention']
    audit.check(type(applied['mode']) is int and applied['mode'] == (1 if fault else 0)
        and applied['indices'] == [0] and type(applied['changed_elements']) is int
        and applied['changed_elements'] == intervention['changed_elements'] == intervention['new_fault_hits'] == int(fault)
        and intervention['selected_elements'] == 1 and intervention['readback_exact'] is True,
        'ACTUAL_NATIVE_APPLY_MODE_SELECTED_INDEX_AND_EFFECT', phase)
    audit.check(events[-1]['native_live'] is False and events[-1]['return_code'] == 0,
        'ACTUAL_NATIVE_RELEASE_THEN_EXACT_TICKET_ABSENT_READBACK', phase)
    return receipt
