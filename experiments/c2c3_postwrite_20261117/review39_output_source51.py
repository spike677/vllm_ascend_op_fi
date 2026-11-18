"""Independent, read-only MODEL_LOCAL 39.output_proxy source51 review.

No runtime imports, statistics/E5 comparison, mechanism execution or authority.
Embedded evidence is reconciled independently; absent raw bytes remain absent.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

SOURCE_COMMIT = '3ec12a7bc3bd5daebc404c70342083238e075a00'
MANIFEST_SHA = '86878d43bc63dc1572b9d571c02def8a4061da66fd66153198748ac3e390866f'
NATIVE_SHA = '8876f2b99e70a9df8201b6ae0ace2cc4be8702afcf6703f9b72d1dda3bc26622'
BASELINE_SHA = 'e88e3f3f248a7f07e62e76a108c28a87a05342ba278de2031ed1a8e9f779cf01'
FAULT_SHA = 'e9cc8a8c0875bc8136b76d86531c5e4b01aa9011da5b200e7865956d064465d0'
DEFAULT_PROMPT = 'Explain what a checksum is in one sentence.'
PRODUCERS = {'aten.mm.default', 'aten.linear.default', 'aten.matmul.default'}
CONSUMERS = {'aten.add.Tensor', 'aten.mul.Tensor', 'aten.mm.default', 'aten.bmm.default', 'aten.addmm.default',
    'aten.sum.dim_IntList', 'aten.linear.default', 'aten.matmul.default', 'aten._softmax.default',
    'aten.softmax.int', 'aten.silu.default', 'aten.gelu.default'}
GEOMETRY = ('shape', 'stride', 'storage_offset', 'dtype', 'device', 'storage_bytes', 'logical_bytes')
SOURCES = ('fault_injection/deployments/postwrite.py', 'fault_injection/deployments/postwrite_boundaries.py',
    'fault_injection/deployments/postwrite_inventory.py', 'fault_injection/deployments/postwrite_evidence.py',
    'fault_injection/deployments/postwrite_structure.py', 'fault_injection/deployments/postwrite_coverage.py',
    'fault_injection/coverage.py', 'fault_injection/coverage_session.py', 'fault_injection/runtime_discovery.py',
    'fault_injection/native_adapters/postwrite/postwrite.cpp', 'fault_injection/native_adapters/postwrite/postwrite.h',
    'experiments/c2c3_postwrite_20261117/natural39_coverage_probe.py', 'experiments/c2c3_postwrite_20261117/model_identity.py')
SUPPORTING_SOURCES = ('src/demoncat/runtime/vllm_ascend/worker_extension/call_binding.py',
    'src/demoncat/runtime/vllm_ascend/worker_extension/serving_context.py',
    'experiments/engineering_20261117/model_capture.py')
SESSION_KEYS = ('schema', 'session_id', 'scope', 'selection', 'trigger', 'owner', 'generation', 'budget', 'discovery',
    'observation', 'planned_targets', 'planned_targets_hit', 'observed_calls', 'observed_eligible_calls', 'hits',
    'missing_outcomes', 'propagation_added_calls', 'calls', 'recovery', 'contract_satisfied', 'rejection_reasons',
    'real_worker_evidence', 'real_pretrained_model_evidence', 'public_qualification_granted')


def digest(value):
    result = hashlib.sha256()
    for block in json.JSONEncoder(sort_keys=True, separators=(',', ':'), allow_nan=False).iterencode(value):
        result.update(block.encode())
    return result.hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''): result.update(block)
    return result.hexdigest()


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


def read(path):
    path = Path(path)
    if path.stat().st_size > 768*1024**2: raise ValueError('BOUNDED_RESULT_FILE_REQUIRED')
    def invalid(value): raise ValueError('NONFINITE_JSON:' + value)
    with path.open(encoding='utf-8') as stream:
        return json.load(stream, object_pairs_hook=unique_pairs, parse_constant=invalid)


def valid_sha(value): return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None
def key(row): return tuple(row['key'])
def geom(row): return {name: (row['output_geometry'] or {}).get(name) for name in GEOMETRY}
def worker(target): return [target[name] for name in ('process_id', 'process_start_identity', 'worker_id', 'rank', 'device_type', 'logical_device')]


class Review:
    def __init__(self): self.checks = 0; self.errors = Counter(); self.examples = {}
    def check(self, value, name, where=''):
        self.checks += 1
        if not value:
            self.errors[name] += 1
            if len(self.examples.setdefault(name, [])) < 3: self.examples[name].append(str(where)[:160])
        return bool(value)
    def summary(self):
        return dict(checks=self.checks, consistent=not self.errors,
                    issues=[dict(code=name, count=count, examples=self.examples[name]) for name, count in sorted(self.errors.items())])


def geometry_valid(meta, *, exclusive=False, device='npu'):
    shape, stride = meta['shape'], meta['stride']
    if len(shape) != len(stride) or not shape or not all(type(n) is int and n > 0 for n in shape): return False
    if not all(type(n) is int and n >= 0 for n in stride) or meta['storage_offset'] < 0: return False
    if meta['dtype'] != 'torch.bfloat16' or meta['device'].split(':')[0] != device: return False
    if meta['logical_bytes'] != math.prod(shape)*2: return False
    span = meta['storage_offset'] + sum((n-1)*s for n,s in zip(shape,stride)) + 1
    if span*2 > meta['storage_bytes']: return False
    if exclusive:
        expected = []; step = 1
        for n in reversed(shape): expected.insert(0, step); step *= n
        return all(n == 1 or s == e for n,s,e in zip(shape,stride,expected)) and meta['storage_offset'] == 0 and meta['logical_bytes'] == meta['storage_bytes'] <= 1024**2
    return True


def reads_indices(view, producer, indices):
    if any(view[name] != producer[name] for name in ('device', 'dtype', 'storage_identity', 'storage_bytes')): return False
    if view['data_ptr'] - producer['data_ptr'] != (view['storage_offset'] - producer['storage_offset'])*2: return False
    dimensions = sorted(((s,n) for s,n in zip(view['stride'], view['shape']) if s), reverse=True)
    if any(s < 0 for s,_ in dimensions): return False
    for index in indices:
        remainder = producer['storage_offset'] + index - view['storage_offset']
        if remainder < 0: return False
        for stride, size in dimensions:
            coordinate, remainder = divmod(remainder, stride)
            if coordinate >= size: return False
        if remainder: return False
    return bool(indices)


def consumers(review, observations, producer, indices, names, where, device, *, operators=None):
    # The historical source51 caller keeps its frozen aten-only default. A
    # separate, pinned public-Core profile supplies and audits any extension.
    operators = CONSUMERS if operators is None else operators
    review.check(bool(observations) and [row['operator'] for row in observations] == names, 'CONSUMER_CENSUS', where)
    for row in observations:
        review.check(row['operator'] in operators and row['original_returned'] is True
            and row['selected_elements_read'] == indices and isinstance(row['module_path'], str) and bool(row['operand_views']),
            'ACTUAL_VALUE_CONSUMER', where)
        for view in row['operand_views']:
            review.check(geometry_valid(view, device=device) and reads_indices(view, producer, indices),
                         'SELECTED_PRODUCER_ELEMENTS_NOT_READ_BY_CONSUMER_VIEW', where)


def inventory(review, item, where, device, *, producers=None, consumers=None):
    producers = PRODUCERS if producers is None else producers
    consumers = CONSUMERS if consumers is None else consumers
    review.check(item['admission_independent'] is True and item['dropped_calls'] == 0, 'INDEPENDENT_INVENTORY_INCOMPLETE', where)
    rows = item['calls']; keys = [key(row) for row in rows]
    modules = item['modules']; module_ids = {row['object_identity'] for row in modules}
    review.check(len(keys) == len(set(keys)) and len(modules) == len(module_ids), 'DUPLICATE_ACTUAL_CALL_OR_MODULE', where)
    review.check(bool(modules) and all(type(row['object_identity']) is int and row['object_identity'] > 0
        and type(row['calls']) is int and row['calls'] > 0 and isinstance(row['path'], str) for row in modules)
        and all(type(count) is int and count > 0 for count in item['operator_counts'].values()), 'ACTUAL_MODULE_AND_OPERATOR_COUNTS', where)
    counts = Counter(row['operator'] for row in rows)
    review.check(all(item['operator_counts'].get(op, 0) == counts[op] for op in producers), 'PROJECTION_OPERATOR_DENOMINATOR', where)
    ordinals = Counter()
    for row in rows:
        signature = (row['module_object_identity'], row['operator'])
        review.check(row['operator'] in producers and row['original_calls'] == 1 and key(row) == (*signature, ordinals[signature])
            and signature[0] in module_ids and any(m['object_identity'] == signature[0] and m['path'] == row['module_path'] for m in modules),
            'ACTUAL_PROJECTION_IDENTITY_OR_ORDINAL', where)
        ordinals[signature] += 1
        if row['eligible']:
            review.check(row['selected_indices'] == [0] and bool(row['consumers']) and not row['rejection']
                and all(name in consumers for name in row['consumers'])
                and geometry_valid(row['output_geometry'], exclusive=True, device=device), 'ELIGIBLE_PROJECTION_GEOMETRY_OR_CONSUMER', where)
    return keys


def producer_geometry(proof):
    """Recompute operation-specific result geometry from actual operand metadata."""
    operation, operands = proof['operator'], proof['operands']
    left, right = (('input', 'weight') if operation == 'aten.linear.default' else
                   ('self', 'mat2' if operation == 'aten.mm.default' else 'other'))
    a, b = operands[left]['shape'], operands[right]['shape']
    if operation == 'aten.mm.default':
        return [a[0], b[1]] if len(a) == len(b) == 2 and a[1] == b[0] else None
    if operation == 'aten.linear.default':
        if not a or len(b) != 2 or a[-1] != b[1]: return None
        if 'bias' in operands and operands['bias']['shape'] != [b[0]]: return None
        return a[:-1]+[b[0]]
    if not a or not b or a[-1] != (b[-2] if len(b) > 1 else b[-1]): return None
    batch = []
    aa, bb = a[:-2], b[:-2]
    for i in range(1, max(len(aa), len(bb))+1):
        av, bv = (aa[-i] if i <= len(aa) else 1), (bb[-i] if i <= len(bb) else 1)
        if av != bv and av != 1 and bv != 1: return None
        batch.insert(0, max(av,bv))
    return batch+(a[-2:-1] if len(a)>1 else [])+(b[-1:] if len(b)>1 else [])


def review_recovery(document, name, *, expected_targets, cpu_contract):
    """Recovery remains independently reviewable after a refused/incomplete C3."""
    r = Review(); group = document[name]; recovery = group['recovery']
    device = 'cpu' if cpu_contract else 'npu'
    first = group['independent_discovery'][0]
    eligible = {key(row): row for row in first['calls'] if row['eligible']}
    rows = recovery['targets']; prototype = rows[0]['lease']['target']
    targets = {digest(reconstructed_target(row, prototype, group['scope'], document['environment']['native_library_sha256'])):row
               for row in eligible.values()}
    lossless = {row['target_id']:row for row in group['per_target_lossless']}
    r.check(len(rows) == len(targets) == len(lossless) == expected_targets
        and set(targets) == set(lossless) == {digest(row['lease']['target']) for row in rows}, 'RECOVERY_COMPLETE_TARGET_SET')
    healthy = group['forward_artifacts'][0]['output_sha256']
    off = [row for row in group['forward_artifacts'] if row['phase'] == 'off']
    r.check(valid_sha(healthy) and len(off) == 1 and off[0].get('output_sha256') == healthy
        and off[0] == group['forward_artifacts'][-1], 'RECOVERY_FULL_OFF_OUTPUT')
    if off:
        inventory(r, off[0]['inventory'], 'recovery:off', device)
        r.check([key(row) for row in off[0]['inventory']['calls']] == [key(row) for row in first['calls']]
            and off[0]['inventory']['modules'] == first['modules'], 'RECOVERY_OFF_ACTUAL_UNIVERSE')
        calls = [call for call in group['calls'] if call['forward_id'] == off[0]['forward_id']]
        r.check(len(calls) == len(first['calls']) and all(call['phase'] == 'OFF_RECOVERY'
            and call['decision']['reason'] == 'OFF_RECOVERY' and call['decision']['admitted'] is False
            and call['receipt']['native_applied'] is False and call['receipt']['original_calls'] == 1
            and call['receipt']['outcome'] == 'PASSTHROUGH' and call['receipt']['effect_sha256'] is None
            and call['receipt']['execution_sha256'] is None for call in calls), 'RECOVERY_OFF_NO_INTERVENTION')
    c1 = document['c1']['off']['actual_target']
    r.check(prototype['process_id'] == c1['pid'] and prototype['model_object_identity'] == c1['model_instance']
        and prototype['model_instance_id'] == str(c1['model_instance']) and prototype['scope_kind'] == 'MODEL_LOCAL'
        and group['owner'] == str(c1['owner']), 'RECOVERY_SAME_MODEL_OWNER')
    r.check(recovery['complete'] is True and recovery['state'] == 'DISABLED_CONFIRMED'
        and recovery['owner'] == group['owner'] and recovery['generation'] == group['generation']
        and type(recovery['inflight']) is int and recovery['inflight'] == 0
        and all(recovery[k] == [] for k in ('held_leases','pending_drains','pending_leases','errors')), 'RECOVERY_ZERO_RESOURCES')
    off_targets = {tid:row['before'] for tid,row in lossless.items()}
    for row in rows:
        lease = row['lease']; tid = digest(lease['target']); reference = targets[tid]
        proof = dict(target_id=tid, off=healthy, native_live_leases=0, off_targets=off_targets)
        r.check(lease['target'] == reconstructed_target(reference, prototype, group['scope'], document['environment']['native_library_sha256'])
            and lease['owner'] == group['owner'] and lease['generation'] == group['generation']
            and lease['lease_id'] == 'projection:'+tid and lease['device_bytes'] == 0
            and lease['host_bytes'] == 2*reference['output_geometry']['logical_bytes']
            and row['drained'] is True and row['released'] is True and row['off_exact'] is True and row['pending_reason'] is None
            and row['off_output_sha256'] == healthy and row['proof_sha256'] == digest(proof), 'RECOVERY_TARGET_PROOF', tid)
    expected_worker = worker(prototype)
    r.check(len(recovery['workers']) == 1, 'RECOVERY_WORKER_DENOMINATOR')
    for row in recovery['workers']:
        proof = dict(worker=expected_worker, owner=group['owner'], generation=group['generation'],
                     healthy_before=healthy, healthy_after=healthy, no_owned_hooks=True)
        r.check(row['worker'] == expected_worker and row['owner'] == group['owner'] and row['generation'] == group['generation']
            and row['baseline_output_sha256'] == row['off_output_sha256'] == healthy
            and all(type(row[k]) is int and row[k] == 0 for k in ('active_leases','device_bytes','host_bytes','inflight_calls','owned_hooks'))
            and row['proof_sha256'] == digest(proof), 'RECOVERY_HEALTH_RESOURCES_PROOF')
    return dict(**r.summary(), target_count=len(rows), source_bound=True, separate_hook_owner_table_readback=False)


def reconstructed_target(row, prototype, scope, native_sha):
    target = {name: prototype[name] for name in ('process_id','process_start_identity','worker_id','scope_kind','rank',
        'device_type','logical_device','model_instance_id','model_object_identity')}
    target.update(module_object_identity=row['module_object_identity'], module_roles=[], operator_api=row['operator'],
        tensor_signature=digest(geom(row)), boundary=dict(schema_sha256=hashlib.sha256(row['schema'].encode()).hexdigest(),
            implementation_sha256=scope['implementation_sha256'], call_token='stable-projection-site:'+digest(row['key']),
            implementation_kind='torch_dispatch', kind='operator_api'), provider_identity='39.output_proxy:bounded-coverage-v1',
        provider_proof=[['native_library_sha256', native_sha]])
    return target


def review_group(document, name, *, expected_targets=140, cpu_contract=False):
    r = Review(); group = document[name]; controls = document['arm'] == 'baseline'; device = 'cpu' if cpu_contract else 'npu'
    discovery = document['independent_discovery']; scope = group['scope']; healthy = group['independent_discovery']
    r.check(len(healthy) == 2 and healthy == discovery['independent_forwards'], 'TWO_INDEPENDENT_HEALTHY_INVENTORIES')
    first, second = healthy; order = inventory(r, first, 'healthy:0', device); inventory(r, second, 'healthy:1', device)
    r.check(order == [key(row) for row in second['calls']] and first['modules'] == second['modules']
        and first['operator_counts'] == second['operator_counts'], 'HEALTHY_ACTUAL_UNIVERSE_CHANGED')
    ref = {key(row): row for row in first['calls']}; eligible = {k for k,row in ref.items() if row['eligible']}
    r.check(len(eligible) == expected_targets == document['discovery']['eligible_targets'] == group['planned_targets'], 'FULL_ELIGIBLE_DENOMINATOR')
    r.check(len(order) == document['discovery']['projection_calls'], 'ALL_PROJECTION_CALLS_RETAINED')
    for row in second['calls']:
        baseline = ref.get(key(row), {})
        r.check(baseline and geom(row) == geom(baseline) and row['eligible'] == baseline['eligible']
                and row['selected_indices'] == baseline['selected_indices'] and row['schema'] == baseline['schema'], 'HEALTHY_TARGET_SIGNATURE_CHANGED')
    structural = discovery['structural_noncall_inventory']; modules = first['modules']; module_ids = {m['object_identity'] for m in modules}
    r.check(structural == document['discovery']['structural_noncall_inventory']
        and len(module_ids | {row['object_identity'] for row in structural}) == discovery['structural_modules'] == document['discovery']['structural_modules'],
        'STRUCTURAL_UNIVERSE_INCOMPLETE')
    r.check(discovery['actual_module_calls'] == modules, 'ACTUAL_MODULE_INVENTORY_CHANGED')
    for row in structural:
        r.check(row['structural_noncall'] is True and row['actual_forward_calls'] == 0
            and row['class_name'] == 'torch.nn.modules.container.ModuleList'
            and row['reason'] == 'EXACT_MODULELIST_UNIMPLEMENTED_FORWARD_WITH_OBSERVED_MEMBERS'
            and row['object_identity'] not in module_ids and row['parent_identity'] in module_ids
            and bool(row['direct_children']) and set(row['direct_children']).issubset(module_ids)
            and valid_sha(row['container_source_sha256']) and valid_sha(row['unimplemented_forward_source_sha256']), 'STRUCTURAL_NONCALL_NOT_PROVEN')
    recovery = group['recovery']; recovered = recovery['targets']; leases = {digest(row['lease']['target']):row['lease'] for row in recovered}
    r.check(len(leases) == len(recovered) == expected_targets, 'RECOVERY_TARGET_DENOMINATOR')
    prototype = recovered[0]['lease']['target']; c1 = document['c1']['off']['actual_target']
    r.check(prototype['process_id'] == c1['pid'] and prototype['model_object_identity'] == c1['model_instance']
        and prototype['model_instance_id'] == str(c1['model_instance']) and prototype['worker_id'] == 'model-local:'+str(c1['model_instance'])
        and prototype['scope_kind'] == 'MODEL_LOCAL' and prototype['rank'] == 0 and prototype['device_type'] == device
        and prototype['logical_device'] == (None if cpu_contract else 0) and (cpu_contract or prototype['process_start_identity'].isdigit())
        and str(c1['owner']) == group['owner'], 'ACTUAL_MODEL_LOCAL_LIFETIME_OR_OWNER')
    targets = {k:reconstructed_target(row, prototype, scope, document['environment']['native_library_sha256']) for k,row in ref.items()}
    target_ids = {k:digest(target) for k,target in targets.items()}; planned = {target_ids[k] for k in eligible}
    r.check(set(leases) == planned and all(leases[target_ids[k]]['target'] == targets[k] for k in eligible), 'TARGET_IDENTITIES_NOT_FROM_ACTUAL_CALLS')
    r.check(group['discovery']['observed_target_ids'] == sorted(target_ids.values()), 'DISCOVERY_DROPPED_UNSUPPORTED_TARGETS')
    expected_worker = worker(prototype)
    r.check(group['discovery']['expected_workers'] == group['discovery']['observed_workers'] == [expected_worker], 'MODEL_LOCAL_WORKER_DENOMINATOR')
    candidates = []
    for k,row in ref.items():
        candidates.append(dict(candidate_id='projection:'+digest(row['key']), identity=targets[k], eligibility=dict(
            status='SUPPORTED' if k in eligible else 'UNSUPPORTED', reasons=[] if k in eligible else [row['rejection'] or 'NO_SELECTED_ELEMENT_VALUE_CONSUMER'])))
    seen = {row['module_object_identity'] for row in ref.values()}
    for module in modules:
        if module['object_identity'] not in seen:
            candidates.append(dict(candidate_id='module:'+str(module['object_identity']), identity=None,
                eligibility=dict(status='UNSUPPORTED', reasons=['COMPLETE_ACTUAL_MODULE_CALL_HAS_NO_ADAPTED_PROJECTION'])))
    for row in structural:
        candidates.append(dict(candidate_id='structural-noncall:'+str(row['object_identity']), identity=None,
            eligibility=dict(status='UNSUPPORTED', reasons=[row['reason']])))
    counts = {status:sum(row['eligibility']['status'] == status for row in candidates)
              for status in ('SUPPORTED','UNSUPPORTED','UNVERIFIED','AMBIGUOUS','NOT_OBSERVED','RESOURCE_LIMITED')}
    candidate_report = dict(candidates=candidates, discovery_complete=True, eligible=len(eligible), discovered=len(candidates),
                            eligibility_counts=counts, execution_authorized=False)
    r.check(discovery['candidates'] == candidate_report, 'CANDIDATE_CLASSIFICATION_NOT_RECONSTRUCTIBLE')
    r.check(group['discovery']['candidates_sha256'] == digest(candidate_report)
        and group['discovery']['artifact_sha256'] == digest(dict(forwards=healthy, structural_noncall_inventory=[structural,structural])),
        'DISCOVERY_ARTIFACT_HASH')
    r.check(group['discovery']['complete'] is True and group['discovery']['source'] == 'independent_workload_observer'
        and group['discovery']['implementation_sha256'] == document['source_sha256']['fault_injection/deployments/postwrite_inventory.py']
        and group['discovery']['workload_sha256'] == scope['workload_sha256'], 'DISCOVERY_SOURCE_BINDING')
    forwards = group['forward_artifacts']; repeat = expected_targets if name == 'c2' else 2
    r.check([row['phase'] for row in forwards] == ['lossless']+['fault']*repeat+['off'], 'COMPLETE_PHASE_MATRIX')
    r.check(group['completed_fault_forwards'] == repeat == group['max_fault_forwards'] and group['provider_failure'] is None,
            'ALL_NATURAL_FORWARDS_COMPLETED')
    r.check(group['trigger'] == ('once' if name == 'c2' else 'every_match') and group['baseline_control'] is controls
        and group['selection'] == 'all_eligible' and not group['public_qualification_granted']
        and not group['real_worker_evidence'] and group['isolated_target_replays'] is (name == 'c2')
        and group['generation'] == (1 if name == 'c2' else 2)
        and group['baseline_controls_grant_qualification'] is False, 'GROUP_SCOPE_OR_CONTROL_LABEL')
    r.check(group['budget']['max_calls'] == repeat*len(order) and 0 < group['budget']['max_seconds'] <= 1500
        and group['budget']['max_concurrency'] == group['simultaneous_lease_limit']
        and (group['simultaneous_lease_limit'] == 1 if name == 'c2' else 1 <= group['simultaneous_lease_limit'] <= expected_targets),
        'BOUNDED_CALL_AND_LEASE_BUDGET')
    frame_map = {}; frame_indices = {}; forward_rows = {}; healthy_output = forwards[0]['output_sha256']
    for index, frame in enumerate(forwards):
        fid = frame['forward_id']; frame_map[fid] = frame; frame_indices[fid] = index
        r.check(fid not in forward_rows and frame['request_id'] == 'model-local-invocation:'+fid.split(':')[-1]
            and fid.startswith('forward:') and (index == 0 or int(fid.split(':')[-1]) == int(forwards[index-1]['forward_id'].split(':')[-1])+1),
            'ACTUAL_INVOCATION_MAPPING_OR_SEQUENCE', fid)
        inv = frame['inventory']; current_order = inventory(r, inv, fid, device)
        r.check(current_order == order and inv['modules'] == modules, 'NATURAL_CALL_DENOMINATOR_CHANGED', fid)
        forward_rows[fid] = {key(row):row for row in inv['calls']}
        for row in inv['calls']:
            before = ref.get(key(row))
            r.check(before and geom(row) == geom(before) and row['schema'] == before['schema']
                    and row['eligible'] == before['eligible'], 'ACTUAL_TARGET_SIGNATURE_DRIFT', fid)
        output = frame.get('output_sha256')
        r.check(valid_sha(output), 'COMPLETE_MODEL_OUTPUT_HASH_MISSING', fid)
        if controls or frame['phase'] != 'fault': r.check(output == healthy_output, 'LOSSLESS_CONTROL_OR_OFF_FULL_OUTPUT_CHANGED', fid)
    lossless_rows = group['per_target_lossless']; lossless = {row['target_id']:row for row in lossless_rows}
    r.check(len(lossless_rows) == len(lossless) == expected_targets and set(lossless) == planned, 'ALL_TARGET_LOSSLESS_DENOMINATOR')
    native_generation = lossless_rows[0]['native_generation']
    expected_generation = (c1['generation']+1 if name == 'c2' else
        document['c2']['per_target_lossless'][0]['native_generation']+len({row['forward_id'] for row in document['c2']['forward_artifacts']}))
    r.check(native_generation == expected_generation, 'NATIVE_GENERATION_NOT_AFTER_C1')
    for k in eligible:
        row = lossless[target_ids[k]]
        r.check(row['key'] == list(k) and valid_sha(row['before']) and row['before'] == row['after']
            and row['original_calls'] == 1 and row['native_generation'] == native_generation
            and row['temporary_readback_verified'] is True and row['released'] is True, 'PER_TARGET_LOSSLESS_EFFECT_OR_RELEASE', str(k))
    executions = group['execution_artifacts']; by_call = {row['call_id']:row for row in executions}
    repeats_per_target = 1 if name == 'c2' else 2
    r.check(len(by_call) == len(executions) == repeats_per_target*expected_targets
        and Counter(row['target_id'] for row in executions) == Counter({tid:repeats_per_target for tid in planned}), 'PER_TARGET_EXECUTION_DISTRIBUTION')
    calls = group['calls']; tickets = []; attempted = set(); current_frame = None; active = 0; hit_count = 0; eligible_calls = 0
    target_keys = {tid:k for k,tid in target_ids.items()}; expected_call_order = [(f['forward_id'], k) for f in forwards[1:] for k in order]
    r.check(len(calls) == len(expected_call_order), 'COMPLETE_CALL_LEDGER_DENOMINATOR')
    for sequence, call in enumerate(calls):
        fid, tid = call['forward_id'], call['target_id']; decision, receipt = call['decision'], call['receipt']; ticket = decision['ticket']
        tickets.append(ticket); k = target_keys.get(tid); frame = frame_map.get(fid); is_off = frame and frame['phase'] == 'off'
        if fid != current_frame: current_frame = fid; active = 0
        actual_eligible = k in eligible
        if actual_eligible and not is_off: eligible_calls += 1
        reason = ('OFF_RECOVERY' if is_off else 'NON_TARGET' if not actual_eligible else
            'TARGET_ALREADY_ATTEMPTED' if name == 'c2' and tid in attempted else
            'CONCURRENCY_BUDGET_EXHAUSTED' if active >= group['simultaneous_lease_limit'] else 'ADMITTED')
        phase = 'OFF_RECOVERY' if is_off else 'POST_FIRST_ANOMALY' if active else 'PRE_FAULT_PROPAGATION'
        r.check(sequence < len(expected_call_order) and (fid,k) == expected_call_order[sequence], 'LEDGER_ACTUAL_CALL_ORDER', call['call_id'])
        r.check(frame and k in ref and call['call_id'] == fid+':'+str(order.index(k)+1)
            and call['request_ids'] == [frame['request_id']] and call['owner'] == group['owner'] and call['generation'] == group['generation']
            and call['worker'] == expected_worker and call['model_instance'] == prototype['model_instance_id']
            and call['rank'] == 0 and call['device'] == prototype['logical_device'] and call['operator'] == ref[k]['operator']
            and call['eligible'] is actual_eligible and call['propagation_added'] is False and call['phase'] == phase,
            'ACTUAL_CALL_TARGET_REQUEST_OR_OWNER_JOIN', call['call_id'])
        r.check(ticket == dict(observer_id=group['discovery']['observer_id'], sequence=sequence, eligible=actual_eligible,
            propagation_added=False, call=dict(target=targets[k], call_id=call['call_id'], forward_context=fid,
                request_ids=call['request_ids'], phase=phase)) and decision['session_id'] == group['session_id']
            and decision['lease'] == leases.get(tid) and receipt['decision'] == decision,
            'DECISION_TICKET_OR_RECEIPT_JOIN', call['call_id'])
        r.check(decision['reason'] == reason and decision['admitted'] is (reason == 'ADMITTED'), 'RECOMPUTED_ADMISSION_SCHEDULE', call['call_id'])
        if reason == 'ADMITTED': active += 1; attempted.add(tid)
        expected_hit = reason == 'ADMITTED' and not controls
        hit_count += receipt['outcome'] == 'HIT'
        r.check(receipt['original_calls'] == 1 and receipt['native_applied'] is expected_hit
            and receipt['outcome'] == ('HIT' if expected_hit else 'RESOURCE_LIMITED' if reason == 'CONCURRENCY_BUDGET_EXHAUSTED' else 'PASSTHROUGH'),
            'ACTUAL_NATIVE_ORIGINAL_ONCE_OUTCOME', call['call_id'])
        if reason == 'ADMITTED':
            effect = by_call.get(call['call_id'])
            r.check(effect is not None, 'ADMITTED_CALL_WITHOUT_EXECUTION_ARTIFACT', call['call_id'])
            if effect is None: continue
            r.check(effect['target_id'] == tid and effect['forward_id'] == fid and effect['request_id'] == frame['request_id']
                and effect['native_generation'] == native_generation+frame_indices[fid] and effect['original_calls'] == 1
                and effect['native_lifetime'] == 'one actual complete forward'
                and effect['application_attempted'] is True and effect['application_returned'] is True
                and effect['native_applied'] is expected_hit and effect['baseline_control'] is controls
                and effect['selected_elements'] == 1 and effect['changed_elements'] == int(expected_hit)
                and effect.get('readback_exact') is True and effect.get('native_recovered') is True
                and valid_sha(effect.get('full_model_output_sha256'))
                and effect.get('full_model_output_sha256') == frame.get('output_sha256'), 'PER_TARGET_NATIVE_EFFECT_OR_RECOVERY', call['call_id'])
            r.check(valid_sha(effect['original_output_sha256']) and valid_sha(effect['mutated_output_sha256'])
                and (effect['original_output_sha256'] != effect['mutated_output_sha256']) is expected_hit,
                'FULL_TARGET_OUTPUT_EFFECT_HASH', call['call_id'])
            if name == 'c2': r.check(effect['original_output_sha256'] == lossless[tid]['before'], 'ISOLATED_TARGET_INPUT_NOT_LOSSLESS', call['call_id'])
            r.check(receipt['execution_sha256'] == digest(effect) and receipt['effect_sha256'] == effect['mutated_output_sha256'],
                    'EXECUTION_OR_EFFECT_ARTIFACT_HASH', call['call_id'])
            producer = forward_rows[fid][k]['output_geometry']
            consumers(r, effect.get('consumer_observations', []), producer, ref[k]['selected_indices'], effect.get('consumers', []), call['call_id'], device)
        else:
            r.check(call['call_id'] not in by_call and receipt['execution_sha256'] is None and receipt['effect_sha256'] is None,
                    'NON_TARGET_OR_OFF_INTERVENTION', call['call_id'])
    r.check(group['hits'] == hit_count == (0 if controls else repeats_per_target*expected_targets)
        and group['planned_targets_hit'] == (0 if controls else expected_targets)
        and group['observed_calls'] == len(calls) and group['observed_eligible_calls'] == eligible_calls
        and group['missing_outcomes'] == [] and group['propagation_added_calls'] == [], 'RECOMPUTED_COVERAGE_COUNTERS')
    observation = group['observation']
    r.check(observation['complete'] is True and observation['dropped_calls'] == 0 and observation['observed_calls'] == len(calls)
        and observation['calls_sha256'] == digest(tickets)
        and observation['artifact_sha256'] == digest(dict(discovery=healthy, forwards=forwards, tickets=tickets)), 'COMPLETE_OBSERVATION_ARTIFACT_HASH')
    r.check(recovery['complete'] is True and recovery['state'] == 'DISABLED_CONFIRMED'
        and recovery['owner'] == group['owner'] and recovery['generation'] == group['generation'] and recovery['inflight'] == 0
        and all(recovery[k] == [] for k in ('held_leases','pending_drains','pending_leases','errors')), 'GROUP_ZERO_RESOURCE_RECOVERY')
    off_targets = {tid:row['before'] for tid,row in lossless.items()}
    for row in recovered:
        lease = row['lease']; tid = digest(lease['target']); k = target_keys[tid]
        proof = dict(target_id=tid, off=healthy_output, native_live_leases=0, off_targets=off_targets)
        r.check(lease['owner'] == group['owner'] and lease['generation'] == group['generation']
            and lease['lease_id'] == 'projection:'+tid and lease['device_bytes'] == 0
            and lease['host_bytes'] == 2*ref[k]['output_geometry']['logical_bytes']
            and row['drained'] is True and row['released'] is True and row['off_exact'] is True and row['pending_reason'] is None
            and row['off_output_sha256'] == healthy_output and row['proof_sha256'] == digest(proof), 'PER_TARGET_OFF_RECOVERY_ARTIFACT', tid)
    r.check(len(recovery['workers']) == 1, 'HEALTH_MODEL_LOCAL_DENOMINATOR')
    for row in recovery['workers']:
        proof = dict(worker=expected_worker, owner=group['owner'], generation=group['generation'],
            healthy_before=healthy_output, healthy_after=healthy_output, no_owned_hooks=True)
        r.check(row['worker'] == expected_worker and row['owner'] == group['owner'] and row['generation'] == group['generation']
            and row['baseline_output_sha256'] == row['off_output_sha256'] == healthy_output
            and all(type(row[k]) is int and row[k] == 0 for k in ('active_leases','device_bytes','host_bytes','inflight_calls','owned_hooks'))
            and row['proof_sha256'] == digest(proof), 'HEALTH_ZERO_RESOURCES_OR_ARTIFACT_HASH')
    all_once = len(executions) == len(eligible) and len({row['target_id'] for row in executions}) == len(eligible)
    r.check(group['provider_failure'] is None and group['all_targets_controlled_once'] is all_once, 'PROVIDER_COMPLETION_STATUS')
    return dict(**r.summary(), natural_targets=len(eligible), complete_projection_denominator=len(order),
        natural_forwards=len(forwards), actual_invocation_count=len(calls), recomputed_hits=hit_count,
        baseline_control=controls, root_fault_outputs_changed=sum(frame.get('output_sha256') != healthy_output for frame in forwards if frame['phase']=='fault'),
        complete_model_output_hash=healthy_output, target_ids=sorted(planned),
        logical_universe_sha256=digest(dict(calls=[dict(module_path=row['module_path'], operator=row['operator'],
            ordinal=row['key'][2], schema=row['schema'], geometry=geom(row), eligible=row['eligible'],
            selected_indices=row['selected_indices'], consumers=row['consumers']) for row in first['calls']],
            actual_modules=[dict(path=row['path'], calls=row['calls']) for row in modules],
            structural_noncall=[dict(path=row['path'], class_name=row['class_name'], reason=row['reason']) for row in structural])),
        source_bound_recovery_record_consistent=not any('RECOVERY' in code or 'RESOURCES' in code for code in r.errors),
        stored_contract_satisfied=group['contract_satisfied'], contract_boolean_used_as_verdict=False)


def review_document(document, *, expected_targets=140, cpu_contract=False):
    r = Review(); c1 = document['c1']; device = 'cpu' if cpu_contract else 'npu'
    r.check(document['schema'] == 'demoncat.postwrite39.natural_coverage_probe.v1'
        and document['mechanism'] == '39.output_proxy' and document['worker_extension'] is False
        and document['public_qualification_granted'] is False and document['evidence_kind'] == 'natural_pretrained_model', 'MODEL_LOCAL_OUTPUT_PROXY_SCOPE')
    r.check(document.get('live_leases') == 0 and not any(document.get(k) for k in ('error','recovery_error','incomplete_group')),
            'FINAL_PROVIDER_RECOVERY_OR_FAILURE')
    natural_first = next(row for row in document['c2']['independent_discovery'][0]['calls'] if row['eligible'])
    q = c1['qualification']
    r.check(q['status'] == 'ACTUAL_CALL_CANDIDATE_SELECTED' and q['count'] == expected_targets
        and q['selection'] == 'first eligible in stable actual execution order' and q['requires_lossless'] is True
        and q['operator'] == natural_first['operator'] and q['module_path'] == natural_first['module_path'],
        'C1_QUALIFICATION_FROM_SAME_ACTUAL_UNIVERSE')
    for label, phase, hits in (('lossless','baseline',0),('fault','fault',1),('off','off',0)):
        row = c1[label]; target = row['actual_target']; proof = target['producer_proof']
        r.check(row['phase'] == phase and row['mechanism'] == '39.output_proxy'
            and row['transport'] == ('cpu_test' if cpu_contract else 'public_acl_native')
            and row['new_fault_hits'] == hits and row['recovery_confirmed'] is True
            and row['exact_baseline_output'] is (label != 'fault'), 'C1_ACTUAL_EFFECT_OR_OFF', label)
        r.check(target['generation'] == {'lossless':1,'fault':2,'off':3}[label]
            and all(target[k] == c1['off']['actual_target'][k] for k in ('pid','model_instance','owner','module_path','operator'))
            and target['module_path'] == natural_first['module_path'] and target['operator'] == natural_first['operator']
            and row['original_operator_calls'] == sum(row['discovery_diagnostics']['operator_counts'].values())
            and row['eligible_with_consumers'] == expected_targets
            and all(row[k] is False for k in ('native_acceptance_inherited','public_worker_qualified','all_eligible_qualified','every_match_qualified')),
            'C1_ACTUAL_IDENTITY_COUNTS_OR_GENERATION', label)
        # These cumulative census refusals include the unselected large
        # lm_head output. They do not describe a failed selected C1 execution;
        # its actual producer, effect and consumer are checked independently.
        transport = row['discovery_diagnostics']['transport_rejections']
        r.check(isinstance(row['rejections'], list)
            and all(isinstance(reason, str) and reason in {item['reason'] for item in transport}
                    for reason in row['rejections']), 'C1_CANDIDATE_REFUSAL_RECONCILIATION', label)
        r.check(target['operator'] in PRODUCERS and proof['operator'] == target['operator']
            and proof['original_return_storage'] is True and proof['returned_tensor']['shape'] == target['shape']
            and proof['expected_output_shape'] == producer_geometry(proof) == target['shape'] and target['logical_indices'] == [0]
            and all(target[k] == proof['returned_tensor'][k] for k in ('stride','storage_offset','dtype'))
            and proof['schema'] == natural_first['schema'] and proof['schema_sha256'] == hashlib.sha256(proof['schema'].encode()).hexdigest()
            and geometry_valid(proof['returned_tensor'], exclusive=True, device=device)
            and target['storage_identity'] == [proof['returned_tensor']['device'], proof['returned_tensor']['storage_identity'],
                proof['returned_tensor']['data_ptr'], proof['returned_tensor']['storage_bytes']]
            and all(geometry_valid(view, device=device) and view['storage_identity'] != proof['returned_tensor']['storage_identity']
                    for view in proof['operands'].values()),
            'C1_ORIGINAL_PRODUCER_RETURN_BINDING', label)
        consumers(r, target['consumer_observations'], proof['returned_tensor'], [0], target['consumer_calls'], label, device)
        if label != 'off':
            native = row['intervention']
            r.check(native['original_executed'] is True and native['selected_elements'] == 1
                and native['changed_elements'] == hits and native['readback_exact'] is True
                and native['new_fault_hits'] == hits and native['boundary'] == 'post_operator_tensor_output', 'C1_NATIVE_POSTWRITE_BOUNDARY', label)
        else: r.check(row['intervention'] is None, 'C1_OFF_HAS_NO_INTERVENTION')
    final = document['final_off']
    r.check(final['phase'] == 'off' and final['new_fault_hits'] == 0 and final['intervention'] is None
        and final['recovery_confirmed'] is True and final['exact_baseline_output'] is True
        and all(final['actual_target'][k] == c1['off']['actual_target'][k] for k in ('pid','model_instance','owner','module_path','operator')),
        'FINAL_SAME_MODEL_NORMAL_OFF')
    final_target = final['actual_target']; final_proof = final_target['producer_proof']
    r.check(final_proof['operator'] == natural_first['operator'] and final_proof['schema'] == natural_first['schema']
        and final_proof['original_return_storage'] is True and final_proof['expected_output_shape'] == producer_geometry(final_proof)
        and final_target['shape'] == final_proof['returned_tensor']['shape'] == final_proof['expected_output_shape']
        and geometry_valid(final_proof['returned_tensor'], exclusive=True, device=device)
        and all(final_target[k] == final_proof['returned_tensor'][k] for k in ('stride','storage_offset','dtype')),
        'FINAL_OFF_ORIGINAL_PRODUCER')
    consumers(r, final_target['consumer_observations'], final_proof['returned_tensor'], [0], final_target['consumer_calls'], 'final_off', device)
    latest = document.get('c3', document['c2'])
    if latest.get('per_target_lossless') and latest.get('forward_artifacts'):
        r.check(final_target['generation'] == latest['per_target_lossless'][0]['native_generation']
            + len({row['forward_id'] for row in latest['forward_artifacts']}), 'FINAL_OFF_NATIVE_GENERATION')
    r.check(valid_sha(c1['qualification']['input_sha256']), 'ACTUAL_FIXED_INPUT_DIGEST')
    groups = {}
    for name in ('c2','c3'):
        if name not in document: continue
        try: groups[name] = review_group(document, name, expected_targets=expected_targets, cpu_contract=cpu_contract)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            groups[name] = dict(consistent=False, issues=[dict(code='INCOMPLETE_OR_MALFORMED_GROUP_EVIDENCE', detail=str(error)[:200])])
        try: groups[name]['recovery'] = review_recovery(document, name, expected_targets=expected_targets, cpu_contract=cpu_contract)
        except (KeyError, TypeError, ValueError, IndexError) as error:
            groups[name]['recovery'] = dict(consistent=False, issues=[dict(code='INCOMPLETE_RECOVERY_EVIDENCE', detail=str(error)[:200])])
        groups[name]['stored_provider_failure'] = document[name].get('provider_failure')
        groups[name]['stored_rejection_reasons'] = document[name].get('rejection_reasons')
    proof = dict(present='c3' in document, issuance_receipt=False, use_receipt=False, same_live_object_receipted=False,
                 source_bound_live_gate='Frozen ProjectionCoverage passes the live C2Proof object to CoverageSession; this does not reconstruct an absent receipt.')
    if 'c3' in document:
        c2, c3 = document['c2'], document['c3']; ledger = c3.get('live_c2_proof', {}); consumed = ledger.get('consumed') or {}
        issued = c2.get('live_c2_proof', {}).get('issued', [])
        proof['issuance_receipt'] = bool(issued); proof['use_receipt'] = bool(consumed)
        if consumed:
            fields = consumed.get('proof', {}); consumer = consumed.get('consumer', {})
            expected = dict(scope=c2['scope'], candidates_sha256=c2['discovery']['candidates_sha256'],
                target_ids=groups['c2'].get('target_ids'), owner=c2['owner'], generation=c2['generation'], session_id=c2['session_id'],
                report_sha256=digest({k:c2[k] for k in SESSION_KEYS}), discovery_sha256=c2['discovery']['artifact_sha256'],
                observation_sha256=c2['observation']['artifact_sha256'], recovery_sha256=digest(c2['recovery']))
            proof['use_fields_consistent'] = (fields == expected and consumed.get('proof_sha256') == digest(fields)
                and consumed.get('exact_issued_object_verified') is True and consumed.get('serializable_authority') is False
                and consumed.get('object_binding', {}).get('pid') == c1['off']['actual_target']['pid']
                and type(consumed.get('object_binding', {}).get('python_object_id')) is int
                and consumer == dict(session_id=c3['session_id'], scope=c3['scope'], owner=c3['owner'], generation=c3['generation'],
                    candidates_sha256=c3['discovery']['candidates_sha256'], target_ids=groups['c2'].get('target_ids'))
                and c3['scope'] == c2['scope'] and c3['owner'] == c2['owner'] and c3['generation'] > c2['generation'])
            proof['same_live_object_receipted'] = proof['use_fields_consistent'] and len(issued) == 1 and all(
                issued[0].get(k) == consumed.get(k) for k in ('proof','proof_sha256','object_binding','exact_issued_object_verified'))
    return dict(**r.summary(), arm=document['arm'], groups=groups, c3_live_proof=proof,
        c1_fault_invocations=1, process_contains_c1_fault=True,
        c1_cumulative_candidate_refusals={label:dict(Counter(c1[label]['rejections']))
            for label in ('lossless','fault','off')},
        c1_final_off_consistent=not r.errors,
        c2_evidence_reconciled=not r.errors and groups.get('c2', {}).get('consistent', False)
            and groups['c2']['recovery']['consistent'],
        c3_coverage_record_consistent=groups.get('c3', {}).get('consistent', False),
        c3_issuance_and_use_reconciled=proof['same_live_object_receipted'],
        scope='MODEL_LOCAL 39.output_proxy; not standard Worker, not 39.matmul_range, not E5',
        evidence_limits=['Full model/target bytes were hashed by frozen runtime code but no tensor bins were preserved for independent rehash.',
            'The Native library is rehashed; build_fingerprint is a joined reference without a separate build-receipt artifact.',
            'Full producer operand proofs are retained for the C1 target; group producers are source-bound to actual dispatcher inventory and geometry.',
            'Recovery hashes/zero-resource receipts are reconciled; no separate final hook/mode/native-owner readback is present.',
            'Raw inputs and per-forward input digests are absent; fixed-input equality is enforced by the frozen runtime guard.',
            'MODEL_LOCAL invocation IDs are driver invocations, never scheduler request IDs.'],
        input_sha256=c1['qualification']['input_sha256'], source_sha256=document['source_sha256'],
        weights_sha256=document['weights']['weights_sha256'], model_source_sha256=document['model_source']['source_sha256'],
        environment_sha256=digest(document['environment']), build_fingerprint=document['build_fingerprint'],
        public_qualification_granted=False, complete_raw_tensor_bytes_rehashed=False, e5_statistics_reviewed=False)


def resolve_recorded(path, mappings):
    source = PurePosixPath(path)
    if '..' in source.parts: raise ValueError('RECORDED_PATH_TRAVERSAL')
    for prefix, target in mappings:
        if source.is_relative_to(prefix):
            root = target.resolve(strict=True); result = (root/Path(*source.relative_to(prefix).parts)).resolve(strict=True)
            if not result.is_relative_to(root): raise ValueError('MAPPED_FILE_ESCAPES_ROOT')
            return result
    raise ValueError('EXPLICIT_PATH_MAPPING_MISSING')


def provenance(document, source_root, mappings, native_library):
    r = Review(); unavailable = []; external = 0
    manifest_path = source_root/'source_manifest.json'; manifest = read(manifest_path)
    r.check(file_sha(manifest_path) == MANIFEST_SHA and manifest['base_commit'] == SOURCE_COMMIT, 'EXACT_SOURCE51_MANIFEST')
    files = {row['path']:row for row in manifest['files']}
    r.check(len(files) == len(manifest['files']) and set(document['source_sha256']) == set(SOURCES), 'COMPLETE_SOURCE_MANIFEST_SET')
    for name in (*SOURCES, *SUPPORTING_SOURCES):
        actual = file_sha(source_root/name)
        r.check(files[name]['sha256'] == actual and files[name]['bytes'] == (source_root/name).stat().st_size
            and (name not in document['source_sha256'] or document['source_sha256'][name] == actual), 'FROZEN_IMPLEMENTATION_BYTES', name)
    r.check(file_sha(native_library) == document['environment']['native_library_sha256'] == NATIVE_SHA, 'EXACT_NATIVE_LIBRARY_BYTES')
    r.check(document['build_fingerprint'] == document['environment']['build_fingerprint'] and valid_sha(document['build_fingerprint']), 'BUILD_FINGERPRINT')
    for name, metadata in (('weights',document['weights']), ('model_source',document['model_source'])):
        r.check(metadata[name+'_sha256' if name == 'weights' else 'source_sha256'] == digest(metadata['manifest']), 'ACTUAL_MODEL_MANIFEST_HASH', name)
    weights = document['weights']; model = PurePosixPath(weights['path']); record = weights['manifest']
    references = {str(model/name):value['sha256'] for name,value in record['files'].items()}
    references[str(model/'config.json')] = record['config_sha256']
    if record['index']: references[str(model/record['index']['file'])] = record['index']['sha256']
    references.update(document['model_source']['manifest']['files'])
    r.check(record['format'] == 'safetensors' and bool(record['files'])
        and record['loader_selection'] == 'from_pretrained(use_safetensors=True, local_files_only=True)'
        and bool(document['model_source']['manifest']['classes'])
        and all(row['source'] in document['model_source']['manifest']['files'] for row in document['model_source']['manifest']['classes']),
        'ACTUAL_WEIGHTS_AND_SOURCE_SELECTION')
    for path, expected in references.items():
        r.check(valid_sha(expected), 'EXTERNAL_ARTIFACT_HASH_FORMAT', path)
        try: actual = file_sha(resolve_recorded(path, mappings))
        except (OSError, ValueError) as error:
            if isinstance(error, ValueError) and str(error) != 'EXPLICIT_PATH_MAPPING_MISSING':
                r.check(False, 'INVALID_EXTERNAL_ARTIFACT_PATH', path)
            unavailable.append(dict(path=path, reason=type(error).__name__)); continue
        external += 1; r.check(actual == expected, 'EXTERNAL_ARTIFACT_BYTES_CHANGED', path)
        weight_record = record['files'].get(str(PurePosixPath(path).relative_to(model))) if PurePosixPath(path).is_relative_to(model) else None
        if weight_record:
            r.check(type(weight_record['bytes']) is int and weight_record['bytes'] > 0
                and resolve_recorded(path, mappings).stat().st_size == weight_record['bytes'], 'WEIGHTS_FILE_SIZE_CHANGED', path)
    for group_name in ('c2',):
        if group_name not in document: continue
        scope = document[group_name]['scope']
        r.check(scope['mechanism'] == '39.output_proxy' and scope['evidence_kind'] == 'natural_pretrained_model'
            and scope['model_weights_sha256'] == weights['weights_sha256']
            and scope['model_source_sha256'] == document['model_source']['source_sha256']
            and scope['implementation_sha256'] == digest(document['source_sha256'])
            and scope['environment_sha256'] == digest(document['environment']), 'GROUP_SCOPE_PROVENANCE', group_name)
        r.check(scope['workload_sha256'] == digest(dict(prompt=DEFAULT_PROMPT,
            input_sha256=document['c1']['qualification']['input_sha256'], max_length=32, use_cache=False)),
            'SOURCE51_PREPARED_WORKLOAD_BINDING', group_name)
    return dict(**r.summary(), external_files_rehashed=external, external_files_unavailable=unavailable,
                external_artifacts_all_available=not unavailable,
                source_manifest_sha256=file_sha(manifest_path), native_sha256=file_sha(native_library))


def forbid_runtime(event, args):
    if event in ('ctypes.dlopen','subprocess.Popen','os.system','socket.connect'): raise RuntimeError('CPU_READ_ONLY_REVIEW:'+event)
    if event == 'import' and args[0].split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
        raise RuntimeError('RUNTIME_IMPORT_REFUSED:'+args[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--baseline-sha256', default=BASELINE_SHA)
    parser.add_argument('--fault', type=Path); parser.add_argument('--fault-sha256', default=FAULT_SHA)
    parser.add_argument('--source-root', type=Path, required=True); parser.add_argument('--native-library', type=Path, required=True)
    parser.add_argument('--path-map', action='append', default=[])
    args = parser.parse_args()
    if args.fault and not valid_sha(args.fault_sha256): parser.error('--fault-sha256 required with --fault')
    mappings = [(PurePosixPath('/source'), args.source_root)]
    for item in args.path_map:
        prefix, target = item.split('=', 1); mappings.append((PurePosixPath(prefix), Path(target)))
    mappings.sort(key=lambda pair:len(pair[0].parts), reverse=True)
    sys.addaudithook(forbid_runtime)
    result = dict(schema='demoncat.postwrite39.source51.independent_review.v1', reviewer_sha256=file_sha(__file__), arms={},
                  public_qualification_granted=False, standard_worker=False, matmul_range=False, e5=False)
    for arm, path, expected in (('baseline',args.baseline,args.baseline_sha256), ('fault',args.fault,args.fault_sha256)):
        if path is None: continue
        try:
            actual = file_sha(path)
            if actual != expected: raise ValueError('PINNED_RESULT_SHA256_DIFFERS')
            document = read(path)
            if document['arm'] != arm: raise ValueError('RESULT_ARM_DIFFERS')
            reviewed = review_document(document)
            reviewed.update(original_path=str(path), original_sha256=actual, provenance=provenance(document,args.source_root,mappings,args.native_library))
            reviewed['consistent'] = reviewed['c2_evidence_reconciled'] and reviewed['provenance']['consistent']
            for summary in reviewed['groups'].values(): summary.pop('target_ids', None)
            result['arms'][arm] = reviewed
            del document
        except Exception as error:
            result['arms'][arm] = dict(consistent=False, error=type(error).__name__+': '+str(error)[:300], original_path=str(path))
    if set(result['arms']) == {'baseline','fault'}:
        left, right = result['arms']['baseline'], result['arms']['fault']
        result['paired_input_source_build_equal'] = all(left.get(k) is not None and left.get(k) == right.get(k) for k in
            ('input_sha256','source_sha256','weights_sha256','model_source_sha256','environment_sha256','build_fingerprint'))
        result['paired_complete_healthy_output_equal'] = (left.get('groups', {}).get('c2', {}).get('complete_model_output_hash') is not None
            and left.get('groups', {}).get('c2', {}).get('complete_model_output_hash') == right.get('groups', {}).get('c2', {}).get('complete_model_output_hash'))
        result['paired_logical_universe_equal'] = (left.get('groups', {}).get('c2', {}).get('logical_universe_sha256') is not None
            and left.get('groups', {}).get('c2', {}).get('logical_universe_sha256') == right.get('groups', {}).get('c2', {}).get('logical_universe_sha256'))
    reconciled = all(row.get('consistent') is True for row in result['arms'].values()) and all(
        value is True for key,value in result.items() if key.startswith('paired_'))
    result['status'] = 'RECONCILED_WITH_EXPLICIT_EVIDENCE_LIMITS' if reconciled else 'INCONSISTENT_OR_INCOMPLETE_EVIDENCE'
    print(json.dumps(result, separators=(',', ':'), allow_nan=False))
    return 0 if reconciled else 2


if __name__ == '__main__': raise SystemExit(main())
