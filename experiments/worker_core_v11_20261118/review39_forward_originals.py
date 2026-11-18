"""Pure saved-frame joins; no observer/injector imports and no live authority."""
from collections import Counter


ROTARY_MODULE = 'vllm_ascend.ops.rotary_embedding'
ROTARY_SHA = '74653da7274de182e8497f7d6b07ea7c6deeeef21e710e34eed5a349b0f671e7'
ROTARY_OPERATOR = 'vllm.npu_rotary_embedding.default'
ROTARY_SCHEMA = ('vllm::npu_rotary_embedding(Tensor positions, Tensor query, Tensor key, '
    'Tensor cos_sin_cache, SymInt head_size, SymInt rotary_dim, bool is_neox_style, '
    'Tensor? offsets=None) -> (Tensor, Tensor)')
ROTARY_REASON = 'ACTUAL_ASCEND_FORWARD_OOT_USES_PUBLIC_ROTARY_NOT_APPLY_MODULE'
FIELDS = ('actual_forward_frames', 'frame_binding_gaps', 'frame_failures',
          'bounded_noncall_proofs', 'frame_observer_scope')


def native_phase(phase):
    if phase in ('observe', 'observe_repeat'):
        return 'observe'
    if phase.startswith(('c2_', 'c3_', 'group_recovery_')):
        tail = phase.rsplit('_', 1)[-1]
        return tail if tail in ('fault', 'health') else 'group_' + tail
    return phase


def check(a, C, group, rows, worker, aliases, *, natural, model_source=None, cpu_profile=None):
    """Reconstruct non-call proof from every complete observed request.

    cpu_profile is a test-only explicit module/SHA/schema, never a CLI input or
    natural profile. Bound external code is matched to frozen model-source
    receipts, not imported or executed by this reader.
    """
    structural = group['discovery']['structural_noncall_inventory']
    inventories = group['independent_discovery'] + [x['inventory'] for x in group['forward_artifacts']]
    newer = 'serving39_output_discovery.py' in group['source_pins']
    present = any(any(key in inv for key in FIELDS) for inv in inventories)
    if not newer and not present:
        a.check(not any(x['reason'] == ROTARY_REASON for x in structural), 'NO_LEGACY_ROTARY_NONCALL_UPGRADE')
        return dict(format='LEGACY_NO_DIRECT_FRAME_OBSERVATION', rotary_children=[], frames=0)
    a.check(newer and all(all(key in inv for key in FIELDS) for inv in inventories),
            'DIRECT_DISCOVERY_SOURCE_AND_COMPLETE_FIELDS')
    a.check(not natural or cpu_profile is None, 'NATURAL_REJECTS_CPU_ROTARY_PROFILE')
    profile = dict(module=ROTARY_MODULE, source_sha256=ROTARY_SHA, operator_schema=ROTARY_SCHEMA)
    if not natural and cpu_profile is not None:
        profile = cpu_profile
    a.check(isinstance(aliases, list) and bool(aliases), 'DIRECT_FRAMES_REQUIRE_FULL_HELD_ALIASES')
    if not aliases:
        return dict(format='MISSING_HELD_ALIASES', rotary_children=[], frames=0)
    alias_pairs = {(x['path'], x['object_identity']) for x in aliases}
    a.check(len(alias_pairs) == len(aliases), 'HELD_ALIASES_UNIQUE')
    model_ids = [x['object_identity'] for x in aliases if x['path'] == '']
    a.check(len(model_ids) == 1, 'FRAME_HELD_ROOT_IDENTITY')
    frozen = (model_source or {}).get('manifest', {}).get('files', {})
    if model_source is not None:
        a.check(C.digest(model_source['manifest']) == model_source['source_sha256'], 'FRAME_MODEL_SOURCE_MANIFEST_HASH')
    by_alias = {}
    for row in aliases:
        by_alias.setdefault(row['object_identity'], []).append(row['path'])
    expected_proofs = group['independent_discovery'][0]['bounded_noncall_proofs']
    a.check(group['independent_discovery'][1]['bounded_noncall_proofs'] == expected_proofs,
            'ROTARY_TWO_HEALTHY_PROOFS_EXACT')
    children = set(); stable = {}; frame_total = 0; threads = set()
    for number, inv in enumerate(inventories):
        where = str(number)
        actual = [row for row in rows if row.get('complete_inventory') == inv]
        a.check(len(actual) == 1, 'FRAME_INVENTORY_JOINS_ONE_ACTUAL_REQUEST', where)
        if len(actual) != 1:
            continue
        row = actual[0]; scope = inv['frame_observer_scope']; modules = inv['modules']
        module_counts = {x['object_identity']: x['calls'] for x in modules}
        a.check(scope['owner'] == row['native_owner'] == group['native_owner'] and
            scope['native_generation'] == row['native_generation'] and scope['pid'] == worker['pid'] and
            scope['model_identity'] == model_ids[0] and C.positive(scope['thread']) and
            scope['phase'] == native_phase(row['phase']) and scope['profile'] == 'CURRENT_THREAD_ACTUAL_BOUND_FORWARD' and
            scope['serializable_authority'] is False and scope['qualification_granted'] is False,
            'ACTUAL_FRAME_WORKER_OWNER_PHASE', where)
        threads.add(scope['thread'])
        events = inv['actual_forward_frames']; gaps = inv['frame_binding_gaps']
        a.check(not inv['frame_failures'] and bool(events) and
            [x['sequence'] for x in events] == list(range(1, len(events)+1)), 'COMPLETE_NORMAL_FRAME_RETURNS', where)
        a.check(inv['bounded_noncall_proofs'] == expected_proofs, 'NONCALL_BRANCH_REPEATS_EVERY_FORWARD', where)
        frame_total += len(events); per_binding = Counter()
        for event in events:
            ident = event['module_object_identity']; source = event['source']; key = (ident, event['attribute'])
            a.check(event['attribute'] in ('forward', '_forward_method') and
                event['entry_kind'] in ('direct_frame', 'existing_module_entry') and
                event['returned'] is event['nonnull_return'] is True and (event['path'], ident) in alias_pairs and
                ident in module_counts and C.sha(source['sha256']) and bool(source['module']) and
                bool(source['path']) and bool(source['function']) and all(C.positive(source[k]) for k in
                    ('first_line', 'function_identity', 'code_identity', 'globals_identity')) and
                type(event['actual_rotary_returns']) is int and event['actual_rotary_returns'] >= 0,
                'ACTUAL_BOUND_FRAME_RETURN_AND_HELD_OBJECT', where)
            binding = {k:event[k] for k in ('path', 'module_object_identity', 'attribute', 'source')}
            stable.setdefault(key, binding)
            a.check(stable[key] == binding, 'FRAME_BOUND_CODE_SOURCE_STABLE', where)
            per_binding[key] += 1
            if event['entry_kind'] == 'direct_frame':
                a.check(frozen.get(source['path']) == source['sha256'], 'DIRECT_FRAME_FROZEN_MODEL_SOURCE_PIN', where)
        for gap in gaps:
            a.check((gap['path'], gap['module_object_identity']) in alias_pairs and
                gap['attribute'] in ('forward', '_forward_method') and isinstance(gap['reason'], str) and bool(gap['reason']),
                'UNVERIFIED_FRAME_GAP_RETAINED', where)
        for ident, count in module_counts.items():
            counts = [per_binding[(ident, name)] for name in ('forward', '_forward_method')]
            gap = any(x['module_object_identity'] == ident for x in gaps)
            a.check(all(n <= count for n in counts) and (max(counts) == count or gap),
                    'ACTUAL_MODULE_FRAME_COUNT_OR_EXPLICIT_GAP', where)
        for proof in inv['bounded_noncall_proofs']:
            parent, child = proof['parent_identity'], proof['child_identity']; binding = proof['delegate_binding']
            source = binding['source']; matching = [x for x in events if
                all(x.get(k) == v for k,v in binding.items())]
            expected = module_counts.get(parent, 0); children.add(child)
            a.check(proof['schema'] == 'demoncat.output39.bounded_rotary_noncall.v1' and
                proof['reason'] == ROTARY_REASON and proof['workload_only'] is True and
                proof['qualification_granted'] is False and child not in module_counts and
                not any(x['module_object_identity'] == child for x in events) and
                type(proof['helper_forward_calls']) is int and proof['helper_forward_calls'] == 0,
                'ROTARY_HELPER_ACTUALLY_UNCALLED', where)
            a.check(binding['attribute'] == '_forward_method' and binding['module_object_identity'] == parent and
                source['module'] == profile['module'] and source['function'] == 'AscendRotaryEmbedding.forward_oot' and
                source['sha256'] == profile['source_sha256'] and frozen.get(source['path']) == source['sha256'] and
                proof['operator'] == ROTARY_OPERATOR and C.positive(proof['operator_identity']) and
                proof['operator_schema'] == profile['operator_schema'] and
                not any(x['module_object_identity'] == parent and x['attribute'] == '_forward_method' for x in gaps),
                'ROTARY_REVIEWED_SOURCE_DELEGATE_AND_API', where)
            a.check(proof['parent_aliases'] == by_alias.get(parent) and proof['child_aliases'] == by_alias.get(child)
                and bool(proof['parent_aliases']) and binding['path'] == proof['parent_aliases'][0],
                'ROTARY_FULL_PARENT_CHILD_ALIAS_SETS', where)
            a.check(C.positive(expected) and len(matching) == expected and
                all(type(proof[k]) is int and proof[k] == expected for k in
                    ('actual_parent_calls', 'actual_delegate_returns', 'actual_public_rotary_returns')) and
                all(x['returned'] is x['nonnull_return'] is True and x['actual_rotary_returns'] == 1 for x in matching),
                'ROTARY_EACH_PARENT_FRAME_HAS_ONE_ACTUAL_API_RETURN', where)
        a.check(sum(x['actual_public_rotary_returns'] for x in inv['bounded_noncall_proofs']) <=
            inv['operator_counts'].get(ROTARY_OPERATOR, 0), 'ROTARY_RETURNS_WITHIN_COMPLETE_API_COUNT', where)
    a.check(len(threads) == 1, 'ONE_HELD_FRAME_THREAD')
    noncall_rows = [x for x in structural if x['reason'] == ROTARY_REASON]
    a.check({x['object_identity'] for x in noncall_rows} == children, 'NO_INVENTED_ROTARY_NONCALL_ROW')
    for row in noncall_rows:
        proofs = [x for x in expected_proofs if x['child_identity'] == row['object_identity']]
        a.check(bool(proofs) and row['actual_bounded_noncall'] == proofs and
            row['class_name'] == profile['module']+'.AscendApplyRotaryEmb' and
            row['parent_identity'] in {x['parent_identity'] for x in proofs} and
            row['path'] == by_alias[row['object_identity']][0] and not row['direct_children'] and
            row['actual_forward_calls'] == 0 and row['structural_noncall'] is True,
            'STRUCTURAL_ROTARY_ROW_REBUILT_FROM_ACTUAL_FRAMES')
    return dict(format='SOURCE_BOUND_ACTUAL_FRAMES', rotary_children=sorted(children),
                frames=frame_total, complete_forward_inventories=len(inventories),
                external_source_files_rehashed=False, qualification_granted=False)
