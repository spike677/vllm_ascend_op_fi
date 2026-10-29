"""Exact FP16 endpoint contrasts and source-plan comparisons; no device imports.

Labels select predeclared comparisons only. They never supply measured values,
actual tiling, or a floating-point reduction order absent from the evidence.
"""
import hashlib
import json
import struct

PROFILES = {'base': (128, 8), 'm_only': (64, 8), 'r_only': (128, 4), 'joint': (64, 4)}
CONTRASTS = {
    'M_at_ratio8': {'m_only': 1, 'base': -1},
    'M_at_ratio4': {'joint': 1, 'r_only': -1},
    'ratio_at_M128': {'r_only': 1, 'base': -1},
    'ratio_at_M64': {'joint': 1, 'm_only': -1},
    'M_ratio_interaction': {'joint': 1, 'm_only': -1, 'r_only': -1, 'base': 1},
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def fp16_units(raw):
    """Every finite binary16 is an exact signed integer multiple of 2**-24."""
    require(len(raw) > 0 and len(raw) % 2 == 0, 'Empty or malformed FP16 endpoint')
    result = []
    for (bits,) in struct.iter_unpack('<H', raw):
        exponent, fraction = (bits >> 10) & 31, bits & 1023
        require(exponent != 31, 'Non-finite FP16 endpoint cannot support finite contrasts')
        magnitude = fraction if exponent == 0 else (1024 + fraction) << (exponent - 1)
        result.append(-magnitude if bits & 32768 else magnitude)
    return result


def exact_contrast(vectors, coefficients):
    require(set(vectors) >= set(coefficients), 'Missing factorial cell')
    require(coefficients and all(type(x) is int for x in coefficients.values()), 'Invalid exact contrast')
    length = len(vectors[next(iter(coefficients))])
    require(length > 0 and all(len(vectors[k]) == length for k in coefficients), 'Endpoint shape mismatch')
    return [sum(c * vectors[k][i] for k, c in coefficients.items()) for i in range(length)]


def describe_contrast(values):
    nonzero = [i for i, x in enumerate(values) if x]
    # This hash commits to exact signed integer values, not rounded summaries.
    serialized = b''.join(struct.pack('<q', x) for x in values)
    return {'elements': len(values), 'nonzero_elements': len(nonzero),
            'max_abs_exact_units': max(map(abs, values), default=0),
            'max_abs': max(map(abs, values), default=0) * 2.0 ** -24,
            'sum_abs_exact_units': sum(map(abs, values)),
            'unit': '2^-24', 'exact_int64_le_sha256': digest(serialized),
            'first_nonzero_flat_index': nonzero[0] if nonzero else None,
            'observation': 'NUMERICAL_DIFFERENCE_OBSERVED' if nonzero else 'NO_NUMERICAL_DIFFERENCE_OBSERVED'}


def factorial(rows):
    """Rows are validated cells, with independently checked actual tiling/bytes.

    An unstable repeat is retained as an observation but prevents factorial
    attribution. Signed zero can change bits without changing exact values.
    """
    groups = {}
    for row in rows:
        key = (row['batch_size'], row['profile'])
        require(key[0] in (1, 8) and key[1] in PROFILES, 'Unexpected factorial cell')
        groups.setdefault(key, []).append(row)
    require(set(groups) == {(b, p) for b in (1, 8) for p in PROFILES}, 'Incomplete 2x2x2 factorial matrix')
    repeat_report, chosen, unstable = [], {}, False
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda x: x['repeat_index'])
        require(len(group) >= 2 and [x['repeat_index'] for x in group] == list(range(len(group))),
                'Every profile/batch needs distinct contiguous repeats beginning at zero')
        require(len({x['call_identity'] for x in group}) == len(group), 'Reused call identity within repeats')
        fields = ('target_output_sha256', 'endpoint_sha256', 'profile_fingerprint')
        stable = all(len({x[k] for x in group}) == 1 for k in fields)
        unstable |= not stable
        repeat_report.append({'batch_size': key[0], 'profile': key[1], 'repeats': len(group),
                              'stable': stable, 'varying_fields': [k for k in fields if len({x[k] for x in group}) != 1]})
        chosen[key] = group[0]
    result = {'status': 'NOT_IDENTIFIED_UNSTABLE_REPEATS' if unstable else 'ANALYZED_FIXED_INPUT_FACTORIAL',
              'repeat_coverage': repeat_report, 'primary_endpoint': 'batch0, query2047, all heads/channels',
              'secondary_endpoint': 'full batch0 output SHA only', 'causal_attribution': 'NOT_IDENTIFIED' if unstable else 'REPEATABLE_INTERVENTION_CONTRASTS_ONLY',
              'micro_arithmetic_root_cause': 'NOT_IDENTIFIED', 'historical_defect_reproduced': False}
    if unstable:
        result['contrasts'] = None
        return result
    unit_vectors = {(b, p): fp16_units(row['endpoint_raw']) for (b, p), row in chosen.items()}
    within, exact = {}, {}
    for b in (1, 8):
        matrix = {p: unit_vectors[b, p] for p in PROFILES}
        within[str(b)] = {}
        for name, coefficients in CONTRASTS.items():
            values = exact_contrast(matrix, coefficients)
            exact[b, name] = values
            within[str(b)][name] = {'coefficients': coefficients, **describe_contrast(values)}
    batch, batch_interactions = {}, {}
    for p in PROFILES:
        values = exact_contrast({'B1': unit_vectors[1, p], 'B8': unit_vectors[8, p]}, {'B8': 1, 'B1': -1})
        batch[p] = {**describe_contrast(values), 'full_target_bit_difference': chosen[1, p]['target_output_sha256'] != chosen[8, p]['target_output_sha256']}
    for name in CONTRASTS:
        values = exact_contrast({'B1': exact[1, name], 'B8': exact[8, name]}, {'B8': 1, 'B1': -1})
        batch_interactions[name] = describe_contrast(values)
    pairs = [('base', 'm_only'), ('r_only', 'joint'), ('base', 'r_only'), ('m_only', 'joint'), ('base', 'joint')]
    bit_effects = {str(b): [{'before': a, 'after': z,
                            'full_target_bit_difference': chosen[b, a]['target_output_sha256'] != chosen[b, z]['target_output_sha256'],
                            'last_query_bit_difference': chosen[b, a]['endpoint_sha256'] != chosen[b, z]['endpoint_sha256']}
                           for a, z in pairs] for b in (1, 8)}
    result.update(contrasts=within, batch_effects=batch, batch_interactions=batch_interactions,
                  bit_effects=bit_effects, all_numeric_contrasts_zero=not any(x for v in exact.values() for x in v) and
                  not any(x['nonzero_elements'] for x in batch.values()))
    return result


def query_orders(trace):
    """Expand audited target-batch task intervals into independently comparable queries.

    The ordered legal key groups describe online update boundaries. Computed
    groups may include masked future keys. Neither is a hardware instruction
    trace or a complete Cube/Softmax implementation reduction-tree description.
    """
    require(trace['every_real_query_owned_once'] is True and trace['each_query_legal_keys_covered_once'] is True,
            'Unproven source-derived query/key coverage')
    require(trace['query_split_between_cores'] is False, 'Split query across cores is outside this experiment')
    dims = trace['profile_dimensions']
    sequence = dims['s1']
    require(sequence == dims['s2'] == 2048, 'Only audited equal-length causal prefill supported')
    orders = {}
    for row in trace['target_batch_tasks']:
        lo, hi = row['query_interval']
        require(0 <= lo < hi <= sequence, 'Invalid target query interval')
        groups = row['computed_key_groups']
        require(groups and groups[0][0] == 0 and all(a[1] == b[0] for a, b in zip(groups, groups[1:])),
                'Computed groups have missing/overlapping keys')
        for q in range(lo, hi):
            legal = [[a, min(b, q + 1)] for a, b in groups if a <= q]
            require(legal and legal[0][0] == 0 and legal[-1][1] == q + 1 and sum(b-a for a,b in legal) == q+1,
                    'A query does not cover every causal key exactly once')
            key = (row['query_head'], q)
            require(key not in orders, 'Duplicate target query owner')
            vec = []
            # Matched kernel ComputeBmm1Tail 1057-1065, Vec1 1280-1283,
            # Softmax shape 1922-1926; audited util.h aligns FP16 to 16.
            # Include computed masked-only groups: these still execute before
            # masking, and must not be confused with duplicate legal key work.
            for first, last in groups:
                width = last-first
                aligned = (width+15)//16*16
                base = min((1024//aligned)*8, hi-lo)
                require(base > 0, 'Unsupported dynamic Vec1 row geometry')
                split_n = (hi-lo+base-1)//base
                index = (q-lo)//base
                real = min(base, hi-lo-index*base)
                vec.append({'computed_key_interval': [first,last], 'computed_width': width,
                            'legal_key_count': max(0, min(last,q+1)-first),
                            'masked_key_count': width-max(0,min(last,q+1)-first),
                            's2_aligned': aligned, 'vec1_base': base, 'vec1_real': real,
                            'real_split_n': split_n, 'row_subtile_index': index,
                            'softmax_shape': [real,aligned], 'softmax_original_shape': [real,width]})
            orders[key] = {'legal_key_groups': legal, 'computed_key_groups': groups,
                           'task_query_interval': [lo, hi], 'aiv': row['aiv'], 'task': row['task'],
                           'use_next_l1': row['use_next_l1'], 'vec1_groups': vec,
                           'softmax_row_geometry': [{k:x[k] for k in ('computed_width','masked_key_count','softmax_shape','softmax_original_shape')} for x in vec],
                           'vec1_row_assignment': [{k:x[k] for k in ('real_split_n','row_subtile_index')} for x in vec]}
    require(set(orders) == {(h, q) for h in range(dims['n1']) for q in range(sequence)}, 'Missing target query/head')
    return orders


def compare_orders(before, after):
    a, b = query_orders(before), query_orders(after)
    require(set(a) == set(b), 'Target query shape differs')
    fields = ('legal_key_groups', 'computed_key_groups', 'task_query_interval', 'aiv', 'use_next_l1',
              'softmax_row_geometry', 'vec1_row_assignment')
    counts = {field: 0 for field in fields}
    examples = {}
    tail = {field: 0 for field in fields}
    for key in sorted(a):
        for field in fields:
            if a[key][field] != b[key][field]:
                counts[field] += 1
                if key[1] == 2047:
                    tail[field] += 1
                examples.setdefault(field, {'head': key[0], 'query': key[1], 'before': a[key][field], 'after': b[key][field]})
    return {'target_queries_compared': len(a), 'changed_query_counts': counts,
            'last_query_changed_head_counts': tail, 'first_changed_examples': examples,
            'legal_key_set_unchanged': True, 'query_split_between_cores': False,
            'floating_point_instruction_order': 'NOT_IDENTIFIED',
            'interpretation': 'Actual source-derived grouping and scheduling only; matching groups do not establish identical arithmetic.'}


def full_sequence_factorial(rows, chunk_elements=65536):
    """All target queries, exact int64 contrasts, bounded memory, CPU NumPy only.

    Rows' private output files must already have been SHA/size verified. This
    function rechecks their SHA while streaming, preventing file replacement
    between the evidence gate and arithmetic. No private values are returned.
    """
    import contextlib
    import numpy as np
    require(chunk_elements > 0 and chunk_elements % 128 == 0, 'Chunk must preserve full query rows')
    groups = {}
    for row in rows:
        groups.setdefault((row['batch_size'], row['profile']), []).append(row)
    require(set(groups) == {(b, p) for b in (1, 8) for p in PROFILES}, 'Incomplete full-output matrix')
    chosen = {}
    for key, group in groups.items():
        require(len(group) >= 2, 'Missing full-output repeat')
        if len({(r['target_output_sha256'], r['profile_fingerprint']) for r in group}) != 1:
            return {'status': 'NOT_IDENTIFIED_UNSTABLE_REPEATS', 'contrasts': None}
        chosen[key] = sorted(group, key=lambda x: x['repeat_index'])[0]
    shape = (16, 2048, 128)
    require(all(tuple(x['target_output_shape']) == shape for x in chosen.values()), 'Unsupported full target shape')
    coefficients = {}
    for b in (1, 8):
        for name, c in CONTRASTS.items():
            coefficients[f'B{b}.{name}'] = {(b, p): v for p, v in c.items()}
    for p in PROFILES:
        coefficients[f'Batch.{p}'] = {(8, p): 1, (1, p): -1}
    for name, c in CONTRASTS.items():
        coefficients[f'BatchInteraction.{name}'] = {(b, p): v * (1 if b == 8 else -1) for b in (1, 8) for p, v in c.items()}
    accum = {k: {'nonzero_elements': 0, 'sum_abs_exact_units': 0, 'max_abs_exact_units': 0,
                 'nonzero_by_head': np.zeros(16, dtype=np.int64),
                 'nonzero_by_query': np.zeros(2048, dtype=np.int64),
                 'max_abs_units_by_query': np.zeros(2048, dtype=np.int64),
                 'hash': hashlib.sha256(), 'first_nonzero_coordinates': []} for k in coefficients}
    for name,c in coefficients.items():
        if len(c)==2 and sorted(c.values())==[-1,1]:
            accum[name].update(bit_different_elements=0,first_bit_different_coordinates=[])
    total = 16 * 2048 * 128
    # A finite FP16 magnitude is below 2**40 units. Eight +/- terms fit int64.
    with contextlib.ExitStack() as stack:
        streams = {key: stack.enter_context(open(row['target_private_path'], 'rb')) for key, row in chosen.items()}
        hashes = {key: hashlib.sha256() for key in chosen}
        for start in range(0, total, chunk_elements):
            count = min(chunk_elements, total-start)
            vectors,raw_bits = {},{}
            for key, stream in streams.items():
                raw = stream.read(count*2)
                require(len(raw) == count*2, 'Truncated private full target output')
                hashes[key].update(raw)
                bits = np.frombuffer(raw, dtype='<u2').astype(np.int64)
                raw_bits[key]=bits
                exp, fraction = (bits >> 10) & 31, bits & 1023
                require(not np.any(exp == 31), 'Non-finite full output')
                values = np.where(exp == 0, fraction, (1024+fraction) << np.maximum(exp-1, 0))
                vectors[key] = np.where(bits & 32768, -values, values)
            query_rows = np.arange(start//128, (start+count)//128)
            heads, queries = query_rows//2048, query_rows % 2048
            for name, c in coefficients.items():
                values = np.zeros(count, dtype=np.int64)
                for key, coefficient in c.items():
                    values += coefficient * vectors[key]
                row, absolute = accum[name], np.abs(values)
                row['hash'].update(values.astype('<i8', copy=False).tobytes())
                nz = np.flatnonzero(values)
                row['nonzero_elements'] += int(nz.size)
                row['sum_abs_exact_units'] += int(absolute.sum(dtype=np.int64))
                row['max_abs_exact_units'] = max(row['max_abs_exact_units'], int(absolute.max(initial=0)))
                counts = np.count_nonzero(values.reshape(-1, 128), axis=1)
                maxima = absolute.reshape(-1, 128).max(axis=1)
                np.add.at(row['nonzero_by_head'], heads, counts)
                np.add.at(row['nonzero_by_query'], queries, counts)
                np.maximum.at(row['max_abs_units_by_query'], queries, maxima)
                for pos in nz[:max(0, 16-len(row['first_nonzero_coordinates']))]:
                    linear = start+int(pos)
                    row['first_nonzero_coordinates'].append([linear//(2048*128), linear//128 % 2048, linear % 128])
                if 'bit_different_elements' in row:
                    left,right=list(c)
                    changed=np.flatnonzero(raw_bits[left]!=raw_bits[right])
                    row['bit_different_elements']+=int(changed.size)
                    for pos in changed[:max(0,16-len(row['first_bit_different_coordinates']))]:
                        linear=start+int(pos)
                        row['first_bit_different_coordinates'].append([linear//(2048*128),linear//128%2048,linear%128])
        for key, stream in streams.items():
            require(stream.read(1) == b'', 'Trailing bytes in private full output')
            require(hashes[key].hexdigest() == chosen[key]['target_output_sha256'], 'Full output changed while streaming')
    for row in accum.values():
        row['exact_int64_le_sha256'] = row.pop('hash').hexdigest()
        for k in ('nonzero_by_head', 'nonzero_by_query', 'max_abs_units_by_query'):
            row[k] = row[k].tolist()
        row['max_abs'] = row['max_abs_exact_units'] * 2.0**-24
        row['elements'] = total
        row['unit'] = '2^-24'
    return {'status': 'ANALYZED_FULL_SEQUENCE_FACTORIAL', 'shape': list(shape), 'contrasts': accum,
            'all_numeric_contrasts_zero': not any(x['nonzero_elements'] for x in accum.values()),
            'source': 'SHA-verified full batch0 private output bytes',
            'floating_point_instruction_cause': 'NOT_IDENTIFIED',
            'outputs_or_tensor_values_embedded': False}
