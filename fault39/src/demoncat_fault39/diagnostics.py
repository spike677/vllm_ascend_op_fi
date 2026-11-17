"""Explicit call-identity diagnostics; never a relaxed first-difference gate.

Inputs are raw collector records and resolved stack/construct context only.
Every record remains in the ledger. A unique source invocation is distinct
from equal arguments, equal tensor descriptions, and numerical equality.
"""
from collections import defaultdict
import hashlib
import json
import math
import re

from .msprobe import ComparisonPolicy

METRICS = ('Max', 'Min', 'Mean', 'Norm')
POLICY = ComparisonPolicy(absolute_tolerance=1e-8, relative_tolerance=1e-5)
RULES = {
    'version': 'source-invocation-diagnostics-v1',
    'identity': 'operator family + exact resolved source stack + exact construct entry',
    'uniqueness': 'exactly one record for that context on each side; repeated contexts are ambiguous',
    'order': 'all shared unique anchors must preserve relative order',
    'missing_context': 'missing/empty stack or missing construct refuses pairing; null construct is disclosed',
    'arguments': 'scalar argument changes keep invocation identity but forbid same-arguments attribution',
    'outputs': 'scalar output changes are observations, not argument identity changes',
    'tensor_statistics': 'only compare matching complete tensor descriptions; preserve null/nonfinite',
    'unpaired': 'retain every record and its surrounding shared anchors; never delete inserted calls',
    'global_first_difference': 'never supplied by this diagnostic; retain the unchanged strict report',
    'formula': 'abs(F-B) > 1e-8 + 1e-5*abs(B)',
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)


def safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return 'NaN' if math.isnan(value) else ('+Inf' if value > 0 else '-Inf')
    if isinstance(value, dict):
        return {key: safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [safe(child) for child in value]
    return value


def digest(value):
    return hashlib.sha256(canonical(safe(value)).encode()).hexdigest()


def family(operator):
    return re.sub(r'\.forward\.[0-9]+$', '.forward.#',
                  re.sub(r'\.[0-9]+\.forward$', '.#.forward', operator))


def resolve_stacks(stack):
    result = {}
    for group in stack.values():
        if (not isinstance(group, list) or len(group) != 2
                or not isinstance(group[0], list) or not isinstance(group[1], list)):
            raise ValueError('UNSUPPORTED_STACK_SCHEMA')
        for operator in group[0]:
            if not isinstance(operator, str) or operator in result:
                raise ValueError('DUPLICATE_OR_INVALID_STACK_OPERATOR')
            result[operator] = group[1]
    return result


def describe(value):
    if isinstance(value, dict):
        return {key: describe(child) for key, child in value.items()
                if not (value.get('type') == 'torch.Tensor' and key in METRICS)}
    if isinstance(value, list):
        return [describe(child) for child in value]
    return value


def records(value, path='', *, tensor):
    if isinstance(value, dict):
        if value.get('type') == 'torch.Tensor':
            if tensor:
                yield path, value
        elif 'type' in value:
            if not tensor:
                yield path, value
        else:
            for key, child in value.items():
                yield from records(child, path + '/' + key, tensor=tensor)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            yield from records(child, path + '/' + str(i), tensor=tensor)


def differences(a, b, path=''):
    if type(a) is not type(b):
        return [dict(path=path, reason='TYPE_CHANGED', baseline=a, fault=b)]
    if isinstance(a, dict):
        out = []
        if list(a) != list(b):
            out.append(dict(path=path, reason='KEYS_OR_ORDER_CHANGED', baseline=list(a), fault=list(b)))
        for key in a:
            if key in b:
                out.extend(differences(a[key], b[key], path + '/' + key))
            else:
                out.append(dict(path=path+'/'+key, reason='MISSING_FROM_FAULT', baseline=a[key]))
        out.extend(dict(path=path+'/'+key, reason='MISSING_FROM_BASELINE', fault=b[key])
                   for key in b if key not in a)
        return out
    if isinstance(a, list):
        out = []
        if len(a) != len(b):
            out.append(dict(path=path, reason='LENGTH_CHANGED', baseline=len(a), fault=len(b)))
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(differences(x, y, path+'/'+str(i)))
        return out
    return [] if a == b else [dict(path=path, reason='VALUE_CHANGED', baseline=a, fault=b)]


def numeric(value):
    if value is None:
        return 'NULL', None
    if isinstance(value, bool):
        return 'INVALID', None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return 'INVALID', None
    if math.isnan(number):
        return 'NAN', None
    return ('INFINITE' if math.isinf(number) else 'FINITE'), number


def compare_metric(a, b):
    result = dict(baseline=safe(a), fault=safe(b), exact_change=None,
                  exceeds_tolerance=None, relative_change=None)
    ak, av = numeric(a)
    bk, bv = numeric(b)
    if 'NULL' in (ak, bk):
        return dict(result, status='UNOBSERVABLE')
    if 'INVALID' in (ak, bk) or 'NAN' in (ak, bk):
        return dict(result, status='UNCOMPARABLE')
    if 'INFINITE' in (ak, bk):
        return dict(result, status='NONFINITE_OBSERVED', exact_change=av != bv)
    delta = bv-av
    threshold = POLICY.absolute_tolerance + POLICY.relative_tolerance*abs(av)
    return dict(result, status='OBSERVABLE_FINITE', exact_change=av != bv,
                signed_difference=delta, absolute_difference=abs(delta), threshold=threshold,
                exceeds_tolerance=abs(delta) > threshold,
                relative_change=abs(delta)/abs(av) if av != 0 else None,
                relative_change_reason=None if av != 0 else 'ZERO_BASELINE_NO_RELATIVE_VALUE')


def _context_map(data, stack, construct):
    groups, unavailable = defaultdict(list), {}
    for name in data:
        if name not in stack or not stack[name]:
            unavailable[name] = 'STACK_MISSING_OR_EMPTY'
        elif name not in construct:
            unavailable[name] = 'CONSTRUCT_MISSING'
        else:
            groups[canonical([family(name), stack[name], construct[name]])].append(name)
    return groups, unavailable


def diagnose_records(baseline, fault, baseline_stack, fault_stack, baseline_construct, fault_construct,
                     *, capture_metadata_aligned):
    """Return a complete diagnostic ledger without claiming global attribution."""
    if type(capture_metadata_aligned) is not bool:
        raise ValueError('EXPLICIT_METADATA_GATE_REQUIRED')
    bs, fs = resolve_stacks(baseline_stack), resolve_stacks(fault_stack)
    bg, bu = _context_map(baseline, bs, baseline_construct)
    fg, fu = _context_map(fault, fs, fault_construct)
    bi = {name: i for i, name in enumerate(baseline)}
    fi = {name: i for i, name in enumerate(fault)}
    candidates = [(left[0], fg[key][0]) for key, left in bg.items()
                  if len(left) == 1 and len(fg.get(key, [])) == 1]
    candidates.sort(key=lambda row: bi[row[0]])
    fault_order = [fi[row[1]] for row in candidates]
    order_equal = fault_order == sorted(fault_order)
    eligible = capture_metadata_aligned and order_equal
    pairs = candidates if eligible else []
    bm, fm = dict(pairs), {right: left for left, right in pairs}
    report = dict(schema='demoncat.msprobe.call_diagnostics.v1', rules=RULES,
        rules_sha256=digest(RULES), strict_acceptance_overridden=False,
        global_first_difference=None, source='RAW_STATISTICS_STACK_CONSTRUCT_ONLY',
        native_logs_read=False, tensor_payloads_read=False, request_mapping='UNPROVEN_REQUEST_MAPPING',
        capture_metadata_aligned=capture_metadata_aligned, anchor_order_preserved=order_equal,
        counts=dict(baseline=len(baseline), fault=len(fault), matched=len(pairs)),
        ledger=dict(baseline=[], fault=[]), pairs=[], unpaired_windows=[],
        changed_tensor_fields=[], changed_scalar_outputs=[], unobservable_fields=[],
        first_local_tensor_change=None)
    for side, data, group, other_groups, unavailable, mapping, stacks, construct in (
            ('baseline', baseline, bg, fg, bu, bm, bs, baseline_construct),
            ('fault', fault, fg, bg, fu, fm, fs, fault_construct)):
        reasons = dict(unavailable)
        for context, names in group.items():
            other = other_groups.get(context, [])
            for name in names:
                if len(names) != 1 or len(other) > 1:
                    reasons[name] = 'AMBIGUOUS_REPEATED_CONTEXT'
                elif not other:
                    reasons[name] = 'NO_SAME_SOURCE_CONTEXT'
        for index, (name, value) in enumerate(data.items()):
            reason = (None if name in mapping else
                      ('CAPTURE_METADATA_UNALIGNED' if not capture_metadata_aligned else
                       'ANCHOR_ORDER_CHANGED' if not order_equal else reasons.get(name)))
            row = dict(index=index, operator=name, record_sha256=digest(value),
                counterpart=mapping.get(name), status='DIAGNOSTIC_PAIR' if name in mapping else 'UNPAIRED',
                reason=reason, construct_null=construct.get(name) is None)
            report['ledger'][side].append(row)
            if name not in mapping:
                positions = [(i, key) for i, key in enumerate(data) if key in mapping]
                prior = [key for i, key in positions if i < index]
                after = [key for i, key in positions if i > index]
                report['unpaired_windows'].append(dict(side=side, index=index, operator=name, reason=reason,
                    previous_shared_anchor=prior[-1] if prior else None,
                    next_shared_anchor=after[0] if after else None,
                    first_raw_stack_frame=(stacks.get(name) or [None])[0],
                    scalar_records=[dict(path=path, record=record) for path, record in records(value, tensor=False)],
                    tensor_observations=[dict(path=path, description=describe(record),
                        statistics={field: safe(record[field]) for field in METRICS})
                        for path, record in records(value, tensor=True)],
                    producer_consumer_identity='UNPROVEN', numerical_comparison_accepted=False))
    for left, right in pairs:
        a, b = baseline[left], fault[right]
        changes = differences(describe(a), describe(b))
        input_changes = [row for row in changes if not row['path'].startswith('/output')]
        item = dict(baseline_operator=left, fault_operator=right, baseline_index=bi[left], fault_index=fi[right],
            identity_status='UNIQUE_SAME_SOURCE_INVOCATION_DIAGNOSTIC_ONLY', same_operator_id=left == right,
            argument_descriptions_unchanged=not input_changes, nonstatistical_changes=changes,
            construct_null=baseline_construct[left] is None,
            numerical_attribution='SAME_ARGUMENT_DESCRIPTIONS_ONLY' if not input_changes
                                  else 'CONDITION_CHANGED_NOT_SAME_ARGUMENTS',
            tensor_description_mismatches=[], scalar_output_changes=[])
        scalars_a, scalars_b = dict(records(a, tensor=False)), dict(records(b, tensor=False))
        for path, record in scalars_a.items():
            if path.startswith('/output/') and path in scalars_b and record != scalars_b[path]:
                out = dict(baseline_operator=left, fault_operator=right, path=path,
                    baseline=record, fault=scalars_b[path], status='OBSERVED_SCALAR_OUTPUT_CHANGE')
                item['scalar_output_changes'].append(out)
                report['changed_scalar_outputs'].append(out)
        tensors_a, tensors_b = dict(records(a, tensor=True)), dict(records(b, tensor=True))
        for path in sorted(set(tensors_a) | set(tensors_b)):
            x, y = tensors_a.get(path), tensors_b.get(path)
            if x is None or y is None or describe(x) != describe(y):
                item['tensor_description_mismatches'].append(path)
                continue
            for field in METRICS:
                values = compare_metric(x[field], y[field])
                entry = dict(baseline_operator=left, fault_operator=right, path=path, field=field,
                    same_argument_descriptions=not input_changes, not_global_first_difference=True, **values)
                if values['exceeds_tolerance'] is True:
                    report['changed_tensor_fields'].append(entry)
                    if report['first_local_tensor_change'] is None:
                        report['first_local_tensor_change'] = entry
                if values['status'] != 'OBSERVABLE_FINITE':
                    report['unobservable_fields'].append(entry)
        report['pairs'].append(item)
    for side, data in (('baseline', baseline), ('fault', fault)):
        ledger = report['ledger'][side]
        assert len(ledger) == len(data) and len({row['operator'] for row in ledger}) == len(data)
    report['status'] = 'DIAGNOSTIC_ONLY_NO_GLOBAL_FIRST_DIFFERENCE'
    return safe(report)
