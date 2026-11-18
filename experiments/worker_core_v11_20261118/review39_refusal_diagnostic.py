"""CPU-only comparison of a pinned compact diagnostic; no qualification grant.

Full Native group originals/build provenance/recovery must still be reviewed
independently. This file does not import a provider or select any fault target.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from .output39_refusal_diagnostic import digest


def _review(value):
    issues = []
    checks = 0
    def check(ok, name):
        nonlocal checks
        checks += 1
        if not ok:
            issues.append(name)
    rows = value.get('records', [])
    check(value.get('schema') == 'demoncat.output39.refusal_diagnostic.v1', 'SCHEMA')
    check(value.get('plan_sha256') == digest(value.get('plan')), 'FROZEN_PLAN_DIGEST')
    check(not value.get('errors'), 'NO_DIAGNOSTIC_ERRORS')
    check(all(value.get(k) is False for k in ('c2','c3','e5','qualification_granted')), 'NO_SERIALIZED_AUTHORITY')
    expected = {('healthy','refused'), ('single_predecessor_fault','predecessor'),
                ('single_predecessor_fault','refused'), ('cumulative_fault','predecessor'),
                ('cumulative_fault','refused')}
    counts = Counter((row.get('arm'), row.get('target')) for row in rows)
    check(set(counts) == expected and all(n == 1 for n in counts.values()), 'EXACT_FIVE_OBSERVATIONS')
    check([row.get('sequence') for row in rows] == list(range(1, len(rows)+1)), 'ACTUAL_OBSERVATION_SEQUENCE')
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        return dict(checks=checks, issues=issues, pattern='UNRESOLVED', c2=False, c3=False, e5=False)
    by = {(row['arm'],row['target']):row for row in rows}
    healthy = by['healthy','refused']
    single = by['single_predecessor_fault','refused']
    cumulative = by['cumulative_fault','refused']
    phases = dict(healthy='c2_lossless', single_predecessor_fault='c2_fault', cumulative_fault='c3_fault')
    request_ids, inputs = {}, set()
    for row in rows:
        binding, span = row.get('binding',{}), row.get('span',{})
        arm, target = row['arm'], row['target']
        mapping = binding.get('request_mapping',{})
        request_id = binding.get('request_id')
        check(bool(request_id) and mapping.get('actual_internal_request_id') == request_id,
              'ACTUAL_REQUEST_JOIN:'+arm+':'+target)
        check(binding.get('phase') == mapping.get('phase') == phases[arm], 'ACTUAL_PHASE:'+arm+':'+target)
        request_ids.setdefault(arm, set()).add(request_id)
        inputs.add(mapping.get('input_sha256'))
        check(span.get('full_span') is True and span.get('raw_payload_saved') is False,
              'COMPLETE_SPAN_NO_PRIVATE_ARRAY:'+arm+':'+target)
        check(span.get('representation') == 'ALL_LOGICAL_SPAN_BF16_BITS_LITTLE_ENDIAN', 'BF16_REPRESENTATION')
        check(type(span.get('bytes')) is int and span.get('bytes') == 2*span.get('elements',-1), 'SPAN_BYTES')
        field_counts = span.get('counts',{})
        check(set(field_counts) == {'finite','positive_infinity','negative_infinity','quiet_nan','signaling_nan'}
              and all(type(n) is int and n >= 0 for n in field_counts.values()), 'CLASS_COUNTS')
        check(sum(field_counts.values()) == span.get('elements'), 'COMPLETE_COUNT_DENOMINATOR')
        check(span.get('nonfinite_elements') == span.get('elements',0)-field_counts.get('finite',0), 'NONFINITE_COUNT')
        first = span.get('first_nonfinite')
        if span.get('nonfinite_elements') == 0:
            check(first is None, 'FINITE_HAS_NO_NONFINITE_INDEX')
        else:
            check(isinstance(first,dict) and type(first.get('index')) is int
                  and 0 <= first['index'] < span['elements'], 'NONFINITE_FIRST_INDEX')
            if isinstance(first,dict):
                bits = int(first.get('bits_hex','-1'),16)
                label = ('negative_infinity' if bits & 0x8000 else 'positive_infinity') if bits & 0x7f == 0 \
                    else ('quiet_nan' if bits & 0x40 else 'signaling_nan')
                check(0 <= bits <= 0xffff and bits & 0x7f80 == 0x7f80
                      and first.get('classification') == label and field_counts.get(label,0)>0,
                      'NONFINITE_FIRST_BIT_CLASSIFICATION')
        check(isinstance(span.get('sha256'),str) and len(span['sha256']) == 64, 'FULL_SPAN_SHA256_PRESENT')
        interval = row.get('interval_monotonic_ns', [])
        check(len(interval) == 2 and all(type(t) is int for t in interval) and interval[0] <= interval[1], 'READ_TIME')
        if row.get('origin') == 'SAME_NATIVE_LEASE_ORIGINAL':
            native, opened, applied = row.get('native',{}), row.get('open_event',{}), row.get('apply_event',{})
            check(all(type(item.get('return_code')) is int for item in (opened,applied)), 'TYPED_NATIVE_RETURN_CODES')
            check(opened.get('return_code') == 0 and opened.get('returned') is True, 'ORIGINAL_OPEN_SUCCEEDED')
            check(all(native.get(k) == opened.get(k) == applied.get(k) for k in ('handle','owner','generation')),
                  'SAME_OPEN_APPLY_ORIGINAL_LEASE')
            check(native.get('owner') == binding.get('native_owner')
                  and native.get('generation') == binding.get('native_generation'), 'ACTUAL_NATIVE_SCOPE')
            check(native.get('span_bytes') == opened.get('bytes') == span.get('bytes'), 'EXACT_ORIGINAL_SPAN')
            check(native.get('tensor_address') == opened.get('data_ptr') == binding.get('geometry',{}).get('data_ptr'),
                  'EXACT_ACTUAL_TENSOR_POINTER')
        if target == 'predecessor':
            applied = row.get('apply_event',{})
            check(row.get('origin') == 'SAME_NATIVE_LEASE_ORIGINAL' and span.get('nonfinite_elements') == 0,
                  'FINITE_PREDECESSOR_ORIGINAL:'+arm)
            check(applied.get('return_code') == 0 and applied.get('mode') == 1
                  and applied.get('indices') == [0] and applied.get('changed_elements') == 1,
                  'ONE_ACTUAL_PREDECESSOR_INF_WRITE:'+arm)
            check(type(applied.get('mode')) is int and type(applied.get('changed_elements')) is int
                  and all(type(index) is int for index in applied.get('indices',[])), 'TYPED_ACTUAL_MUTATION')
        if (arm, target) != ('cumulative_fault','refused'):
            consumers = row.get('consumer_observations', [])
            check(bool(consumers) and all(c.get('original_returned') is True for c in consumers), 'ACTUAL_CONSUMER:'+arm+':'+target)
            check(any(0 in c.get('selected_elements_read',[]) and any(
                view.get('storage_identity') == binding.get('geometry',{}).get('storage_identity')
                for view in c.get('operand_views',[])) for c in consumers), 'SAME_STORAGE_VALUE_READ:'+arm+':'+target)
    check(all(len(ids)==1 for ids in request_ids.values()) and len(set.union(*request_ids.values())) == 3,
          'THREE_DISTINCT_ACTUAL_REQUESTS')
    check(len(inputs) == 1 and isinstance(next(iter(inputs)),str) and len(next(iter(inputs))) == 64,
          'SAME_ACTUAL_MODEL_INPUT_BYTES')
    check(len({tuple(row['binding'].get('target_key',[])) for row in (healthy,single,cumulative)}) == 1,
          'SAME_REFUSED_PRODUCER_IDENTITY')
    check(healthy.get('origin') == single.get('origin') == 'ACTUAL_FRESH_READ_SPAN', 'CONTROL_FRESH_ORIGIN')
    check(cumulative.get('origin') == 'SAME_NATIVE_LEASE_ORIGINAL', 'REFUSAL_SAME_ORIGINAL_ORIGIN')
    check(healthy['span']['nonfinite_elements'] == 0 and single['span']['nonfinite_elements'] > 0,
          'HEALTHY_FINITE_SINGLE_PREDECESSOR_NONFINITE')
    check(single['span']['sha256'] == cumulative['span']['sha256'] != healthy['span']['sha256'],
          'COMPLETE_COUNTERFACTUAL_SPAN_MATCH')
    first_single, first_cumulative = by['single_predecessor_fault','predecessor'], by['cumulative_fault','predecessor']
    check(first_single['span']['sha256'] == first_cumulative['span']['sha256']
          and first_single['binding']['target_key'] == first_cumulative['binding']['target_key'],
          'SAME_PREDECESSOR_ORIGINAL_AND_OBJECT')
    check(cumulative.get('apply_event',{}).get('return_code') == 4
          and cumulative.get('native',{}).get('applied') is False, 'UNCHANGED_NATIVE_REFUSAL')
    check(cumulative.get('consumer_status') == 'NOT_REACHED_AFTER_ACTUAL_APPLY_REFUSAL'
          and cumulative.get('consumer_observations') == [], 'FAILED_CONSUMER_NOT_INVENTED')
    return dict(checks=checks, issues=issues,
        pattern='BF16_COUNTERFACTUAL_PATTERN_CONSISTENT' if not issues else 'UNRESOLVED',
        explicit_cpu_contract=value.get('explicit_cpu_contract'),
        requires_independent=['complete original RPC/enqueue/request joins and fixed inputs/weights',
            'all Native calls in single-F and cumulative request, no hidden intervention',
            'actual fresh DSO/source/build/mapped-address provenance', 'OFF/health/owner/lease/trace cleanup'],
        causal_qualification_granted=False, c2=False, c3=False, e5=False)


def review(value):
    try:
        return _review(value)
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        return dict(checks=0, issues=['MALFORMED_DIAGNOSTIC:'+type(error).__name__], pattern='UNRESOLVED',
                    causal_qualification_granted=False, c2=False, c3=False, e5=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = args.report.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != args.sha256:
        raise SystemExit('ORIGINAL_DIAGNOSTIC_SHA_MISMATCH')
    result = review(json.loads(raw))
    result['original_sha256'] = actual
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')
    return int(bool(result['issues']))


if __name__ == '__main__':
    raise SystemExit(main())
