"""Raw-only fixed-tolerance review, with an independent Decimal cross-check.

No argument accepts launch records, injector logs, expected locations or tensor
payloads. Only two statistics triplets and their config/collector are read.
"""
import argparse
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
from pathlib import Path

RTOL, ATOL = Decimal('1e-5'), Decimal('1e-8')
METRICS = ('Max', 'Mean', 'Min', 'Norm')
COMPARATOR_SHA = 'd5633600f8bde98590c4eaf367eb16d76d4e6c9ad5b0cb2f82d434ae6a377ca2'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_rows(document):
    result = []
    def visit(value, path):
        if isinstance(value, dict):
            if value.get('type') == 'torch.Tensor':
                require(set(METRICS).issubset(value), 'INCOMPLETE_STATISTICS')
                result.append((path, value))
            else:
                for key, child in value.items():
                    visit(child, path + '/' + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + '/' + str(index))
    for operator, value in document['data'].items():
        visit(value, operator)
    return result


def classify(value, dtype, field):
    if value is None:
        return 'null', None
    if type(value) is bool:
        return ('finite', Decimal(int(value))) if dtype == 'torch.bool' and field in ('Max', 'Min') else ('unsupported', None)
    try:
        require(type(value) in (int, Decimal) or isinstance(value, str) and
                value.strip().lower() in ('nan', '+nan', '-nan', 'inf', '+inf', '-inf',
                                          'infinity', '+infinity', '-infinity'), 'UNSUPPORTED_STATISTIC')
        number = Decimal(value)
    except (ValueError, TypeError):
        return 'unsupported', None
    if number.is_nan():
        return 'nan', None
    if number.is_infinite():
        return '-inf' if number.is_signed() else '+inf', None
    return 'finite', number


def direct_decimal_compare(baseline, fault):
    left, right = tensor_rows(baseline), tensor_rows(fault)
    require(len(left) == len(right) and all(a[0] == b[0] for a, b in zip(left, right)), 'DIRECT_RECORD_ALIGNMENT_DIFFERS')
    exact, exceeded, nulls, incomparable = [], [], [], []
    finite_count = 0
    for (identity, a), (_, b) in zip(left, right):
        require(a['dtype'] == b['dtype'] and a['shape'] == b['shape'], 'DIRECT_TENSOR_STRUCTURE_DIFFERS')
        for field in METRICS:
            ak, av = classify(a[field], a['dtype'], field)
            bk, bv = classify(b[field], b['dtype'], field)
            row = dict(identity=identity, field=field,
                baseline=str(av) if ak == 'finite' else None if ak == 'null' else ak,
                fault=str(bv) if bk == 'finite' else None if bk == 'null' else bk)
            if 'null' in (ak, bk):
                nulls.append(row); continue
            if any(kind in ('nan', 'unsupported') for kind in (ak, bk)):
                incomparable.append(row); continue
            if ak == bk == 'finite':
                finite_count += 1
                with localcontext() as context:
                    context.prec = max(80, len(av.as_tuple().digits) + len(bv.as_tuple().digits)
                                       + abs(av.as_tuple().exponent - bv.as_tuple().exponent) + 30)
                    difference = abs(bv - av)
                    threshold = ATOL + RTOL * abs(av)
                    relative = difference / abs(av) if av else None
                    changed, over = av != bv, difference > threshold
                    row.update(absolute_difference=str(difference), tolerance=str(threshold),
                        relative_change=str(relative) if relative is not None else None,
                        relative_percent=str(relative * 100) if relative is not None else None,
                        exceedance=over, magnitude_field=field in ('Max', 'Min', 'Norm'),
                        reaches_0_1_percent=field in ('Max', 'Min', 'Norm') and relative is not None and relative >= Decimal('0.001'))
            else:
                changed = over = ak != bk
                row.update(absolute_difference=None, tolerance=None, relative_change=None,
                           relative_percent=None, exceedance=over, magnitude_field=False, reaches_0_1_percent=False)
            if changed:
                exact.append(row)
            if over:
                exceeded.append(row)
    return dict(finite_pairs=finite_count, exact_changes=exact, exceedances=exceeded,
        unobservable_fields=nulls, incomparable_fields=incomparable,
        first_exact_change=exact[0] if exact else None, first_exceedance=exceeded[0] if exceeded else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fault', type=Path, required=True)
    parser.add_argument('--comparator', type=Path, required=True)
    parser.add_argument('--rtol', type=Decimal, default=RTOL)
    parser.add_argument('--atol', type=Decimal, default=ATOL)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    require(args.rtol == RTOL and args.atol == ATOL, 'FIXED_TOLERANCE_REQUIRED')
    require(digest(args.comparator) == COMPARATOR_SHA, 'FROZEN_COMPARATOR_CHANGED')
    spec = importlib.util.spec_from_file_location('frozen_raw_comparator', args.comparator)
    comparator = importlib.util.module_from_spec(spec); spec.loader.exec_module(comparator)
    require(comparator.RTOL == RTOL and comparator.ATOL == ATOL, 'COMPARATOR_TOLERANCE_DIFFERS')
    options = dict(baseline=args.baseline/'observation/step0/rank0', fault=args.fault/'observation/step0/rank0',
        baseline_config=args.baseline/'config.json', fault_config=args.fault/'config.json',
        baseline_collector=args.baseline/'collector_identity.json', fault_collector=args.fault/'collector_identity.json')
    report = comparator.compare_raw_statistics(**options)
    self_options = dict(options, fault=options['baseline'], fault_config=options['baseline_config'],
                        fault_collector=options['baseline_collector'])
    self_report = comparator.compare_raw_statistics(**self_options)
    direct = None
    if report['aligned']:
        baseline, _ = comparator.read_json(options['baseline']/'dump.json')
        fault, _ = comparator.read_json(options['fault']/'dump.json')
        direct = direct_decimal_compare(baseline, fault)
        require(report['exact_change_count'] == len(direct['exact_changes']) and
                report['exceedance_count'] == len(direct['exceedances']) and
                report['unobservable_fields'] == [dict(index=row['index'], **value) for row, value in
                    zip(report['unobservable_fields'], direct['unobservable_fields'])] and
                len(report['unobservable_fields']) == len(direct['unobservable_fields']) and
                len(report['incomparable_fields']) == len(direct['incomparable_fields']), 'INDEPENDENT_DECIMAL_COUNTS_DISAGREE')
        for name in ('first_exact_change', 'first_exceedance'):
            left, right = report[name], direct[name]
            require((left is None) == (right is None) and (left is None or
                    (left['identity'], left['field']) == (right['identity'], right['field'])), 'INDEPENDENT_FIRST_LOCATION_DISAGREES')
    summary = comparator.compact_summary(report)
    summary.update(review_schema='demoncat.agent7.fa_statistics_review.v1',
        review_script_sha256=digest(__file__), comparator_source_sha256=COMPARATOR_SHA,
        explicit_rtol=str(args.rtol), explicit_atol=str(args.atol),
        independent_decimal_crosscheck=direct,
        baseline_self=dict(status=self_report['status'], exact_change_count=self_report.get('exact_change_count'),
                           exceedance_count=self_report.get('exceedance_count'), first_exceedance=self_report['first_exceedance']),
        oracle_summary_read=False, runtime_or_hit_logs_read=False, full_tensor_payloads_read=False,
        supplied_location_used=False, statistics_only_phase=True,
        e5_positive_localization=report['first_exceedance'] is not None,
        full_model_acceptance=False, c2=False, c3=False,
        magnitude_target_reached=bool(direct and any(row['reaches_0_1_percent'] for row in direct['exact_changes'])))
    args.out.mkdir(exist_ok=False)
    for name, value in (('full.json', report), ('baseline_self.json', self_report), ('summary.json', summary)):
        with (args.out/name).open('x', encoding='utf-8') as stream:
            json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(status=report['status'], exact=report.get('exact_change_count'),
        exceeded=report.get('exceedance_count'), null_fields=len(report.get('unobservable_fields', [])),
        first_exceedance=report['first_exceedance'], independent_crosscheck=direct is not None)))


if __name__ == '__main__':
    main()
