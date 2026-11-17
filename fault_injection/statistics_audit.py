"""Raw-only, fixed-tolerance statistics audit with explicit missing information.

This module accepts no provider, hit, target, expected-operator or tensor input.
JSON numbers are evaluated as Decimal; output numerical values are decimal text
so the report cannot silently round the original statistics. Raw bytes are never
rewritten. A numerical location is an observable difference, not a root-cause
claim or a C2/C3 qualification.
"""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path


RTOL, ATOL = Decimal('1e-5'), Decimal('1e-8')
METRICS = ('Max', 'Mean', 'Min', 'Norm')
RAW_FILES = ('dump.json', 'stack.json', 'construct.json')


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('DUPLICATE_JSON_KEY:' + key)
        result[key] = value
    return result


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    value = json.loads(raw, parse_float=Decimal, parse_constant=Decimal,
                       object_pairs_hook=_pairs)
    return value, dict(path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def _number(value):
    if value is None:
        return 'null', None
    if isinstance(value, bool):
        return 'unsupported', None
    if isinstance(value, (int, Decimal)):
        number = Decimal(value)
    elif isinstance(value, str) and value.strip().lower() in (
            'nan', '+nan', '-nan', 'inf', '+inf', '-inf',
            'infinity', '+infinity', '-infinity'):
        number = Decimal(value)
    else:
        return 'unsupported', None
    if number.is_nan():
        return 'nan', None
    if number.is_infinite():
        return '-inf' if number.is_signed() else '+inf', None
    return 'finite', number


def _label(value):
    kind, number = _number(value)
    return str(number) if kind == 'finite' else (None if kind == 'null' else kind)


def _finite_difference(baseline, fault):
    # Include all input digits and exponent separation when subtracting near the
    # fixed tolerance. Default Decimal precision must not round a barely larger
    # value onto the tolerance boundary.
    left, right = baseline.as_tuple(), fault.as_tuple()
    with localcontext() as context:
        context.prec = max(50, len(left.digits) + len(right.digits)
                           + abs(left.exponent - right.exponent) + 20)
        absolute = abs(fault - baseline)
        tolerance = ATOL + RTOL * abs(baseline)
        relative = absolute / abs(baseline) if baseline else None
        return absolute, tolerance, relative


def _capture(directory, config, collector):
    directory = Path(directory)
    documents, files = {}, {}
    for name, path in [(name, directory/name) for name in RAW_FILES] + [('config', config), ('collector', collector)]:
        documents[name], files[name] = read_json(path)
    doc = documents['dump.json']
    if (not isinstance(doc, dict) or doc.get('task') != 'statistics' or doc.get('level') != 'mix'
            or doc.get('framework') != 'pytorch' or doc.get('dump_data_dir') is not None
            or not isinstance(doc.get('data'), dict)):
        raise ValueError('RAW_PYTORCH_STATISTICS_MIX_REQUIRED')
    configuration = dict(documents['config'])
    configuration.pop('dump_path', None)
    if (configuration.get('task') != 'statistics' or configuration.get('level') != 'mix'
            or configuration.get('statistics', {}).get('summary_mode') != 'statistics'):
        raise ValueError('CONFIGURATION_NOT_STATISTICS_MIX')
    collector_identity = dict(documents['collector'])
    collector_identity.pop('package', None)
    if not isinstance(collector_identity.get('files'), dict) or not collector_identity['files']:
        raise ValueError('COLLECTOR_IDENTITY_INCOMPLETE')
    rows, structure = [], []
    def walk(value, path):
        if isinstance(value, dict):
            if value.get('type') == 'torch.Tensor':
                if not set(METRICS).issubset(value):
                    raise ValueError('INCOMPLETE_RAW_TENSOR_STATISTICS:' + path)
                structure.append((path, {k: v for k, v in value.items() if k not in METRICS}))
                rows.append((path, {name: value[name] for name in METRICS}))
            else:
                structure.append((path, ['dict', list(value)]))
                for name, child in value.items():
                    walk(child, path + '/' + name)
        elif isinstance(value, list):
            structure.append((path, ['list', len(value)]))
            for index, child in enumerate(value):
                walk(child, path + '/' + str(index))
        else:
            structure.append((path, value))
    for operator, values in doc['data'].items():
        walk(values, operator)
    return dict(records=rows, structure=structure, stack=documents['stack.json'],
                construct=documents['construct.json'], config=configuration,
                collector=collector_identity, files=files)


def _inventory(rows):
    counts = dict(finite=0, null=0, nan=0, positive_inf=0, negative_inf=0, unsupported=0)
    exceptional = []
    for index, (identity, values) in enumerate(rows):
        for field in METRICS:
            kind, _ = _number(values[field])
            counts[{'+inf': 'positive_inf', '-inf': 'negative_inf'}.get(kind, kind)] += 1
            if kind != 'finite':
                exceptional.append(dict(index=index, identity=identity, field=field,
                                        kind=kind, value=_label(values[field])))
    return dict(counts=counts, exceptional_fields=exceptional)


def compare_raw_statistics(*, baseline, fault, baseline_config, fault_config,
                           baseline_collector, fault_collector):
    """Compare complete raw captures without runtime or expected-location hints."""
    b = _capture(baseline, baseline_config, baseline_collector)
    f = _capture(fault, fault_config, fault_collector)
    inventories = {name: _inventory(capture['records']) for name, capture in (('baseline', b), ('fault', f))}
    result = dict(schema='demoncat.raw_statistics_audit.v1', task='statistics', level='mix',
        summary_mode='statistics', rtol='0.00001', atol='0.00000001', reference='baseline',
        comparison_rule='abs(F-B) > atol + rtol*abs(B)', numerical_encoding='decimal_text',
        native_logs_read=False, private_tensors_read=False, expected_location_read=False,
        files=dict(baseline=b['files'], fault=f['files']), full_dump_inventory=inventories,
        comparator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        public_qualification_granted=False, root_cause_identified=False)
    mismatches = [name for name in ('config', 'collector', 'structure', 'stack', 'construct') if b[name] != f[name]]
    if mismatches:
        return dict(result, status='ALIGNMENT_MISMATCH', aligned=False, mismatched=mismatches,
                    first_exact_change=None, first_exceedance=None,
                    whole_dump_comparable=False, baseline_self_comparable=False,
                    baseline_self_status='NOT_EVALUATED_DUE_TO_PAIR_ALIGNMENT',
                    e5_statistics_comparable=False)
    if not b['records']:
        return dict(result, status='INSUFFICIENT_STATISTICS', aligned=True,
                    first_exact_change=None, first_exceedance=None, whole_dump_comparable=False,
                    baseline_self_comparable=False, baseline_self_status='INSUFFICIENT_STATISTICS',
                    e5_statistics_comparable=False)
    nulls, incomparable, first_exact, first_exceedance = [], [], None, None
    compared, exact_count, exceeded_count = 0, 0, 0
    for index, ((identity, left), (_, right)) in enumerate(zip(b['records'], f['records'])):
        for field in METRICS:
            ka, a = _number(left[field]); kb, value = _number(right[field])
            row = dict(index=index, identity=identity, field=field, baseline=_label(left[field]), fault=_label(right[field]))
            if 'null' in (ka, kb):
                nulls.append(row)
                continue
            if any(kind in ('nan', 'unsupported') for kind in (ka, kb)):
                incomparable.append(dict(row, baseline_kind=ka, fault_kind=kb))
                continue
            compared += 1
            if ka == kb == 'finite':
                absolute, limit, relative = _finite_difference(a, value)
                changed, exceeded = a != value, absolute > limit
                row.update(absolute_difference=str(absolute), tolerance=str(limit),
                    relative_change=str(relative) if relative is not None else None,
                    reason='FINITE_STATISTIC_DIFFERS')
            else:
                changed = exceeded = ka != kb
                row.update(absolute_difference=None, tolerance=None, relative_change=None,
                           reason='NONFINITE_TRANSITION' if changed else 'SAME_SIGNED_INFINITY')
            if changed:
                exact_count += 1
                if first_exact is None:
                    first_exact = row
            if exceeded:
                exceeded_count += 1
                if first_exceedance is None:
                    first_exceedance = dict(row, preceding_unobservable_fields=len(nulls),
                                            preceding_incomparable_fields=len(incomparable))
    self_bad = sum(inventories['baseline']['counts'][k] for k in ('nan', 'unsupported'))
    self_observable = sum(inventories['baseline']['counts'][k] for k in ('finite', 'positive_inf', 'negative_inf'))
    comparable = not incomparable and compared > 0
    status = ('INCOMPARABLE_STATISTICS' if incomparable else
              'FIRST_OBSERVABLE_STATISTICS_DIFFERENCE' if first_exceedance else
              'NO_OBSERVABLE_STATISTICS_DIFFERENCE')
    return dict(result, status=status, aligned=True, records_compared=len(b['records']),
        fields_compared=compared, exact_change_count=exact_count, exceedance_count=exceeded_count,
        first_exact_change=first_exact, first_exceedance=first_exceedance,
        unobservable_fields=nulls, incomparable_fields=incomparable,
        baseline_self_comparable=self_bad == 0 and self_observable > 0,
        baseline_self_status='INCOMPARABLE_STATISTICS' if self_bad else (
            'NO_OBSERVABLE_FIELDS' if not self_observable else 'NO_OBSERVABLE_STATISTICS_DIFFERENCE'),
        whole_dump_comparable=comparable, e5_statistics_comparable=comparable and self_bad == 0,
        entire_tensor_values_compared=False,
        first_location_is_observable_only=first_exceedance is not None)


def compact_summary(report):
    """Publishable summary; preserve exceptional-field lists in the full report."""
    result = dict(report)
    for key in ('unobservable_fields', 'incomparable_fields'):
        if key in result:
            result[key + '_count'] = len(result.pop(key))
    result['full_dump_inventory'] = {group: dict(counts=value['counts'],
        exceptional_fields_count=len(value['exceptional_fields']))
        for group, value in report['full_dump_inventory'].items()}
    result['exceptional_field_details_retained_in_full_report'] = True
    return result
