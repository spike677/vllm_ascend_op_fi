"""Second diagnostic report over raw statistics; the strict comparator is unchanged."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from demoncat_fault39 import diagnostics
from compare_model_capture import actual_steps, compare_groups
from compare_msprobe import load_raw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def diagnose_groups(baseline, fault, rank=0):
    baseline, fault = Path(baseline), Path(fault)
    strict = compare_groups(baseline, fault, diagnostics.POLICY, rank)
    bsteps, fsteps = actual_steps(baseline), actual_steps(fault)
    result = dict(schema='demoncat.msprobe.call_diagnostics.group.v1',
        status='DIAGNOSTICS_ONLY_STRICT_RESULT_RETAINED', strict_original_result=strict,
        strict_acceptance_overridden=False, global_first_difference=None,
        native_logs_read=False, tensor_payloads_read=False, rules=diagnostics.RULES,
        rules_sha256=diagnostics.digest(diagnostics.RULES), steps=[],
        implementation_sha256={
            'diagnostics.py': sha(inspect.getfile(diagnostics)),
            'diagnose_call_sequence.py': sha(__file__),
            'compare_msprobe.py': sha(Path(__file__).with_name('compare_msprobe.py')),
            'compare_model_capture.py': sha(Path(__file__).with_name('compare_model_capture.py'))},
        limitations=['No global first difference is inferred by diagnostic pairing',
            'Shared callsite does not prove scheduler request mapping or producer-consumer storage identity',
            'Unchanged argument descriptions/statistics do not establish complete input equality',
            'Null and paired NaN are unobservable/uncomparable, never equality',
            'Collector step numbers are preserved, not interpreted as decode tokens'])
    for step in sorted(set(bsteps) | set(fsteps)):
        row = dict(step=step, rank=rank)
        if step not in bsteps or step not in fsteps:
            row.update(status='UNPAIRED_STEP', missing_from='baseline' if step not in bsteps else 'fault')
        else:
            prior = next(item['report'] for item in strict['steps'] if item['step'] == step)
            metadata = ('comparison' in prior or (prior.get('status') == 'ALIGNMENT_MISMATCH'
                         and prior.get('mismatched') in ('structure', 'stack', 'construct')))
            bp, fp = bsteps[step]/f'rank{rank}', fsteps[step]/f'rank{rank}'
            br, fr = load_raw(bp), load_raw(fp)
            bd = json.loads((bp/'dump.json').read_bytes())['data']
            fd = json.loads((fp/'dump.json').read_bytes())['data']
            row.update(diagnostics.diagnose_records(bd, fd, br['stack'], fr['stack'],
                br['construct'], fr['construct'], capture_metadata_aligned=metadata))
            row['baseline_raw_sha256'], row['fault_raw_sha256'] = br['hashes'], fr['hashes']
        result['steps'].append(row)
    return result


def compact(report):
    out = {key: value for key, value in report.items() if key != 'steps'}
    out['summary_only'] = True
    out['steps'] = []
    for row in report['steps']:
        keep = {key: value for key, value in row.items()
                if key not in ('ledger', 'pairs', 'unobservable_fields', 'rules')}
        if 'ledger' in row:
            keep['complete_record_ledger_sha256'] = diagnostics.digest(row['ledger'])
            keep['ledger_preserved_in_private_details'] = True
            keep['unobservable_fields_count'] = len(row['unobservable_fields'])
            keep['unobservable_fields'] = [{key: field[key] for key in
                ('baseline_operator', 'fault_operator', 'path', 'field', 'status')}
                for field in row['unobservable_fields']]
            keep['paired_invocation_changes'] = [item for item in row['pairs']
                if item['nonstatistical_changes'] or item['tensor_description_mismatches']]
        out['steps'].append(keep)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--fault', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private-details-output', type=Path, required=True)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--rtol', type=float, required=True)
    parser.add_argument('--atol', type=float, required=True)
    parser.add_argument('--prior-strict-report', type=Path)
    args = parser.parse_args()
    if args.rtol != 1e-5 or args.atol != 1e-8:
        parser.error('This reviewed diagnostic requires rtol=1e-5 and atol=1e-8')
    if args.output.resolve() == args.private_details_output.resolve():
        parser.error('Separate summary and private detail paths required')
    if args.output.exists() or args.private_details_output.exists():
        parser.error('Refusing to overwrite an existing report')
    result = diagnose_groups(args.baseline, args.fault, args.rank)
    if args.prior_strict_report:
        original = json.loads(args.prior_strict_report.read_bytes())
        result['prior_strict_report'] = dict(sha256=sha(args.prior_strict_report),
            same_as_recomputed=original == result['strict_original_result'],
            status=original.get('status'), first_difference=original.get('first_difference'))
    for path, value in ((args.private_details_output, result), (args.output, compact(result))):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(diagnostics.safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status=result['status'], strict=result['strict_original_result']['status'],
        steps=[dict(step=row['step'], counts=row.get('counts'),
                    scalar_changes=len(row.get('changed_scalar_outputs', [])),
                    tensor_changes=len(row.get('changed_tensor_fields', []))) for row in result['steps']])))


if __name__ == '__main__':
    main()
