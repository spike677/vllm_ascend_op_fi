"""Seal B/F/F/OFF complete-model cases from the accepted fixed-layout baseline."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-case', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    baseline = json.loads(args.baseline_case.read_bytes())
    if baseline['schema'] != 'fault33.pa.replay.case.v1' or baseline['mode'] != 'off' or baseline['operator_variant'] != 'private_vendor':
        raise ValueError('Use the sealed replay baseline')
    args.output.mkdir(exist_ok=False, parents=True)
    rows = []
    for name, mode in [('model_baseline', 'off'), ('model_fault_r0', 'inner1024'),
                       ('model_fault_r1', 'inner1024'), ('model_restore', 'off')]:
        case = dict(baseline, schema='fault33.pa.model.case.v1', case_id=name, mode=mode,
                    repeats=1, execution_kind='model_paged_adapter')
        path = args.output / (name + '.json')
        path.write_text(json.dumps(case, indent=2) + '\n')
        rows.append({'case_id': name, 'mode': mode, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'eligible_decode_ifa_calls': 28, 'request_layer_pairs': 224})
    (args.output / 'matrix_manifest.json').write_text(json.dumps({
        'schema': 'fault33.pa.model.matrix.v1', 'cases': rows,
        'scope': 'Real HF model with explicit paged IFA decode adapter, no serving scheduler.',
        'prefill_tokens': 16383, 'teacher_forced_decode_tokens': 1, 'requests': 8,
        'formal_statistics_groups': ['baseline', 'fault'],
        'repeat_and_restore': 'Same frozen prefix and teacher token; two fault fresh processes and separate OFF restoration.'
    }, indent=2) + '\n')
    print(json.dumps({'status': 'SEALED_NOT_EXECUTED', 'cases': len(rows), 'eligible_decode_calls': 112}))


if __name__ == '__main__':
    main()
