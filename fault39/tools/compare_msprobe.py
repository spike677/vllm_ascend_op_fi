"""Independent comparison of raw msprobe statistics/mix; no native log input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from demoncat_fault39.msprobe import ComparisonPolicy, first_difference

METRICS = {'Max', 'Min', 'Mean', 'Norm'}


def load_raw(directory):
    directory = Path(directory)
    raw = {name: (directory / name).read_bytes() for name in ('dump.json', 'stack.json', 'construct.json')}
    doc = json.loads(raw['dump.json'])
    if doc.get('task') != 'statistics' or doc.get('level') != 'mix' or doc.get('framework') != 'pytorch':
        raise ValueError('EXPECTED_RAW_PYTORCH_STATISTICS_MIX')
    if doc.get('dump_data_dir') is not None or not isinstance(doc.get('data'), dict):
        raise ValueError('UNSUPPORTED_RAW_DUMP_SCHEMA')
    records, structure = [], []
    def walk(value, path):
        if isinstance(value, dict):
            if value.get('type') == 'torch.Tensor':
                if not METRICS.issubset(value):
                    raise ValueError('INCOMPLETE_RAW_TENSOR_STATISTICS: ' + path)
                # Preserve the complete non-statistical tensor description for
                # alignment. Never compare shape/dtype as numerical statistics.
                structure.append((path, {k: v for k, v in value.items() if k not in METRICS}))
                records.append({'identity': path, 'statistics': {k: value[k] for k in ('Max', 'Min', 'Mean', 'Norm')}})
            else:
                structure.append((path, ['dict', list(value)]))
                for key, child in value.items():
                    walk(child, path + '/' + key)
        elif isinstance(value, list):
            structure.append((path, ['list', len(value)]))
            for index, child in enumerate(value):
                walk(child, path + '/' + str(index))
        else:
            structure.append((path, value))
    for operator, value in doc['data'].items():
        walk(value, operator)
    return {'records': records, 'structure': structure,
            'stack': json.loads(raw['stack.json']), 'construct': json.loads(raw['construct.json']),
            'hashes': {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()}}


def compare(baseline, fault, policy, *, baseline_config, fault_config, baseline_collector, fault_collector):
    b, f = load_raw(baseline), load_raw(fault)
    evidence = {'baseline': b['hashes'], 'fault': f['hashes'], 'source': 'RAW_MSPROBE_STATISTICS_MIX',
                'native_logs_read': False, 'private_tensors_read': False}
    metadata = {}
    for name, path in [('baseline_config', baseline_config), ('fault_config', fault_config),
                       ('baseline_collector', baseline_collector), ('fault_collector', fault_collector)]:
        raw = Path(path).read_bytes()
        metadata[name] = json.loads(raw)
        evidence[name + '_sha256'] = hashlib.sha256(raw).hexdigest()
    for group in ('config', 'collector'):
        left, right = dict(metadata['baseline_' + group]), dict(metadata['fault_' + group])
        if group == 'config':
            left.pop('dump_path', None); right.pop('dump_path', None)
            if left.get('task') != 'statistics' or left.get('level') != 'mix' or left.get('statistics', {}).get('summary_mode') != 'statistics':
                return dict(evidence, status='CONFIGURATION_NOT_STATISTICS_MIX')
        else:
            left.pop('package', None); right.pop('package', None)
            if not left.get('files'):
                return dict(evidence, status='COLLECTOR_IDENTITY_INCOMPLETE')
        if left != right:
            return dict(evidence, status='ALIGNMENT_MISMATCH', mismatched=group)
    for field in ('structure', 'stack', 'construct'):
        if b[field] != f[field]:
            return dict(evidence, status='ALIGNMENT_MISMATCH', mismatched=field)
    return dict(evidence, comparison=first_difference(b['records'], f['records'], policy))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--fault', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for name in ('baseline-config', 'fault-config', 'baseline-collector', 'fault-collector'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--atol', type=float, default=1e-8)
    a = p.parse_args()
    result = compare(a.baseline, a.fault, ComparisonPolicy(a.atol, a.rtol),
                     baseline_config=a.baseline_config, fault_config=a.fault_config,
                     baseline_collector=a.baseline_collector, fault_collector=a.fault_collector)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
