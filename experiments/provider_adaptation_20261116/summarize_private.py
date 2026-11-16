"""Export allowlisted results and hashes, never raw captures or tensor arrays.

This runs after the independent comparator. Native receipts are summarized
separately and cannot supply or override the comparator's first difference.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def summarize_case(path):
    x = json.loads(path.read_text())
    allowed = ('status', 'arm', 'kind', 'scope', 'stage', 'full_pretrained_model',
               'off_exact', 'input_unchanged', 'inputs_unchanged', 'changed',
               'repeat_stable', 'writer_exact', 'residual_restored',
               'consumer_changed', 'supported', 'claims', 'native_calls',
               'actual_policy_applied', 'logits_max_abs', 'top1_equal')
    out = {k: x[k] for k in allowed if k in x}
    out['receipt_sha256'] = sha(path)
    coverage = x.get('coverage')
    if coverage:
        out['routing'] = {
            'candidate_count': len(coverage['candidates']),
            'event_status_counts': dict(Counter(e['status'] for e in coverage['events'])),
            'all_discovered_candidates_covered': coverage['all_discovered_candidates_covered'],
            'all_observed_supported_calls_routed': coverage['all_observed_supported_calls_routed'],
            'cleanup_completed': coverage['cleanup_completed'],
            'report_scope': coverage['report_scope'],
        }
    if isinstance(x.get('native'), list):
        out['native'] = [{k: v for k, v in n.items() if k in
                          ('rank', 'supported', 'claims', 'completed', 'pending',
                           'events', 'native_poisoned')} for n in x['native']]
    if 'native_completed' in x:
        out['completed_native_events'] = len(x['native_completed'])
        out['all_native_events_pass'] = all(e['pass'] for e in x['native_completed'])
    if isinstance(x.get('healthy'), list):
        observed = x.get('observed') or x.get('runs', [[]])[-1]
        out['request_count'] = len(x['healthy'])
        for field in ('tokens', 'text', 'logprobs'):
            out['changed_' + field + '_rows'] = sum(
                a[field] != b[field] for a, b in zip(x['healthy'], observed))
    if x.get('error'):
        # Traceback may contain private addresses. Keep its integrity only.
        out['error_sha256'] = hashlib.sha256(str(x['error']).encode()).hexdigest()
    return out


def summarize_comparison(path):
    x = json.loads(path.read_text())
    out = {k: x[k] for k in ('status', 'source', 'native_logs_read',
           'private_tensors_read', 'rank', 'comparison_policy', 'baseline_steps',
           'fault_steps', 'all_steps_aligned', 'blocking_steps', 'limitations') if k in x}
    out['private_report_sha256'] = sha(path)
    out['steps'] = []
    for step in x.get('steps', []):
        c = step['report'].get('comparison', {})
        out['steps'].append(dict(step=step['step'], status=step['status'],
            identity=c.get('identity'), difference=c.get('difference'),
            unobservable_count=len(c.get('unobservable_fields', []))))
    return out


def summarize_model39(path):
    x = json.loads(path.read_text())
    out = {k: x[k] for k in ('status', 'stage', 'tp', 'ranks', 'baseline_stable',
                             'native_build_profiles') if k in x}
    out['receipt_sha256'] = sha(path)
    out['phases'] = []
    for phase in x.get('phases', []):
        if 'validated' not in phase:
            continue
        item = {k: phase[k] for k in ('label', 'validated', 'recovery_verified') if k in phase}
        item['native_verdicts'] = {str(rank): value.get('verdict')
            for rank, value in phase.get('native_inspection', {}).items()}
        out['phases'].append(item)
    out['cleanup_status'] = x.get('cleanup', {}).get('status')
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    names = ['provider20-model-a1', 'provider20-smollm-a1', 'provider26-npu-a1',
             'provider26-model-a2', 'provider26-fia-model-a1', 'provider26-final-a2',
             'quant-model-a2', 'provider33-model-a2',
             'model-851-lossless-a1', 'model-851-bf-a1',
             'tp2-851-lossless-a1', 'tp2-851-bf-a1', 'model-91-lossless-a2', 'model-91-bf-a2']
    names += [f'npu-{v}-a3' for v in ('851', '90', '91', '92')]
    out = dict(schema='demoncat.provider.acceptance_summary.v1',
               private_root=str(root), raw_statistics_published=False,
               first_difference_source='Independent raw statistics reports; no native logs used',
               cases={}, comparisons={}, model39={}, capture_hashes={}, missing_capture_files=[])
    for name in names:
        directory = root / name
        if not directory.exists():
            continue
        for path in sorted(directory.glob('*/result.json')):
            out['cases'][str(path.relative_to(root))] = summarize_case(path)
        for path in sorted(directory.glob('*/rank*/provider_result.json')):
            x = json.loads(path.read_text())
            c = x['coverage']
            out['cases'][str(path.relative_to(root))] = dict(
                receipt_sha256=sha(path), rank=x['rank'], supported=x['supported'],
                claims=x['claims'], completed=len(x['completed']), pending=x['pending'],
                has_error=bool(x['error']), candidate_count=len(c['candidates']),
                event_status_counts=dict(Counter(e['status'] for e in c['events'])),
                passthrough_reasons=dict(Counter(e['reason'] for e in c['events'] if e['status']=='passthrough')),
                all_observed_supported_calls_routed=c['all_observed_supported_calls_routed'],
                all_discovered_candidates_covered=c['all_discovered_candidates_covered'])
        for path in sorted(directory.glob('*independent*.json')):
            out['comparisons'][str(path.relative_to(root))] = summarize_comparison(path)
        for path in sorted(directory.glob('*/validation/validation.json')) + sorted(directory.glob('*/lossless/validation.json')):
            out['model39'][str(path.relative_to(root))] = summarize_model39(path)
        for path in sorted(directory.rglob('dump.json')):
            # The complete triplet and capture identity are required even for
            # an inconclusive numerical comparison; no private bytes exported.
            group = path.parent.parent.parent.parent
            for file in [path.parent / (n + '.json') for n in ('dump', 'stack', 'construct')] + [
                    group / 'config.json', group / 'collector_identity.json']:
                if not file.is_file():
                    out['missing_capture_files'].append(str(file.relative_to(root)))
                    continue
                out['capture_hashes'][str(file.relative_to(root))] = dict(
                    sha256=sha(file), bytes=file.stat().st_size)
    out['fault39_operator_matrix'] = json.loads((root / 'fault39-npu-summary-a3.json').read_text())
    out['provider33_native_launch'] = json.loads((root / 'provider33-model-a2/native_launch_summary.json').read_text())
    out['builds'] = {}
    for relative in [f'build-{v}-a5/build_manifest.json' for v in ('851', '90', '91', '92')] + [
            'build20-a1/build.json', 'build26-a2/build.json', 'quant-scoped-a1/runtime/build.json']:
        path = root / relative
        out['builds'][relative] = dict(sha256=sha(path), manifest=json.loads(path.read_text()))
    with args.output.open('x') as stream:
        json.dump(out, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.output), cases=len(out['cases']),
                         comparisons=len(out['comparisons']), capture_files=len(out['capture_hashes']))))


if __name__ == '__main__':
    main()
