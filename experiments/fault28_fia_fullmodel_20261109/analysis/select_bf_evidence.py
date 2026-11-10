"""Select the two already captured FIA groups without rewriting raw evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

PARENT = '2aa4432ce92a2c878f6fb6ce3a648b1049df4b48'
CASES = ('model_b8_baseline', 'model_b8_fault_r1')
ANALYSIS = {'graph_evidence.py', 'statistics_values.py', 'coverage.py',
            'public_plan.py', 'extract_native.py'}
REPORTS = {'reports/fault_r1_msprobe.json', 'reports/fault_r1_complete.json',
           *(f'reports/coverage/{name}.json' for name in CASES)}


def extended(path):
    path = path.absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def selected(name):
    parts = Path(name).parts
    if parts[0] == 'cases':
        return parts[1] in CASES
    if parts[0] in ('build', 'collector'):
        return True
    if parts[0] == 'analysis':
        return len(parts) == 2 and parts[1] in ANALYSIS
    if parts[0] == 'runtime_source':
        return len(parts) == 2 or (len(parts) == 3 and parts[1] == 'setup'
                                 and parts[2] in {n + '.json' for n in CASES})
    return name in REPORTS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    source = extended(args.source).resolve(strict=True)
    destination = extended(args.destination)
    assert not destination.exists() and not destination.resolve().is_relative_to(source)
    sums = {}
    for line in (source / 'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        assert name not in sums and not Path(name).is_absolute() and '..' not in Path(name).parts
        sums[name] = digest
    files = {f.relative_to(source).as_posix(): f for f in source.rglob('*') if f.is_file()}
    assert set(files) == set(sums) | {'SHA256SUMS'}
    for name, path in files.items():
        assert not path.is_symlink() and path.resolve().is_relative_to(source)
        if name != 'SHA256SUMS':
            assert sha(path) == sums[name], name
    parent = json.loads((source / 'reports/acceptance.json').read_bytes())
    assert parent['status'] == 'PASS_FULLMODEL_FIA_COVERAGE_FIRST_DIFFERENCE_AND_RESTORE'
    assert parent['cases'] == 10
    mapping = json.loads((source / 'source_mapping.json').read_bytes())
    for row in mapping['files']:
        path = files[row['published']]
        assert row['original_bytes_preserved'] and sha(path) == row['sha256']
        assert path.stat().st_size == row['bytes']
    destination.mkdir(parents=True)
    for name, path in files.items():
        if not selected(name):
            continue
        target = destination / name
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        assert sha(target) == sums[name], name
    # These new publication tools are not mislabeled as frozen capture sources.
    for name in ('select_bf_evidence.py', 'verify_public_bundle.py'):
        shutil.copyfile(Path(__file__).with_name(name), destination / 'analysis' / name)
    save(destination / 'primary_BF.json', {
        'baseline': 'cases/' + CASES[0], 'fault': 'cases/' + CASES[1],
        'delivery': 'primary_bf_only', 'same_batch': 8, 'same_causal_rule': True,
        'both_routes': 'explicit registered npu_fused_infer_attention_score.default',
        'baseline_m': 512, 'fault_m': 128,
        'independent_analysis': 'analysis/graph_evidence.py'})
    pair = next(p for p in parent['primary_pairs']
                if (p['baseline'], p['fault']) == CASES)
    save(destination / 'reports/acceptance.json', {
        'schema': 'fault28.fia_fullmodel.primary_bf_acceptance.v1',
        'status': 'PASS_PRIMARY_BF_FIA_COVERAGE_FIRST_DIFFERENCE_AND_LOGITS',
        'model': parent['model'], 'scope': parent['scope'],
        'cases': 2, 'model_graph_executions': 6, 'fia_node_executions': 168,
        'logical_compiled_fia_nodes': 56, 'layer_request_executions': 1344,
        'all_supported_layers_requests_covered': True,
        'coverage': [row for row in parent['coverage'] if row['case_id'] in CASES],
        'primary_pairs': [pair],
        'all_parameters_inputs_unchanged': parent['all_parameters_inputs_unchanged'],
        'first_token_logits_propagation': parent['first_token_logits_propagation'],
        'all_eight_request_logits_changed': parent['all_eight_request_logits_changed'],
        'matrix_sha256': parent['matrix_sha256'],
        'historical_Pangu_version_reproduced': False,
        'limits': [
            'This is a subset of completed 2026-11-09 captures, not a new NPU run.',
            'Both B and F use the explicit FIA adapter and shared causal rule; B is not default SDPA.',
            'Independent F, OFF, B1 and alias-neutrality datasets remain in the parent commit, outside this two-group delivery.',
            'Private complete-array reports are retained; their tensor payloads are not public.',
            'Native six-digit CSV alone cannot distinguish the earliest FIA difference; uniform disclosed FP64 statistics and actual graph aliases are needed.',
            'No unique Reduce-instruction or historical Pangu attribution.']})
    save(destination / 'source_mapping.json', {
        'schema': mapping['schema'], 'files': [r for r in mapping['files'] if selected(r['published'])],
        'scope': mapping['scope'], 'selection': 'primary_bf_only',
        'parent_commit': PARENT, 'parent_source_mapping_sha256': sha(source / 'source_mapping.json')})
    save(destination / 'publication_scope.json', {
        'schema': 'fault28.fia_fullmodel.primary_bf_selection.v1',
        'case_ids': list(CASES), 'parent_commit': PARENT,
        'parent_artifact': 'artifacts/attention_tiling/fault28_fia_fullmodel_statistics_20261109/data',
        'parent_source_mapping_sha256': sha(source / 'source_mapping.json'),
        'parent_sha256sums_sha256': sha(source / 'SHA256SUMS'),
        'parent_acceptance_sha256': sha(source / 'reports/acceptance.json'),
        'all_parent_files_verified_before_selection': True,
        'all_retained_capture_bytes_unchanged': True, 'native_calls': 0,
        'selection_source_sha256': sha(Path(__file__)),
        'new_generated_files': ['primary_BF.json', 'reports/acceptance.json', 'source_mapping.json',
                                'publication_scope.json', 'SHA256SUMS',
                                'analysis/select_bf_evidence.py', 'analysis/verify_public_bundle.py'],
        'scope': 'Only two FIA capture groups. No default-SDPA, B1, independent F, OFF or noalias capture group is included. Runtime_source contains only the two case configs; full original source manifests still identify the historical frozen runtime.'})
    paths = sorted(f for f in destination.rglob('*') if f.is_file())
    (destination / 'SHA256SUMS').write_text(''.join(
        sha(f) + '  ' + f.relative_to(destination).as_posix() + '\n' for f in paths), encoding='utf-8')
    print(json.dumps({'status': 'SELECTED_TWO_ORIGINAL_FIA_GROUPS', 'case_ids': list(CASES),
                      'files': len(paths) + 1, 'mapped_originals': sum(selected(r['published']) for r in mapping['files']),
                      'bytes': sum(f.stat().st_size for f in paths) + (destination / 'SHA256SUMS').stat().st_size}))


if __name__ == '__main__':
    main()
