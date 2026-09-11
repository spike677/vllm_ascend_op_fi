"""Offline source/documentation inventory and release checks; never starts services.

All outputs are new files. No historical manifests, evidence or other worktrees
are rewritten. This helper is not a runtime, installer or fault controller.
"""
from __future__ import annotations
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_sha(path):
    """Git LF text bytes, not a claim that a CRLF working file is byte-identical."""
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def write_new(path, data):
    with path.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def git(root, *args):
    return subprocess.check_output(['git', '-c', 'safe.directory=' + root.as_posix(),
        '-C', str(root), *args], encoding='utf-8').strip()


def inventory():
    trees, tree = [], {}
    for line in git(ROOT, 'worktree', 'list', '--porcelain').splitlines() + ['']:
        if not line:
            if tree:
                trees.append(tree)
            tree = {}
        else:
            k, _, v = line.partition(' ')
            tree[k] = v
    for tree in trees:
        path = Path(tree['worktree'])
        tree['status_porcelain'] = git(path, 'status', '--porcelain=v1')
        changed = git(path, 'diff', '--name-only', 'HEAD').splitlines()
        extra = git(path, 'ls-files', '--others', '--exclude-standard').splitlines()
        tree['unique_pending_files'] = [{'path': n, 'sha256': sha(path / n), 'bytes': (path / n).stat().st_size,
            'kind': 'untracked' if n in extra else 'modified', 'disposition': 'PRESERVE_IN_SOURCE_WORKTREE'}
            for n in sorted(set(changed + extra)) if (path / n).is_file()]
        tree['markdown'] = [{'path': n, 'sha256': sha(path / n), 'bytes': (path / n).stat().st_size}
            for n in git(path, 'ls-files', '*.md').splitlines() if (path / n).is_file()]
    return {'schema': 'SOURCE_BASELINE_INVENTORY_V1', 'canonical_root': 'D:/project/demoncatpfi',
        'worktrees': trees, 'NPU_used': False, 'branches_merged': [],
        'note': 'Point-in-time inventory. Concurrent Research activity is not modified by this task.'}


def local_links(root):
    docs = list((root / 'docs').rglob('*.md'))
    docs += [root / 'README.md', root / 'CHANGELOG.md']
    for folder in ('src', 'tools', 'tests', 'profiles', 'experiments', 'artifacts'):
        docs += list((root / folder).rglob('README.md'))
    links = []
    for file in sorted(set(docs)):
        if not file.is_file():
            continue
        text = re.sub(r'(?ms)^(`{3,}|~{3,})[^\n]*\n.*?^\1[^\n]*$', '', file.read_text(encoding='utf-8-sig'))
        targets = re.findall(r'\[[^\]]*\]\(([^)]+)\)', text)
        targets += re.findall(r'(?m)^\[[^\]]+\]:\s*(\S+)', text)
        for target in targets:
            target = target.strip().strip('<>')
            if re.match(r'(?:https?:|mailto:|#|//)', target, re.I):
                continue
            if re.match(r'(?:[A-Za-z]:|file:)', target):
                links.append({'source': file.relative_to(root).as_posix(), 'target': target, 'exists': False,
                              'reason': 'ABSOLUTE_LOCAL_LINK_NOT_PORTABLE'})
                continue
            part = unquote(urlsplit(target).path)
            path = root / part.lstrip('/') if part.startswith('/') else file.parent / part
            links.append({'source': file.relative_to(root).as_posix(), 'target': target, 'exists': path.exists()})
    return {'checked': len(links), 'broken': [r for r in links if not r['exists']]}


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    errors = []
    for name, expected in manifest['source_files'].items():
        p = ROOT / name
        if not p.is_file() or source_sha(p) != expected:
            errors.append(name)
    dep = manifest['reference_dependency']
    path = ROOT / dep['path']
    reference_ok = path.is_file() and sha(path) == dep['sha256']
    workspace = json.loads((ROOT / 'demoncatpfi.code-workspace').read_text())
    links = local_links(ROOT)
    helps = []
    for name in manifest['recommended_cli']:
        p = subprocess.run([sys.executable, str(ROOT / name), '--help'], capture_output=True,
            text=True, encoding='utf-8', cwd=ROOT)
        helps.append({'path': name, 'exit_code': p.returncode, 'output': (p.stdout + p.stderr)[-4000:]})
    valid = not errors and reference_ok and not links['broken'] and all(h['exit_code'] == 0 for h in helps)
    valid = valid and workspace['folders'] == [{'path': '.'}]
    return {'status': 'PASS' if valid else 'FAIL', 'NPU_used': False,
        'source_mismatches': errors, 'reference_hash_matches': reference_ok,
        'workspace_relative_single_root': workspace['folders'] == [{'path': '.'}],
        'links': links, 'help_checks': helps}


def freeze(inventory_path, evidence_root, output_dir):
    """Generate small source and hash-only evidence records; never copies raw."""
    before = json.loads(inventory_path.read_text(encoding='utf-8'))
    trees = [{k: t[k] for k in ('worktree', 'HEAD', 'branch', 'status_porcelain', 'unique_pending_files')}
             for t in before['worktrees']]
    current = git(ROOT, 'rev-parse', 'HEAD')
    source_paths = []
    for name in ('src', 'tools', 'tests', 'profiles', 'experiments/fake_requant_reference'):
        source_paths += [p for p in (ROOT / name).rglob('*')
                        if p.is_file() and p.suffix in ('.py', '.json', '.cjs') and '__pycache__' not in p.parts]
    source_files = {p.relative_to(ROOT).as_posix(): source_sha(p) for p in sorted(source_paths)}
    reference = 'experiments/fake_requant_reference/source/fake_requant_reference_extension.py'
    live = json.loads((evidence_root / 'evidence_v1/live_source_audit.json').read_text())
    raw_checks = [{'path': 'evidence_v1/' + n, 'sha256': h,
                   'exists': (evidence_root / 'evidence_v1' / n).is_file(),
                   'matches': (evidence_root / 'evidence_v1' / n).is_file()
                   and sha(evidence_root / 'evidence_v1' / n) == h}
                  for n, h in live['raw_files'].items()]
    python_checks = []
    for n, expected in live['source_files'].items():
        if n.endswith('.py'):
            p = ROOT / n
            python_checks.append({'path': n, 'live_sha256': expected,
                'release_lf_sha256': source_sha(p) if p.is_file() else None,
                'raw_worktree_sha256': sha(p) if p.is_file() else None,
                'lf_matches_live': p.is_file() and source_sha(p) == expected})
    differences = [r['path'] for r in python_checks if not r['lf_matches_live']]
    expected_existing_differences = ['src/ascend_fi/inspect/dataset_effect.py', 'tools/private_demo_service.py']
    archive_path = inventory_path.parent / 'client_live_v1.tar.gz'
    archive_sha = sha(archive_path)
    if archive_sha != 'e89c8c59fffd355fecd0ee52372d6b4a553e754dfd5175dd349cb1b17db5eba9':
        raise ValueError('LIVE_ARCHIVE_HASH_MISMATCH')
    source_diffs = {}
    with tarfile.open(archive_path, 'r:gz') as archive:
        for name in differences:
            original = archive.extractfile('client/' + name).read()
            if hashlib.sha256(original).hexdigest() != live['source_files'][name]:
                raise ValueError('ARCHIVE_VS_LIVE_AUDIT_MISMATCH:' + name)
            source_diffs[name] = ''.join(difflib.unified_diff(original.decode().splitlines(True),
                (ROOT / name).read_text(encoding='utf-8').splitlines(True),
                fromfile='live/' + name, tofile='release/' + name))
    preserved_path = Path('D:/project/demoncatpfi-worktrees/preserved-main-before-source-baseline-v1')
    main_before = next(t for t in trees if t['worktree'] == 'D:/project/demoncatpfi')
    preservation = [{'path': f['path'], 'expected_sha256': f['sha256'],
        'matches': (preserved_path / f['path']).is_file() and sha(preserved_path / f['path']) == f['sha256']}
        for f in main_before['unique_pending_files']]
    write_new(output_dir / 'worktree_inventory.json', {
        'schema': 'SOURCE_BASELINE_WORKTREE_SNAPSHOT_V1', 'before': trees,
        'before_markdown_count_by_worktree': {t['worktree']: len(t['markdown']) for t in before['worktrees']},
        'scope': 'Point-in-time snapshot; no whole-branch merges; concurrent branches untouched',
        'main_preservation': {'worktree': str(preserved_path),
            'branch': 'codex/preserve-main-before-source-baseline-v1',
            'stash_retained': '4f3f1698720a1e2fb788767c0ea5d959cd40881d', 'file_checks': preservation}})
    extra = ['evidence_v1/live_source_audit.json', 'evidence_v1/service_stopped.json',
             'user_report_v1/demo_walkthrough_zh.md', 'user_report_v1/numerical_effect.json',
             'user_report_v1/effect/evaluation_result.json', 'user_report_v1/replay_provenance.json']
    write_new(output_dir / 'evidence_index.json', {
        'schema': 'EXISTING_QWEN06B_EVIDENCE_HASH_INDEX_V1',
        'private_package_name': 'qwen06b_demo_20260911',
        'contains_raw_responses_or_credentials': False,
        'raw_checks_against_original_audit': raw_checks,
        'additional_existing_files': {n: sha(evidence_root / n) for n in extra},
        'live_python_comparison': python_checks,
        'python_differences': differences,
        'difference_interpretation': 'Existing before consolidation: dataset_effect aggregation correction; starter device/source guards moved before lease and TRITON_CACHE_DIR set under session. Startup/cache changes not revalidated live. Injection/selector/lifecycle owner/scoring unchanged.',
        'archive_sha256': archive_sha, 'source_diffs': source_diffs,
        'new_live_run': False})
    if (any(not r['matches'] for r in raw_checks + preservation) or differences != expected_existing_differences):
        raise ValueError('EVIDENCE_OR_PRESERVATION_MISMATCH; inspect generated index, no release certification')
    return {'schema': 'SOURCE_BASELINE_MANIFEST_V1', 'release_id': 'SOURCE_BASELINE_V1_20260911',
        'branch': 'codex/source-baseline-v1', 'tag': 'source-baseline-v1-20260911',
        'tested_source_freeze_commit': current, 'live_base_commit': live['base_HEAD'],
        'commit_identity': 'git rev-parse source-baseline-v1-20260911^{commit}',
        'source_hash_mode': 'UTF8_TEXT_LF_GIT_ATTRIBUTES; strict raw bytes for reference dependency',
        'source_files': source_files,
        'reference_dependency': {'path': reference, 'sha256': sha(ROOT / reference),
            'source_commit': '1d715725f78a64ffb56bf8d1d75ec37275aef51f', 'import': 'git archive; no math edits'},
        'recommended_cli': ['tools/private_demo_service.py', 'tools/prepare_private_demo_suite.py',
            'tools/interactive_inspect.py', 'tools/inspect_model_targets.py',
            'tools/run_model_target_plan_live.py', 'tools/summarize_private_demo.py',
            'tools/audit_private_demo.py', 'tools/audit_source_baseline.py'],
        'new_NPU_use': False, 'runtime_behavior_changes_since_freeze': False,
        'known_limits': ['full_request_window unproven', 'DeepSeek unvalidated',
            'no TargetPlan general probability transaction', 'remove failure and early launch cleanup limitations retained',
            'scalar helper eps order differs from tensor path at tiny scales'],
        'evidence_index': 'evidence_index.json', 'worktree_inventory': 'worktree_inventory.json'}


def offline_tests():
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'tests')]),
               PYTHONIOENCODING='utf-8')
    rows = []
    for directory in ('tests', 'experiments/fake_requant_reference/tests'):
        p = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', directory, '-v'],
                           cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8')
        rows.append({'directory': directory, 'exit_code': p.returncode, 'output': p.stdout + p.stderr})
    return {'status': 'PASS' if all(r['exit_code'] == 0 for r in rows) else 'FAIL',
            'NPU_used': False, 'platform': sys.platform, 'python': sys.version, 'results': rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['inventory', 'verify', 'links', 'freeze', 'tests'])
    p.add_argument('--manifest', type=Path)
    p.add_argument('--inventory', type=Path)
    p.add_argument('--evidence-root', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.action == 'verify' and not a.manifest:
        p.error('verify requires --manifest')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.action == 'freeze' and (not a.inventory or not a.evidence_root):
        p.error('freeze requires --inventory and --evidence-root')
    result = (inventory() if a.action == 'inventory' else verify(a.manifest) if a.action == 'verify'
              else offline_tests() if a.action == 'tests'
              else freeze(a.inventory, a.evidence_root, a.out.parent) if a.action == 'freeze'
              else local_links(ROOT))
    write_new(a.out, result)
    print(json.dumps({'output': str(a.out), 'status': result.get('status'),
                      'broken_links': result.get('broken', result.get('links', {}).get('broken'))}, ensure_ascii=False))
    if result.get('status') == 'FAIL':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
