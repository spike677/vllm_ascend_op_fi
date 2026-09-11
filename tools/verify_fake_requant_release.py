"""Offline release audit and raw Qwen replay; never contact or control a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ascend_fi.contracts import EvaluationResult
from ascend_fi.evaluation.distortion.contracts import DistortionProbe, content_hash
from ascend_fi.evaluation.distortion.evaluator import compare
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate
from ascend_fi.evaluation.distortion.report import write_report
from ascend_fi.inspect.plan_builder import verify_plan
from ascend_fi.inspect.plan_executor import plan_window_gate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write('\n')


def within(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('PATH_OUTSIDE_EVIDENCE_OR_SOURCE_ROOT')
    return path


def source_equivalence(root, captured):
    """Only presentation/tests may differ; no general 'accept changed code' flag."""
    if not captured.get('files'):
        raise ValueError('EMPTY_CAPTURED_SOURCE_MANIFEST')
    rows = []
    for name, digest in captured['files'].items():
        path = within(root, name)
        actual = sha(path) if path.is_file() else None
        category = ('MATCH' if actual == digest else 'PRESENTATION_ONLY'
                    if name == 'src/ascend_fi/evaluation/distortion/report.py' and actual else
                    'TEST_ONLY' if name.startswith('tests/') and actual else 'LIVE_REVALIDATION_REQUIRED')
        rows.append({'path': name, 'captured_sha256': digest, 'release_sha256': actual, 'category': category})
    return {'status': 'PASS' if all(r['category'] != 'LIVE_REVALIDATION_REQUIRED' for r in rows) else 'FAIL',
            'captured_files': len(rows), 'exact_matches': sum(r['category'] == 'MATCH' for r in rows),
            'rows': rows, 'scope': 'captured deployment files; external dependencies separately pinned'}


def file_map_mismatches(root, expected):
    if not expected:
        raise ValueError('EMPTY_RELEASE_FILE_MAP')
    return [name for name, digest in expected.items()
            if not within(root,name).is_file() or sha(within(root,name)) != digest]


def exact_responses(rows, phases):
    base = rows[phases[0]]
    return (isinstance(base.get('response_text'), str) and isinstance(base.get('token_pieces'), list)
            and all(rows[p].get('http_status') == 200
                and rows[p].get('response_text') == base['response_text']
                and rows[p].get('token_pieces') == base['token_pieces'] for p in phases))


def audit_qwen(evidence, suite):
    """Recompute from raw request records, not a prior PASS label."""
    root = evidence/'model_target_acceptance'
    if suite.get('hash') != content_hash(suite):
        raise ValueError('FROZEN_SUITE_HASH_MISMATCH')
    plan, identity, pair = (read(root/f) for f in ('target_plan.json', 'identity.json', 'pair.json'))
    verify_plan(plan)
    # Every response must agree with its independently saved raw record.
    all_rows = list(pair['pair'].values()) + list(identity['validation']['raw'].values())
    for row in all_rows:
        marker = '/model_target_acceptance/'
        source = row.get('source_ref', '')
        if marker not in source:
            raise ValueError('RAW_RESPONSE_REFERENCE_MISSING')
        raw = within(root, source.split(marker, 1)[1])
        if read(raw) != row or row.get('response_sha256') != 'sha256:' + hashlib.sha256(row['response_text'].encode()).hexdigest():
            # Historical collector stores a bare SHA in some versions.
            if read(raw) != row or row.get('response_sha256') != hashlib.sha256(row['response_text'].encode()).hexdigest():
                raise ValueError('RAW_RESPONSE_OR_TEXT_HASH_MISMATCH')
    id_raw = identity['validation']['raw']
    id_gate = plan_window_gate(plan, id_raw['identity'], identity=True)
    gate = plan_window_gate(plan, pair['pair']['fault'])
    if (id_gate['mechanism'] != 'PASS' or gate['mechanism'] != 'PASS'
            or not exact_responses(id_raw, ('baseline', 'disabled', 'identity', 'recovery'))
            or not exact_responses(pair['pair'], ('baseline', 'recovery'))):
        raise ValueError('QWEN_MINIMAL_IDENTITY_MECHANISM_OR_RECOVERY_FAILED')
    final, state = read(evidence/'final_environment.json'), read(root/'final_state.json')
    if (cleanup_gate(state['query'], state['healthy'], len(plan['workers']))['status'] != 'PASS'
            or final.get('final_health') is not True or final.get('lock_released') is not True
            or final.get('fault20_control', {}).get('enabled') != 0):
        raise ValueError('CLEANUP_EVIDENCE_MISSING')
    data = read(root/'effect/evaluation_result.json')
    probe_id = pair['pair']['baseline']['sample_id']
    probe = next(p for p in suite['probes'] if p['probe_id'] == probe_id)
    for row in pair['pair'].values():
        req = row['request']
        if (req != pair['pair']['baseline']['request']
                or req.get('messages') != [{'role': 'user', 'content': probe['prompt']}]
                or any(req.get(k) != v for k,v in probe['generation_config'].items())
                or any(req.get(k) != suite['request_protocol'][k] for k in ('chat_template_kwargs','logprobs','top_logprobs'))):
            raise ValueError('FROZEN_REQUEST_MISMATCH')
    saved = data['distortion']['cycles'][0]['matrix'][0]
    for phase, key in (('fault', 'effect'), ('recovery', 'recovery_effect')):
        replay = compare(DistortionProbe.from_dict(probe), pair['pair']['baseline'], pair['pair'][phase], suite['thresholds'])
        if replay != saved[key]:
            raise ValueError('SCORER_REPLAY_DIFFERENT; preserve and review before release')
    if data['model_targeting']['plan_sha256'] != plan['plan_sha256']:
        raise ValueError('REPORT_PLAN_MISMATCH')
    return {'status': 'QWEN_MINIMAL_FUNCTIONAL_ACCEPTANCE_PASS', 'plan_sha256': plan['plan_sha256'],
            'targets': len(plan['selected_modules']), 'identity': id_gate['mechanism'],
            'mechanism': gate['mechanism'], 'target_call_window': gate['target_call_window'],
            'full_request_window': gate['full_request_window'], 'recovery_exact': True,
            'raw_replay': 'PASS', 'complete_lossy_cycles': 1,
            'reference_backends': [w['backend'] for w in plan['workers']],
            'actual_arm': plan['prospective_arm'], 'dataset': suite['dataset'],
            'suite_hash': suite['hash'], 'thresholds': suite['thresholds'],
            'request_protocol': pair['pair']['baseline']['request'],
            'scorer': data['scorer'], 'environment_record': 'final_environment.json',
            'environment_is_historical_not_current_probe': True}, data


def run(argv, cwd=ROOT, env=None):
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=180)
    return {'command': argv, 'returncode': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True, help='immutable downloaded live_result directory')
    parser.add_argument('--suite', type=Path, required=True, help='frozen local MT-Bench suite JSON')
    parser.add_argument('--output', type=Path, required=True, help='new directory outside source evidence')
    parser.add_argument('--manifest', type=Path, help='verify an existing release file hash manifest')
    args = parser.parse_args()
    evidence, output = args.evidence.resolve(), args.output.resolve()
    if output == evidence or output.is_relative_to(evidence):
        parser.error('output must be outside original evidence')
    output.mkdir(parents=True, exist_ok=False)
    before = {p.relative_to(evidence).as_posix(): sha(p) for p in evidence.rglob('*') if p.is_file()}
    checks = {'source_equivalence': source_equivalence(ROOT, read(evidence/'source_manifest.json'))}
    if args.manifest:
        release = read(args.manifest)
        if release.get('hash') != content_hash(release):
            raise ValueError('RELEASE_MANIFEST_HASH_MISMATCH')
        checks['release_file_hashes'] = {'mismatch': file_map_mismatches(ROOT, release['source_files'])}
        checks['release_evidence_hashes'] = {'mismatch': file_map_mismatches(evidence, release['qwen_evidence_files'])}
        checks['release_suite_hash'] = sha(args.suite) == release['frozen_suite_file_sha256']
    qwen, result = audit_qwen(evidence, read(args.suite))
    checks['qwen_minimum'] = qwen
    # Source-layout installation: no editable pip metadata or internet dependency.
    stage = output/'source_install'
    for folder in ('src', 'tools', 'profiles', 'tests'):
        shutil.copytree(ROOT/folder, stage/folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # Existing contract tests dereference historical source/evidence pointers;
    # include exactly those fixtures, not the large raw dump directories.
    fixture_paths = {'artifacts/evaluation/qwen_2card_fault_effect_v1/v4/candidate.json'}
    for path in (ROOT/'profiles/capabilities').glob('*.json'):
        canonical = read(path)['canonical']
        fixture_paths.update(p for group in ('source','build','evidence') for p in canonical[group])
    for name in sorted(fixture_paths):
        source, dest = within(ROOT, name), within(stage, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    checks['source_install_supporting_fixtures'] = {p: sha(ROOT/p) for p in sorted(fixture_paths)}
    env = {**os.environ, 'PYTHONPATH': os.pathsep.join((str(stage/'src'), str(stage/'tests'))), 'PYTHONIOENCODING': 'utf-8'}
    checks['source_install_import'] = run([sys.executable, '-c',
        'from ascend_fi.inspect.plan_executor import TargetPlanExecutor; from ascend_fi.inspect.model_adapters import *; '
        'from ascend_fi.evaluation.distortion.evaluator import SCORER_VERSION; print(SCORER_VERSION)'], stage, env)
    checks['unit_tests'] = run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-q'], stage, env)
    checks['cli_help'] = {name: run([sys.executable, 'tools/'+name, '--help'], stage, env) for name in (
        'interactive_inspect.py', 'inspect_model_targets.py', 'run_model_target_plan_live.py',
        'run_distortion_evaluation.py', 'verify_fake_requant_release.py')}
    checks['diff_check'] = run(['git', 'diff', '--check'])
    checks['branch'] = run(['git', 'branch', '--show-current'])['stdout'].strip()
    checks['python_version'] = sys.version
    checks['python_executable'] = sys.executable
    checks['HEAD_before_freeze'] = run(['git', 'rev-parse', 'HEAD'])['stdout'].strip()
    checks['raw_evidence_unchanged'] = all(sha(evidence/p) == h for p,h in before.items())
    checks['source_files'] = {p.relative_to(ROOT).as_posix(): sha(p) for folder in ('src','tools','tests','profiles')
        for p in sorted((ROOT/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json','.cjs')}
    checks['evidence_files'] = before
    checks['source_install_sha256'] = content_hash(checks['source_files'])
    passed = (checks['source_equivalence']['status'] == 'PASS' and checks['raw_evidence_unchanged']
        and all(checks[k]['returncode'] == 0 for k in ('source_install_import','unit_tests','diff_check'))
        and all(v['returncode'] == 0 for v in checks['cli_help'].values())
        and not checks.get('release_file_hashes',{}).get('mismatch')
        and not checks.get('release_evidence_hashes',{}).get('mismatch')
        and checks.get('release_suite_hash', True))
    checks['status'] = 'TOOL_RELEASE_MINIMUM_READY' if passed else 'RELEASE_CANDIDATE_INCOMPLETE'
    checks['boundaries'] = {'new_inference_requests': 0, 'npu_lock_acquired': False,
        'native_deepseek': 'PENDING_CONCRETE_INSTANCE', 'independent_step_coverage': 'DEFERRED_UNPROVEN',
        'gsm8k_ppl_three_cycles_for_m20': 'NOT_MEASURED_NOT_A_RELEASE_PREREQUISITE'}
    save(output/'release_audit.json', checks)
    write_report(output/'qwen_replay', EvaluationResult.from_dict(result), {
        'operation': 'READONLY_RAW_GATES_AND_SCORER_REPLAY', 'source': str(evidence),
        'files_read': before, 'suite_sha256': sha(args.suite), 'report_source_sha256': sha(ROOT/'src/ascend_fi/evaluation/distortion/report.py'),
        'new_model_experiment': False, 'original_evidence_modified': False})
    print(json.dumps({'status': checks['status'], 'source_equivalence': checks['source_equivalence']['status'],
        'qwen': qwen['status'], 'tests': checks['unit_tests']['stderr'], 'output': str(output)}, ensure_ascii=True))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
