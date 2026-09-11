"""Linux host: IDENTITY, one lossy request, finally stop ONLY the leased container.

No launch, no resource acquisition, no scan, no restart. Uses the existing
private_demo_service lifecycle. Failure preserves artifacts and never retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from private_demo_service import command, stop, write


def run_once_host(session, suite, plan, identity_probe, probe, out, *, timeout=14400):
    session, out = Path(session).resolve(), Path(out).resolve()
    if out == session or not out.is_relative_to(session) or out.exists():
        raise ValueError('FRESH_OUTPUT_DIRECTORY_INSIDE_OWN_SESSION_REQUIRED')
    d = json.loads((session / 'deployment.json').read_text())
    owner = json.loads(Path(d['lock_owner_path']).read_text())
    if (owner.get('lease_id') != d['lease_id'] or owner.get('result_dir') != str(session)
            or owner.get('owner') != d['owner']):
        raise RuntimeError('EXCLUSIVE_LEASE_NOT_OWNED')
    info = json.loads(command(['docker', 'inspect', d['container_name']]))[0]
    if (info['Config']['Labels'].get('ascend-fi-lease') != d['lease_id'] or not info['State']['Running']):
        raise RuntimeError('OWN_RUNNING_MODEL_CONTAINER_REQUIRED')
    source = Path(d['source_root']).resolve()
    suite, plan = Path(suite).resolve(), Path(plan).resolve()
    for path in (suite, plan):
        if not path.is_file() or not (path.is_relative_to(session) or path.is_relative_to(source)):
            raise ValueError('INPUT_MUST_BE_IN_EXISTING_MOUNTED_SESSION_OR_SOURCE')
    if (session / 'single_lossy_instance_reserved.json').exists():
        raise RuntimeError('INSTANCE_ALREADY_RESERVED_OR_CONSUMED')
    contract = {'lease_id': d['lease_id'], 'container': d['container_name'], 'output': str(out),
                'policy': 'ONE_LOSSY_THEN_FULL_OWNED_CONTAINER_STOP',
                'suite_sha256': hashlib.sha256(suite.read_bytes()).hexdigest(),
                'plan_sha256_file': hashlib.sha256(plan.read_bytes()).hexdigest()}
    write(session / 'single_lossy_host_contract.json', contract)
    argv = ['docker', 'exec', d['container_name'], 'python3',
            str(source / 'tools/run_model_target_plan_live.py'), '--session', str(session),
            '--deployment', str(session / 'deployment.json'), '--execution-policy', 'stop-instance',
            '--suite', str(suite), '--plan', str(plan), '--identity-probe', identity_probe,
            '--probe', probe, '--out', str(out)]
    final = {'execution_policy': contract['policy'], 'stop_confirmed': False,
             'hot_recovery': 'NOT_MEASURED', 'restart_text_recovery': 'NOT_MEASURED'}
    try:
        result = subprocess.run(argv, timeout=timeout, check=False)
        final['driver_exit_code'] = result.returncode
    except BaseException as exc:
        final['driver_error'] = repr(exc)
        raise
    finally:
        try:
            stop(session)  # Existing exact lease + container-label checks, Docker whole-container stop.
            final.update(stop_confirmed=True, lease_released=True,
                         status='INSTANCE_STOPPED_TEXT_RECOVERY_NOT_MEASURED')
        except Exception as exc:
            final.update(stop_error=repr(exc), status='STOP_UNCONFIRMED_KEEP_LEASE_NO_FURTHER_REQUESTS')
        out.mkdir(parents=True, exist_ok=True)
        write(out / 'final_instance_state.json', final)
        keys = {'structure': 'structural_reports.json', 'binding': 'model_target_reports.json',
                'plan': 'target_plan.json', 'preview': 'target_preview.md', 'identity': 'identity.json',
                'execution_preview': 'execution_preview.json',
                'raw_pair': 'single_run.json', 'evaluation': 'effect/evaluation_result.json',
                'dashboard': 'effect/effect_dashboard.html', 'cleanup': 'final_state.json',
                'instance_stop': 'final_instance_state.json'}
        entries = {key: {'path': value, 'present': (out / value).exists(),
                        'sha256': hashlib.sha256((out / value).read_bytes()).hexdigest() if (out / value).is_file() else None}
                   for key, value in keys.items()}
        for key, file in [('plan_input', plan), ('binding_input', plan.parent / 'model_target_reports.json'),
                          ('preview_input', plan.parent / 'target_preview.md')]:
            entries[key] = {'path': os.path.relpath(file, out), 'present': file.is_file(),
                            'sha256': hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None}
        write(out / 'artifact_index.json', {'entries': entries, 'source_plan': str(plan),
              'binding_source': 'model_target_reports.json beside source plan; if absent explicitly uncollected',
              'lifecycle': final})
        with (out / 'README.md').open('x', encoding='utf-8') as f:
            f.write('# 单次有损推理产物\n\n' + final['status'] + '\n\n'
                '本目录只有一个 baseline/fault pair；没有热恢复或重启后文本对照。完整周期 N=0。\n\n')
            for key, entry in entries.items():
                f.write(f'- [{key}]({entry["path"]})\n' if entry['present'] else f'- {key}：未采集（不造记录）。\n')
    if not final['stop_confirmed'] or final.get('driver_exit_code') != 0:
        raise RuntimeError('SINGLE_REQUEST_OR_STOP_FAILED; inspect final_instance_state.json')
    return final


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True)
    p.add_argument('--suite', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--identity-probe', required=True)
    p.add_argument('--probe', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--confirm-stop-instance', action='store_true', required=True)
    args = p.parse_args()
    run_once_host(args.session, args.suite, args.plan, args.identity_probe, args.probe, args.out)


if __name__ == '__main__':
    main()
