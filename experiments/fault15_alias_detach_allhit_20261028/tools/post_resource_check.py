"""Read-only post-run resource/cleanup check; never stop or remove a container.

Run on the 124 host after the bounded controller finishes. Background containers
are explicitly device-capable; an idle snapshot never proves device exclusivity.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(attempt):
    completion = read(attempt / 'completion.json')
    identity = read(attempt / 'container_identity.json')
    policy = read(attempt / 'resource_policy.json')
    manifest = read(attempt / 'source_manifest.json')
    frozen = attempt / 'source'
    for relative, expected in manifest.items():
        path = (frozen / relative).resolve()
        require(frozen.resolve() in path.parents and sha(path) == expected, 'Frozen source changed: ' + relative)
    spec = importlib.util.spec_from_file_location('fault15_allhit_frozen_controller', frozen / 'controller.py')
    controller = importlib.util.module_from_spec(spec)
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec.loader.exec_module(controller)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    controller.ALLOW_IDLE_IDS = set(policy['idle_allowlist_ids'])
    controller.IDLE_BASELINES.update(policy['baseline_metadata'])
    controller.ATTEMPT_EVIDENCE = None  # Never append to the completed attempt.
    current = controller.resources(completion['card'], strict=False)
    owned = identity['id']
    require(re.fullmatch('[0-9a-f]{64}', owned), 'Invalid recorded owned container ID')
    require(completion.get('owned_container_id') == owned, 'Completion owned container ID differs')
    listing = subprocess.run(['docker', 'ps', '-a', '-q', '--no-trunc'],
                             capture_output=True, text=True, timeout=15)
    require(listing.returncode == 0 and all(re.fullmatch('[0-9a-f]{64}', value)
                                          for value in listing.stdout.split()), 'Cannot verify container inventory')
    inspected = subprocess.run(['docker', 'inspect', owned], capture_output=True, text=True, timeout=15)
    absent = (owned not in listing.stdout.split() and inspected.returncode == 1
              and inspected.stdout.strip() in ('', '[]')
              and inspected.stderr.strip() in ('Error: No such object: ' + owned,
                  'Error: No such container: ' + owned,
                  'Error response from daemon: No such container: ' + owned))
    require(absent, 'Exact owned container absence was not proven')
    require(identity['image'] == controller.IMAGE and identity['labels'].get('demoncat.task') == controller.TASK
            and identity['labels'].get('demoncat.controller_token'), 'Recorded owned identity differs')
    require(completion['status'] == 'PASS' and not completion['cleanup_errors']
            and completion['owned_container_removed'] and completion['device_lock_acquired']
            and completion['device_lock_released'] and completion['external_container_mutations'] is False,
            'Controller did not finish and clean up successfully')
    require([stage['arm'] for stage in completion['stages']] == ['B', 'F']
            and all(stage['rc'] == 0 and stage['independent_acceptance'] for stage in completion['stages']),
            'B/F independent acceptance is incomplete')
    require(read(attempt / 'baseline_gate.json')['status'] == 'PASS_BASELINE'
            and read(attempt / 'acceptance.json')['status'].startswith('PASS_'), 'Independent analyzer did not pass')
    return {'status': ('PASS_RUN_AND_CLEANUP' if current['eligible']
                       else 'PASS_RUN_AND_CLEANUP_CURRENT_RESOURCES_CHANGED'),
            'checked_epoch': time.time(), 'attempt': str(attempt),
            'owned_container_id': owned, 'owned_absent': absent,
            'absence_checks': {'docker_ps_all_rc': listing.returncode, 'inspect_rc': inspected.returncode,
                               'inspect_stderr': inspected.stderr},
            'completion_sha256': sha(attempt / 'completion.json'),
            'source_manifest_sha256': sha(attempt / 'source_manifest.json'),
            'current_resources': current, 'idle_review_source': policy['review_source'],
            'checker_sha256': sha(Path(__file__)),
            'scope': 'Checks completed-run status and exact owned cleanup; current resources are a later snapshot, not proof of exclusivity.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Post-resource output must be fresh')
    try:
        report = check(args.attempt_path.resolve())
    except Exception:
        report = {'status': 'FAIL_POST_RESOURCE_CHECK', 'checked_epoch': time.time(),
                  'traceback': traceback.format_exc(), 'checker_sha256': sha(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({key: report[key] for key in ('status', 'owned_absent', 'traceback') if key in report}))
    return 0 if report['status'].startswith('PASS_') else 1


if __name__ == '__main__':
    raise SystemExit(main())
