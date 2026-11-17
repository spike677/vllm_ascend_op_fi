"""Run engineering and affected regression suites with an auditable CPU receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest


REPO = Path(__file__).resolve().parents[2]
REGRESSIONS = (
    'test_coverage_session', 'test_coverage_planning', 'test_coverage_audit',
    'test_boundary_targets', 'test_auto_adaptation', 'test_statistics_audit',
    'test_evidence_manager', 'test_evidence_scope', 'test_natural_tensor_observer',
    'test_tensor_callable', 'test_postwrite_providers', 'test_postwrite_natural_kv',
    'test_postwrite_natural_boundaries', 'test_postwrite_coverage',
    'test_postwrite_capture_window', 'test_postwrite_coverage_probe',
    'test_postwrite41_coverage', 'test_postwrite41_worker_contract',
    'test_postwrite_native_snapshot', 'test_paged33_loading', 'test_native_build',
    'test_writer26_inventory', 'test_quant_inventory', 'test_quant_group',
    'test_conv3d_natural', 'test_coverage_proof_authority',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--engineering-only', action='store_true')
    args = parser.parse_args()
    for path in (REPO, REPO/'src', REPO/'fault39/src', REPO/'tests', REPO/'tests/worker_extension'):
        sys.path.insert(0, str(path))
    suite = unittest.TestSuite()
    for name in ('cpu', 'native_mock', 'linux_loader', 'numerical_reference'):
        directory = Path(__file__).parent / name
        if directory.exists():
            suite.addTests(unittest.TestLoader().discover(str(directory), pattern='test_*.py'))
    if not args.engineering_only:
        suite.addTests(unittest.TestLoader().loadTestsFromNames(REGRESSIONS))
    start = time.monotonic()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    packages = {}
    for name in ('torch', 'numpy'):
        try:
            packages[name] = importlib.import_module(name).__version__
        except ImportError:
            packages[name] = None
    sources = {}
    for folder in ('engineering_closure', 'fault_injection', 'src/demoncat', 'tests'):
        for path in sorted((REPO/folder).rglob('*.py')):
            sources[path.relative_to(REPO).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema='demoncat.engineering.validation.v1',
        recorded_at=datetime.now(timezone.utc).isoformat(), python=sys.version,
        executable=sys.executable, platform=platform.platform(), packages=packages,
        tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
        skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
        failure_details=[dict(test=str(test), traceback=detail) for test, detail in result.failures + result.errors],
        elapsed_seconds=round(time.monotonic()-start, 3), success=result.wasSuccessful(),
        source_sha256=sources, npu_calls=0, new_mechanism_qualification_granted=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('tests_run', 'failures', 'errors', 'skipped', 'success')}))
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
