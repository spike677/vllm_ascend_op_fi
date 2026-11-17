"""Queue preparation is device-free and cannot turn a preflight into C2/C3."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments/c2c3_closure_20261117'


def module(filename):
    spec = importlib.util.spec_from_file_location(filename.removesuffix('.py'), EXPERIMENT / filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class MetadataPreflightTests(unittest.TestCase):
    def test_case_preparation_preserves_no_fault_scope_and_distinct_output(self):
        for filename in ('providers20_prepare_group_case.py', 'providers39_prepare_cases.py'):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / 'case.json'
                argv = [filename, '--installed', '/installed-source', '--native', '/native-source',
                        '--probe', '/probe-source.py', '--build-sha', 'b' * 64, '--source-commit', 'a' * 40,
                        '--out', str(out), '--metadata-preflight']
                with patch('sys.argv', argv):
                    module(filename).main()
                case, = json.loads(out.read_text())
                self.assertTrue(case['id'].endswith('-metadata-preflight'))
                self.assertTrue(case['command'].endswith(' --metadata-preflight'))
                self.assertFalse('--statistics' in case['command'] or '--c3' in case['command'])
                self.assertTrue(case['review']['metadata_preflight_only'])
                self.assertFalse(case['review']['c2'] or case['review']['c3'])
                self.assertIn('Zero fault hits', case['review']['expected_hits'])
                self.assertEqual(case['review']['build_fingerprint'], 'b' * 64)

    def test_probe_and_preparer_reject_statistics_or_c3_before_importing_vllm(self):
        for filename in ('providers20_worker_probe.py', 'providers39_worker_probe.py',
                         'providers20_prepare_group_case.py', 'providers39_prepare_cases.py'):
            for incompatible in ('--statistics', '--c3'):
                with self.subTest(filename=filename, incompatible=incompatible):
                    argv = [filename, '--metadata-preflight', incompatible]
                    if 'prepare' in filename:
                        argv += ['--installed', '/installed', '--native', '/native', '--probe', '/probe.py',
                                 '--build-sha', 'b' * 64, '--source-commit', 'a' * 40, '--out', '/unused.json']
                    with patch('sys.argv', argv), contextlib.redirect_stderr(io.StringIO()) as errors:
                        with self.assertRaises(SystemExit) as failure:
                            module(filename).main()
                    self.assertEqual(failure.exception.code, 2)
                    self.assertIn('Metadata preflight has no fault/statistics/C3 phase', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
