import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'fault39' / 'tools'))
from queue_contract import validate_cases, LABEL
from run_queue_124 import cleanup_owned, read_stale_scope, scoped_gate


def case():
    return dict(id='sample', image='image', command='exec python3 /source/probe.py',
                mounts=[['/home/source', '/source', 'ro']],
                review=dict(experiment_id='sample', provider='39.matmul_range',
                            model='Qwen3', cann='8.5.1', soc='910B4',
                            physical_cards=[2], ranks=[0], worker='standard_vllm',
                            build_fingerprint='reviewed-manifest', expected_hits='observer-defined',
                            resource_budget=dict(timeout_seconds=900, memory_gib=32, cpus=8),
                            recovery_plan='same-process OFF + zero resources'))


class QueueContract(unittest.TestCase):
    def test_no_scope_keeps_stale_review_disabled(self):
        self.assertEqual(read_stale_scope(None, '/home/test/attempt-one'), (None, None))
        gate = Mock(return_value={'idle': False})
        self.assertFalse(scoped_gate(gate, [5], None, '/home/test/attempt-one')['idle'])
        gate.assert_called_once_with([5], review_stale=False,
                                     stale_allocation_scope=None,
                                     attempt_scope='/home/test/attempt-one')

    def test_scope_binds_attempt_and_preserves_original_bytes(self):
        scope = dict(schema='demoncat.stale_allocation_scope.v1',
                     attempt_scope='/home/test/attempt-one',
                     pod_uid='33991897-e3ed-4910-aa2c-835b9fe682a0',
                     checkpoint_sha256='b' * 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scope.json'
            raw = (json.dumps(scope, indent=2) + '\n').encode()
            path.write_bytes(raw)
            self.assertEqual(read_stale_scope(path, scope['attempt_scope']), (scope, raw))
            with self.assertRaises(ValueError):
                read_stale_scope(path, '/home/test/attempt-two')
            gate = Mock(return_value={'idle': False})
            self.assertFalse(scoped_gate(gate, [5], scope, scope['attempt_scope'])['idle'])
            gate.assert_called_once_with([5], review_stale=True,
                                         stale_allocation_scope=scope,
                                         attempt_scope=scope['attempt_scope'])

    def test_duplicate_or_unbounded_scope_rejected_before_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'scope.json'
            path.write_bytes(b'{"pod_uid":"one","pod_uid":"two"}')
            with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
                read_stale_scope(path, '/home/test/attempt-one')
            path.write_bytes(b' ' * 4097)
            with self.assertRaisesRegex(ValueError, 'BOUNDED'):
                read_stale_scope(path, '/home/test/attempt-one')

    def test_valid_bounded_review(self):
        self.assertEqual(validate_cases([case()], [2])[0]['id'], 'sample')

    def test_missing_review_and_resource_mismatch_refused(self):
        for field in list(case()['review']):
            value = case()
            del value['review'][field]
            with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
                validate_cases([value], [2])
        with self.assertRaisesRegex(ValueError, 'MISMATCH'):
            validate_cases([case()], [5])

    def test_duplicate_cases_and_over_budget_refused(self):
        with self.assertRaises(ValueError):
            validate_cases([case(), case()], [2])
        with self.assertRaises(ValueError):
            validate_cases([case()] * 9, [2])
        value = case()
        value['review']['resource_budget']['timeout_seconds'] = 1801
        with self.assertRaises(ValueError):
            validate_cases([value], [2])

    def test_foreign_writes_and_evidence_shadowing_refused(self):
        for mount in [['/home/model', '/model', 'rw'],
                      ['/tmp/fake', '/evidence', 'ro'],
                      ['/tmp/fake', '/../source', 'ro']]:
            value = case()
            value['mounts'].append(mount)
            with self.assertRaises(ValueError):
                validate_cases([value], [2])

    def test_cleanup_never_stops_foreign_owner(self):
        response = type('Response', (), {'returncode': 0, 'stdout': json.dumps([
            {'Config': {'Labels': {'dcat.task': 'foreign'}},
             'State': {'Running': True}}])})()
        with patch('run_queue_124.subprocess.run', return_value=response), \
                patch('run_queue_124.command') as cmd:
            with self.assertRaisesRegex(RuntimeError, 'OWNERSHIP_CHANGED'):
                cleanup_owned('sample')
            cmd.assert_not_called()

    def test_container_cleanup_is_not_recovery(self):
        info = {'Config': {'Labels': {'dcat.task': LABEL}}, 'State': {'Running': False}}
        with patch('run_queue_124.inspect_owned', return_value=info), \
                patch('run_queue_124.command', return_value='') as cmd:
            result = cleanup_owned('sample')
        self.assertFalse(result['recovery_proven'])
        cmd.assert_called_once_with(['docker', 'rm', 'sample'])


if __name__ == '__main__':
    unittest.main()
