import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from queue_contract import validate_cases, LABEL
from run_queue_124 import cleanup_owned


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
