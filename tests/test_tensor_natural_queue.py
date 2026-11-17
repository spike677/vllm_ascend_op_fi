import unittest
from experiments.c2c3_closure_20261117.prepare_tensor_natural_cases import cases


class TensorQueueTests(unittest.TestCase):
    def test_fresh_three_worker_cases_share_source_path_scope_and_request_script(self):
        rows = cases('/home/17-fault-c2c3-20261117/source-a22', 'worker18-statistics')
        self.assertEqual(len(rows), 3)
        self.assertEqual([row['command'].split('--arm ')[-1] for row in rows], ['baseline', 'baseline', 'fault'])
        self.assertTrue(all(row['mounts'] == rows[0]['mounts'] and row['env'] == rows[0]['env'] for row in rows))
        self.assertTrue(all(row['review']['physical_cards'] == [2] and row['review']['ranks'] == 1 for row in rows))
        self.assertTrue(all('/evidence/probe' in row['command'] for row in rows))

    def test_moe_only_real_configured_weights_and_no_injection(self):
        row, = cases('/home/17-fault-c2c3-20261117/source-a21', 'moe8-census', 5)
        self.assertEqual(row['review']['expected_hits'], 0)
        self.assertEqual(row['review']['physical_cards'], [5])
        self.assertIn('DeepSeek-V2-Lite-W8A8', row['review']['model'])
        self.assertIn('--memory .55 --moe-trace', row['command'])
        self.assertEqual(len(row['mounts']), 2)

    def test_unbounded_or_foreign_resource_input_is_refused(self):
        for source, card in (('/source/../elsewhere', 2), ('relative', 2), ('/source', 0)):
            with self.assertRaises(ValueError):
                cases(source, 'worker18-statistics', card)

    def test_actual_moe_group_requires_runtime_counts_and_full_recovery(self):
        row, = cases('/home/17-fault-c2c3-20261117/source-next', 'moe8-group')
        self.assertIn('natural_moe_worker_probe.py', row['command'])
        self.assertNotEqual(row['review']['expected_hits'], 0)
        self.assertIn('Runtime-qualified', row['review']['expected_hits'])
        self.assertIn('no layer/router changes', row['review']['recovery_plan'])


if __name__ == '__main__':
    unittest.main()
