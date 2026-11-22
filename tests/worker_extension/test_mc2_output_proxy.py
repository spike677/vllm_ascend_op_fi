import unittest
import torch
from fault_injection.engines.region_lease import RegionLeases, exact
from fault_injection.sites.mc2_output8 import MC2OutputConsumer, observe_combine_return, shifted_rows


class MC2OutputTests(unittest.TestCase):
    def setup_mode(self):
        completions = []
        leases = RegionLeases('scope', 1, stream=lambda: 's', complete=completions.append)
        return MC2OutputConsumer(leases), completions

    def test_dynamic_groups_padding_no_cross_request_or_expert_metadata_change(self):
        x = torch.arange(24.).reshape(6, 4)
        y, indices = shifted_rows(x, ['b', 'a', 'b', None, 'a', 'c'])
        self.assertEqual(indices, [2, 4, 0, 3, 1, 5])
        self.assertTrue(torch.equal(y, x[indices]))
        self.assertEqual(shifted_rows(x, ['x'] * 6)[1], [1, 2, 3, 4, 5, 0])

    def test_actual_combine_output_consumed_and_exactly_restored_two_scopes(self):
        for scope in range(2):
            mode, completions = self.setup_mode()
            native = torch.arange(12.).reshape(3, 4)
            before = native.clone()
            metadata = {'expert_ids': object(), 'ep_counts': object()}
            calls = []
            def original(**kwargs):
                calls.append(kwargs)
                return native
            with mode:
                result = observe_combine_return(original, (), metadata, mode=mode,
                    context=dict(actual_scheduler_join=True,
                        operator='npu_moe_distribute_combine_v2', request_rows=['r'] * 3,
                        site='live.combine', fault=True))
                self.assertIs(result, native)
                consumed = result + 1
            self.assertTrue(torch.equal(consumed, before[[1, 2, 0]] + 1))
            self.assertTrue(exact(native, before))
            self.assertEqual(calls, [metadata])
            self.assertIs(calls[0]['ep_counts'], metadata['ep_counts'])
            e = mode.events[0]
            self.assertTrue(e['applied'] and e['consumed'] and e['restored'])
            self.assertEqual(e['changed'], 12)
            self.assertGreaterEqual(len(completions), 4)
            mode.close()

    def test_one_row_decode_and_inactive_rank_do_not_create_calls(self):
        mode, _ = self.setup_mode()
        with mode:
            mode.watch(torch.ones(1, 4), ['r'], site='decode', fault=True)
        self.assertFalse(mode.events[0]['eligible'])
        mode.close()
        other, _ = self.setup_mode()
        with other:
            torch.ones(2) + 1
        self.assertEqual(other.events, [])
        other.close()

    def test_inplace_consumer_refused_and_retains_lease(self):
        mode, _ = self.setup_mode()
        native = torch.arange(8.).reshape(2, 4)
        with mode:
            mode.watch(native, ['r', 'r'], site='combine', fault=True)
            with self.assertRaisesRegex(RuntimeError, 'WRITES_PROTECTED'):
                native.add_(1)
        with self.assertRaisesRegex(RuntimeError, 'RETAINS_LEASE'):
            mode.close()

    def test_allocation_from_output_is_not_value_consumption(self):
        mode, _ = self.setup_mode()
        native = torch.arange(8.).reshape(2, 4)
        with mode:
            mode.watch(native, ['r', 'r'], site='combine', fault=True)
            with self.assertRaisesRegex(RuntimeError, 'METADATA'):
                native.new_empty((2, 4))
        self.assertFalse(mode.events[0]['consumed'])

    def test_unjoined_or_wrong_mapping_refused(self):
        mode, _ = self.setup_mode()
        with self.assertRaisesRegex(RuntimeError, 'JOIN_UNPROVEN'):
            observe_combine_return(lambda: None, (), {}, mode=mode, context={})
        with self.assertRaisesRegex(RuntimeError, 'ROW_MAPPING'):
            shifted_rows(torch.ones(2, 3), ['one'])


if __name__ == '__main__':
    unittest.main()
