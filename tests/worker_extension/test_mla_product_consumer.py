import unittest
import torch
from fault_injection.sites.mla28 import verify_consumer_row,MLACopyMode
from types import SimpleNamespace
from fault_injection.product_contract import summarize_request


class MLAConsumerTests(unittest.TestCase):
    def test_observe_existing_native_copy_then_projection_reads_destination(self):
        source=torch.arange(12.).reshape(3,4);dest=torch.zeros_like(source)
        lease=dict(replacement=source,event={});site=SimpleNamespace(pending={'layer':lease})
        with MLACopyMode(site):dest[:]=source
        self.assertTrue(lease['event']['copy_exact'])
        verify_consumer_row(dest,lease['consumer_input'],2)
        self.assertTrue(torch.equal(dest,source))
        with self.assertRaisesRegex(RuntimeError,'STORAGE_NOT_LEASED'):
            verify_consumer_row(dest.clone(),lease['consumer_input'],2)
    def test_equal_but_unrelated_copy_is_not_evidence(self):
        source=torch.ones(3,4);lease=dict(replacement=source,event={});dest=torch.zeros_like(source)
        with MLACopyMode(SimpleNamespace(pending={'layer':lease})):dest.copy_(source.clone())
        self.assertNotIn('consumer_input',lease)
    def test_same_leased_row_and_view_accepted_equal_copy_refused(self):
        x = torch.arange(12.).reshape(3, 4)
        verify_consumer_row(x, x, 2)
        verify_consumer_row(x.view_as(x), x, 2)
        with self.assertRaisesRegex(RuntimeError, 'STORAGE_NOT_LEASED'):
            verify_consumer_row(x.clone(), x, 2)
        with self.assertRaisesRegex(RuntimeError, 'LAYOUT_MISMATCH'):
            verify_consumer_row(x.reshape(2, 6), x, 1)

    def test_product_keeps_mla_consumer_evidence_and_call_element_counts(self):
        event = dict(eligible=True, applied=True, consumed=True, restored=True, changed=1,
                     changed_elements=512, before_sha256='a', restored_sha256='a',
                     consumer_operator='module::RowParallelLinear.forward',
                     consumer_storage_verified=True, consumer_device_completion='SAME_STREAM_EVENT_SYNCHRONIZED',
                     representation='absorbed_decode_latent')
        result = summarize_request('r', 0, [event], completed=True)
        self.assertEqual(result['coverage']['changed'], 1)
        self.assertEqual(result['changed_elements'], 512)
        self.assertTrue(result['sites'][0]['consumer_storage_verified'])
        self.assertEqual(result['sites'][0]['representation'], 'absorbed_decode_latent')


if __name__ == '__main__': unittest.main()
