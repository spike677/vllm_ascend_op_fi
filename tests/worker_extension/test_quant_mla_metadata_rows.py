"""Real CPU producer calls and object joins; not an Ascend metadata acceptance."""
import hashlib
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode, _get_current_dispatch_mode_stack

from demoncat.runtime.vllm_ascend.worker_extension.quant_mla_metadata_rows import MlaMetadataRows
from demoncat.runtime.vllm_ascend.worker_extension.quant_source_rows import source_bindings
from demoncat.runtime.vllm_ascend.worker_extension.quant_row_provenance import RequestRowProvenance


def cpu_rotary(positions, use_cache=False):
    cos = _cos_cache[positions].unsqueeze(1).unsqueeze(2)
    sin = _sin_cache[positions].unsqueeze(1).unsqueeze(2)
    if not use_cache:
        return cos, sin
    count = positions.size(0)
    _cos_mla[:count, ...] = cos
    _sin_mla[:count, ...] = sin
    return _cos_mla[:count, ...], _sin_mla[:count, ...]


class CpuBuilder:
    def build(self, common_prefix_len, common_attn_metadata):
        positions = common_attn_metadata.positions[:8].long()
        cos, sin = cpu_rotary(positions, use_cache=True)
        return NS(input_positions=positions, cos=cos[:8], sin=sin[:8])


class CpuRunner:
    def __init__(self):
        self.builder = CpuBuilder()

    def build(self, positions):
        return self.builder.build(0, NS(positions=positions))


class EarlyMetadataTests(unittest.TestCase):
    def setUp(self):
        global _cos_cache, _sin_cache, _cos_mla, _sin_mla
        self.inference = torch.inference_mode()
        self.inference.__enter__()
        self.addCleanup(lambda: self.inference.__exit__(None, None, None))
        _cos_cache = torch.arange(64, dtype=torch.float32).reshape(16, 4)
        _sin_cache = -_cos_cache
        _cos_mla, _sin_mla = torch.empty(16, 1, 1, 4), torch.empty(16, 1, 1, 4)
        self.positions = torch.arange(8, dtype=torch.int64)
        self.scope = dict(request_ids=['cpu-' + str(i) for i in range(8)], row_token_positions=list(range(8)))
        self.runner = CpuRunner()
        pins = {__name__: (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          {'CpuBuilder.build': 'builder', 'cpu_rotary': 'rotary'})}
        self.observer = MlaMetadataRows(self.runner, *source_bindings(pins), simulation=True)
        self.graph = RequestRowProvenance()
        self.addCleanup(self.cleanup)

    def cleanup(self):
        if self.graph.active:
            self.graph.end()
        if self.observer.active:
            self.observer.close()

    def produce(self, positions=None):
        self.observer.begin(self.scope['request_ids'])
        output = self.runner.build(self.positions if positions is None else positions)
        self.observer.close(scope=self.scope, actual_positions=self.positions)
        self.graph.begin('cpu-decode')
        self.graph.watched_storages.extend(self.observer.watched_storages())
        return output

    def test_actual_index_buffer_and_metadata_return_bind_without_reexecution(self):
        output = self.produce()
        proof = self.observer.consumer(output, self.graph)
        self.assertTrue(proof['verified'])
        self.assertTrue(proof['actual_index_operand_identity_verified'])
        self.assertEqual(proof['actual_rows'], self.scope)
        report = self.observer.report()
        self.assertTrue(report['complete'], report)
        self.assertEqual(sum(row['operator'] == 'aten.index.Tensor' for row in report['events']), 2)
        self.assertTrue(all(row['original_calls'] == 1 for row in report['events']))
        self.assertFalse(report['model_or_operator_reexecution'] or report['native_qualified'])
        self.assertIsNone(sys.getprofile())

    def test_equal_values_new_position_storage_cannot_borrow_actual_rows(self):
        output = self.produce(self.positions.clone())
        with self.assertRaisesRegex(RuntimeError, 'POSITION_OWNER_REQUIRED'):
            self.observer.consumer(output, self.graph)

    def test_equal_value_cos_clone_is_not_the_producer_output(self):
        output = self.produce()
        output.cos = output.cos.clone()
        with self.assertRaisesRegex(RuntimeError, 'NO_LIVE_ACTUAL_ROTARY'):
            self.observer.consumer(output, self.graph)

    def test_reused_output_buffer_written_after_producer_invalidates_consumer(self):
        self.observer.begin(self.scope['request_ids'])
        output = self.runner.build(self.positions)
        _cos_mla.add_(0)  # Identical values still have a later independent write.
        self.observer.close(scope=self.scope, actual_positions=self.positions)
        self.graph.begin('cpu-decode')
        with self.assertRaisesRegex(RuntimeError, 'NO_LIVE_ACTUAL_ROTARY'):
            self.observer.consumer(output, self.graph)

    def test_model_interval_write_epoch_invalidates_inference_buffer(self):
        output = self.produce()
        output.cos.add_(0)
        with self.assertRaisesRegex(RuntimeError, 'NO_LIVE_ACTUAL_ROTARY'):
            self.observer.consumer(output, self.graph)

    def test_model_position_write_even_with_identical_values_invalidates_producer_join(self):
        output = self.produce()
        self.positions.add_(0)
        with self.assertRaisesRegex(RuntimeError, 'POSITIONS_WRITTEN'):
            self.observer.consumer(output, self.graph)

    def test_fabricated_metadata_object_cannot_relabel_producer_aliases(self):
        output = self.produce()
        clone = NS(**vars(output))
        with self.assertRaisesRegex(RuntimeError, 'RETURN_AND_POSITION_OWNER'):
            self.observer.consumer(clone, self.graph)

    def test_budget_retains_missing_count_and_refuses_proof_without_reexecuting(self):
        self.observer.max_calls = 1
        output = self.produce()
        self.assertGreater(self.observer.report()['missing_calls'], 0)
        self.assertFalse(self.observer.report()['complete'])
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE_METADATA_INTERVAL'):
            self.observer.consumer(output, self.graph)

    def test_report_mutation_cannot_change_internal_producer_lease(self):
        output = self.produce()
        report = self.observer.report()
        report['boundaries'][1]['links'][0]['output']['stride'].reverse()
        report['events'][0]['original_calls'] = 900
        self.assertTrue(self.observer.consumer(output, self.graph)['verified'])
        self.assertEqual(self.observer.report()['events'][0]['original_calls'], 1)

    def test_end_of_forward_releases_live_producers_and_report_cannot_authorize_later_use(self):
        output = self.produce()
        self.assertTrue(self.observer.consumer(output, self.graph)['verified'])
        self.observer.release()
        report = self.observer.report()
        self.assertEqual(report['live_producer_records'], 0)
        self.assertEqual(len(report['boundaries']), 2)
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE_METADATA_INTERVAL'):
            self.observer.consumer(output, self.graph)

    def test_close_before_pop_failure_retains_mode_and_retries_after_profile_removal(self):
        self.observer.begin(self.scope['request_ids'])
        mode = self.observer.mode
        with patch.object(type(mode), '__exit__', side_effect=RuntimeError('TRANSIENT_EXIT_FAILURE')):
            with self.assertRaisesRegex(RuntimeError, 'TRANSIENT_EXIT_FAILURE'):
                self.observer.close()
        report = self.observer.report()
        self.assertTrue(report['active'] and report['cleanup_pending'])
        self.assertEqual((report['owned_profile_callbacks'], report['owned_dispatch_modes']), (0, 1))
        self.assertIsNone(sys.getprofile())
        self.assertIs(_get_current_dispatch_mode(), mode)
        self.observer.close()
        self.assertFalse(self.observer.active)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.assertTrue(torch.equal(torch.ones(2) + 1, torch.full((2,), 2.0)))

    def test_close_after_pop_failure_does_not_exit_the_removed_mode_twice(self):
        self.observer.begin(self.scope['request_ids'])
        mode = self.observer.mode
        original_exit = type(mode).__exit__
        def pop_then_raise(value, *args):
            original_exit(value, *args)
            raise RuntimeError('AFTER_POP_FAILURE')
        with patch.object(type(mode), '__exit__', pop_then_raise):
            with self.assertRaisesRegex(RuntimeError, 'AFTER_POP_FAILURE'):
                self.observer.close()
        self.assertTrue(self.observer.active)
        self.assertIsNone(self.observer.mode)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.observer.close()
        self.assertFalse(self.observer.active)

    def test_staged_close_refuses_foreign_profile_or_thread_then_original_owner_retries(self):
        self.observer.begin(self.scope['request_ids'])
        mode = self.observer.mode
        with patch.object(type(mode), '__exit__', side_effect=RuntimeError('RETRY')):
            with self.assertRaisesRegex(RuntimeError, 'RETRY'):
                self.observer.close()
        foreign = lambda *args: None
        sys.setprofile(foreign)
        try:
            with self.assertRaisesRegex(RuntimeError, 'OWNER_OR_INFLIGHT_CHANGED'):
                self.observer.close()
            self.assertIs(sys.getprofile(), foreign)
            self.assertIs(_get_current_dispatch_mode(), mode)
        finally:
            sys.setprofile(None)
        errors = []
        def foreign_thread():
            try:
                self.observer.close()
            except RuntimeError as error:
                errors.append(str(error))
        thread = threading.Thread(target=foreign_thread)
        thread.start()
        thread.join()
        self.assertEqual(len(errors), 1)
        self.assertIn('OWNER_OR_INFLIGHT_CHANGED', errors[0])
        self.observer.close()
        self.assertEqual(_get_current_dispatch_mode_stack(), [])


if __name__ == '__main__':
    unittest.main()
