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
        positions = positions[:8]
        cos, sin = cpu_rotary(positions, use_cache=True)
        return NS(input_positions=positions, cos=cos[:8], sin=sin[:8])


class CpuRunner:
    def __init__(self):
        self.builder = CpuBuilder()

    def build(self, positions):
        return self.builder.build(0, NS(positions=positions))


class CpuChangedBuilder:
    def __init__(self, start=0, end=8, step=1, clone=False, mutate=False, second=True,
                 second_start=0, second_step=1, side_slice=False, replace=False, third=False, clone_middle=False):
        self.start, self.end, self.step, self.clone, self.mutate = start, end, step, clone, mutate
        self.second, self.second_start, self.second_step = second, second_start, second_step
        self.side_slice, self.replace = side_slice, replace
        self.third, self.clone_middle = third, clone_middle

    def build(self, common_prefix_len, common_attn_metadata):
        positions = common_attn_metadata.positions[self.start:self.end:self.step].long()
        if self.clone_middle:
            positions = positions.clone()
        if self.second:
            positions = positions[self.second_start:8:self.second_step]
        if self.third:
            positions = positions[:8]
        if self.clone:
            positions = positions.clone()
        if self.mutate:
            common_attn_metadata.positions.add_(0)
        cos, sin = cpu_rotary(positions, use_cache=True)
        if self.side_slice:
            other = common_attn_metadata.positions[:8]
            if self.replace:
                positions = other
        return NS(input_positions=positions, cos=cos[:8], sin=sin[:8])


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

    def padded_source(self, *, versioned=False):
        with torch.inference_mode(not versioned):
            source = torch.zeros(272, dtype=torch.int64)
            source[:8] = torch.arange(8)
            if versioned:
                for _ in range(5):
                    source.add_(0)
            self.positions = source[:8]
        return source

    def use_changed_builder(self, **settings):
        self.runner.builder = CpuChangedBuilder(**settings)
        pins = {__name__: (hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          {'CpuChangedBuilder.build': 'builder', 'cpu_rotary': 'rotary'})}
        self.observer = MlaMetadataRows(self.runner, *source_bindings(pins), simulation=True)

    def test_actual_padded_common_prefix_slice_proves_unpadded_metadata_return(self):
        source = self.padded_source()
        output = self.produce(source)
        proof = self.observer.consumer(output, self.graph)
        chain = proof['position_lineage']
        self.assertEqual(chain['source']['shape'], [272])
        self.assertEqual(chain['slice_output']['shape'], [8])
        self.assertEqual(chain['source']['storage_address'], chain['slice_output']['storage_address'])
        self.assertEqual(chain['actual_slice_arguments'], [0, 0, 8, 1])
        self.assertTrue(chain['actual_return_object_verified'])
        self.assertFalse(chain['values_used_as_authority'])
        self.assertEqual(chain['source_to_return_actual_slice_count'], 2)
        first, second = chain['slice_chain']
        self.assertLess(first['event'], second['event'])
        self.assertEqual(first['completed_output'], second['input'])
        self.assertEqual(second['completed_output']['object_id'], id(output.input_positions))

    def test_padded_position_version_drift_after_return_refuses_even_same_values(self):
        source = self.padded_source(versioned=True)
        output = self.produce(source)
        chain = self.observer.consumer(output, self.graph)['position_lineage']
        self.assertEqual(chain['source']['version'], 6)
        self.assertEqual(chain['slice_dispatch_output']['version'], 0)
        self.assertEqual(chain['slice_output']['version'], 6)
        self.assertEqual(chain['slice_dispatch_output']['object_id'], chain['slice_output']['object_id'])
        self.assertTrue(all(link['dispatch_output']['version'] == 0 and link['completed_output']['version'] == 6
                            for link in chain['slice_chain']))
        self.assertTrue(all(chain['slice_chain'][0]['event'] < event < chain['slice_chain'][1]['event']
                            for event in chain['identity_cast_events']))
        self.graph.end()
        source.add_(0)  # No dispatch observer: the real shared version still changes.
        self.graph.begin('version-drift')
        with self.assertRaisesRegex(RuntimeError, 'COMPLETE_METADATA_INTERVAL'):
            self.observer.consumer(output, self.graph)

    def test_padded_inference_alias_write_between_producer_and_close_refuses(self):
        source = self.padded_source()
        self.observer.begin(self.scope['request_ids'])
        output = self.runner.build(source)
        source.add_(0)  # Inference tensors have no _version: all storage aliases share an epoch.
        self.observer.close(scope=self.scope, actual_positions=self.positions)
        self.graph.begin('decode')
        with self.assertRaisesRegex(RuntimeError, 'POSITION_OWNER_REQUIRED'):
            self.observer.consumer(output, self.graph)

    def test_changed_prefix_start_end_or_step_cannot_borrow_rows(self):
        for settings in ({'start': 1, 'end': 9}, {'end': 16, 'step': 2}, {'end': 7}):
            with self.subTest(settings=settings):
                self.use_changed_builder(**settings)
                output = self.produce(self.padded_source())
                with self.assertRaisesRegex(RuntimeError, 'POSITION_OWNER_REQUIRED'):
                    self.observer.consumer(output, self.graph)
                self.assertIn('PREFIX_SLICE_CHANGED', self.observer.rows[0]['reason'])
                self.graph.end()

    def test_equal_value_clone_after_slice_has_no_actual_return_object_chain(self):
        self.use_changed_builder(clone=True)
        output = self.produce(self.padded_source())
        with self.assertRaisesRegex(RuntimeError, 'POSITION_OWNER_REQUIRED'):
            self.observer.consumer(output, self.graph)
        self.assertIn('NO_OBSERVED_PREFIX_SLICE', self.observer.rows[0]['reason'])

    def test_independent_later_same_storage_slice_cannot_replace_either_actual_edge(self):
        self.use_changed_builder(side_slice=True)
        output = self.produce(self.padded_source(versioned=True))
        proof = self.observer.consumer(output, self.graph)
        chain = proof['position_lineage']['slice_chain']
        source_id = chain[0]['input']['object_id']
        side = [event for event in self.observer.events if event['operator'] == 'aten.slice.Tensor'
                and event['inputs'][0]['object_id'] == source_id and event['sequence'] > chain[1]['event']]
        self.assertEqual(len(side), 1)
        self.assertNotEqual(side[0]['outputs'][0]['object_id'], chain[1]['completed_output']['object_id'])
        self.assertEqual(side[0]['outputs'][0]['storage_address'], chain[1]['completed_output']['storage_address'])

    def test_unreviewed_one_or_three_slices_clone_middle_and_wrong_second_slice_refuse(self):
        for settings in ({'second': False}, {'third': True}, {'clone_middle': True},
                         {'second_start': 1}, {'second_step': 2}, {'side_slice': True, 'replace': True}):
            with self.subTest(settings=settings):
                self.use_changed_builder(**settings)
                output = self.produce(self.padded_source())
                with self.assertRaisesRegex(RuntimeError, 'POSITION_OWNER_REQUIRED'):
                    self.observer.consumer(output, self.graph)
                builder = self.observer.report()['boundaries'][0]
                self.assertEqual(builder['status'], 'UNVERIFIED')
                self.assertEqual(builder['returned_positions_metadata']['object_id'], id(output.input_positions))
                self.graph.end()

    def test_retained_middle_object_metadata_cannot_drift_outside_observed_graph(self):
        source = self.padded_source()
        output = self.produce(source)
        self.assertTrue(self.observer.consumer(output, self.graph)['verified'])
        middle = self.observer.rows[0]['position_chain_tensors'][1]
        self.graph.end()
        middle.set_(source.untyped_storage(), 1, (8,), (1,))
        self.graph.begin('changed-middle')
        with self.assertRaisesRegex(RuntimeError, 'POSITION_OWNER_REQUIRED'):
            self.observer.consumer(output, self.graph)

    def test_position_write_inside_builder_invalidates_entry_epoch(self):
        self.use_changed_builder(mutate=True)
        output = self.produce(self.padded_source())
        with self.assertRaisesRegex(RuntimeError, 'POSITION_OWNER_REQUIRED'):
            self.observer.consumer(output, self.graph)
        self.assertIn('POSITIONS_CHANGED_DURING_BUILDER', self.observer.rows[0]['reason'])

    def test_equal_value_cos_clone_is_not_the_producer_output(self):
        output = self.produce()
        output.cos = output.cos.clone()
        with self.assertRaisesRegex(RuntimeError, 'NO_LIVE_ACTUAL_ROTARY'):
            self.observer.consumer(output, self.graph)

    def test_replacing_returned_position_object_with_equal_extent_alias_refuses(self):
        output = self.produce(self.padded_source())
        self.assertIsNot(output.input_positions, self.positions)
        output.input_positions = self.positions
        with self.assertRaisesRegex(RuntimeError, 'RETURN_AND_POSITION_OWNER'):
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
        self.runner.build(self.positions)
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
        self.runner.build(self.positions)
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
        self.runner.build(self.positions)
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

    def test_arming_without_actual_builder_never_installs_dispatch_mode(self):
        self.observer.begin(self.scope['request_ids'])
        self.assertIsNone(self.observer.mode)
        self.assertEqual(_get_current_dispatch_mode_stack(), [])
        self.observer.close()
        self.assertEqual(self.observer.report()['actual_calls'], 0)

    def test_rpc_arm_then_real_inference_scope_allows_next_request_tensor_factory(self):
        # Match the serving boundary: begin_capture RPC is outside the runner's
        # guard, while producer calls and model pre-hook are inside it. A fresh
        # subprocess isolates PyTorch TLS if the regression returns.
        import subprocess
        program = '''
import torch, numpy as np
from test_quant_mla_metadata_rows import EarlyMetadataTests
from torch.utils._python_dispatch import _get_current_dispatch_mode_stack
case=EarlyMetadataTests('test_actual_index_buffer_and_metadata_return_bind_without_reexecution')
case.setUp()
try:
 for attempt in range(2):
  with torch.inference_mode(False):
   case.observer.begin(case.scope['request_ids'])
   assert _get_current_dispatch_mode_stack() == []
   with torch.inference_mode():
    padded=case.padded_source()
    output=case.runner.build(padded)
    case.observer.close(scope=case.scope,actual_positions=case.positions)
    case.graph.begin('decode-'+str(attempt))
    assert case.observer.consumer(output,case.graph)['verified']
    case.graph.end()
   assert _get_current_dispatch_mode_stack() == []
   assert torch.equal(torch.from_numpy(np.array([1,2])),torch.tensor([1,2]))
finally:
 case.doCleanups()
'''
        result = subprocess.run([sys.executable, '-B', '-c', program], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
