"""Real CPU modules with explicit Native-v2 mocks; never Native qualification."""
import ctypes as C
from pathlib import Path
import tempfile
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.writer26_group import Config, Call, Lease, Receipt, State, RuntimeState, MODES, packet, key_words
from fault_injection.deployments.writer26_experiment import Writer26Experiment, classify_candidate
from demoncat.runtime.vllm_ascend.worker_extension.serving import complete_tensor
from test_writer26_inventory import CpuReader, CpuWriterModel


class CpuV2:
    mechanism = '26.residual_stomp'
    fia = False
    sha256 = 'b' * 64
    build_identity = 'c' * 64

    def __init__(self, reader):
        self.reader = reader
        self.config = Config(version=2, size=C.sizeof(Config), mechanism=1, owner_hi=1, owner_lo=2,
                             generation=1, thread=threading.get_native_id())
        self.admit = self.finished = self.transaction = None
        self.writer_calls = self.sequence = 0
        self.last = self.current_output = None
        self.installed = self.sealed = False
        self.passes, self.executes, self.receipts = [], [], {}
        self.skip_callback = False

    def state(self):
        core = packet(State(version=2, size=C.sizeof(State), installed=int(self.installed),
            pending=int(self.transaction is not None), sealed=int(self.sealed)))
        value = packet(RuntimeState(version=2, size=C.sizeof(RuntimeState)))
        value['core'] = core
        return value

    def install(self, admit, finished):
        self.admit, self.finished, self.installed = admit, finished, True
        self.sealed = False

    def passthrough(self, call):
        assert call.token not in self.passes
        self.passes.append(call.token)

    def execute(self, call, mode, target, allocation):
        assert self.transaction is None
        self.executes.append((call.token, mode, target))
        self.transaction = dict(call=Call.from_buffer_copy(call), mode=mode, target=target,
                                residual_before=self.last.clone())
        return dict(target=key_words(target), call=packet(call))

    def acknowledge(self, token):
        assert self.transaction['call'].token == token
        self.receipts[token]['released'] = 1
        self.transaction = None
        return dict(self.receipts[token])

    def seal(self, targets):
        assert len(targets) == 2
        self.sealed = True

    def close(self):
        assert self.transaction is None
        self.installed = False
        self.admit = self.finished = None

    def writer(self, x):
        self.writer_calls += 1
        output = x.clone()
        self.current_output = output
        self.reader.raw.all_runs += 1
        if self.fia:
            self.reader.raw.non_target_runs += 1
        else:
            self.reader.raw.qmm_runs += 1
        event = self.reader.append(3 if self.fia else 1, output=output, residual=self.last, producer=self.sequence,
            cache=0 if self.fia else 100 + self.writer_calls % 2, registration=0 if self.fia else 1, dtype=27, format=2)
        event.flags |= 0 if self.fia else 4
        call = Call(version=2, size=C.sizeof(Call), mechanism=2 if self.fia else 1, dimensions=2, device=0, dtype=27, format=2,
            token=event.call, thread=threading.get_native_id(), independent_call=event.call,
            registration=event.registration, cache=event.cache, producer_sequence=self.sequence, output=output.data_ptr(),
            output_bytes=output.untyped_storage().nbytes(), residual=self.last.data_ptr(),
            residual_bytes=self.last.untyped_storage().nbytes(), shape=(C.c_int64 * 4)(*output.shape, 0, 0))
        if self.admit is not None and not self.skip_callback:
            self.admit(C.pointer(call))
        if self.transaction is not None and self.transaction['mode'] == 'fault':
            self.last.view(-1)[-16:] += 1
        for kind in ((4, 5, 6) if self.fia else (2,)):
            self.reader.append(kind, output=output, residual=self.last, producer=self.sequence,
                call=event.call, cache=event.cache, registration=1, dtype=27, format=2)
        return output

    def producer(self, x, residual):
        self.sequence += 1
        output = x + residual
        for kind in (7, 8, 9):
            self.reader.append(kind, output=output, residual=residual, producer=self.sequence,
                               call=self.sequence, dtype=27, format=2)
        if self.transaction:
            tx = self.transaction
            residual.copy_(tx['residual_before'])
            call = tx['call']
            receipt = Receipt(version=2, size=C.sizeof(Receipt), mode=MODES[tx['mode']], mechanism=call.mechanism,
                owner_hi=1, owner_lo=2, generation=self.config.generation, token=call.token, target=(C.c_uint64 * 4)(*key_words(tx['target'])),
                writer_calls=1, consumer_calls=1, writer_complete=1, consumer_complete=1, output_readback_exact=1,
                padding_readback_exact=1, residual_restored_exact=1, changed_residual_bytes=32,
                independent_call=call.independent_call, registration=1, cache=call.cache,
                producer_sequence=call.producer_sequence, output=call.output, output_bytes=call.output_bytes,
                residual=call.residual, residual_bytes=call.residual_bytes, stream=123, executor=456,
                replacement=999 if tx['mode'] == 'lossless' else call.output, reference_equal=1, lossless_qualified=1)
            self.receipts[call.token] = packet(receipt)
            assert self.finished(C.pointer(receipt)) == 0
        self.last = output
        return output * .5, torch.ones(1), output


class WriterV2Tests(unittest.TestCase):
    def test_exact_ctypes_abi(self):
        self.assertEqual([C.sizeof(t) for t in (Config, Call, Lease, Receipt, State, RuntimeState)], [72, 144, 248, 296, 136, 240])
        with self.assertRaisesRegex(RuntimeError, 'FULL_TARGET_SHA256_REQUIRED'):
            key_words('f' * 63)

    def run_experiment(self, stage, *, skip=False, budget=8, erase_output=False, fia=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        reader = CpuReader()
        native = CpuV2(reader)
        if fia:
            native.mechanism, native.fia, native.config.mechanism = '26.fia_destination', True, 2
        native.skip_callback = skip
        model = CpuWriterModel(native).eval()
        experiment = Writer26Experiment(model, native, reader, worker_identity=dict(rank=0),
            folder=Path(temporary.name) / 'experiment', stage=stage, max_targets=budget)
        # Explicit synthetic allocator fixture, only the independent CPU flow test.
        def allocation(call, row):
            row['allocation'] = dict(residual_block=[call.residual, call.output],
                                     output_block=[call.output, call.output + call.output_bytes], shift=32)
        original_post = experiment._post
        def post(module, args, kwargs, output):
            original_post(module, args, kwargs, output)
            if erase_output:
                experiment.output_refs.clear()
                for row in experiment.callbacks:
                    row.pop('writer_output_retained', None)
        with patch.object(torch.ops._C_ascend, 'npu_add_rms_norm_bias', native.producer, create=True), \
                patch.object(experiment, '_allocation', allocation), patch.object(experiment, '_post', post), \
                patch.dict(sys.modules, {'torch_npu': NS(get_npu_format=lambda t: 2)}):
            experiment.install()
            with torch.inference_mode():
                for index in range(40):
                    if experiment.phase == 'complete':
                        break
                    experiment.begin('request-' + str(index), 'e' * 64)
                    output = model(torch.ones(4, 16, dtype=torch.bfloat16))
                    _, digest, finite = complete_tensor(output)
                    experiment.end(digest, model_output_finite=finite)
            self.assertEqual(experiment.phase, 'complete')
            experiment.close()
        return experiment, native

    def test_two_targets_all_candidate_effects_and_same_process_health(self):
        experiment, native = self.run_experiment('effects')
        self.assertEqual(len(experiment.targets), 2)
        self.assertEqual([mode for _, mode, _ in native.executes],
                         ['reference', 'lossless', 'off'] * 2 + ['fault', 'off'] * 2)
        self.assertEqual(len(experiment.rows), 13)
        self.assertTrue(experiment.recovered)
        self.assertFalse(experiment.report()['c2'])
        self.assertEqual(experiment.report()['qualified_candidate_count'], 2)
        self.assertTrue(all(len(row['candidates']) == 2 for row in experiment.rows))
        self.assertEqual(experiment.rows[-1]['phase'], 'health')
        self.assertFalse(native.state()['core']['installed'])
        self.assertFalse(experiment.output_refs or experiment.retained or experiment.handles)

    def test_missing_native_callbacks_stay_in_denominator(self):
        experiment, native = self.run_experiment('effects', skip=True)
        self.assertEqual(len(experiment.candidate_inventory), 2)
        self.assertTrue(all(row['status'] == 'UNVERIFIED' for row in experiment.candidate_inventory))
        self.assertFalse(native.executes)
        self.assertIn('EMPTY_OR_OVER_BUDGET_COMPLETE_RESOLVED_TARGET_SET', experiment.blockers)

    def test_fia_candidate_lifetimes_use_actual_gws_execution_consumer(self):
        experiment, native = self.run_experiment('effects', fia=True)
        self.assertTrue(experiment.recovered)
        self.assertEqual(len(experiment.targets), 2)
        self.assertTrue(all(row['native']['kind_name'] == 'fia_before' for row in experiment.rows[0]['candidates']))
        self.assertEqual(len(native.executes), 10)

    def test_missing_live_writer_output_never_qualifies(self):
        experiment, native = self.run_experiment('effects', erase_output=True)
        self.assertFalse(experiment.targets or native.executes)
        self.assertTrue(all('PUBLIC_WRITER_OUTPUT_NOT_RETAINED' in row['reasons'] for row in experiment.candidate_inventory))

    def test_budget_does_not_choose_first_target(self):
        experiment, native = self.run_experiment('effects', budget=1)
        self.assertEqual(len(experiment.targets), 2)
        self.assertFalse(native.executes)
        self.assertEqual(len(experiment.rows), 3)

    def test_missing_residual_is_distinct_from_missing_public_lineage(self):
        native = dict(residual=0, residual_bytes=0, producer=0, flags=5)
        self.assertEqual(classify_candidate(dict(native=native, callback=None, reasons=['NO_CALLBACK']))[0], 'UNSUPPORTED')
        native['producer'] = 1
        self.assertEqual(classify_candidate(dict(native=native, callback=None, reasons=['NO_CALLBACK']))[0], 'UNVERIFIED')


if __name__ == '__main__':
    unittest.main()
