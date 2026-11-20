"""CPU contracts for the existing capture's explicit live-borrow mode.

Only the Native query is an explicit fake. Real CPU call objects and the
existing Attention wrapper exercise entry, return, owner retention and retry.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from demoncat.runtime.vllm_ascend.worker_extension import attention28_prepared_input as P
from demoncat.runtime.vllm_ascend.worker_extension import attention28_replay_binding as B
from demoncat.runtime.vllm_ascend.worker_extension import attention28_typed_plan as T
from demoncat.runtime.vllm_ascend.worker_extension.attention_execution import bind_actual_api
import test_attention28_replay_binding as fixtures
import test_attention28_typed_plan as typed_fixtures


class BorrowCaptureContracts(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.ReplayInputContracts();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.f.model.attn.q=self.f.model.attn.q.contiguous();self.f.model.attn.k=self.f.model.attn.k.contiguous()
        self.root=Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.library_path=self.root/'libcust_opapi.so';self.library_path.write_bytes(b'CPU_FAKE_NOT_ELF')
        self.native=typed_fixtures.FakePlanLibrary()
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.paged33_loading.symbol_owner',
            return_value=dict(path=str(self.library_path.resolve()),sha256='b'*64,address=17)))
        self.enterContext(patch.object(P,'EVIDENCE_ROOT',self.root))
        self.enterContext(patch.object(P,'identity',side_effect=B.identity))
        self.env={};self.enterContext(patch.object(P.os,'environ',self.env))
        self.state=dict(schema='demoncat.tnd28.execute_borrow.v1',first_error=0,pid=B.identity()['pid'],
            tid=B.identity()['native_tid'],left=1,complete=1,drained=1,terminal_uncertain=0,fault_authorized=False)
        self.query_error=None;self.query_owner=None;self.query_calls=0
        def query(token):
            self.query_calls+=1
            if self.query_error:raise self.query_error
            source=self.capture.receipt
            value={k:source[k] for k in ('token','owner','generation','module_object_id','request_sha256','input_sha256')}
            if self.query_owner:value['owner']=self.query_owner
            value['borrow']=dict(self.state)
            return value
        query.library,query.path,query.sha256=self.native,self.library_path.resolve(),'b'*64
        self.enterContext(patch.object(P,'_open_borrow_query',return_value=query))
        header='a'*64
        build=dict(status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',variant='tnd_prefill_denominator_replay',
            tnd28_execute_borrow_query=dict(symbol='dcat_tnd28_execute_borrow_query_v1',library_sha256='b'*64,
                live_public_plan_symbols=list(T.SYMBOLS),
                read_only=True,fault_authorized=False),
            tnd28_execute_borrow_header_sha256=header,tnd28_compiled_input_observer=dict(
                schema='demoncat.tnd28.compiled_input_source.v1',observer_only=True,fault_authorized=False,
                execute_borrow_header_sha256=header))
        path=self.root/'completion.json';path.write_text(json.dumps(build))
        self.capture=P.PreparedInputCapture(self.f.worker,build_completion=path,
            build_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),evidence_dir=self.root/'capture',
            execute_borrow_library='/explicit-cpu-fake-library',execute_borrow_sha256='b'*64)
        def inside_original():
            if self.capture.live_plan is None:return
            typed=typed_fixtures.typed_packet(self.capture.live_plan,self.library_path)
            if self.native.emit(1,typed)!=1 or self.native.emit(2,typed)!=1:
                raise RuntimeError('EXPLICIT_CPU_TYPED_GUARD_REFUSED')
        self.f.during_original=inside_original
        self.addCleanup(self.cleanup_capture)

    def cleanup_capture(self):
        # Fixture-only restoration of a read-only query seam, never SDK retry.
        self.state.update(complete=1,drained=1,terminal_uncertain=0,first_error=0)
        self.query_error=self.query_owner=None
        if self.capture.active is not None:self.capture.stop()

    def enter(self,row):
        _,original,args,kwargs=self.f.session.live_api_calls[-1]
        self.capture.prepare_plan(row)
        return self.capture.enter('b'*32,row,bind_actual_api(args,kwargs),{'explicit_cpu_capture':True})

    def test_existing_natural_call_binds_plan_and_reads_actual_return_before_release(self):
        def callback(row,once):
            self.enter(row)
            self.assertEqual(self.env['DCAT28_INPUT_EXECUTE_BORROW'],'normal_guard')
            self.assertIs(self.capture.live_plan.actual['value'],self.f.model.attn.v)
            output=once();self.capture.returned(output);self.capture.stop()
            self.assertIsNone(self.capture.live_plan);self.assertIsNone(self.capture.refs)
            self.assertFalse(self.capture.receipt['execute_borrow']['borrow']['fault_authorized'])
            return output
        self.f.run_call(callback)
        self.assertEqual(self.env,{})
        self.assertEqual(self.query_calls,2)

    def test_drain_uncertainty_retains_owner_and_never_calls_raw_pointer_retry(self):
        def callback(row,once):
            self.enter(row);output=once();self.state.update(complete=0,drained=0,terminal_uncertain=1,first_error=8)
            with self.assertRaisesRegex(RuntimeError,'TERMINAL_UNCERTAIN'):self.capture.returned(output)
            with self.assertRaisesRegex(RuntimeError,'TERMINAL_UNCERTAIN'):self.capture.stop()
            self.assertIsNotNone(self.capture.live_plan);self.assertIsNotNone(self.capture.refs)
            self.assertTrue(self.capture.active);self.assertIn('TERMINAL_UNCERTAIN',self.capture.first_error)
            return output
        self.f.run_call(callback)

    def test_transient_live_query_error_can_retry_without_reexecuting_original(self):
        def callback(row,once):
            self.enter(row);output=once();self.capture.returned(output)
            self.query_error=OSError('query result unavailable')
            with self.assertRaises(OSError):self.capture.stop()
            self.query_error=None;self.capture.stop()
            self.assertIn('query result unavailable',self.capture.first_error)
            return output
        self.f.run_call(callback)
        self.assertEqual(self.f.calls,1)

    def test_foreign_native_owner_or_generation_cannot_release(self):
        def callback(row,once):
            self.enter(row);output=once();self.query_owner='different'
            with self.assertRaisesRegex(RuntimeError,'PLAN_RETURN_CHANGED'):self.capture.stop()
            self.assertIsNotNone(self.capture.refs);self.query_owner=None;self.capture.stop()
            return output
        self.f.run_call(callback)

    def test_failure_before_original_releases_local_plan_without_native_query(self):
        def callback(row,once):
            with patch.object(B,'capture_inputs',side_effect=MemoryError('partial plan read')):
                with self.assertRaises(MemoryError):self.enter(row)
            self.assertIsNotNone(self.capture.replay_binding.active)
            self.capture.stop()
            self.assertIsNone(self.capture.replay_binding.active)
            self.assertEqual(self.query_calls,0)
            return once()
        self.f.run_call(callback)

    def test_no_value_readback_is_inserted_inside_the_native_window(self):
        def callback(row,once):
            self.capture.prepare_plan(row)
            _,original,args,kwargs=self.f.session.live_api_calls[-1]
            # This seam marks the real StockTndPrefillCensus scope boundary:
            # prepare_plan was outside; enter/returned must perform no .cpu().
            with patch.object(B,'capture_inputs',side_effect=AssertionError('value read inside Native scope')):
                self.capture.enter('b'*32,row,bind_actual_api(args,kwargs),{'explicit_cpu_capture':True})
                output=once();self.capture.returned(output)
                self.assertFalse(self.capture.replay_binding.snapshot()['last_check_read_input_values'])
            self.capture.stop();return output
        self.f.run_call(callback)

    def test_typed_rejection_before_parent_allocation_cleans_without_execute_query(self):
        def callback(row,once):
            self.enter(row)
            def reject_inside():
                typed=typed_fixtures.typed_packet(self.capture.live_plan,self.library_path)
                typed['value']['members'][0]['offset']=0
                self.assertEqual(self.native.emit(1,typed),0)
                raise RuntimeError('CPU_NATIVE_PREALLOCATION_REFUSAL')
            self.f.during_original=reject_inside
            try:return once()
            finally:
                self.capture.stop()
                self.assertEqual(self.query_calls,0)
                self.assertIsNone(self.capture.refs)
                self.assertIsNone(self.capture.live_plan)
                self.assertIsNone(self.native.callback)
        with self.assertRaisesRegex(RuntimeError,'PREALLOCATION_REFUSAL'):self.f.run_call(callback)
        self.assertEqual(self.f.calls,1)
        self.assertEqual(self.env,{})

    def test_bind_reply_failure_keeps_actual_callback_until_owned_clear(self):
        def callback(row,once):
            original=self.native.bind
            def bind_then_fail(*args):
                original(*args)
                return -1
            self.native.__dict__[T.SYMBOLS[0]].call=bind_then_fail
            with self.assertRaisesRegex(RuntimeError,'TND_LIVE_TYPED_BIND_RETURN:-1'):self.enter(row)
            self.assertIsNotNone(self.native.callback)
            self.assertIsNotNone(self.capture.live_plan)
            self.capture.stop()
            self.assertIsNone(self.native.callback)
            self.assertEqual(self.query_calls,0)
            return once()
        self.f.run_call(callback)


if __name__=='__main__':unittest.main()
