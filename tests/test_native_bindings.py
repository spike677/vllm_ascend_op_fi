"""Native-provider plumbing tests on CPU, not a substitute for NPU proof."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from fault_injection.core import Decision, UnsupportedCall
from fault_injection.native import (NativeCall, NativeProof, NativeSession, REQUIREMENTS,
                                    inspect_native, validate_proofs)

HAS_TORCH = importlib.util.find_spec('torch') is not None


class FixtureRuntime:
    validation_scope = 'CPU_TEST_FIXTURE_NOT_NATIVE_INJECTION'

    def __init__(self, directory):
        self.directory = Path(directory)
        self.callback = None
        self.counter = self.original_calls = self.aborts = 0
        self.supported = True
        self.omit = None
        self.execute_twice = self.finish_error = self.abort_error = False
        self.last_call = None
        self.skip_callback = False

    def install(self, callback, candidates, mechanism, arm):
        self.callback, self.mechanism, self.arm = callback, mechanism, arm
        self.paths = [x[0] for x in candidates]
        def remove():
            self.callback = None
        return remove

    def check(self, call, mechanism, arm):
        return Decision(self.supported, 'CPU fixture capability')

    def execute(self, call, invoke_once, mechanism, arm):
        out = invoke_once()
        if self.execute_twice:
            invoke_once()
        return out

    def finish(self, call, mechanism, arm):
        if self.finish_error:
            raise RuntimeError('fixture observer failed')
        return dict(call_id=call.call_id, native_launch_verified=False, fixture=True)

    def abort(self, call, mechanism, arm):
        self.aborts += 1
        if self.abort_error:
            raise RuntimeError('fixture cleanup failed')

    def invoke(self, module, original, *, operator='aclnnMatmul'):
        def execute():
            self.original_calls += 1
            return original()
        if self.callback is None or self.skip_callback:
            return execute()
        self.counter += 1
        call_id = f'fixture-{self.counter}'
        proofs = []
        for kind in REQUIREMENTS[self.mechanism]:
            if kind == self.omit:
                continue
            receipt = dict(schema='demoncat.native.proof.v1', call_id=call_id,
                           owner_id=id(module), pid=os.getpid(), kind=kind,
                           mechanism=self.mechanism, operator=operator, status='verified',
                           details={'fixture_only': True})
            path = self.directory / (call_id + '-' + kind + '.json')
            data = json.dumps(receipt).encode()
            path.write_bytes(data)
            proofs.append(NativeProof(kind, str(path), hashlib.sha256(data).hexdigest()))
        call = NativeCall(module, call_id, operator, proofs=tuple(proofs))
        self.last_call = call
        return self.callback(call, execute)


@unittest.skipUnless(HAS_TORCH, 'CPU torch required; native bridge tests not executed')
class NativeBindingTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.runtime = FixtureRuntime(self.temp.name)
        runtime = self.runtime
        class Projection(torch.nn.Linear):
            def forward(self, x):
                return runtime.invoke(self, lambda: torch.nn.functional.linear(x, self.weight, self.bias))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unusual_tree = torch.nn.ModuleList([Projection(4,4), Projection(4,4)])
            def forward(self, x):
                for module in self.unusual_tree:
                    x = module(x)
                return x
        self.model = Model().eval()
        self.x = torch.randn(2,4)

    def test_missing_provider_rejects_before_hooks_or_model_execution(self):
        with self.assertRaises(UnsupportedCall):
            with NativeSession(self.model, '20.buffer_substitute'):
                pass
        self.assertEqual(self.runtime.original_calls, 0)
        self.assertTrue(all(not module._forward_hooks for module in self.model.modules()))

    def test_actual_object_discovery_and_every_call_coverage(self):
        names = [r['path'] for r in inspect_native(self.model)['candidates']]
        self.assertEqual(names, ['unusual_tree.0', 'unusual_tree.1'])
        baseline = self.model(self.x)
        with NativeSession(self.model, '20.buffer_substitute', self.runtime) as session:
            self.assertTrue(self.torch.equal(self.model(self.x),baseline))
            self.assertTrue(self.torch.equal(self.model(self.x),baseline))
            report = session.assert_coverage()
            self.assertEqual(len(report['events']),4)
            self.assertEqual([f['index'] for f in report['forwards']], [0, 1])
            self.assertTrue(all(f['all_discovered_candidates_covered'] for f in report['forwards']))
            self.assertTrue(all(row['actual_original_calls']==1 for row in report['events']))
            self.assertFalse(report['native_fault_hit_proven'])
            self.assertFalse(report['npu_validation_claimed'])
        self.assertIsNone(self.runtime.callback)
        self.assertTrue(self.torch.equal(self.model(self.x),baseline))

    def test_other_model_and_outside_forward_are_not_injected(self):
        module = self.torch.nn.Linear(4,4)
        with NativeSession(self.model,'39.output_proxy',self.runtime) as session:
            self.runtime.invoke(module,lambda: self.x)
            self.runtime.invoke(self.model.unusual_tree[0],lambda:self.x)
            self.assertEqual(session.report()['events'],[])
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_shared_submodule_called_without_owner_model_is_not_injected(self):
        other = self.torch.nn.Sequential(self.model.unusual_tree[0])
        with NativeSession(self.model,'39.output_proxy',self.runtime) as session:
            other(self.x)
            self.assertEqual(session.report()['events'],[])
            self.model(self.x)
            self.assertEqual(len(session.report()['events']),2)

    def test_missing_proof_fails_before_native_execution(self):
        self.runtime.omit = 'live_storage'
        with NativeSession(self.model,'39.matmul_range',self.runtime) as session:
            with self.assertRaisesRegex(UnsupportedCall,'live_storage'):
                self.model(self.x)
            self.assertEqual(self.runtime.original_calls,0)
            self.assertEqual(session.report()['events'][0]['status'],'failed')

    def test_proof_hash_and_call_identity_and_owner_are_bound(self):
        with NativeSession(self.model,'39.matmul_range',self.runtime):self.model(self.x)
        call=self.runtime.last_call
        proof=call.proofs[0]
        path=Path(proof.path)
        receipt=json.loads(path.read_bytes());receipt['owner_id']+=1
        raw=json.dumps(receipt).encode();path.write_bytes(raw)
        with self.assertRaisesRegex(UnsupportedCall,'identity changed'):validate_proofs(call,'39.matmul_range')
        revised=tuple(NativeProof(p.kind,p.path,hashlib.sha256(raw).hexdigest()) if p is proof else p for p in call.proofs)
        changed=NativeCall(call.owner,call.call_id,call.operator,proofs=revised)
        with self.assertRaisesRegex(UnsupportedCall,'stale, mismatched'):validate_proofs(changed,'39.matmul_range')

    def test_operator_mechanisms_cannot_be_interchanged(self):
        with NativeSession(self.model,'39.output_proxy',self.runtime):self.model(self.x)
        call=self.runtime.last_call
        with self.assertRaises(UnsupportedCall):validate_proofs(call,'33.paged_decode')
        self.assertNotEqual(REQUIREMENTS['26.residual_stomp'],REQUIREMENTS['26.fia_destination'])

    def test_passthrough_is_explicit_and_not_coverage(self):
        self.runtime.supported=False
        with NativeSession(self.model,'39.output_proxy',self.runtime,unsupported='passthrough') as session:
            self.model(self.x)
            self.assertTrue(all(e['status']=='passthrough' for e in session.report()['events']))
            with self.assertRaises(UnsupportedCall):session.assert_coverage()

    def test_no_retry_after_native_failure_and_cleanup_is_attempted(self):
        self.runtime.execute_twice=True
        with self.assertRaises(UnsupportedCall):
            with NativeSession(self.model,'39.output_proxy',self.runtime) as session:self.model(self.x)
        self.assertEqual(self.runtime.original_calls,1)
        self.assertEqual(self.runtime.aborts,1)
        self.assertIsNone(self.runtime.callback)
        self.assertEqual(session.report()['events'][0]['status'],'failed')

    def test_observer_and_cleanup_failure_are_both_retained(self):
        self.runtime.finish_error=self.runtime.abort_error=True
        with self.assertRaisesRegex(RuntimeError,'scoped cleanup failed'):
            with NativeSession(self.model,'39.output_proxy',self.runtime) as session:self.model(self.x)
        event=session.report()['events'][0]
        self.assertIn('observer failed',event['reason'])
        self.assertIn('cleanup failed',event['cleanup_error'])
        self.assertIsNotNone(self.runtime.callback)
        self.assertTrue(session.report()['cleanup_pending'])
        with self.assertRaisesRegex(UnsupportedCall,'cleanup is pending'):
            self.model(self.x)
        self.runtime.abort_error=False
        session.close()
        self.assertIsNone(self.runtime.callback)
        self.assertTrue(session.report()['events'][0]['abort_recovery_completed'])
        self.assertIn('cleanup failed',session.report()['events'][0]['cleanup_error'])

    def test_no_native_callback_is_never_a_pass(self):
        with NativeSession(self.model,'39.output_proxy',self.runtime) as session:
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
            self.assertEqual(len(session.report()['unobserved_candidates']),2)

    def test_later_dynamic_branch_cannot_inherit_previous_coverage(self):
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            self.model(self.x)
            session.assert_coverage()
            self.model.forward = lambda x: self.model.unusual_tree[0](x)
            self.model(self.x)
            report = session.report()
            self.assertTrue(report['forwards'][0]['all_discovered_candidates_covered'])
            self.assertEqual(report['forwards'][1]['unexecuted_candidates'], ['unusual_tree.1'])
            self.assertFalse(report['all_discovered_candidates_covered'])
            with self.assertRaises(UnsupportedCall): session.assert_coverage()

    def test_each_candidate_invocation_requires_native_observation(self):
        original = self.model.forward
        def partial(x):
            x = original(x)
            self.runtime.skip_callback = True
            try:
                return self.model.unusual_tree[0](x)
            finally:
                self.runtime.skip_callback = False
        self.model.forward = partial
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            self.model(self.x)
            report = session.report()
            self.assertEqual(len(report['events']), 2)
            self.assertEqual(len(report['module_invocations']), 3)
            self.assertEqual(report['module_invocations'][-1]['observed_calls'], 0)
            self.assertEqual(report['forwards'][0]['unobserved_candidates'], ['unusual_tree.0'])
            with self.assertRaises(UnsupportedCall): session.assert_coverage()

    def test_root_failure_after_routed_calls_is_incomplete_and_scope_recovers(self):
        original = self.model.forward
        def failing(x):
            original(x)
            raise RuntimeError('later model failure')
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            self.model.forward = failing
            with self.assertRaisesRegex(RuntimeError, 'later model failure'): self.model(self.x)
            self.model.forward = original
            self.model(self.x)
            report = session.report()
            self.assertEqual(report['forwards'][0]['status'], 'no_output_or_exception')
            self.assertTrue(report['forwards'][1]['all_discovered_candidates_covered'])
            self.assertEqual([e['forward'] for e in report['events']], [0, 0, 1, 1])
            with self.assertRaises(UnsupportedCall): session.assert_coverage()

    def test_recursive_root_rejection_does_not_pop_outer_scope(self):
        original = self.model.forward
        def recursive(x):
            with self.assertRaisesRegex(UnsupportedCall, 'recursive'): self.model(x)
            return original(x)
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            self.model.forward = recursive
            self.model(self.x)
            self.model.forward = original
            self.model(self.x)
            report = session.report()
            self.assertEqual(len(report['forwards']), 2)
            self.assertTrue(all(f['all_discovered_candidates_covered'] for f in report['forwards']))
            self.assertEqual(report['rejected_root_calls'][0]['reason'], 'Recursive root call')
            with self.assertRaises(UnsupportedCall): session.assert_coverage()

    def test_root_candidate_recursive_rejection_preserves_outer_invocation(self):
        model = self.model.unusual_tree[0]
        original = model.forward
        def recursive(x):
            with self.assertRaises(UnsupportedCall): model(x)
            return original(x)
        model.forward = recursive
        with NativeSession(model, '39.output_proxy', self.runtime) as session:
            model(self.x)
            report = session.report()
            self.assertEqual(len(report['events']), 1)
            self.assertEqual(report['module_invocations'][0]['status'], 'returned')
            self.assertTrue(report['forwards'][0]['all_discovered_candidates_covered'])
            self.assertFalse(report['all_discovered_candidates_covered'])

    def test_foreign_thread_root_is_rejected_before_provider_execution(self):
        original = self.model.forward
        errors = []
        def foreign():
            try: self.model(self.x)
            except BaseException as exc: errors.append(exc)
        def concurrent(x):
            thread = threading.Thread(target=foreign)
            thread.start(); thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            return original(x)
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            self.model.forward = concurrent
            self.model(self.x)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], UnsupportedCall)
            self.assertEqual(self.runtime.original_calls, 2)
            self.assertTrue(session.report()['forwards'][0]['all_discovered_candidates_covered'])
            self.assertEqual(len(session.report()['rejected_root_calls']), 1)
            with self.assertRaises(UnsupportedCall): session.assert_coverage()

    def test_earlier_local_pre_hook_failure_is_closed_by_always_call(self):
        def failing(module, args): raise RuntimeError('pre hook failure')
        handle = self.model.register_forward_pre_hook(failing)
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            with self.assertRaisesRegex(RuntimeError, 'pre hook failure'): self.model(self.x)
            handle.remove()
            self.model(self.x)
            report = session.report()
            self.assertEqual(report['forwards'][0]['status'], 'no_output_or_exception')
            self.assertTrue(report['forwards'][1]['all_discovered_candidates_covered'])
            with self.assertRaises(UnsupportedCall): session.assert_coverage()

    def test_forward_hook_failure_with_nonempty_output_is_not_success(self):
        def failing(module, args, output): raise RuntimeError('post hook failure')
        handle = self.model.register_forward_hook(failing)
        self.addCleanup(handle.remove)
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            with self.assertRaisesRegex(RuntimeError, 'post hook failure'): self.model(self.x)
            report = session.report()
            self.assertEqual(len(report['events']), 2)
            self.assertEqual(report['forwards'][0]['status'], 'no_output_or_exception')
            self.assertIn('post hook failure', report['forwards'][0]['errors'][0])
            with self.assertRaises(UnsupportedCall): session.assert_coverage()

    def test_forward_in_existing_exception_handler_is_not_misclassified(self):
        with NativeSession(self.model, '39.output_proxy', self.runtime) as session:
            try:
                raise ValueError('unrelated caught error')
            except ValueError:
                self.model(self.x)
            session.assert_coverage()

    def test_all_six_bridge_contracts_route_and_reject_missing_mechanism_proof(self):
        # These are CPU ownership/protocol fixtures, never native executions.
        torch = self.torch
        specifications = [
            ('20.buffer_substitute', 'linear', 'aclnnMatmul'),
            ('26.residual_stomp', 'quantized', 'aclnnQuantMatmulWeightNz'),
            ('26.fia_destination', 'attention', 'aclnnFusedInferAttentionScore'),
            ('33.paged_decode', 'attention', 'aclnnIncreFlashAttention'),
            ('39.matmul_range', 'linear', 'aclnnMatmul'),
            ('39.output_proxy', 'linear', 'aclnnMatmul'),
        ]
        for fault, structure, operator in specifications:
            for missing in (None, REQUIREMENTS[fault][-1]):
                with self.subTest(fault=fault, missing=missing):
                    directory = Path(self.temp.name) / (fault + '-' + str(missing))
                    directory.mkdir()
                    runtime = FixtureRuntime(directory)
                    runtime.omit = missing
                    class Linear(torch.nn.Linear):
                        def forward(self, x):
                            return runtime.invoke(self, lambda: torch.nn.functional.linear(
                                x, self.weight, self.bias), operator=operator)
                    class QuantMethod:
                        def apply(self, owner, x):
                            return runtime.invoke(owner, lambda: x + 1, operator=operator)
                    class Quantized(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.quant_method = QuantMethod()
                        def forward(self, x):
                            return self.quant_method.apply(self, x)
                    class NativeImpl:
                        def forward(self, owner, x):
                            return runtime.invoke(owner, lambda: x + 1, operator=operator)
                    class NativeAttention(torch.nn.Module):
                        def __init__(self):
                            super().__init__()
                            self.impl = NativeImpl()
                            self.num_heads, self.num_kv_heads, self.head_size = 2, 1, 4
                        def forward(self, x):
                            return self.impl.forward(self, x)
                    candidate = (Linear(4, 4) if structure == 'linear' else
                                 Quantized() if structure == 'quantized' else NativeAttention())
                    model = torch.nn.Sequential(candidate).eval()
                    with NativeSession(model, fault, runtime) as session:
                        if missing is not None:
                            with self.assertRaisesRegex(UnsupportedCall, missing): model(self.x)
                            self.assertEqual(runtime.original_calls, 0)
                            self.assertEqual(session.report()['events'][0]['status'], 'failed')
                            with self.assertRaises(UnsupportedCall): session.assert_coverage()
                        else:
                            model(self.x)
                            report = session.assert_coverage()
                            event = report['events'][0]
                            self.assertEqual(event['operator'], operator)
                            self.assertEqual(event['status'], 'routed')
                            self.assertEqual(event['actual_original_calls'], 1)
                            self.assertEqual(event['provider_scope'], 'CPU_TEST_FIXTURE_NOT_NATIVE_INJECTION')
                            self.assertEqual({p['kind'] for p in event['proofs']}, set(REQUIREMENTS[fault]))
                            self.assertFalse(report['native_fault_hit_proven'])
                            self.assertFalse(report['npu_validation_claimed'])


if __name__=='__main__':unittest.main()
