"""Real CPU torch operations exercise automatic wiring, not NPU acceptance."""
import importlib.util
import threading
import unittest

from fault_injection.alias import AliasSession
from fault_injection.core import UnsupportedCall

HAS_TORCH = importlib.util.find_spec('torch') is not None


class SimulatedFormat:
    simulation = True

    def __init__(self):
        self.converted = set()

    def format(self, tensor):
        return 2 if tensor.untyped_storage()._cdata in self.converted else 29

    def convert(self, tensor):
        result = tensor.clone()
        self.converted.add(result.untyped_storage()._cdata)
        return result

    def synchronize(self, tensor):
        pass


@unittest.skipUnless(HAS_TORCH, 'CPU torch required; no alias integration executed')
class AliasAutoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)

    def model(self, *, size=(8,8), write=True, consume=True, fail=False, count=1):
        torch = self.torch

        class UnnamedOperation(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('accumulator', torch.arange(256, dtype=torch.float32).reshape(16,16).to(torch.bfloat16))

            def forward(self):
                result = None
                for _ in range(count):
                    view = torch.as_strided(self.accumulator, size, (16,1), 17)
                    if write:
                        view.copy_(torch.full(size, -4., dtype=torch.bfloat16))
                    if fail:
                        raise RuntimeError('model failure after view')
                    result = self.accumulator + 1 if consume else view
                return result

        class ArbitraryModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.unusual_name = UnnamedOperation()

            def forward(self):
                return self.unusual_name()

        return ArbitraryModel()

    def session(self, model, **kwargs):
        return AliasSession(model, _format_provider=SimulatedFormat(), **kwargs)

    def test_automatic_live_base_binding_and_write_loss(self):
        torch = self.torch
        baseline, fault = self.model(), self.model()
        original = torch.as_strided
        with torch.no_grad(), self.session(baseline, arm='baseline') as b:
            bout = baseline()
            b.assert_coverage()
        with torch.no_grad(), self.session(fault) as f:
            fout = fault()
            f.assert_coverage()
        self.assertIs(torch.as_strided, original)
        self.assertEqual(b.report()['events'][0]['path'], 'unusual_name')
        self.assertFalse(torch.equal(bout, fout))
        self.assertTrue(torch.equal(bout[1:9,1:9], torch.full((8,8),-3.,dtype=torch.bfloat16)))
        self.assertTrue(torch.equal(fout, torch.arange(256,dtype=torch.float32).reshape(16,16).to(torch.bfloat16)+1))
        bw = b.report()['events'][0]['writes'][0]
        fw = f.report()['events'][0]['writes'][0]
        self.assertTrue(bw['view_write_correct'] and bw['base_matches_expected_update'])
        self.assertTrue(fw['view_write_correct'] and fw['base_target_retains_prewrite_values'])
        self.assertFalse(fw['base_matches_expected_update'])
        self.assertTrue(fw['non_target_unchanged'])
        self.assertTrue(f.report()['original_base_consumer_observed'])
        self.assertFalse(f.report()['npu_validation_claimed'])
        self.assertFalse(f.report()['model_acceptance_claimed'])

    def test_every_matching_call_and_forward_triggers(self):
        model = self.model(count=3)
        with self.torch.no_grad(), self.session(model) as session:
            model(); model()
        report = session.report()
        self.assertEqual(report['routed_calls'], 6)
        self.assertEqual([e['forward'] for e in report['events']], [0,0,0,1,1,1])
        self.assertTrue(all(len(e['writes']) == 1 for e in report['events']))

    def test_missing_write_or_consumer_is_not_model_acceptance(self):
        for write, consume in ((False,True),(True,False)):
            model = self.model(write=write,consume=consume)
            with self.torch.no_grad(), self.session(model) as session: model()
            report = session.report()
            self.assertFalse(report['model_acceptance_claimed'])
            self.assertEqual(report['write_observed'],write)
            self.assertFalse(report['original_base_consumer_observed'])

    def test_default_provider_never_detaches_cpu_tensors(self):
        model = self.model()
        with self.torch.no_grad(), AliasSession(model) as session:
            model()
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
        self.assertEqual(session.report()['routed_calls'],0)

    def test_independent_storage_is_required(self):
        class IdentityFormat(SimulatedFormat):
            def convert(self,tensor):return tensor
        model = self.model()
        with self.torch.no_grad(), AliasSession(model,_format_provider=IdentityFormat()) as session:
            with self.assertRaises(UnsupportedCall):model()
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
        self.assertEqual(session.report()['events'][0]['status'],'failed')

    def test_unsupported_private_geometry_is_explicit(self):
        model = self.model(size=(4,4))
        with self.torch.no_grad(), self.session(model) as session:
            with self.assertRaises(UnsupportedCall):model()
        with self.torch.no_grad(), self.session(model,unsupported='passthrough') as session:
            model()
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
        self.assertEqual(session.report()['events'][0]['status'],'passthrough')

    def test_other_model_outside_root_is_not_injected(self):
        model, other = self.model(), self.model()
        with self.torch.no_grad(), self.session(model) as session:
            other()
            model.unusual_name()
        self.assertEqual(session.report()['events'],[])

    def test_exception_restores_callable_and_hooks(self):
        model = self.model(fail=True)
        original = self.torch.as_strided
        with self.assertRaisesRegex(RuntimeError,'model failure'):
            with self.torch.no_grad(),self.session(model) as session:model()
        self.assertIs(self.torch.as_strided,original)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for _,m in model.named_modules()))

    def test_nested_session_rejected_without_stealing_outer(self):
        model = self.model()
        with self.torch.no_grad(),self.session(model) as outer:
            installed = self.torch.as_strided
            with self.assertRaises(RuntimeError):
                with self.session(self.model()):pass
            self.assertIs(self.torch.as_strided,installed)
            model()
            outer.assert_coverage()

    def test_failed_copy_observation_cannot_leave_coverage_pass(self):
        model = self.model(write=False, consume=False)
        torch = self.torch
        def broken_write():
            view = torch.as_strided(model.unusual_name.accumulator, (8,8), (16,1), 17)
            view.copy_(torch.ones(3,3,dtype=torch.bfloat16))
            return view
        model.unusual_name.forward = broken_write
        with torch.no_grad(), self.session(model) as session:
            with self.assertRaises(RuntimeError):model()
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
        event = session.report()['events'][0]
        self.assertEqual(event['status'],'failed')
        self.assertTrue(event['observation_errors'])
        self.assertFalse(session.report()['model_acceptance_claimed'])

    def test_metadata_factory_and_overwrite_are_not_data_consumers(self):
        torch = self.torch
        for overwrite in (False, True):
            model = self.model(write=False,consume=False)
            def only_write_and_metadata():
                base = model.unusual_name.accumulator
                view = torch.as_strided(base,(8,8),(16,1),17)
                view.copy_(torch.full((8,8),-4.,dtype=torch.bfloat16))
                return base.zero_() if overwrite else torch.empty_like(base)
            model.unusual_name.forward = only_write_and_metadata
            with torch.no_grad(),self.session(model) as session:model()
            self.assertFalse(session.report()['original_base_consumer_observed'])
            self.assertGreater(session.report()['unclassified_base_uses'],0)

    def test_cleanup_failure_restores_other_resources_and_retains_lease_for_retry(self):
        model = self.model()
        original = self.torch.as_strided
        session = self.session(model)
        session.__enter__()
        real_handle = session._hooks[0]
        class RaisingHandle:
            first = True
            def remove(self):
                real_handle.remove()
                if self.first:
                    self.first = False
                    raise RuntimeError('hook cleanup fixture')
        session._hooks[0] = RaisingHandle()
        with self.assertRaisesRegex(RuntimeError,'hook cleanup fixture'):
            session.close()
        self.assertIs(self.torch.as_strided,original)
        self.assertTrue(all(not m._forward_hooks and not m._forward_pre_hooks for _,m in model.named_modules()))
        self.assertEqual(len(session.report()['cleanup_errors']),1)
        self.assertFalse(session._closed)
        with self.assertRaisesRegex(RuntimeError,'owns the public API'):
            self.session(self.model()).__enter__()
        session.close()
        self.assertTrue(session._closed)
        with self.session(self.model()):pass

    def test_recursive_rejection_preserves_outer_scope(self):
        model = self.model()
        def recursive_forward(nested=False):
            if not nested:
                try:model(nested=True)
                except UnsupportedCall:pass
            return model.unusual_name()
        model.forward = recursive_forward
        with self.torch.no_grad(), self.session(model) as session:
            model()
            report = session.report()
            self.assertEqual(report['routed_calls'],1)
            self.assertEqual(report['forwards'],[dict(index=0,status='returned')])
            self.assertEqual(len(report['rejected_root_calls']),1)
            self.assertTrue(report['events'][0]['writes'][0]['base_target_retains_prewrite_values'])
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
            self.assertFalse(session._tls.stack)

    def test_wrong_thread_root_is_rejected_without_mutating_owner_scope(self):
        model = self.model()
        errors = []
        def other_thread():
            try:model()
            except BaseException as exc:errors.append(exc)
        with self.torch.no_grad(), self.session(model) as session:
            thread = threading.Thread(target=other_thread)
            thread.start();thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors),1)
            self.assertIsInstance(errors[0],UnsupportedCall)
            self.assertEqual(session.report()['events'],[])
            model()
            self.assertEqual(session.report()['routed_calls'],1)
            with self.assertRaises(UnsupportedCall):session.assert_coverage()
        self.assertFalse(session.report()['npu_validation_claimed'])

    def test_raising_forward_hook_invalidates_routed_call_coverage(self):
        model = self.model()
        def fail_after_output(module,args,output):
            raise ValueError('post output hook failure')
        hook = model.register_forward_hook(fail_after_output)
        try:
            with self.torch.no_grad(),self.session(model) as session:
                with self.assertRaisesRegex(ValueError,'post output hook failure'):model()
                self.assertEqual(session.report()['routed_calls'],1)
                self.assertEqual(session.report()['forwards'][0]['status'],'no_output_or_exception')
                with self.assertRaises(UnsupportedCall):session.assert_coverage()
        finally:hook.remove()

    def test_existing_exception_context_is_not_a_failed_forward(self):
        model = self.model()
        with self.torch.no_grad(),self.session(model) as session:
            try:raise ValueError('already handled')
            except ValueError:
                model()
                session.assert_coverage()
        self.assertEqual(session.report()['forwards'][0]['status'],'returned')

    def test_existing_pre_hook_failure_does_not_reuse_previous_coverage(self):
        model=self.model();fail=[False]
        def earlier_hook(module,args):
            if fail[0]:raise ValueError('earlier pre hook')
        handle=model.register_forward_pre_hook(earlier_hook)
        try:
            with self.torch.no_grad(),self.session(model) as session:
                model();session.assert_coverage();fail[0]=True
                with self.assertRaisesRegex(ValueError,'earlier pre hook'):model()
                self.assertEqual([f['status'] for f in session.report()['forwards']],['returned','no_output_or_exception'])
                with self.assertRaises(UnsupportedCall):session.assert_coverage()
        finally:handle.remove()


if __name__=='__main__':unittest.main()
