"""Dependency-free dispatch/lifecycle checks; these are not native NPU tests."""
import types
import threading
import unittest
from unittest.mock import patch

from fault_injection.attention import AttentionSession, inspect_model
from fault_injection.core import Decision, UnsupportedCall


class Registry:
    _global_mapping = {}

    @classmethod
    def register(cls, key, value):
        cls._global_mapping[key] = value

    def get_interface(self, name, default):
        return self._global_mapping.get(name, default)

    def __getitem__(self, name):
        return self._global_mapping[name]


class Masks:
    _global_mapping = {"eager": lambda *a, **kw: (a, kw), "sdpa": lambda *a, **kw: (a, kw)}

    @classmethod
    def register(cls, name, value):
        cls._global_mapping[name] = value


REGISTRY = Registry()


def eager(module, q, k, v, mask, **kwargs):
    module.received = (q, k, v, mask, kwargs)
    return "original", None


class AttentionBlock:
    def __init__(self, config):
        self.config = config

    def forward(self, q, k, v, mask, **kwargs):
        selected = REGISTRY.get_interface(self.config._attn_implementation, eager)
        return selected(self, q, k, v, mask, **kwargs)


class LegacyAttention:
    def __init__(self, config):
        self.config = config

    def forward(self, q, k, v, mask, **kwargs):
        selected = eager
        if self.config._attn_implementation != "eager":
            selected = REGISTRY[self.config._attn_implementation]
        return selected(self, q, k, v, mask, **kwargs)


class BypassAttention:
    def __init__(self, config):
        self.config = config

    def forward(self, q, k, v, mask):
        return eager(self, q, k, v, mask)


class Model:
    def __init__(self, modules):
        self.modules = modules
        self.pre_hooks, self.post_hooks = {}, {}
        self.selected_indices = None
        self.before = None
        self.fail_after = False

    def register_forward_pre_hook(self, hook, prepend=False):
        key = object()
        self.pre_hooks[key] = hook
        if prepend:
            self.pre_hooks = {key: hook, **self.pre_hooks}
        return types.SimpleNamespace(remove=lambda: self.pre_hooks.pop(key, None))

    def register_forward_hook(self, hook, always_call=False):
        key = object()
        self.post_hooks[key] = (hook, always_call)
        return types.SimpleNamespace(remove=lambda: self.post_hooks.pop(key, None))

    def __call__(self, *args, **kwargs):
        output, called = None, set()
        try:
            for hook in tuple(self.pre_hooks.values()):
                hook(self, args)
            output = self.forward(*args, **kwargs)
            for key, (hook, _) in tuple(self.post_hooks.items()):
                called.add(key)
                hook(self, args, output)
            return output
        except BaseException:
            for key, (hook, always_call) in tuple(self.post_hooks.items()):
                if always_call and key not in called:
                    hook(self, args, output)
            raise

    def forward(self, *args, **kwargs):
        if self.before is not None:
            self.before()
        indices = range(len(self.modules)) if self.selected_indices is None else self.selected_indices
        output = [self.modules[i][1].forward(*args, **kwargs) for i in indices]
        if self.fail_after:
            raise ValueError("root failed after attention")
        return output

    def named_modules(self):
        yield "", self
        yield from self.modules


class Tensor:
    def __init__(self, shape):
        self.shape = shape


class Backend:
    def __init__(self, supported=True, fail=False):
        self.calls = []
        self.supported, self.fail = supported, fail

    def check(self, call):
        return Decision(supported=self.supported, reason="fixture capability")

    def run(self, call):
        self.calls.append(call)
        if self.fail:
            raise ValueError("backend failed")
        return "routed", None


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.patch = patch("fault_injection.attention._interfaces", return_value=(Registry, Masks))
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.old_attention = dict(Registry._global_mapping)
        self.old_masks = dict(Masks._global_mapping)
        self.config = types.SimpleNamespace(_attn_implementation="eager")
        self.q, self.k, self.v = Tensor((3, 6, 7, 16)), Tensor((3, 2, 9, 16)), Tensor((3, 2, 9, 11))
        self.mask = object()

    def tearDown(self):
        self.assertEqual(Registry._global_mapping, self.old_attention)
        self.assertEqual(Masks._global_mapping, self.old_masks)

    def test_structural_discovery_and_actual_dynamic_geometry(self):
        blocks = [AttentionBlock(self.config), LegacyAttention(self.config)]
        model = Model([("decoder.unusual.attn", blocks[0]), ("nested.other", blocks[1])])
        discovery = inspect_model(model)
        self.assertEqual(len(discovery["candidates"]), 2)
        self.assertEqual(discovery["actual_calls_observed"], 0)
        backend = Backend()
        with AttentionSession(model, backend) as session:
            self.assertFalse(session.report()["all_discovered_candidates_routed"])
            self.assertIs(Masks._global_mapping[self.config._attn_implementation], self.old_masks["eager"])
            self.assertEqual(model(self.q, self.k, self.v, self.mask, scaling=.3, dropout=0, cache_position="real"),
                             [("routed", None)] * 2)
            report = session.assert_coverage()
            self.assertEqual(report["modules"]["nested.other"]["request_slot_executions"], 3)
            call = backend.calls[0]
            self.assertIs(call.query, self.q)
            self.assertIs(call.attention_mask, self.mask)
            self.assertEqual(call.scaling, .3)
            self.assertEqual(call.kwargs, {"cache_position": "real"})
            self.assertEqual(call.value.shape[-1], 11)
        self.assertEqual(self.config._attn_implementation, "eager")

    def test_missing_layers_do_not_count_as_covered(self):
        first, second = AttentionBlock(self.config), AttentionBlock(self.config)
        model = Model([("x", first), ("y", second)])
        model.selected_indices = [0]
        with AttentionSession(model, Backend()) as session:
            model(self.q, self.k, self.v, self.mask)
            with self.assertRaises(UnsupportedCall):
                session.assert_coverage()
            self.assertEqual(session.report()["modules"]["y"]["observed_calls"], 0)

    def test_unsupported_strict_and_passthrough_are_distinct(self):
        block = AttentionBlock(self.config)
        model = Model([("x", block)])
        with AttentionSession(model, Backend(False)) as session:
            with self.assertRaises(UnsupportedCall):
                model(self.q, self.k, self.v, self.mask)
            self.assertEqual(session.report()["events"][0]["status"], "unsupported")
        with AttentionSession(model, Backend(False), unsupported="passthrough") as session:
            self.assertEqual(model(self.q, self.k, self.v, self.mask, scaling=.4, dropout=.2, cache="x"), [("original", None)])
            self.assertIs(block.received[3], self.mask)
            self.assertEqual(block.received[4], {"scaling": .4, "dropout": .2, "cache": "x"})
            self.assertEqual(session.report()["events"][0]["status"], "passthrough")
            self.assertFalse(session.report()["all_discovered_candidates_routed"])

    def test_runtime_exception_restores_configuration(self):
        block = AttentionBlock(self.config)
        model = Model([("a", block)])
        with self.assertRaises(ValueError):
            with AttentionSession(model, Backend(fail=True)) as session:
                model(self.q, self.k, self.v, self.mask)
        self.assertEqual(self.config._attn_implementation, "eager")
        self.assertEqual(session.report()["events"][0]["status"], "failed")
        self.assertEqual(model.pre_hooks, {})
        self.assertEqual(model.post_hooks, {})

    def test_unidentified_shared_config_model_is_not_injected(self):
        selected, foreign = AttentionBlock(self.config), AttentionBlock(self.config)
        backend = Backend()
        with AttentionSession(Model([("selected", selected)]), backend):
            with self.assertRaises(UnsupportedCall):
                foreign.forward(self.q, self.k, self.v, self.mask)
            self.assertEqual(backend.calls, [])

    def test_shared_child_outside_selected_root_uses_original_backend(self):
        block = AttentionBlock(self.config)
        model, foreign = Model([("selected", block)]), Model([("shared", block)])
        backend = Backend()
        with AttentionSession(model, backend) as session:
            self.assertEqual(foreign(self.q, self.k, self.v, self.mask), [("original", None)])
            self.assertEqual(model.forward(self.q, self.k, self.v, self.mask), [("original", None)])
            self.assertEqual(session.report()["events"], [])
            with self.assertRaises(UnsupportedCall):
                session.assert_coverage()
            self.assertEqual(model(self.q, self.k, self.v, self.mask), [("routed", None)])
            self.assertEqual(foreign(self.q, self.k, self.v, self.mask), [("original", None)])
            self.assertEqual(len(backend.calls), 1)
            self.assertEqual(len(session.assert_coverage()["forwards"]), 1)

    def test_later_forward_cannot_reuse_earlier_layer_coverage(self):
        model = Model([("a", AttentionBlock(self.config)), ("b", AttentionBlock(self.config))])
        with AttentionSession(model, Backend()) as session:
            model(self.q, self.k, self.v, self.mask)
            session.assert_coverage()
            model.selected_indices = [0]
            model(self.q, self.k, self.v, self.mask)
            report = session.report()
            self.assertTrue(report["forwards"][0]["all_discovered_candidates_routed"])
            self.assertFalse(report["forwards"][1]["all_discovered_candidates_routed"])
            with self.assertRaises(UnsupportedCall):
                session.assert_coverage()

    def test_root_failure_after_routing_is_not_covered(self):
        model = Model([("a", AttentionBlock(self.config))])
        with AttentionSession(model, Backend()) as session:
            model(self.q, self.k, self.v, self.mask)
            model.fail_after = True
            with self.assertRaisesRegex(ValueError, "root failed"):
                model(self.q, self.k, self.v, self.mask)
            report = session.report()
            self.assertEqual(report["modules"]["a"]["routed_calls"], 2)
            self.assertEqual(report["forwards"][1]["status"], "no_output_or_exception")
            with self.assertRaises(UnsupportedCall):
                session.assert_coverage()

    def test_earlier_posthook_failure_with_output_is_not_covered(self):
        model = Model([("a", AttentionBlock(self.config))])

        def fail(module, args, output):
            raise ValueError("posthook failed")

        handle = model.register_forward_hook(fail)
        with AttentionSession(model, Backend()) as session:
            with self.assertRaisesRegex(ValueError, "posthook failed"):
                model(self.q, self.k, self.v, self.mask)
            self.assertEqual(session.report()["forwards"][0]["status"], "no_output_or_exception")
            with self.assertRaises(UnsupportedCall):
                session.assert_coverage()
        handle.remove()

    def test_existing_caught_exception_does_not_mark_successful_forward_failed(self):
        model = Model([("a", AttentionBlock(self.config))])
        try:
            raise ValueError("unrelated caught exception")
        except ValueError:
            with AttentionSession(model, Backend()) as session:
                model(self.q, self.k, self.v, self.mask)
                session.assert_coverage()

    def test_caught_recursive_root_rejection_preserves_outer_scope(self):
        model = Model([("a", AttentionBlock(self.config))])

        def recurse_once():
            model.before = None
            with self.assertRaisesRegex(RuntimeError, "Recursive"):
                model(self.q, self.k, self.v, self.mask)

        model.before = recurse_once
        with AttentionSession(model, Backend()) as session:
            self.assertEqual(model(self.q, self.k, self.v, self.mask), [("routed", None)])
            report = session.report()
            self.assertEqual(len(report["forwards"]), 1)
            self.assertTrue(report["forwards"][0]["all_discovered_candidates_routed"])
            self.assertEqual(report["rejected_root_calls"][0]["reason"], "recursive_root_forward")
            self.assertFalse(report["all_discovered_candidates_routed"])

    def test_other_thread_root_is_rejected_and_does_not_reuse_coverage(self):
        model = Model([("a", AttentionBlock(self.config))])
        errors = []

        def worker():
            try:
                model(self.q, self.k, self.v, self.mask)
            except Exception as error:
                errors.append(error)

        with AttentionSession(model, Backend()) as session:
            model(self.q, self.k, self.v, self.mask)
            thread = threading.Thread(target=worker)
            thread.start()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], RuntimeError)
            self.assertEqual(len(session.report()["events"]), 1)
            with self.assertRaises(UnsupportedCall):
                session.assert_coverage()

    def test_bypassed_registry_is_explicitly_rejected(self):
        model = Model([("custom", BypassAttention(self.config))])
        report = inspect_model(model)
        self.assertEqual(report["candidates"], [])
        self.assertEqual(report["unsupported_modules"][0]["path"], "custom")
        with self.assertRaises(UnsupportedCall):
            AttentionSession(model, Backend()).__enter__()

    def test_unresolved_shared_config_is_rejected_even_for_passthrough(self):
        model = Model([("good", AttentionBlock(self.config)), ("custom", BypassAttention(self.config))])
        with self.assertRaises(UnsupportedCall):
            AttentionSession(model, Backend(), unsupported="passthrough").__enter__()
        self.assertEqual(self.config._attn_implementation, "eager")

    def test_unsupported_backend_preflight_is_atomic(self):
        other = types.SimpleNamespace(_attn_implementation="flash_attention_2")
        model = Model([("a", AttentionBlock(self.config)), ("b", AttentionBlock(other))])
        with self.assertRaises(UnsupportedCall):
            AttentionSession(model, Backend()).__enter__()
        self.assertEqual(self.config._attn_implementation, "eager")
        self.assertEqual(other._attn_implementation, "flash_attention_2")

    def test_nested_sessions_fail_before_mutation(self):
        block = AttentionBlock(self.config)
        model = Model([("a", block)])
        with AttentionSession(model, Backend()):
            route = self.config._attn_implementation
            with self.assertRaises(RuntimeError):
                AttentionSession(model, Backend()).__enter__()
            self.assertEqual(self.config._attn_implementation, route)

    def test_inactive_route_refuses_execution(self):
        block = AttentionBlock(self.config)
        session = AttentionSession(Model([("a", block)]), Backend())
        with session:
            wrapper = Registry._global_mapping[self.config._attn_implementation]
        with self.assertRaises(RuntimeError):
            wrapper(block, self.q, self.k, self.v, self.mask)

    def test_registration_failure_rolls_back_existing_configs(self):
        other = types.SimpleNamespace(_attn_implementation="eager")
        model = Model([("a", AttentionBlock(self.config)), ("b", AttentionBlock(other))])
        original_register = Masks.register
        attempts = []

        def fail_second(name, value):
            attempts.append(name)
            if len(attempts) == 2:
                raise ValueError("mask registration failed")
            original_register(name, value)

        with patch.object(Masks, "register", side_effect=fail_second):
            with self.assertRaisesRegex(ValueError, "mask registration failed"):
                AttentionSession(model, Backend()).__enter__()
        self.assertEqual(self.config._attn_implementation, "eager")
        self.assertEqual(other._attn_implementation, "eager")
        self.assertEqual(model.pre_hooks, {})
        self.assertEqual(model.post_hooks, {})


if __name__ == "__main__":
    unittest.main()
