"""Document host protocol tests; the real inner framework is unavailable."""
import unittest

from fault_injection.integration import (ContractCleanupError, DocumentContractAdapter,
                                        HOST_API, INTEGRATION_STATUS, integration_manifest)
from fault_injection.core import UnsupportedCall


class MockHost:
    api = HOST_API

    def __init__(self):
        self.catalog, self.bindings, self.calls = {}, {}, []
        self.fail_register_after = None
        self.fail_binding_remove = set()
        self.fail_catalog_remove = set()

    def catalog_register_mechanism(self, owner, definition):
        key = definition["mechanism"]
        if key in self.catalog:
            raise ValueError("CATALOG_OWNER_CONFLICT")
        self.catalog[key] = (owner, definition)
        self.calls.append(("add_catalog", key))
        if self.fail_register_after == ("catalog", key):
            raise RuntimeError("registration reply lost")

    def catalog_unregister_mechanism(self, owner, mechanism):
        if mechanism in self.fail_catalog_remove:
            raise RuntimeError("catalog remove unavailable")
        if mechanism in self.catalog and self.catalog[mechanism][0] != owner:
            raise ValueError("FOREIGN_CATALOG_OWNER")
        self.catalog.pop(mechanism, None)
        self.calls.append(("remove_catalog", mechanism))

    def binding_register(self, owner, definition, backend_factory):
        key = definition["binding"]
        if key in self.bindings:
            raise ValueError("BINDING_OWNER_CONFLICT")
        self.bindings[key] = (owner, definition, backend_factory)
        self.calls.append(("add_binding", key))
        if self.fail_register_after == ("binding", key):
            raise RuntimeError("binding reply lost")

    def binding_unregister(self, owner, binding):
        if binding in self.fail_binding_remove:
            raise RuntimeError("binding remove unavailable")
        if binding in self.bindings and self.bindings[binding][0] != owner:
            raise ValueError("FOREIGN_BINDING_OWNER")
        self.bindings.pop(binding, None)
        self.calls.append(("remove_binding", binding))


class HostContractTests(unittest.TestCase):
    def test_manifest_has_fifteen_distinct_variants_and_keeps_blockers(self):
        manifest = integration_manifest()
        self.assertEqual(manifest["mechanism_count"], 15)
        self.assertNotIn("primitive_count", manifest)
        self.assertEqual(manifest["status"], INTEGRATION_STATUS)
        rows = {row["mechanism"]: row for row in manifest["mechanisms"]}
        self.assertEqual(len(rows), 15)
        self.assertFalse(any(x.startswith("39.") for x in rows))
        for name in ("8.restore_offset", "22.wrong_row", "25.reduce_scatter",
                     "28.fa.legal_tiling", "41.tail_column"):
            self.assertFalse(rows[name]["model_session_route_available"])
        for name in ("20.buffer_substitute", "26.residual_stomp", "26.fia_destination",
                     "33.paged_decode", "quant.batch_shared_scale"):
            self.assertTrue(rows[name]["deployment_runtime_required"])
        self.assertTrue(all(row["target"] == "auto" for row in rows.values()))
        self.assertTrue(all(row["runtime_acceptance_inherited"] is False for row in rows.values()))

    def test_explicit_protocol_and_owner_required(self):
        with self.assertRaises(ValueError):
            DocumentContractAdapter(object())
        for owner in ("", " ", 3, False):
            with self.subTest(owner=owner), self.assertRaises(ValueError):
                DocumentContractAdapter(MockHost(), owner)
        incomplete = type("Incomplete", (), {"api": HOST_API})()
        with self.assertRaises(TypeError):
            DocumentContractAdapter(incomplete)

    def test_install_and_idempotent_uninstall_preserve_other_owner(self):
        host = MockHost()
        host.catalog["foreign"] = ("other", {})
        host.bindings["foreign.binding"] = ("other", {}, object())
        adapter = DocumentContractAdapter(host)
        adapter.install()
        self.assertTrue(adapter.query()["installed"])
        with self.assertRaisesRegex(RuntimeError, "ALREADY_INSTALLED"):
            adapter.install()
        adapter.uninstall()
        self.assertTrue(adapter.uninstall()["contract_adapter_removed"])
        self.assertEqual(set(host.catalog), {"foreign"})
        self.assertEqual(set(host.bindings), {"foreign.binding"})
        self.assertFalse(adapter.query()["inner_framework_integration_verified"])

    def test_registration_failure_after_application_is_rolled_back(self):
        for kind, key in (("catalog", "15.alias_detach"),
                          ("binding", "model_fault.15.alias_detach")):
            with self.subTest(kind=kind):
                host = MockHost(); host.fail_register_after = (kind, key)
                adapter = DocumentContractAdapter(host)
                with self.assertRaisesRegex(RuntimeError, "reply lost"):
                    adapter.install()
                self.assertEqual(host.catalog, {})
                self.assertEqual(host.bindings, {})
                self.assertFalse(adapter.query()["pending_host_entries"])
                host.fail_register_after = None
                adapter.install(); adapter.uninstall()

    def test_failed_rollback_retains_binding_dependency_and_retries(self):
        host = MockHost()
        key = "model_fault.15.alias_detach"
        host.fail_register_after = ("binding", key)
        host.fail_binding_remove.add(key)
        adapter = DocumentContractAdapter(host)
        with self.assertRaises(ContractCleanupError):
            adapter.install()
        self.assertEqual(adapter.query()["state"], "ROLLBACK_PENDING")
        self.assertEqual(set(host.bindings), {key})
        self.assertEqual(set(host.catalog), {"15.alias_detach"})
        with self.assertRaisesRegex(RuntimeError, "CLEANUP_PENDING"):
            adapter.install()
        with self.assertRaisesRegex(RuntimeError, "BINDING_NOT_ACTIVE"):
            host.bindings[key][2](object())
        host.fail_binding_remove.clear()
        adapter.uninstall()
        self.assertEqual(host.bindings, {})
        self.assertEqual(host.catalog, {})

    def test_uninstall_partial_catalog_failure_is_retryable(self):
        host = MockHost(); adapter = DocumentContractAdapter(host)
        adapter.install()
        host.fail_catalog_remove.add("17.spatial_padding")
        with self.assertRaises(ContractCleanupError):
            adapter.uninstall()
        self.assertEqual(host.bindings, {})
        self.assertEqual(set(host.catalog), {"17.spatial_padding"})
        self.assertEqual(adapter.query()["state"], "UNINSTALL_PENDING")
        host.fail_catalog_remove.clear()
        adapter.uninstall()
        self.assertEqual(host.catalog, {})

    def test_lost_unregister_reply_keeps_intent_until_idempotent_retry(self):
        host = MockHost(); adapter = DocumentContractAdapter(host)
        adapter.install()
        original = host.binding_unregister
        key = "model_fault.17.spatial_padding"
        def lost_reply(owner, binding):
            original(owner, binding)
            if binding == key:
                raise RuntimeError("remove reply lost after application")
        host.binding_unregister = lost_reply
        with self.assertRaises(ContractCleanupError):
            adapter.uninstall()
        self.assertEqual(host.bindings, {})
        self.assertEqual(set(host.catalog), {"17.spatial_padding"})
        self.assertTrue(any(e["name"] == key for e in adapter.query()["pending_host_entries"]))
        host.binding_unregister = original
        adapter.uninstall()
        self.assertEqual(host.catalog, {})

    def test_foreign_collision_never_deletes_foreign_entry(self):
        host = MockHost()
        definition = {"mechanism": "8.restore_offset"}
        host.catalog["8.restore_offset"] = ("other", definition)
        adapter = DocumentContractAdapter(host, "ours")
        with self.assertRaises(ContractCleanupError):
            adapter.install()
        self.assertEqual(host.catalog["8.restore_offset"], ("other", definition))
        with self.assertRaises(ContractCleanupError):
            adapter.uninstall()
        # Only the real owner removes its entry. Our retry is then idempotent.
        host.catalog_unregister_mechanism("other", "8.restore_offset")
        adapter.uninstall()

    def test_bound_backend_requires_clean_and_cannot_switch_mechanism(self):
        import torch
        from fault_injection.lifecycle import FaultConfig
        host = MockHost(); adapter = DocumentContractAdapter(host)
        adapter.install()
        create = host.bindings["model_fault.18.api_half"][2]
        model = torch.nn.RMSNorm(8).eval()
        x = torch.ones(1, 8); expected = model(x)
        backend = create(model); backend.prepare()
        self.assertEqual(adapter.query()["created_backend_count"], 1)
        with self.assertRaisesRegex(ValueError, "MECHANISM_MISMATCH"):
            backend.register(FaultConfig("17.spatial_padding"))
        backend.register(FaultConfig("18.api_half"))
        self.assertTrue(torch.equal(model(x), expected))
        with self.assertRaisesRegex(RuntimeError, "CLEAN_REQUIRED"):
            adapter.uninstall()
        backend.enable()
        try:
            self.assertTrue(torch.equal(model(x), expected * 0.5))
            with self.assertRaisesRegex(RuntimeError, "CLEAN_REQUIRED"):
                adapter.uninstall()
        finally:
            backend.clean()
        self.assertTrue(torch.equal(model(x), expected))
        adapter.uninstall()
        self.assertEqual(adapter.query()["created_backend_count"], 0)
        for operation in (lambda: create(model), backend.prepare,
                          lambda: backend.register(FaultConfig("18.api_half")), backend.enable):
            with self.assertRaisesRegex(RuntimeError, "BINDING_NOT_ACTIVE"):
                operation()
        adapter.install()
        with self.assertRaisesRegex(RuntimeError, "BINDING_NOT_ACTIVE"):
            create(model)
        adapter.uninstall()

    def test_blocked_binding_is_discoverable_without_installing_hooks(self):
        import torch
        from fault_injection.lifecycle import FaultConfig
        host = MockHost(); adapter = DocumentContractAdapter(host)
        adapter.install()
        model = torch.nn.Identity()
        backend = host.bindings["model_fault.8.restore_offset"][2](model)
        backend.prepare()
        # Some lifecycle implementations reject at register, others at enable;
        # in either case this binding cannot bypass the existing unsupported route.
        try:
            with self.assertRaises(UnsupportedCall):
                backend.register(FaultConfig("8.restore_offset"))
                backend.enable()
            self.assertFalse(model._forward_hooks)
            self.assertFalse(model._forward_pre_hooks)
        finally:
            backend.clean()
            adapter.uninstall()

    def test_unknown_cleanup_state_prevents_any_host_removal(self):
        import torch
        host = MockHost(); adapter = DocumentContractAdapter(host)
        adapter.install()
        backend = host.bindings["model_fault.18.api_half"][2](torch.nn.RMSNorm(8).eval())
        original = backend.query
        backend.query = lambda: {"registered": False, "cleanup_confirmed": None}
        before = len(host.calls)
        with self.assertRaisesRegex(RuntimeError, "CLEAN_REQUIRED"):
            adapter.uninstall()
        self.assertEqual(len(host.calls), before)
        self.assertTrue(adapter.query()["installed"])
        backend.query = original
        adapter.uninstall()

    def test_cleaned_model_references_retained_until_successful_host_drain(self):
        import gc
        import weakref
        import torch
        host = MockHost(); adapter = DocumentContractAdapter(host)
        adapter.install()
        model = torch.nn.RMSNorm(8).eval()
        reference = weakref.ref(model)
        backend = host.bindings["model_fault.18.api_half"][2](model)
        backend.clean()
        del model, backend
        host.fail_catalog_remove.add("18.api_half")
        with self.assertRaises(ContractCleanupError):
            adapter.uninstall()
        gc.collect()
        self.assertIsNotNone(reference())
        self.assertEqual(adapter.query()["created_backend_count"], 1)
        host.fail_catalog_remove.clear()
        adapter.uninstall()
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(adapter.query()["created_backend_count"], 0)


if __name__ == "__main__":
    unittest.main()
