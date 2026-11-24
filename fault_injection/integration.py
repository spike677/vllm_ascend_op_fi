"""Versioned host wiring contract for model-scoped mechanism adapters.

The current internal ``src/demoncat`` is unavailable. These are deliberately
named document-protocol methods, not claims about that framework's API. Host
unregister operations must be idempotent for absent entries and must refuse a
different owner. Registration can fail after applying: intent is recorded
before calling the host so rollback can reconcile an uncertain reply.
"""
from __future__ import annotations

from copy import deepcopy
import threading
from typing import Protocol
import uuid

from .catalog import CAPABILITIES, DEPLOYMENT_RUNTIME_ROUTES, MODEL_SESSION_ROUTES
from .capabilities import describe_mechanism

HOST_API = "demoncat.model_faults.host_contract.v1"
INTEGRATION_STATUS = "INNER_FRAMEWORK_INTEGRATION_PENDING"


class HostContract(Protocol):
    api: str

    def catalog_register_mechanism(self, owner: str, definition: dict): ...
    def catalog_unregister_mechanism(self, owner: str, mechanism: str): ...
    def binding_register(self, owner: str, definition: dict, backend_factory): ...
    def binding_unregister(self, owner: str, binding: str): ...


def integration_manifest():
    """Describe variants, without inventing fifteen inner-framework primitives."""
    mechanisms = []
    for capability in CAPABILITIES:
        if capability.name.startswith("39.") or capability.integration == 'continuous_product_only':
            continue
        mechanisms.append({
            "mechanism": capability.name,
            "binding": "model_fault." + capability.name,
            "target": "auto",
            "capability": describe_mechanism(capability.name),
            "model_session_route_available": capability.integration in MODEL_SESSION_ROUTES,
            "deployment_runtime_required": capability.integration in DEPLOYMENT_RUNTIME_ROUTES,
            "arming_is_conditional": True,
            "runtime_acceptance_inherited": False,
            "status": INTEGRATION_STATUS,
        })
    return {
        "api": HOST_API,
        "schema_version": 1,
        "status": INTEGRATION_STATUS,
        "mechanism_count": len(mechanisms),
        "mechanisms": mechanisms,
        "fault39": "Separate demoncat_fault39 package and host contract; unchanged",
        "backend": "fault_injection.lifecycle.ModelFaultBackend",
        "lifecycle": ["prepare", "register", "enable", "disable", "query", "clean", "unregister"],
        "registration_starts_disabled": True,
        "clean_scope": "own model session only; no reload_weights or global fault reset",
        "requires_host_implementation": ["catalog", "binding_registry", "controller", "worker_dispatch"],
        "host_unregister_contract": "Idempotent if absent; reject foreign ownership; retain uncertain operations for retry",
        "evidence_scope": "Host protocol and routing only, not native launch, msprobe first difference or model acceptance",
    }


class ContractCleanupError(RuntimeError):
    """Host entries remain owned/pending; retry uninstall after resolving errors."""

    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__("HOST_CLEANUP_PENDING: " + "; ".join(errors))


class DocumentContractAdapter:
    """Owner-scoped registry transaction; it never silently drains a model.

Created backends are retained until their registration and cleanup states are
confirmed. A controller must clean/unregister them before removing bindings.
Failed host removal leaves a retryable intent ledger. Old factory references
and backend objects cannot arm after their binding generation is removed.
"""

    def __init__(self, host: HostContract, owner=None):
        if getattr(host, "api", None) != HOST_API:
            raise ValueError("EXPLICIT_HOST_CONTRACT_REQUIRED")
        for name in ("catalog_register_mechanism", "catalog_unregister_mechanism",
                     "binding_register", "binding_unregister"):
            if not callable(getattr(host, name, None)):
                raise TypeError("HOST_METHOD_REQUIRED: " + name)
        owner = "model-fault-contract-" + uuid.uuid4().hex if owner is None else owner
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("NONEMPTY_OWNER_REQUIRED")
        self.host, self.owner = host, owner
        self._lock = threading.RLock()
        self._ledger, self._backends = [], []
        self._generation = 0
        self._state = "NEW"
        self.installed = False

    def _check_binding(self, generation, binding):
        if (not self.installed or self._state != "INSTALLED"
                or generation != self._generation
                or not any(x["kind"] == "binding" and x["name"] == binding for x in self._ledger)):
            raise RuntimeError("BINDING_NOT_ACTIVE")

    def _factory(self, definition, generation):
        adapter, mechanism, binding = self, definition["mechanism"], definition["binding"]

        def create(model, runtime=None):
            from .lifecycle import ModelFaultBackend

            class BoundModelFaultBackend(ModelFaultBackend):
                def prepare(self, *args, **kwargs):
                    with adapter._lock:
                        adapter._check_binding(generation, binding)
                        return super().prepare(*args, **kwargs)

                def register(self, config):
                    with adapter._lock:
                        adapter._check_binding(generation, binding)
                        selected = config.get("mechanism") if isinstance(config, dict) else getattr(config, "mechanism", None)
                        if selected != mechanism:
                            raise ValueError("BINDING_MECHANISM_MISMATCH")
                        return super().register(config)

                def enable(self, *args, **kwargs):
                    with adapter._lock:
                        adapter._check_binding(generation, binding)
                        return super().enable(*args, **kwargs)

            with adapter._lock:
                adapter._check_binding(generation, binding)
                backend = BoundModelFaultBackend(model, runtime=runtime)
                adapter._backends.append(backend)
                return backend

        return create

    def install(self):
        with self._lock:
            if self.installed:
                raise RuntimeError("ADAPTER_ALREADY_INSTALLED")
            if self._ledger:
                raise RuntimeError("HOST_CLEANUP_PENDING")
            self._generation += 1
            self._state = "INSTALLING"
            manifest = integration_manifest()
            try:
                for definition in manifest["mechanisms"]:
                    self._ledger.append({"kind": "catalog", "name": definition["mechanism"],
                                         "binding": definition["binding"]})
                    self.host.catalog_register_mechanism(self.owner, deepcopy(definition))
                    self._ledger.append({"kind": "binding", "name": definition["binding"],
                                         "mechanism": definition["mechanism"]})
                    self.host.binding_register(self.owner, deepcopy(definition),
                                               self._factory(definition, self._generation))
            except BaseException as error:
                self.installed = False
                errors = self._drain()
                self._state = "ROLLBACK_PENDING" if self._ledger else "NEW"
                if errors:
                    raise ContractCleanupError(["install: " + repr(error)] + errors) from error
                raise
            self.installed, self._state = True, "INSTALLED"
            return manifest

    def _drain(self):
        errors, blocked_bindings = [], set()
        for entry in list(reversed(self._ledger)):
            if entry["kind"] == "catalog" and entry["binding"] in blocked_bindings:
                continue
            try:
                if entry["kind"] == "binding":
                    self.host.binding_unregister(self.owner, entry["name"])
                else:
                    self.host.catalog_unregister_mechanism(self.owner, entry["name"])
            except BaseException as error:
                errors.append(entry["kind"] + " " + entry["name"] + ": " + repr(error))
                if entry["kind"] == "binding":
                    blocked_bindings.add(entry["name"])
            else:
                self._ledger.remove(entry)
        return errors

    def uninstall(self):
        with self._lock:
            # Query failure is deliberately fail-closed. No registry entry is
            # removed until every created backend confirms its own release.
            for backend in self._backends:
                state = backend.query()
                if state.get("registered") is not False or state.get("cleanup_confirmed") is not True:
                    raise RuntimeError("OWNED_BACKEND_CLEAN_REQUIRED")
            self.installed, self._state = False, "UNINSTALL_PENDING"
            errors = self._drain()
            if errors:
                raise ContractCleanupError(errors)
            # All owned registrations and cleanup are confirmed. Do not keep
            # models alive across a later host-adapter installation merely to
            # retain old factory objects; generation checks already revoke them.
            self._backends.clear()
            self._state = "REMOVED"
            return self.query()

    def query(self):
        with self._lock:
            return {"status": INTEGRATION_STATUS, "state": self._state,
                    "owner": self.owner, "generation": self._generation,
                    "installed": self.installed, "pending_host_entries": [dict(x) for x in self._ledger],
                    "created_backend_count": len(self._backends),
                    "contract_adapter_removed": self._state == "REMOVED" and not self._ledger,
                    "inner_framework_integration_verified": False}
