"""Document-only host adapter; internal src/demoncat is unavailable here.

These method names describe a deliberately versioned test protocol. They are not
claims about signatures in the inaccessible inner framework. A host integration
must implement this protocol and separately validate real lifecycle wiring.
"""
from typing import Protocol
from .backend import NativeBackend
from .schema import INTEGRATION_STATUS, PATTERNS, PRIMITIVE

HOST_API = "demoncat_fault39.host_contract.v1"


class HostContract(Protocol):
    api: str
    def catalog_register_primitive(self, owner: str, definition: dict): ...
    def catalog_unregister_primitive(self, owner: str, primitive: str): ...
    def binding_register(self, owner: str, definition: dict, backend_factory): ...
    def binding_unregister(self, owner: str, binding: str): ...


def integration_manifest():
    return {"api": HOST_API, "schema_version": 1, "status": INTEGRATION_STATUS,
            "fault_id": 39, "primitive": PRIMITIVE, "primitive_count": 1,
            "mechanism": "finite_bf16_input_value_transformation",
            "patterns": list(PATTERNS), "target": "auto",
            "binding": "fault39.native.matmul", "backend": "demoncat_fault39.backend.NativeBackend",
            "requires_host_implementation": ["catalog", "binding_registry", "controller", "worker_extension"],
            "lifecycle": ["prepare", "register", "enable", "disable", "query", "unregister", "clean"],
            "clean_scope": "fault39_owner_only", "output_proxy": False,
            "native_interposition_required": True, "runtime_acceptance_inherited": False}


class DocumentContractAdapter:
    def __init__(self, host: HostContract, owner="fault39.contract.adapter", *, qualified=False):
        if getattr(host, "api", None) != HOST_API:
            raise ValueError("EXPLICIT_HOST_CONTRACT_REQUIRED")
        self.host, self.owner = host, owner
        self.installed = False
        self._qualified = qualified
        self._generation = 0
        self._backends = []

    def _qualified_factory(self):
        from .host_qualification import QualifiedFault39Backend

        generation = self._generation

        def active():
            return self.installed and generation == self._generation

        def create(workers, **binding):
            if not active():
                raise RuntimeError("FAULT39_BINDING_NOT_ACTIVE")
            backend = QualifiedFault39Backend(workers, binding_active=active, **binding)
            self._backends.append(backend)
            return backend

        return create

    def install(self):
        if self.installed:
            raise RuntimeError("ADAPTER_ALREADY_INSTALLED")
        manifest = integration_manifest()
        self._generation += 1
        factory = NativeBackend
        if self._qualified:
            factory = self._qualified_factory()
            manifest["backend"] = "demoncat_fault39.host_qualification.QualifiedFault39Backend"
            manifest["qualification"] = "SCOPED_SINGLE_WORKER_LOSSLESS"
        self.host.catalog_register_primitive(self.owner, manifest)
        try:
            self.host.binding_register(self.owner, manifest, factory)
        except Exception:
            self.host.catalog_unregister_primitive(self.owner, PRIMITIVE)
            raise
        self.installed = True
        return manifest

    def uninstall(self):
        if self.installed:
            for backend in self._backends:
                state = backend.query()
                if state["config"] is not None or state["generations"] or state["uncertain_workers"]:
                    raise RuntimeError("OWNED_BACKEND_CLEAN_REQUIRED")
            self.host.binding_unregister(self.owner, "fault39.native.matmul")
            self.host.catalog_unregister_primitive(self.owner, PRIMITIVE)
            self.installed = False
            self._backends.clear()
        return {"status": INTEGRATION_STATUS, "contract_adapter_removed": True}
