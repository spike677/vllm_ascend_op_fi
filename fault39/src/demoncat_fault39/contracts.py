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
    def __init__(self, host: HostContract, owner="fault39.contract.adapter"):
        if getattr(host, "api", None) != HOST_API:
            raise ValueError("EXPLICIT_HOST_CONTRACT_REQUIRED")
        self.host, self.owner = host, owner
        self.installed = False

    def install(self):
        if self.installed:
            raise RuntimeError("ADAPTER_ALREADY_INSTALLED")
        manifest = integration_manifest()
        self.host.catalog_register_primitive(self.owner, manifest)
        try:
            self.host.binding_register(self.owner, manifest, NativeBackend)
        except Exception:
            self.host.catalog_unregister_primitive(self.owner, PRIMITIVE)
            raise
        self.installed = True
        return manifest

    def uninstall(self):
        if self.installed:
            self.host.binding_unregister(self.owner, "fault39.native.matmul")
            self.host.catalog_unregister_primitive(self.owner, PRIMITIVE)
            self.installed = False
        return {"status": INTEGRATION_STATUS, "contract_adapter_removed": True}
