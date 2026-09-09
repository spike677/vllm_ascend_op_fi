"""Optional single startup entrypoint, requiring the independently installed R1 module.

No runtime dlopen, method replacement or code copying. Import only at new service startup.
"""
import hashlib
import importlib.util
import os
from pathlib import Path

from .extension import InspectWorkerExtension

spec = importlib.util.find_spec('fake_requant_reference_extension')
expected = os.environ.get('ASCEND_FI_REFERENCE_SHA256', '')
if not spec or not spec.origin or len(expected) != 64 or hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest() != expected:
    raise RuntimeError('PINNED_RESEARCH_REFERENCE_DEPENDENCY_REQUIRED')

from fake_requant_reference_extension import FakeRequantReferenceExtension

if hasattr(FakeRequantReferenceExtension, 'demoncat_inspect'):
    raise RuntimeError('WORKER_EXTENSION_METHOD_COLLISION')


class InteractiveWorkerExtension(InspectWorkerExtension, FakeRequantReferenceExtension):
    """Use this ONE --worker-extension-cls when explicit IDENTITY delegation is needed."""
