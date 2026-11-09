"""Explicit FA Python dependency closure, hashed from current source bytes.

Unrelated future adapters are not part of this closure. Changes to shared core,
FA adapter or this verifier invalidate an older validated certificate. The
helper contains paths, not a literal hash of itself, so no self-hash cycle is
introduced.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


PYTHON_SOURCE_FILES = (
    '__init__.py',
    '__main__.py',
    'models.py',
    'config.py',
    'policies.py',
    'registry.py',
    'certificates.py',
    'protocol.py',
    'session.py',
    'discovery.py',
    'certificate_tools.py',
    'source_fingerprints.py',
    'adapters/__init__.py',
    'adapters/fa_schema.py',
    'adapters/flash_attention.py',
)


def python_source_fingerprints() -> dict[str, str]:
    """Return the fixed relative-path -> actual SHA256 map; missing files fail."""
    root = Path(__file__).resolve(strict=True).parent
    result = {}
    for name in PYTHON_SOURCE_FILES:
        path = root.joinpath(*name.split('/')).resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Python source dependency escapes package or is not a file: ' + name)
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result
