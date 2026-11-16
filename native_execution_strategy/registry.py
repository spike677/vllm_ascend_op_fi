"""Explicit adapter/build resolution; 'validated' is a pinned alias, not a version.

Registration metadata is supplied by a trusted adapter provider after checking
its certificate. It does not replace runtime file hashes/ABI/resource guards.
New source or build fingerprints require a new validation record.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AdapterSelector
from .models import BuildFingerprints, EvidenceRef, require


@dataclass(frozen=True)
class AdapterRegistration:
    name: str
    version: str
    fingerprints: BuildFingerprints
    validation_state: str = 'experimental'
    validation_evidence: tuple[EvidenceRef, ...] = ()
    route: str | None = None

    def __post_init__(self) -> None:
        require(isinstance(self.name, str) and bool(self.name), 'Registration name is required')
        require(isinstance(self.version, str) and bool(self.version) and self.version != 'validated', 'Registration needs a concrete version')
        require(isinstance(self.fingerprints, BuildFingerprints), 'Registration needs explicit source/build fingerprints')
        require(self.validation_state in ('experimental', 'validated'), 'Invalid validation_state')
        require(self.route is None or (isinstance(self.route, str) and bool(self.route)), 'Invalid adapter route')
        require(all(isinstance(v, EvidenceRef) for v in self.validation_evidence), 'Invalid certificate evidence')
        object.__setattr__(self, 'validation_evidence', tuple(self.validation_evidence))
        if self.validation_state == 'validated':
            require(self.fingerprints.certificate_sha256 is not None and bool(self.validation_evidence),
                    'Validated registration requires an explicit certificate SHA and evidence')
            require(any(e.sha256 == self.fingerprints.certificate_sha256 for e in self.validation_evidence),
                    'Validation evidence must include the actual certificate SHA')


@dataclass(frozen=True)
class ResolvedAdapter:
    registration: AdapterRegistration
    adapter: Any
    requested_version: str

    @property
    def version(self) -> str:
        return self.registration.version

    @property
    def fingerprints(self) -> BuildFingerprints:
        return self.registration.fingerprints

    def as_dict(self) -> dict:
        r = self.registration
        return {'name': r.name, 'requested_version': self.requested_version, 'resolved_version': r.version,
                'route': r.route,
                'validation_state': r.validation_state, 'fingerprints': r.fingerprints.as_dict(),
                'validation_evidence': [{'path': e.path, 'sha256': e.sha256} for e in r.validation_evidence]}


class AdapterRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[AdapterRegistration, Any]] = {}

    def register(self, registration: AdapterRegistration, adapter: Any) -> None:
        require(isinstance(registration, AdapterRegistration), 'AdapterRegistration is required')
        key = registration.name, registration.version
        require(key not in self._entries, 'A concrete adapter version is already registered; fingerprints cannot be replaced')
        require(getattr(adapter, 'name', None) == registration.name and getattr(adapter, 'version', None) == registration.version,
                'Adapter identity differs from its registration')
        self._entries[key] = registration, adapter

    def resolve(self, selector: AdapterSelector) -> ResolvedAdapter:
        require(isinstance(selector, AdapterSelector), 'AdapterSelector is required')
        candidates = [entry for (name, version), entry in self._entries.items() if name == selector.name and
                      (entry[0].validation_state == 'validated' if selector.version == 'validated' else version == selector.version)]
        if selector.route is not None:
            candidates = [entry for entry in candidates if entry[0].route == selector.route]
        if selector.expected_fingerprints is not None:
            expected = selector.expected_fingerprints
            candidates = [entry for entry in candidates if entry[0].fingerprints.source_sha256 == expected.source_sha256 and
                          entry[0].fingerprints.build_sha256 == expected.build_sha256 and
                          (expected.certificate_sha256 is None or entry[0].fingerprints.certificate_sha256 == expected.certificate_sha256)]
        require(bool(candidates), 'No adapter matches the requested version, validation state and fingerprints')
        require(len(candidates) == 1, 'Ambiguous validated version: specify a concrete version or exact fingerprints')
        registration, adapter = candidates[0]
        return ResolvedAdapter(registration, adapter, selector.version)

    def registrations(self) -> tuple[AdapterRegistration, ...]:
        return tuple(self._entries[k][0] for k in sorted(self._entries))
