"""Read-only, SHA-bound local certificate files; no network or device access."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from .config import _unique_pairs, fields
from .models import BuildFingerprints, EvidenceRef, json_value, require, sha256
from .source_fingerprints import PYTHON_SOURCE_FILES, python_source_fingerprints


def local_path(root: str | Path, relative: str) -> Path:
    """Resolve a package-relative path and reject traversal/symlink escape."""
    require(isinstance(relative, str) and bool(relative), 'Bound path must be a nonempty relative string')
    require('\\' not in relative and ':' not in relative and '\x00' not in relative, 'Bound paths use relative POSIX syntax')
    part = PurePosixPath(relative)
    require(not part.is_absolute() and all(p not in ('', '.', '..') for p in relative.split('/')),
            'Bound path must not contain traversal, absolute roots or empty components')
    base = Path(root).resolve(strict=True)
    require(base.is_dir(), 'Certificate package root must be a directory')
    path = base.joinpath(*part.parts).resolve(strict=True)
    require(path.is_relative_to(base) and path.is_file(), 'Bound file escapes certificate package root or is not a file')
    return path


def read_bound_file(root: str | Path, relative: str, expected_sha256: str,
                    *, expected_bytes: int | None = None, max_bytes: int = 16 * 1024 * 1024) -> bytes:
    """Verify the actual bytes that are consumed, not only manifest assertions."""
    sha256(expected_sha256, 'Bound file SHA256')
    path = local_path(root, relative)
    size = path.stat().st_size
    require(type(max_bytes) is int and max_bytes > 0 and size <= max_bytes, 'Bound file exceeds read limit')
    if expected_bytes is not None:
        require(type(expected_bytes) is int and expected_bytes >= 0 and size == expected_bytes, 'Bound file size differs')
    raw = path.read_bytes()
    require(len(raw) == size and hashlib.sha256(raw).hexdigest() == expected_sha256, 'Bound file SHA256 differs')
    return raw


def strict_json(raw: bytes, label: str = 'JSON document') -> dict[str, Any]:
    def reject_constant(value):
        raise ValueError(f'{label} contains nonfinite number {value}')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_pairs, parse_constant=reject_constant)
    require(type(value) is dict, f'{label} must be a JSON object')
    return value


def read_bound_json(root: str | Path, relative: str, expected_sha256: str,
                    *, expected_bytes: int | None = None) -> dict[str, Any]:
    return strict_json(read_bound_file(root, relative, expected_sha256, expected_bytes=expected_bytes), relative)


@dataclass(frozen=True)
class LoadedCertificate:
    data: dict[str, Any]
    fingerprints: BuildFingerprints
    evidence: tuple[EvidenceRef, ...]
    path: Path


RUNTIME = {'source_commit': 'c77c0829f1f0f0b8f882db1d9b051a758b605472',
           'cann_build': 'V100R001C11SPC001B243', 'bridge_abi': 'nes.fa.cann9.1.s1s2.v1', 'device': 'Ascend910B4'}


def _certificate_path(path: str | Path, evidence_root: str | Path) -> Path:
    base = Path(evidence_root).resolve(strict=True)
    p = Path(path)
    if p.is_absolute():
        p = p.resolve(strict=True)
        require(p.is_relative_to(base), 'Certificate must be inside evidence_root')
        relative = p.relative_to(base).as_posix()
    else:
        relative = str(path).replace('\\', '/')
    return local_path(base, relative)


def _contexts(value: Any, *, validated: bool) -> dict[tuple[int, str], dict]:
    require(type(value) is list and bool(value), 'Certificate contexts must be a nonempty list')
    found, requests = {}, set()
    for raw_context in value:
        ctx = fields(raw_context, {'request_count', 'strategies', 'input_contract'}, {'request_count', 'strategies'}, 'certificate context')
        count = ctx['request_count']
        require(type(count) is int and count in (1, 8) and count not in requests, 'Duplicate or unsupported certified request_count')
        requests.add(count)
        contract = ctx.get('input_contract')
        require(not validated or type(contract) is dict, 'Validated context requires the complete native input contract')
        if contract is not None:
            from .adapters.fa_schema import validate_input_contract
            validate_input_contract(contract)
            require(contract['inputParams.bSize'] == count, 'Input contract batch differs from explicit request_count')
        require(type(ctx['strategies']) is list and bool(ctx['strategies']), 'Certified strategies must be a nonempty list')
        semantics = set()
        for raw_strategy in ctx['strategies']:
            strategy = fields(raw_strategy, {'strategy_id', 'parameters'}, {'strategy_id', 'parameters'}, 'certified strategy')
            name = strategy['strategy_id']
            require(isinstance(name, str) and bool(name) and (count, name) not in found, 'Duplicate/invalid certified strategy_id')
            params = fields(strategy['parameters'], {'query_tile', 'key_group_ratio', 'key_tile', 'execution_plan'},
                            {'query_tile', 'key_group_ratio', 'key_tile', 'execution_plan'}, 'certified parameters')
            require(type(params['query_tile']) is int and params['query_tile'] in (64, 128), 'Unsupported certified query tile')
            require(type(params['key_group_ratio']) is int and params['key_group_ratio'] in (4, 8), 'Unsupported certified key ratio')
            require(type(params['key_tile']) is int and params['key_tile'] == 128, 'Unsupported certified key tile')
            key = params['query_tile'], params['key_group_ratio']
            require(key not in semantics, 'Duplicate semantic recipe under different strategy IDs')
            semantics.add(key)
            plan = params['execution_plan']
            require(plan is None or (type(plan) is dict and bool(plan)), 'Execution plan must be nonempty object or explicit null')
            if validated:
                require(type(plan) is dict and bool(plan), 'Validated strategies require a complete observed execution plan')
            json_value(plan, 'execution_plan')
            if plan is not None:
                from .adapters.fa_schema import validate_execution_plan
                validate_execution_plan(plan)
                require(plan['coreParams.s1BaseSize'] == params['query_tile'] and
                        plan['coreParams.nRatio'] == params['key_group_ratio'] and
                        plan['coreParams.s2BaseSize'] == params['key_tile'],
                        'Certified recipe disagrees with its actual complete plan')
            found[count, name] = {'request_count': count, 'input_contract': contract, 'strategy': strategy}
    return found


def load_certificate(path: str | Path, *, evidence_root: str | Path,
                     allow_experimental: bool = False) -> LoadedCertificate:
    """Read a locally trusted certificate and bind every referenced report.

The evidence root is explicit; certificates and reports must resolve within it.
Experimental recipes are never promoted by a caller flag. This reader checks
evidence bindings, not the truth of arbitrary externally authored report text.
"""
    require(type(allow_experimental) is bool, 'allow_experimental must be explicit bool')
    p = _certificate_path(path, evidence_root)
    require(p.stat().st_size <= 16 * 1024 * 1024, 'Certificate exceeds 16 MiB')
    raw = p.read_bytes()
    d = strict_json(raw, 'certificate')
    fields(d, {'schema', 'id', 'adapter', 'validation_state', 'fingerprints', 'runtime', 'validation_evidence', 'contexts', 'projection_schema_sha256', 'python_source_files'},
           {'schema', 'id', 'adapter', 'validation_state', 'fingerprints', 'runtime', 'validation_evidence', 'contexts', 'projection_schema_sha256'}, 'certificate')
    require(d['schema'] == 'nes.fa.validation_certificate.v1', 'Unsupported certificate schema')
    from .adapters.fa_schema import POD_SCHEMA_SHA256
    require(d['projection_schema_sha256'] == POD_SCHEMA_SHA256,
            'Certificate must bind the audited 205-field POD projection schema')
    require(isinstance(d['id'], str) and bool(d['id']), 'Certificate id is required')
    adapter = fields(d['adapter'], {'name', 'version', 'route'}, {'name', 'version', 'route'}, 'certificate adapter')
    require(adapter['name'] == 'FlashAttentionScore' and adapter['route'] == 'S1S2', 'Unsupported certified adapter/route')
    require(isinstance(adapter['version'], str) and bool(adapter['version']) and adapter['version'] != 'validated',
            'Certificate adapter version must be concrete')
    require(d['validation_state'] in ('experimental', 'validated'), 'Invalid certificate validation_state')
    validated = d['validation_state'] == 'validated'
    require(validated or allow_experimental, 'Experimental certificate requires allow_experimental=True')
    python_files = d.get('python_source_files')
    require(not validated or type(python_files) is dict, 'Validated certificate requires Python source fingerprints')
    if python_files is not None:
        require(type(python_files) is dict and set(python_files) == set(PYTHON_SOURCE_FILES),
                'Python source fingerprints must cover the exact fixed FA dependency closure')
        for name, value in python_files.items():
            sha256(value, 'Python source SHA256: ' + name)
        require(python_files == python_source_fingerprints(), 'Current Python source fingerprints differ from certificate')
    declared = fields(d['fingerprints'], {'source_sha256', 'build_sha256'}, {'source_sha256', 'build_sha256'}, 'certificate fingerprints')
    fingerprints = BuildFingerprints(declared['source_sha256'], declared['build_sha256'], hashlib.sha256(raw).hexdigest())
    fields(d['runtime'], set(RUNTIME), set(RUNTIME), 'certificate runtime')
    require(d['runtime'] == RUNTIME, 'Certificate runtime/ABI is not the supported frozen FA source')
    supported = _contexts(d['contexts'], validated=validated)
    require(type(d['validation_evidence']) is list, 'validation_evidence must be a list')
    require(not validated or bool(d['validation_evidence']), 'Validated certificate needs an external passing report')
    references, seen_paths, covered = [EvidenceRef(str(p), fingerprints.certificate_sha256)], set(), {}
    for raw_ref in d['validation_evidence']:
        ref = fields(raw_ref, {'path', 'sha256'}, {'path', 'sha256'}, 'validation evidence')
        real = local_path(evidence_root, ref['path'])
        require(real != p and real not in seen_paths, 'Duplicate or self-referential certificate evidence')
        seen_paths.add(real)
        data = read_bound_file(evidence_root, ref['path'], ref['sha256'])
        references.append(EvidenceRef(str(real), ref['sha256']))
        if validated:
            report = strict_json(data, 'validation report')
            require(report.get('status') == 'PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1', 'External validation report did not pass the native FA phase')
            require(report.get('source_sha256') == fingerprints.source_sha256 and report.get('build_sha256') == fingerprints.build_sha256,
                    'External report belongs to another source/build; old build evidence cannot validate this certificate')
            require(report.get('python_source_files') == python_files,
                    'External validation report Python source fingerprints differ or are missing')
            actual = _contexts(report.get('validated_contexts'), validated=True)
            for key, strategy in actual.items():
                require(key not in covered or covered[key] == strategy, 'External reports disagree on a validated strategy')
                covered[key] = strategy
    if validated:
        require(all(key in covered and covered[key] == strategy for key, strategy in supported.items()),
                'External passing reports do not cover every certified context/strategy/complete plan')
    return LoadedCertificate(d, fingerprints, tuple(references), p)


def load_catalog(path: str | Path, *, evidence_root: str | Path,
                 allow_experimental: bool = False) -> tuple[LoadedCertificate, ...]:
    """Every entry fixes the certificate SHA; no latest-version/network lookup."""
    p = _certificate_path(path, evidence_root)
    require(p.stat().st_size <= 1048576, 'Catalog exceeds 1 MiB')
    d = strict_json(p.read_bytes(), 'adapter catalog')
    fields(d, {'schema', 'certificates'}, {'schema', 'certificates'}, 'catalog')
    require(d['schema'] == 'nes.adapter_catalog.v1' and type(d['certificates']) is list, 'Unsupported catalog schema')
    result, seen_paths, identities = [], set(), set()
    for raw_ref in d['certificates']:
        ref = fields(raw_ref, {'path', 'sha256'}, {'path', 'sha256'}, 'catalog certificate')
        certificate_path = local_path(evidence_root, ref['path'])
        require(certificate_path not in seen_paths, 'Duplicate certificate path in catalog')
        seen_paths.add(certificate_path)
        read_bound_file(evidence_root, ref['path'], ref['sha256'])
        loaded = load_certificate(certificate_path, evidence_root=evidence_root, allow_experimental=allow_experimental)
        require(loaded.fingerprints.certificate_sha256 == ref['sha256'], 'Certificate changed during catalog load')
        identity = tuple(loaded.data['adapter'][k] for k in ('name', 'version', 'route'))
        require(identity not in identities, 'Catalog contains ambiguous concrete adapter identity')
        identities.add(identity)
        result.append(loaded)
    return tuple(result)


def resolve_catalog(path: str | Path, *, evidence_root: str | Path, version: str = 'validated',
                    operator: str = 'FlashAttentionScore', route: str = 'S1S2',
                    allow_experimental: bool = False) -> LoadedCertificate:
    """Resolve a unique verified certificate; experimental never satisfies validated."""
    require(isinstance(version, str) and bool(version), 'A version selector is required')
    # Catalogs may retain experimental history. Reading its bounded, SHA-checked
    # metadata does not authorize using it; the selected result is checked below.
    entries = load_catalog(path, evidence_root=evidence_root, allow_experimental=True)
    matches = [e for e in entries if e.data['adapter']['name'] == operator and e.data['adapter']['route'] == route and
               (e.data['validation_state'] == 'validated' if version == 'validated' else e.data['adapter']['version'] == version)]
    require(len(matches) == 1, 'Catalog version selector did not resolve to exactly one verified certificate')
    require(matches[0].data['validation_state'] == 'validated' or allow_experimental,
            'Selecting an experimental certificate requires allow_experimental=True')
    return matches[0]
