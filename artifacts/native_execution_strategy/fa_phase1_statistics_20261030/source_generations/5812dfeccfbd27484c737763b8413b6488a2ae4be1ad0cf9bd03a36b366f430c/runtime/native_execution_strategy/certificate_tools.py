"""Explicit experiment bootstrap and evidence-backed certificate issuance.

Bootstrap recipes retain experimental status. Only complete actual validation
reports may issue a validated certificate for the exact same source/build.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from .adapters.fa_schema import POD_SCHEMA_SHA256, decode, input_contract
from .certificates import RUNTIME, load_certificate, strict_json
from .models import require


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write('\n')


def prepare_experimental(package_root: Path, vendor_manifest: Path, readiness: Path,
                         reference_bundle: Path) -> Path:
    """Use prior legal recipes as a bootstrap, never inherit old runtime status."""
    package_root = package_root.resolve(strict=True)
    vendor = strict_json(vendor_manifest.read_bytes())
    ready = strict_json(readiness.read_bytes())
    require(vendor['schema'] == 'nes.fa.private_vendor.v1' and ready['status'] == 'READY_EXPERIMENTAL_CPU_ONLY', 'New private CPU stage required')
    require(file_sha(vendor_manifest) == ready['vendor_manifest_sha256'] and ready['npu_executed'] is False,
            'Build receipt/inventory mismatch')
    require(vendor['policy_source_manifest_sha256'] == ready['policy_source_manifest_sha256'], 'Build/source provenance mismatch')
    data = package_root / 'data'
    proof = data / 'evidence/cpu_build_ready.json'
    require(not proof.exists(), 'Fresh certificate generation required')
    proof.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(readiness, proof)
    contexts = []
    for count in (1, 8):
        path = reference_bundle / f'attempts/private_base_a3/base_b{count}_r0/host_events.jsonl'
        events = [strict_json(line.encode()) for line in path.read_text(encoding='utf-8').splitlines()]
        require(len(events) == 1 and events[0]['callback_status'] == 0, 'Prior native recipe provenance missing')
        pod = decode(bytes.fromhex(events[0]['raw_hex']))
        contexts.append({'request_count': count, 'input_contract': input_contract(pod),
            'strategies': [{'strategy_id': f'fa.s1s2.m{m}.r{ratio}',
                            'parameters': {'query_tile': m, 'key_group_ratio': ratio, 'key_tile': 128, 'execution_plan': None}}
                           for m, ratio in ((128, 8), (64, 8), (128, 4), (64, 4))]})
    certificate = {'schema': 'nes.fa.validation_certificate.v1', 'id': 'fa_s1s2_experimental_20261030',
        'adapter': {'name': 'FlashAttentionScore', 'version': 'cann9.1-s1s2-candidate-20261030', 'route': 'S1S2'},
        'validation_state': 'experimental',
        'fingerprints': {'source_sha256': ready['policy_source_manifest_sha256'], 'build_sha256': file_sha(vendor_manifest)},
        'runtime': RUNTIME, 'projection_schema_sha256': POD_SCHEMA_SHA256,
        'validation_evidence': [{'path': 'data/evidence/cpu_build_ready.json', 'sha256': file_sha(proof)}], 'contexts': contexts}
    output = data / 'fa_s1s2_experimental.json'
    write_new(output, certificate)
    load_certificate(output, evidence_root=package_root, allow_experimental=True)
    write_new(data / 'catalog.json', {'schema': 'nes.adapter_catalog.v1',
                                    'certificates': [{'path': output.relative_to(package_root).as_posix(), 'sha256': file_sha(output)}]})
    return output


def issue_validated(package_root: Path, bootstrap: Path, validation_report: Path) -> Path:
    package_root = package_root.resolve(strict=True)
    loaded = load_certificate(bootstrap, evidence_root=package_root, allow_experimental=True)
    report = strict_json(validation_report.read_bytes())
    require(report['status'] == 'PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1'
            and report['source_sha256'] == loaded.fingerprints.source_sha256
            and report['build_sha256'] == loaded.fingerprints.build_sha256, 'Report does not validate this exact bridge')
    proof = package_root / 'data/evidence/fa_phase1_validation.json'
    require(not proof.exists(), 'Cannot overwrite a validation generation')
    proof.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(validation_report, proof)
    data = dict(loaded.data)
    data.update(id='fa_cann91_s1s2_20261030_v1', validation_state='validated',
                adapter={'name': 'FlashAttentionScore', 'version': 'cann9.1-s1s2-v1', 'route': 'S1S2'},
                contexts=report['validated_contexts'],
                python_source_files=report['python_source_files'],
                validation_evidence=[{'path': 'data/evidence/fa_phase1_validation.json', 'sha256': file_sha(proof)}])
    output = package_root / 'data/fa_cann91_s1s2_v1.json'
    write_new(output, data)
    # This loader checks every full plan and every bound coverage item, not only status.
    load_certificate(output, evidence_root=package_root)
    catalog = strict_json((package_root / 'data/catalog.json').read_bytes())
    catalog['certificates'].append({'path': output.relative_to(package_root).as_posix(), 'sha256': file_sha(output)})
    path = package_root / 'data/catalog.json'
    temporary = path.with_suffix('.new')
    write_new(temporary, catalog)
    temporary.replace(path)
    return output
