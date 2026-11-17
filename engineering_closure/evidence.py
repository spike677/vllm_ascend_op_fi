"""Join original evidence references without turning saved reports into admission."""
import csv
import json
from pathlib import Path
import re

from fault_injection.catalog import CAPABILITIES, resolve
from fault_injection.coverage_audit import audit_saved_coverage, index_artifacts
from fault_injection.evidence import digest
from fault_injection.statistics_audit import compare_raw_statistics

from . import BASE_COMMIT


STAGES = ('source', 'build', 'native_inventory', 'target_binding', 'baseline',
          'fault', 'numerical_effect', 'off', 'recovery', 'independent_audit')
SCOPE_FIELDS = ('mechanism', 'model', 'cann', 'soc', 'tp', 'source_commit',
                'implementation_sha256', 'environment_sha256', 'workload_sha256')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _local_path(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('EVIDENCE_PATH_OUTSIDE_ROOT')
    return path


def _reference(root, reference):
    """Resolve an explicitly scoped local artifact and verify its original bytes."""
    if not isinstance(reference, dict) or not isinstance(reference.get('path'), str):
        raise ValueError('EVIDENCE_PATH_REQUIRED')
    path = _local_path(root, reference['path'])
    expected = reference.get('sha256', '')
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('EVIDENCE_SHA256_REQUIRED')
    if digest(path) != expected:
        raise ValueError('EVIDENCE_HASH_MISMATCH:' + reference['path'])
    return path


def _document(path, pointer=''):
    value = json.loads(path.read_bytes())
    if pointer:
        if not isinstance(pointer, str) or not pointer.startswith('/'):
            raise ValueError('JSON_POINTER_REQUIRED')
        for part in pointer.split('/')[1:]:
            key = part.replace('~1', '/').replace('~0', '~')
            value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def audit_bundle(bundle, *, artifact_root):
    """Check one mechanism scope, raw bytes, saved coverage and raw statistics.

    Stage references retain the author's claims. Only known audit schemas receive
    a recomputed result. No serialized report creates a live C2 proof.
    """
    if bundle.get('schema') != 'demoncat.engineering.evidence_bundle.v1':
        raise ValueError('UNSUPPORTED_EVIDENCE_BUNDLE')
    scope = bundle.get('scope', {})
    if any(scope.get(key) in (None, '') for key in SCOPE_FIELDS):
        raise ValueError('COMPLETE_EVIDENCE_SCOPE_REQUIRED')
    resolve(scope['mechanism'])
    if type(scope['tp']) is not int or scope['tp'] < 1:
        raise ValueError('POSITIVE_TP_REQUIRED')
    for key in ('implementation_sha256', 'environment_sha256', 'workload_sha256'):
        if re.fullmatch('[0-9a-f]{64}', str(scope[key])) is None:
            raise ValueError('INVALID_SCOPE_HASH:' + key)
    root = Path(artifact_root).resolve(strict=True)
    stages = bundle.get('stages', {})
    if set(stages) - set(STAGES):
        raise ValueError('UNKNOWN_EVIDENCE_STAGE')
    result = dict(schema='demoncat.engineering.evidence_audit.v1', scope=scope,
                  scope_labels_verification='AUTHOR_SUPPLIED_REQUIRES_ENVIRONMENT_AND_MODEL_REVIEW',
                  stages={}, errors=[], coverage=None, statistics=None,
                  engineering_coverage=None, mechanism_coverage=None,
                  public_qualification_granted=False)
    verified = []
    for name in STAGES:
        references = stages.get(name, [])
        if not isinstance(references, list):
            raise ValueError('EVIDENCE_STAGE_REFERENCES_MUST_BE_LIST')
        rows = []
        for reference in references:
            row = dict(reference=reference, bytes_verified=False,
                       semantic_review='PENDING')
            try:
                path = _reference(root, reference)
                row.update(bytes_verified=True, bytes=path.stat().st_size)
                verified.append(path)
            except (OSError, ValueError) as error:
                result['errors'].append(name + ':' + str(error))
            rows.append(row)
        result['stages'][name] = dict(status='REFERENCED' if rows else 'MISSING', files=rows)
    if bundle.get('coverage'):
        try:
            ref = bundle['coverage']
            report = _document(_reference(root, ref), ref.get('pointer', ''))
            for key in ('mechanism', 'implementation_sha256', 'environment_sha256', 'workload_sha256'):
                if report.get('scope', {}).get(key) != scope[key]:
                    raise ValueError('COVERAGE_SCOPE_MISMATCH:' + key)
            c2 = None
            if bundle.get('c2'):
                c2_ref = bundle['c2']
                c2 = _document(_reference(root, c2_ref), c2_ref.get('pointer', ''))
            result['coverage'] = audit_saved_coverage(
                report, artifact_index=index_artifacts(verified), c2_report=c2)
            result['errors'].extend(result['coverage']['errors'])
        except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
            result['errors'].append('COVERAGE:' + str(error))
    if bundle.get('statistics'):
        try:
            captures = bundle['statistics']
            options = {}
            for phase in ('baseline', 'fault'):
                refs = captures[phase]
                paths = {name: _reference(root, refs[name]) for name in
                         ('dump.json', 'stack.json', 'construct.json', 'config', 'collector')}
                directory = paths['dump.json'].parent
                if any(paths[name] != directory / name for name in
                       ('dump.json', 'stack.json', 'construct.json')):
                    raise ValueError('STATISTICS_TRIPLET_DIRECTORY_MISMATCH')
                options[phase] = directory
                options[phase + '_config'] = paths['config']
                options[phase + '_collector'] = paths['collector']
            result['statistics'] = compare_raw_statistics(**options)
        except (OSError, ValueError, KeyError, TypeError) as error:
            result['errors'].append('STATISTICS:' + str(error))
    result['missing_stages'] = [name for name, row in result['stages'].items()
                                if row['status'] == 'MISSING']
    result['reference_integrity_verified'] = bool(verified) and not result['errors']
    return result


def audit_checkpoint(repo):
    """Check the inherited publication and the new mechanism accounting locally."""
    repo = Path(repo).resolve()
    folder = repo / 'engineering_closure'
    rows = read_csv(folder / 'mechanism_matrix.csv')
    mechanisms = {item.name for item in CAPABILITIES}
    names = [row['mechanism'] for row in rows]
    if len(names) != 17 or set(names) != mechanisms or len(set(names)) != len(names):
        raise ValueError('EXACTLY_17_UNIQUE_MECHANISMS_REQUIRED')
    manifest_path = repo / 'experiments/c2c3_closure_20261117/publication_manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    matched, changed, missing = [], [], []
    for item in manifest['files']:
        path = _local_path(repo, item['path'])
        if not path.is_file():
            missing.append(item['path'])
        elif digest(path) != item['sha256']:
            changed.append(item['path'])
        else:
            matched.append(item['path'])
    references = []
    for row in rows:
        for name in row['evidence_paths'].split('|'):
            path = _local_path(repo, name)
            references.append(dict(mechanism=row['mechanism'], path=name,
                available=path.is_file(), sha256=digest(path) if path.is_file() else None))
    return dict(schema='demoncat.engineering.checkpoint.v1', baseline_commit=BASE_COMMIT,
        mechanism_count=len(rows),
        historical_reported_c2=[row['mechanism'] for row in rows if row['c2'] == 'SCOPED_PASS'],
        historical_reported_c3=[row['mechanism'] for row in rows if row['c3'] == 'SCOPED_PASS'],
        publication=dict(manifest_sha256=digest(manifest_path), matched=len(matched),
                         modified_since_publication=changed, missing=missing),
        mechanisms=rows, evidence_references=references, original_remote_raw_bytes_verified=False,
        new_mechanism_qualification_granted=False,
        engineering_coverage='See executable validation receipt and per-adapter capability report')
