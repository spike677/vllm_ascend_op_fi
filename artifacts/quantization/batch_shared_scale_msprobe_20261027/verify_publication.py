#!/usr/bin/env python3
"""Verify publication bytes and ZIP membership using only the Python stdlib.

This is an integrity/structure check. CPU tensor semantics are checked separately
by verify_msprobe.py from audit_support.zip (requires NumPy).
"""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import stat
import zipfile

STAGES = ('H', 'H1', 'F', 'F2', 'R')
CASES = ('letter_counts_B2', 'even_count_B4')
NATIVE = 'evidence/model_capture01/msprobe_dump/'
REMOVED_COLUMNS = {'pid', 'msprobe_tensor_path', 'debug_json_path', 'canonical_tensor_path', 'manifest_path'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name and '\0' not in name,
            'Unsafe path: ' + str(name))
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and all(x not in ('', '.', '..') for x in name.split('/')),
            'Unsafe relative path: ' + name)
    return name


def original_relative(value):
    require(isinstance(value, str) and value.startswith('/audit/'), 'Unexpected original capture path')
    return safe(value[len('/audit/'):])


def audit_allowed(name):
    p = PurePosixPath(safe(name))
    for run in ('model_B01', 'model_capture01'):
        prefix = 'evidence/' + run
        if name in (prefix + '/result.json', prefix + '/control.json'):
            return True
        if p.parent.as_posix() == prefix and p.name.endswith('.control.json'):
            return True
        if name.startswith(prefix + '/operators/') and p.suffix in ('.jsonl', '.pt'):
            return True
    return (name in ('evidence/msprobe_cpu_review_remote.json', 'prior_result.json', 'verify_msprobe.py')
            or (p.parent.as_posix() == 'source_versions/model_B01/model' and p.suffix == '.py')
            or (name.startswith('legacy/src/') and p.suffix == '.py')
            or (p.parent.as_posix() == 'legacy/experiments/fake_requant_reference/source' and p.suffix == '.py')
            or (p.parent.as_posix() == 'verification_support' and p.suffix == '.py'))


def verify(root):
    root = Path(root).resolve()
    inventory = json.loads((root / 'SHA256.json').read_text(encoding='utf-8'))
    require(inventory['schema'] == 'batch_shared_scale_msprobe_publication_v1', 'Wrong publication schema')
    entries = inventory['files']
    names = [safe(x['path']) for x in entries]
    require(len(names) == len(set(names)) == len({x.casefold() for x in names}), 'Duplicate publication path')
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual == set(names) | {'SHA256.json'}, 'Publication file set differs from inventory')
    for entry in entries:
        path = root / entry['path']
        require(not path.is_symlink() and path.resolve().is_relative_to(root), 'Publication path escapes root')
        data = path.read_bytes()
        require(len(data) == entry['bytes'] and digest(data) == entry['sha256'], 'File hash mismatch: ' + entry['path'])
    archives = inventory['archives']
    require(set(archives) == {'data/' + x + '.zip' for x in STAGES} | {'audit_support.zip'}, 'Unexpected ZIP set')
    union = {}; audit = {}; session_keys = set()
    for archive, metadata in archives.items():
        expected = metadata['members']
        expected_by_name = {safe(x['path']): x for x in expected}
        require(len(expected_by_name) == len(expected), 'Duplicate ZIP manifest member')
        payload = (root / archive).read_bytes()
        require(len(payload) == metadata['bytes'] and digest(payload) == metadata['sha256'], 'ZIP hash mismatch')
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            infos = z.infolist(); member_names = [safe(x.filename) for x in infos]
            require(len(member_names) == len(set(member_names)) == len({x.casefold() for x in member_names}), 'Duplicate ZIP member')
            require(set(member_names) == set(expected_by_name), 'ZIP member set mismatch')
            members = {}
            for info in infos:
                require(not info.is_dir() and stat.S_ISREG(info.external_attr >> 16), 'Non-regular ZIP member')
                require(not info.flag_bits & 1, 'Encrypted ZIP member')
                expected_member = expected_by_name[info.filename]
                require(info.file_size == expected_member['bytes'], 'ZIP size mismatch')
                data = z.read(info)
                require(digest(data) == expected_member['sha256'] == expected_member['source_sha256'],
                        'ZIP member differs from verified source: ' + info.filename)
                members[info.filename] = data
        if archive == 'audit_support.zip':
            require(all(audit_allowed(x) for x in members), 'Non-allowlisted audit file')
            require(sum(PurePosixPath(x).parent.as_posix() == 'source_versions/model_B01/model' for x in members) == 9,
                    'Expected nine frozen model Python sources')
            audit = members
            continue
        stage = PurePosixPath(archive).stem
        require(len(members) == 38 and all(x.startswith(NATIVE) for x in members), 'Wrong native stage member count')
        counts = Counter('pt' if x.endswith('.debug.pt') else PurePosixPath(x).name for x in members)
        require(counts == {'pt': 28, 'debug.json': 4, 'manifest.json': 4, 'capture_config.json': 2}, 'Wrong native stage composition')
        for name, data in members.items():
            if name in union:
                require(name.endswith('/capture_config.json') and union[name] == data, 'Unexpected cross-stage duplicate')
            union[name] = data
            if not name.endswith('/manifest.json'):
                continue
            manifest = json.loads(data)
            meta = manifest['run_metadata']; generation = meta['generation_id']
            match = re.fullmatch(r'(letter_counts_B2|even_count_B4)_(H|H1|F|F2|R)', generation)
            require(match is not None and match[2] == stage and meta['rank'] in (0, 1) and manifest['status'] == 'PASS',
                    'Session stage/rank/status mismatch')
            key = (match[1], stage, meta['rank']); require(key not in session_keys, 'Duplicate session'); session_keys.add(key)
            tensors = [value for record in manifest['records'] for value in record['tensors'].values()]
            require(len(tensors) == 7 and len({x['file'] for x in tensors}) == 7, 'Expected seven session tensors')
            for tensor in tensors:
                pt = original_relative(tensor['file']); debug = original_relative(tensor['debug_json'])
                require(pt in members and debug in members, 'Missing referenced native payload')
                require(digest(members[pt]) == tensor['file_sha256'] and len(members[pt]) == tensor['file_bytes'], 'Session tensor hash mismatch')
                require(digest(members[debug]) == tensor['debug_json_sha256'], 'Session debug hash mismatch')
    require(len(union) == 182, 'Expected 182 distinct native files')
    require(session_keys == {(c, s, r) for c in CASES for s in STAGES for r in (0, 1)}, 'Incomplete session matrix')
    require(sum(x.endswith('.debug.pt') for x in union) == 140 and sum(x.endswith('/debug.json') for x in union) == 20
            and sum(x.endswith('/manifest.json') for x in union) == 20, 'Wrong native union counts')
    with (root / 'capture_index.csv').open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream); rows = list(reader)
        require(not REMOVED_COLUMNS.intersection(reader.fieldnames or []), 'Private index columns remain')
    require(len(rows) == 140 and len({x['msprobe_tensor_relative_to_archive'] for x in rows}) == 140, 'Wrong public index size')
    for row in rows:
        for column, source in (('msprobe_tensor_relative_to_archive', union), ('debug_json_relative_to_archive', union),
                               ('manifest_relative_to_archive', union), ('canonical_tensor_relative_to_archive', audit)):
            require(safe(row[column]) in source, 'Index reference is missing: ' + row[column])
        require(digest(union[row['msprobe_tensor_relative_to_archive']]) == row['file_sha256'], 'Index tensor hash mismatch')
    with (root / 'paired_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        paired = list(csv.DictReader(stream))
    require(len(paired) == 4 and {(x['case'], int(x['rank'])) for x in paired} == {(c, r) for c in CASES for r in (0, 1)},
            'Expected four paired metric rows')
    summary = json.loads((root / 'verification_summary.json').read_text(encoding='utf-8'))
    require(summary['status'] == 'PASS' and summary['sessions'] == 20 and summary['msprobe_tensor_files'] == 140,
            'Summary counts/status mismatch')
    require(summary['observer_transparency']['status'] == 'RAW_AND_RESPONSE_EXACT', 'Transparency status mismatch')
    return {'status': 'PASS', 'native_union_files': 182, 'native_tensors': 140, 'debug_json_files': 20,
            'session_manifests': 20, 'rank_configs': 2, 'audit_support_files': len(audit),
            'capture_index_rows': 140, 'paired_metric_rows': 4, 'archives': 6,
            'scope': 'stdlib byte hashes and publication structure; run verify_msprobe.py for CPU tensor semantics'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
