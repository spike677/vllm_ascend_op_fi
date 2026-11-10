"""Verify the two FIA archives; optionally reconstruct the complete public data."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extended(path):
    path = path.absolute()
    return Path('\\\\?\\' + str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def verify(root, destination=None):
    root = extended(root).resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_bytes())
    assert manifest['schema'] == 'fault28.fia_statistics.data_branch.v1'
    assert manifest['case_ids'] == ['model_b8_baseline', 'model_b8_fault_r1']
    assert len(manifest['archives']) == 2
    target = extended(destination) if destination is not None else None
    if target is not None:
        assert not target.exists(), 'Extraction requires a new destination.'
    digests, total_bytes, sums_text = {}, 0, None
    for archive in manifest['archives']:
        relative = PurePosixPath(archive['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        source = root / relative
        assert sha(source) == archive['sha256'] and source.stat().st_size == archive['bytes']
        with zipfile.ZipFile(source) as bundle:
            members = bundle.infolist()
            assert len(members) == archive['files']
            for member in members:
                path = PurePosixPath(member.filename)
                assert not path.is_absolute() and '..' not in path.parts
                assert path.parts[0] == 'data' and len(path.parts) > 1
                assert '\\' not in member.filename and ':' not in member.filename
                assert not member.is_dir() and not stat.S_ISLNK(member.external_attr >> 16)
                assert member.filename not in digests, 'Archives must not overlap.'
                payload = bundle.read(member)
                assert len(payload) == member.file_size
                digests[member.filename] = hashlib.sha256(payload).hexdigest()
                total_bytes += len(payload)
                if member.filename == 'data/SHA256SUMS':
                    sums_text = payload.decode('utf-8')
                if target is not None:
                    output = target / path
                    assert output.resolve().is_relative_to(target.resolve())
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('xb') as stream:
                        stream.write(payload)
    assert sums_text is not None
    expected = {}
    for line in sums_text.splitlines():
        digest, name = line.split('  ', 1)
        assert name not in expected
        expected[name] = digest
    actual = {name.removeprefix('data/'): value for name, value in digests.items()
              if name != 'data/SHA256SUMS'}
    assert actual == expected, 'Complete data inventory or original bytes differ.'
    assert len(digests) == manifest['data_files'] and total_bytes == manifest['data_bytes']
    assert digests['data/SHA256SUMS'] == manifest['source']['data_sha256sums_sha256']
    assert digests['data/source_mapping.json'] == manifest['source']['source_mapping_sha256']
    for preview in manifest['previews']:
        local = root / preview['published']
        assert sha(local) == digests['data/' + preview['original']]
        assert sha(local) == preview['sha256']
    case_ids = {PurePosixPath(name).parts[2] for name in digests
                if name.startswith('data/cases/')}
    assert case_ids == set(manifest['case_ids'])
    return {'status': 'PASS_TWO_ARCHIVES_EXACT_ORIGINAL_DATA_AND_PREVIEWS',
            'data_files': len(digests), 'data_bytes': total_bytes, 'archives': 2,
            'preview_files': len(manifest['previews']), 'case_ids': sorted(case_ids),
            'extracted': target is not None, 'native_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract-to', type=Path)
    args = parser.parse_args()
    result = verify(Path(__file__).absolute().parent, args.extract_to)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
