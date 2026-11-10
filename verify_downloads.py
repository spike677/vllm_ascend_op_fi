"""Verify the two graph-native statistics archives, optionally restoring data/."""
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
    return Path('\\\\?\\'+str(path)) if os.name == 'nt' and not str(path).startswith('\\\\?\\') else path


def verify(root, destination=None):
    root = extended(root).resolve(strict=True)
    manifest = json.loads((root/'manifest.json').read_bytes())
    assert manifest['schema'] == 'fault28.fia_fullgraph_operator_statistics.data_branch.v2'
    assert manifest['group_names'] == ['baseline','fault']
    assert manifest['actual_steps'] == [0,1] and len(manifest['archives']) == 2
    target = extended(destination) if destination is not None else None
    if target is not None:
        assert not target.exists(), 'Use a fresh extraction destination.'
    digests, sizes, total, sums_text, mapping = {}, {}, 0, None, None
    indexed = {}
    for archive in manifest['archives']:
        rel = PurePosixPath(archive['path'])
        assert not rel.is_absolute() and '..' not in rel.parts
        source = root/rel
        assert source.stat().st_size == archive['bytes'] and sha(source) == archive['sha256']
        with zipfile.ZipFile(source) as bundle:
            members = bundle.infolist()
            assert len(members) == archive['files']
            for member in members:
                path = PurePosixPath(member.filename)
                assert not path.is_absolute() and '..' not in path.parts and len(path.parts) > 1
                assert path.parts[0] == 'data' and '\\' not in member.filename and ':' not in member.filename
                assert not member.is_dir() and not stat.S_ISLNK(member.external_attr >> 16)
                assert member.filename not in digests
                assert path.suffix.lower() not in {'.pt','.pth','.bin','.npy','.npz','.safetensors','.so','.o','.log'}
                payload = bundle.read(member)
                assert len(payload) == member.file_size
                digests[member.filename] = hashlib.sha256(payload).hexdigest()
                sizes[member.filename] = len(payload)
                total += len(payload)
                if member.filename == 'data/SHA256SUMS':
                    sums_text = payload.decode('utf-8')
                if member.filename == 'data/source_mapping.json':
                    mapping = json.loads(payload)
                if '/derived/' in member.filename and path.name == 'operators.json':
                    d = json.loads(payload)
                    assert d['schema'] == 'fault28.DERIVED_GE_operator_statistics.v1'
                    assert d['is_native_msprobe_dump_json'] is False
                    assert len(d['operators']) == d['all_optimized_graph_nodes']
                    assert len({n['name'] for n in d['operators']}) == len(d['operators'])
                    indexed[member.filename] = d
                if target is not None:
                    output = target/path
                    assert output.resolve().is_relative_to(target.resolve())
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with output.open('xb') as stream:
                        stream.write(payload)
    assert sums_text is not None and mapping is not None
    expected = {}
    for line in sums_text.splitlines():
        digest, name = line.split('  ', 1)
        assert name not in expected
        expected[name] = digest
    assert {n.removeprefix('data/'): d for n, d in digests.items() if n != 'data/SHA256SUMS'} == expected
    assert len(digests) == manifest['data_files'] and total == manifest['data_bytes']
    assert digests['data/SHA256SUMS'] == manifest['source']['data_sha256sums_sha256']
    assert digests['data/source_mapping.json'] == manifest['source']['source_mapping_sha256']
    seen = set()
    for row in mapping['files']:
        path = 'data/'+row['published']
        assert row['published'] not in seen and row['original_bytes_preserved']
        seen.add(row['published'])
        assert digests[path] == row['sha256'] and sizes[path] == row['bytes']
    for row in manifest['previews']:
        assert sha(root/row['published']) == row['sha256'] == digests['data/'+row['original']]
    for row in manifest['derived_indexes']:
        index = json.loads((root/row['published']).read_bytes())
        original = indexed['data/'+row['original']]
        assert index['generated_index'] and index['is_native_msprobe_dump_json'] is False
        assert index['source_sha256'] == digests['data/'+row['original']]
        assert index['all_optimized_graph_nodes'] == original['all_optimized_graph_nodes']
        assert index['operators'] == [{k: op[k] for k in ('graph_index','name','type','status')}
                                     for op in original['operators']]
    roles = {PurePosixPath(name).parts[2] for name in digests if name.startswith('data/cases/')}
    assert roles == {'baseline','fault'} and len(indexed) == 6
    return {'status': 'PASS_TWO_FULLGRAPH_ARCHIVES_ORIGINAL_BYTES_AND_DERIVED_INDEXES',
            'archives': 2, 'groups': sorted(roles), 'data_files': len(digests),
            'data_bytes': total, 'derived_per_operator_files': len(indexed),
            'actual_steps': [0,1], 'extracted': target is not None, 'native_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--extract-to', type=Path)
    a = p.parse_args()
    print(json.dumps(verify(Path(__file__).absolute().parent, a.extract_to), indent=2))


if __name__ == '__main__':
    main()
