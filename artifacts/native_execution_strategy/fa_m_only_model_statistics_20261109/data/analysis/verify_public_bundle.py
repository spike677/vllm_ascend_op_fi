"""Verify every published original byte and reject private/binary payloads."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path

FORBIDDEN={'.pt','.pth','.bin','.npy','.npz','.safetensors','.so','.o','.a','.run','.log'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.bundle.absolute()
    if os.name=='nt' and not str(root).startswith('\\\\?\\'):root=Path('\\\\?\\'+str(root))
    root=root.resolve(strict=True)
    sums={}
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,relative=line.split('  ',1)
        assert relative not in sums and len(digest)==64
        path=Path(relative)
        assert not path.is_absolute() and '..' not in path.parts
        sums[relative]=digest
    actual={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
    assert set(actual)==set(sums)|{'SHA256SUMS'},'Unexpected or missing bundle files'
    for name,file in actual.items():
        assert not file.is_symlink() and file.resolve().is_relative_to(root)
        assert file.suffix.lower() not in FORBIDDEN
        assert not any(part in ('private_reference','private_output','inputs','__pycache__') for part in file.relative_to(root).parts)
        if name!='SHA256SUMS':assert sha(file)==sums[name],'Changed public file: '+name
    manifest=json.loads((root/'manifest.json').read_bytes())
    assert manifest['schema']=='fault28.m_only.public_evidence.v1'
    originals=set()
    for item in manifest['files']:
        file=actual[item['published']]
        assert item['published'] not in originals
        originals.add(item['published'])
        assert file.stat().st_size==item['bytes'] and sums[item['published']]==item['sha256'] and item['original_bytes_preserved']
    assert all(row['included_in_public_bundle'] is False for row in manifest['remote_only'])
    configurations=[json.loads(actual[f'cases/{name}/config.json'].read_bytes()) for name in ('main_b1','main_b8')]
    assert all(c['task']=='statistics' and c['level']=='mix' and c['statistics']['summary_mode']=='statistics' for c in configurations)
    comparable=[]
    for name,configuration in zip(('main_b1','main_b8'),configurations):
        copy=dict(configuration);assert copy.pop('dump_path')==f'/evidence/{name}/observation';comparable.append(copy)
    assert comparable[0]==comparable[1],'B1/B8 collector settings differ beyond evidence path'
    # Full cases have ten untouched statistics/stack/construct triplets each.
    observations=Counter(name.split('/')[1] for name in actual if '/observation/' in name and name.endswith('/dump.json'))
    assert observations=={'main_b1':10,'main_b8':10}
    # Only collectors/configuration dictionaries can contain a task key. Check
    # original dump schemas directly instead of rewriting their content.
    for name,file in actual.items():
        if '/observation/' in name and name.endswith('/dump.json'):
            dump=json.loads(file.read_bytes())
            assert dump['task']=='statistics' and dump['level']=='mix' and dump['dump_data_dir'] is None
    report={'schema':'fault28.m_only.public_bundle_verification.v1','status':'PASS_PUBLIC_ORIGINAL_BYTES_AND_ALLOWLIST',
            'files':len(actual),'original_files':len(originals),'bytes':sum(p.stat().st_size for p in actual.values()),
            'manifest_sha256':sums['manifest.json'],'statistics_steps':dict(observations),
            'remote_only_entries':len(manifest['remote_only']),'private_tensors_published':False,
            'analysis_source_sha256':sha(Path(__file__))}
    output=a.output.absolute()
    if os.name=='nt' and not str(output).startswith('\\\\?\\'):output=Path('\\\\?\\'+str(output))
    assert not output.exists() and not output.resolve().is_relative_to(root)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(report))


if __name__=='__main__':main()
