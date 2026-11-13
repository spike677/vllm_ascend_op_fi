"""Check the byte manifest and the two statistics ZIPs; optional fresh extraction."""
import argparse,hashlib,json,zipfile
from pathlib import Path,PurePosixPath

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--extract-to',type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parent;checked=0
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);relative=PurePosixPath(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        path=root/Path(*relative.parts)
        assert path.is_file() and sha(path)==digest,'Checksum mismatch: '+name
        checked+=1
    if a.extract_to:
        destination=a.extract_to.resolve();assert not destination.exists(),'Use a fresh extraction directory'
        destination.mkdir(parents=True)
    groups={}
    for arm in ['baseline','fault']:
        expected={p.relative_to(root/arm).as_posix():p for p in (root/arm).rglob('*') if p.is_file()}
        prefix='data/cases/'+arm+'/'
        with zipfile.ZipFile(root/'downloads'/(arm+'.zip')) as z:
            assert len(z.namelist())==len(set(z.namelist()))
            assert set(z.namelist())=={prefix+name for name in expected}
            for name,path in expected.items():
                data=z.read(prefix+name);assert data==path.read_bytes(),'ZIP differs: '+name
                if a.extract_to:
                    out=destination/'data/cases'/arm/Path(*PurePosixPath(name).parts)
                    out.parent.mkdir(parents=True,exist_ok=True)
                    with out.open('xb') as f:f.write(data)
        steps=sorted(p.name for p in (root/arm/'observation').iterdir())
        assert steps==['step0','step1','step2']
        groups[arm]={'files':len(expected),'steps':steps,'zip_sha256':sha(root/'downloads'/(arm+'.zip'))}
    print(json.dumps({'status':'PASS_CHECKSUMS_AND_BF_ZIP_RAW_BYTES','checked_files':checked,'groups':groups}))

if __name__=='__main__':main()
