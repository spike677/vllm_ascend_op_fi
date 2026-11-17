"""Frozen Python package inputs; raw experiments and Native vendors excluded."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
p.add_argument('--base-archive',type=Path);p.add_argument('--replace',action='append',default=[])
p.add_argument('--add',action='append',default=[]);a=p.parse_args()
root=Path(__file__).resolve().parents[2]
paths={root/name for name in ('pyproject.toml','src/pyproject.toml','fault39/pyproject.toml')}
for directory in ('src/demoncat','fault_injection','native_execution_strategy','fault39/src/demoncat_fault39'):
    paths.update(p for p in (root/directory).rglob('*') if p.is_file()
                 and p.suffix in ('.py','.json','.md') and '__pycache__' not in p.parts)
manifest={}
contents={}
if a.base_archive:
    with tarfile.open(a.base_archive) as source:
        for item in source:
            if not item.isfile() or Path(item.name).is_absolute() or '..' in Path(item.name).parts:
                raise ValueError('Unsafe base archive')
            contents[item.name]=source.extractfile(item).read()
    for name in a.replace:
        if name not in contents:raise ValueError('Replace only an existing frozen source member')
        contents[name]=(root/name).read_bytes().replace(b'\r\n',b'\n')
    for name in a.add:
        if name in contents or (root/name) not in paths or '..' in Path(name).parts:
            raise ValueError('Add only an enumerated new package member')
        contents[name]=(root/name).read_bytes().replace(b'\r\n',b'\n')
else:
    contents={path.relative_to(root).as_posix():path.read_bytes().replace(b'\r\n',b'\n') for path in paths}
with a.out.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as archive:
    for name,data in sorted(contents.items()):
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
        archive.addfile(info,io.BytesIO(data))
        manifest[name]=hashlib.sha256(data).hexdigest()
a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(files=len(contents),bytes=a.out.stat().st_size,
                     sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())))
