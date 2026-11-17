"""Create an immutable, source-only build input; never include vendor binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[2]
selected = []
selected.append(root/'fault_injection/deployments/pta_resolver.py')
selected.append(root/'fault_injection/deployments/matmul.py')
for folder in ('fault39/native', 'fault39/tools', 'fault_injection/native_adapters', 'tests/native',
               'fault20_fast_validation/source/l4_buffer_substitute_allhit_v1',
               'fault20_fast_validation/source/l4_metadata_lifecycle_proxy',
               'fault20_fast_validation/source/fault26_readonly_trace_v1'):
    for path in sorted((root / folder).iterdir()):
        if path.is_file() and path.suffix in ('.py', '.cpp', '.h', '.inc', '.sh'):
            selected.append(path)
a.out.parent.mkdir(parents=True, exist_ok=True)
with a.out.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as archive:
    for path in selected:
        archive.add(path, arcname=path.relative_to(root).as_posix())
manifest = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in selected}
a.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps(dict(files=len(selected), bytes=a.out.stat().st_size,
                     sha256=hashlib.sha256(a.out.read_bytes()).hexdigest())))
