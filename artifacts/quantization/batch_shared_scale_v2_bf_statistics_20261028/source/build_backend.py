"""Build inside the frozen CPU-only container; requires no NPU devices."""
from pathlib import Path
import hashlib
import json
import subprocess

root = Path(__file__).resolve().parent
cann = Path('/usr/local/Ascend/cann-9.1.0')
out = root / 'runtime/vendor/op_api/lib/libcust_opapi.so'
out.parent.mkdir(parents=True, exist_ok=True)
command = ['c++', '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror', '-fPIC', '-shared',
           str(root / 'native_backend.cpp'), '-I'+str(cann/'include'),
           '-L'+str(cann/'lib64'), '-lascendcl', '-lnnopbase', '-ldl', '-pthread',
           '-Wl,-z,defs', '-Wl,-soname,libcust_opapi.so', '-o', str(out)]
subprocess.run(command, check=True)
metadata = {'command': command, 'files': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [root/'native_backend.cpp', out]}}
(root/'runtime/build.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata, indent=2))
