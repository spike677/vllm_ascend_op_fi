"""CPU-only scoped deployment build; preserve original arithmetic provenance."""
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
source=Path(__file__).resolve().parents[2]/'fault_injection/native_adapters/batch_scale_native.cpp'
root=Path(sys.argv[1]);root.mkdir(exist_ok=False)
control=Path('/work/quant_backend_control.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(control)=='5ce739c4272840f110044cfe898827fe403b5730273258e8ba0eda437685b38c'
shutil.copy2(control,root/'backend_control.py');shutil.copy2(source,root/'native_backend.cpp')
cann=Path('/usr/local/Ascend/cann-9.1.0')
out=root/'runtime/vendor/op_api/lib/libcust_opapi.so';out.parent.mkdir(parents=True)
cmd=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared',str(root/'native_backend.cpp'),
     '-I'+str(cann/'include'),'-L'+str(cann/'lib64'),'-lascendcl','-lnnopbase','-ldl','-pthread',
     '-Wl,-z,defs','-Wl,-soname,libcust_opapi.so','-o',str(out)]
subprocess.run(cmd,check=True)
manifest=dict(schema='demoncat.quant.thread_scope.v1',command=cmd,source_sha256=sha(source),library_sha256=sha(out),
    parent_native_sha256='895e635d4674dae95c03869bb9da4d66617519ea056f5e2b4c671bf7e3c9675c',control_sha256=sha(control))
(root/'runtime/build.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(build_sha256=sha(root/'runtime/build.json'),**manifest)))
