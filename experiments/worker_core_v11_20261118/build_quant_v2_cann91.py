"""Separate no-device CANN9.1 V2 ABI build; no runtime routing qualification."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

out=Path('/build/quant-v2');out.mkdir()
source=Path('/input/batch_scale_v2_only.cpp');control=Path('/input/batch_scale_control_v2.py')
inventory=Path('/input/batch_scale_inventory.h')
if not control.is_file():raise RuntimeError('PORTABLE_CONTROL_REQUIRED')
if not inventory.is_file():raise RuntimeError('MATCHED_INDEPENDENT_INVENTORY_SOURCE_REQUIRED')
cann=Path('/usr/local/Ascend/cann-9.1.0/aarch64-linux')
real=cann.parent/'opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so'
if not real.is_file():raise RuntimeError('REAL_OPAPI_MISSING')
library=out/'runtime/vendor/op_api/lib/libcust_opapi.so';library.parent.mkdir(parents=True)
cmd=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-fPIC','-shared',str(source),
    '-I'+str(cann/'include'),'-L'+str(cann/'lib64'),'-lascendcl','-lnnopbase','-ldl','-pthread',
    '-Wl,-z,defs','-Wl,-soname,libcust_opapi.so','-o',str(library)]
subprocess.run(cmd,check=True)
symbols=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True)
if any(x+'GetWorkspaceSize' in symbols for x in ('aclnnDynamicQuant','aclnnDynamicQuantV3','aclnnDynamicQuantV4')):
    raise RuntimeError('UNREVIEWED_API_EXPORT')
for name in ('aclnnDynamicQuantV2GetWorkspaceSize','aclnnDynamicQuantV2','CanUsePTACache','dcat_dq_thread_scope_v1','dcat_dq_state_v1',
             'dcat_dq_inventory_state_v1','dcat_dq_inventory_read_v1'):
    if name not in symbols:raise RuntimeError('MISSING_TYPED_EXPORT:'+name)
shutil.copy2(source,out/'native_backend.cpp');shutil.copy2(control,out/'backend_control.py')
shutil.copy2(inventory,out/'batch_scale_inventory.h')
record=dict(schema='demoncat.quant.v2_portable.v1',command=cmd,actual_npu_calls=0,
    source_sha256=sha(source),library_sha256=sha(library),control_sha256=sha(control),
    inventory_source_sha256=sha(inventory),
    real_opapi=str(real),real_opapi_sha256=sha(real),
    v2_header_sha256=sha(cann/'include/aclnnop/aclnn_dynamic_quant_v2.h'),
    status='COMPILED_NOT_NPU_VALIDATED',supported_api='aclnnDynamicQuantV2',
    deployment='CANN9.1.0/v026-installed-c9c9432b',
    limitations=['No V1/V3/V4 exports','Actual torch_npu call routing and Worker B8 decode mapping not validated'],
    nnopbase_sha256=sha(cann/'lib64/libnnopbase.so'),ascendcl_sha256=sha(cann/'lib64/libascendcl.so'))
(out/'runtime/build.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(build_sha256=sha(out/'runtime/build.json'),**record)))
