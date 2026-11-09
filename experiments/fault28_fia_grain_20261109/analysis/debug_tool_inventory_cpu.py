"""Inspect SDK tools in an owned CPU-only container, without NPU devices."""
import json
from pathlib import Path
import subprocess
import uuid

IMAGE='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
command=['docker','run','--rm','--name','f28-grain-cpu-'+uuid.uuid4().hex[:12],
         '--label','demoncat.task=fault28-fia-grain-cpu-inspection','--network','none',
         '--memory','2g','--cpus','2','--entrypoint','bash',IMAGE,'-c',
         "timeout 30s find /usr/local/Ascend -name '*show_kernel_debug*' -o -name '*printf*' -o -name 'kernel_operator_dump_tensor_intf.h' | head -30"]
result=subprocess.run(command,capture_output=True,text=True,timeout=60)
print(json.dumps({'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr[-2000:],
                  'npu_devices_mapped':False}))
