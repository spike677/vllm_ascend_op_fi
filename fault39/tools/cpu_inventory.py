"""Read-only build-environment evidence. Does not import torch or initialize NPU."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    root=Path('/usr/local/Ascend/cann-8.5.1/aarch64-linux')
    files=[root/'include/aclnn/opdev/op_cache.h',root/'pkg_inc/base/dlog_pub.h',root/'lib64/libopapi.so']
    result=dict(schema='demoncat_fault39.build_inventory.v1',actual_npu_calls=0,
        machine=platform.machine(),python=platform.python_version(),pid=os.getpid(),
        packages={name:importlib.metadata.version(name) for name in ('torch','torch-npu','vllm','vllm-ascend')},
        files={str(f):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files},
        compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
        toolkit_install_info=(root/'ascend_toolkit_install.info').read_text())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
