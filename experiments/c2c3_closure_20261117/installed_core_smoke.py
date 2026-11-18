"""Verify installed dependency closure on CPU, without a source-tree fallback."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--installed',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if list(Path('/dev').glob('davinci*')) or Path('/dev/devmm_svm').exists():
        raise RuntimeError('NO_NPU_DEVICES_REQUIRED')
    root=args.installed.resolve(strict=True)
    names=('demoncat.runtime.vllm_ascend.worker_extension.host',
        'demoncat.runtime.vllm_ascend.worker_extension.registry',
        'fault_injection.deployments.writer26_public',
        'fault_injection.deployments.quant_group_c1',
        'fault_injection.deployments.postwrite_structure',
        'engineering_closure')
    files=[]
    for name in names:
        module=importlib.import_module(name);path=Path(module.__file__).resolve(strict=True)
        if not path.is_relative_to(root):raise RuntimeError('NOT_INSTALLED:'+name)
        files.append(dict(module=name,path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    report=dict(status='INSTALLED_CPU_IMPORTS_PASS',actual_npu_calls=0,modules=files,
                native_or_model_qualified=False)
    with args.out.open('x') as stream:json.dump(report,stream,indent=2)


if __name__=='__main__':main()
