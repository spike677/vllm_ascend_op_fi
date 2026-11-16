"""Verify independently installed pure-Python packages, without a Worker/NPU."""
import argparse
import hashlib
import importlib
import json
import platform
from pathlib import Path
import sys
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--installed', type=Path, required=True)
    p.add_argument('--wheels', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    installed = a.installed.resolve()
    sys.path.insert(0, str(installed))
    modules = []
    for name in ('demoncat.runtime.vllm_ascend.controller',
                 'demoncat.runtime.vllm_ascend.worker_extension.runtime',
                 'fault_injection', 'fault_injection.environment',
                 'fault_injection.model_discovery', 'native_execution_strategy',
                 'demoncat_fault39'):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(installed):
            raise RuntimeError('Source checkout leaked into installed import: ' + name)
        modules.append(dict(module=name, installed_path=path.relative_to(installed).as_posix(), sha256=sha(path)))
    wheels = []
    for path in sorted(a.wheels.glob('*.whl')):
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            prohibited = [n for n in names if n.endswith(('.so','.dll','.pt','.bin','.safetensors'))
                          or Path(n).name in ('dump.json','stack.json','construct.json')]
            if prohibited:
                raise RuntimeError('Unexpected raw/native payload: ' + repr(prohibited))
            wheels.append(dict(file=path.name, bytes=path.stat().st_size, sha256=sha(path),
                               members=len(names), prohibited_payloads=prohibited))
    if len(wheels) != 3:
        raise RuntimeError('Expected three independent distributions')
    a.out.write_text(json.dumps(dict(status='PASS_LOCAL_PACKAGE_INSTALL_IMPORT_ONLY',
        real_worker_loaded_from_wheel=False, platform=platform.platform(), actual_npu_calls=0,
        modules=modules, wheels=wheels), indent=2)+'\n')


if __name__ == '__main__':
    main()
