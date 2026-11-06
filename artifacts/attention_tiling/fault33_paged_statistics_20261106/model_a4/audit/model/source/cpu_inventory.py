"""Inspect callable schemas on a CPU-only container; never initialize an NPU."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    import torch
    import torch_npu
    result = {'schema': 'fault33.cpu_inventory.v1', 'npu_calls': 0,
              'python': sys.version, 'torch': torch.__version__,
              'torch_npu': torch_npu.__version__, 'operators': {}, 'files': {}}
    for name in ('npu_incre_flash_attention', 'npu_fused_infer_attention_score',
                 '_npu_paged_attention', '_npu_paged_attention_splitfuse'):
        fn = getattr(torch_npu, name, None)
        record = {'available': fn is not None, 'doc': getattr(fn, '__doc__', None)}
        try:
            record['schema'] = str(getattr(torch.ops.npu, name)._schemas)
        except Exception as exc:
            record['schema_error'] = type(exc).__name__ + ': ' + str(exc)
        result['operators'][name] = record
    for name in ('torch-npu', 'vllm-ascend', 'vllm', 'msprobe', 'transformers'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    for package in (torch, torch_npu):
        p = Path(package.__file__)
        result['files'][package.__name__] = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
