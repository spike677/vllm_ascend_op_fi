"""Read graph compiler schemas and installed source paths without NPU calls."""
import importlib.metadata as md
import importlib.util
import json
from pathlib import Path
import sys


def main():
    import torch
    import torch_npu
    result = {'schema': 'fault28.fia_graph.cpu_inventory.v1', 'npu_calls': 0,
              'python': sys.version, 'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
              'torch_npu_root': str(Path(torch_npu.__file__).parent), 'sys_path': sys.path,
              'packages': {}, 'operators': {}, 'torchair_candidates': []}
    for name in ('torch', 'torch-npu', 'torchair', 'torch-air', 'vllm', 'transformers'):
        try:
            result['packages'][name] = md.version(name)
        except md.PackageNotFoundError:
            result['packages'][name] = None
    spec = importlib.util.find_spec('torchair')
    result['torchair_spec'] = spec.origin if spec else None
    for base in (Path(torch_npu.__file__).parent, Path('/usr/local/lib'), Path('/usr/local/Ascend')):
        patterns = ('dynamo/torchair', 'lib/torchair', 'python*/site-packages/torchair',
                    'python*/site-packages/torch_npu/dynamo/torchair', '*/*/python/site-packages/torchair')
        for pattern in patterns:
            for p in base.glob(pattern):
                result['torchair_candidates'].append(str(p))
    for name in ('npu_fused_infer_attention_score', 'npu_prompt_flash_attention'):
        result['operators'][name] = {'available': hasattr(torch_npu, name),
            'schema': str(getattr(torch.ops.npu, name)._schemas) if hasattr(torch_npu, name) else None}
    result['cann_versions']={}
    for p in Path('/usr/local/Ascend').glob('*/version.info'):
        if p.is_file(): result['cann_versions'][str(p)]=p.read_text()[:4000]
    for p in Path('/usr/local/Ascend').glob('*/*/version.info'):
        if p.is_file(): result['cann_versions'][str(p)]=p.read_text()[:4000]
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
