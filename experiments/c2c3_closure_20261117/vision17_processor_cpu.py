"""Device-free processor diagnostic; no model or explicit NPU computation.

The current vLLM image still imports torch_npu from Transformers even with
device-backend autoload disabled. Without driver libraries that import fails.
Even when libraries are present, import may attempt runtime device discovery;
retain those warnings and never describe this as zero runtime API calls.
No accelerator device nodes are supplied. Passing this diagnostic would still
not qualify model inference or any fault.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if any(Path('/dev').glob('davinci*')):
        raise RuntimeError('DEVICE_FREE_CONTAINER_REQUIRED')
    if os.environ.get('TORCH_DEVICE_BACKEND_AUTOLOAD') != '0':
        raise RuntimeError('CPU_ONLY_BACKEND_IMPORT_REQUIRED')
    if sha(args.metadata) != '2fcf3612194e166fae921a35892bce9f424b6095a4edf24fc1e5ebfbf5eb7395':
        raise RuntimeError('PINNED_OFFICIAL_METADATA_REQUIRED')
    sys.path[:0] = [str(args.source), str(args.source/'src')]
    from experiments.c2c3_closure_20261117.natural_vision_input import (
        require_geometry, IMAGE_SHA256, SOURCE_SHA256, PROMPT,
    )
    metadata = json.loads(args.metadata.read_bytes())
    entries = {item['rfilename']: item for item in metadata['siblings']}
    files = {}
    for path in args.model.iterdir():
        if path.suffix not in ('.json', '.txt') or path.name == 'download_receipt.json':
            continue
        data = path.read_bytes()
        blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        if entries[path.name]['blobId'] != blob:
            raise RuntimeError('PINNED_CONFIG_CHANGED:' + path.name)
        files[path.name] = sha(path)
    if sha(args.image) != IMAGE_SHA256:
        raise RuntimeError('PINNED_EXISTING_IMAGE_REQUIRED')
    geometry = require_geometry(json.loads((args.model/'config.json').read_bytes()))
    import torch
    import transformers
    from PIL import Image
    from transformers import AutoProcessor
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    if sha(inspect.getfile(modeling_qwen3_vl)) != SOURCE_SHA256:
        raise RuntimeError('ACTUAL_MODEL_SOURCE_CHANGED')
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    with Image.open(args.image) as opened:
        image = opened.convert('RGB')
    messages = [dict(role='user', content=[dict(type='image'), dict(type='text', text=PROMPT)])]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[image], return_tensors='pt')
    fields = {}
    for key, value in encoded.items():
        if not isinstance(value, torch.Tensor):
            raise RuntimeError('UNEXPECTED_PROCESSOR_NON_TENSOR_FIELD')
        if str(value.device) != 'cpu':
            raise RuntimeError('CPU_PROCESSOR_OUTPUT_REQUIRED')
        raw = value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        fields[key] = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(raw).hexdigest())
    receipt = dict(schema='demoncat.vision17.cpu_processor.v1',
        status='REAL_CPU_PROCESSOR_PREFLIGHT_NOT_MODEL_OR_NPU_ACCEPTANCE',
        transformers=transformers.__version__, config_files=files, geometry=geometry,
        model_source_sha256=SOURCE_SHA256, processor_source_sha256=sha(inspect.getfile(type(processor))),
        image_sha256=IMAGE_SHA256, image_size=list(image.size), image_mode=image.mode,
        prompt=PROMPT, processor_overrides={}, outputs=fields,
        explicit_npu_compute_invocations=0, runtime_discovery_may_be_attempted=True,
        cpu_environment={'TORCH_DEVICE_BACKEND_AUTOLOAD': '0'},
        model_constructed=False, weights_loaded=False, raw_tensors_saved=False, c2=False, c3=False)
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(receipt, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps(dict(status=receipt['status'], outputs=fields,
        explicit_npu_compute_invocations=0, runtime_discovery_may_be_attempted=True)))


if __name__ == '__main__':
    main()
