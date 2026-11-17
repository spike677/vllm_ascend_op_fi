"""Pinned real image prefill for the existing Qwen3-VL implementation.

This loads the complete pretrained model and its actual image processor. It
does not construct a patch module, change convolution geometry, or manufacture
image patches. The scope is MODEL_LOCAL, never a standard Worker request.
"""
import inspect
import json
from pathlib import Path

from experiments.c2c3_closure_20261117.natural_tensor_census import sha_file

MODEL_ID = 'Qwen/Qwen3-VL-2B-Instruct'
MODEL_REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
WEIGHTS_SHA256 = '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'
WEIGHTS_BYTES = 4255140312
SOURCE_SHA256 = 'dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da'
IMAGE_SHA256 = '077e3d965090c9028c69c00931675f42e1acc815c6eb450ab291b3b72d211a8e'
PROMPT = 'Describe this image in one sentence.'


def require_geometry(config):
    vision = config.get('vision_config', {})
    expected = dict(patch_size=16, temporal_patch_size=2, in_channels=3,
                    hidden_size=1024, spatial_merge_size=2)
    if (config.get('model_type') != 'qwen3_vl' or
            config.get('architectures') != ['Qwen3VLForConditionalGeneration'] or
            any(vision.get(key) != value for key, value in expected.items())):
        raise RuntimeError('ORIGINAL_PRETRAINED_QWEN3VL_PATCH16_GEOMETRY_REQUIRED')
    return expected


def load_vision(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    if args.image is None:
        raise ValueError('Existing pinned image is required')
    source = Path(inspect.getfile(modeling_qwen3_vl))
    if sha_file(source) != SOURCE_SHA256:
        raise RuntimeError('UNREVIEWED_QWEN3VL_ACTUAL_MODEL_SOURCE')
    weights = args.model / 'model.safetensors'
    if weights.stat().st_size != WEIGHTS_BYTES or sha_file(weights) != WEIGHTS_SHA256:
        raise RuntimeError('PINNED_COMPLETE_QWEN3VL_PRETRAINED_WEIGHTS_REQUIRED')
    if sha_file(args.image) != IMAGE_SHA256:
        raise RuntimeError('PINNED_EXISTING_NATURAL_IMAGE_REQUIRED')
    config = json.loads((args.model / 'config.json').read_text())
    geometry = require_geometry(config)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
                                             trust_remote_code=False)
    with Image.open(args.image) as opened:
        image = opened.convert('RGB')
    # Normal public processor path with its downloaded defaults. No manual
    # resizing, patch extraction, temporal duplication, or geometry override.
    messages = [dict(role='user', content=[dict(type='image'), dict(type='text', text=PROMPT)])]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    encoded = processor(text=[text], images=[image], return_tensors='pt')
    provenance = dict(scope='NATURAL_PRETRAINED_QWEN3VL_IMAGE_PREFILL', standard_vllm_worker=False,
        model_id=MODEL_ID, public_revision=MODEL_REVISION, weight_sha256=WEIGHTS_SHA256,
        weights_bytes=WEIGHTS_BYTES, source_file=str(source), source_sha256=SOURCE_SHA256,
        config_sha256=sha_file(args.model/'config.json'), image_sha256=IMAGE_SHA256,
        image_size=list(image.size), image_mode=image.mode, prompt=PROMPT,
        processor_type=type(processor).__module__+'.'+type(processor).__qualname__,
        processor_source_sha256=sha_file(inspect.getfile(type(processor))),
        pretrained_vision_geometry=geometry, geometry_modified=False,
        processor_overrides={}, tensor_input_shapes={key:list(value.shape) for key,value in encoded.items()
            if isinstance(value,torch.Tensor)})
    network = Qwen3VLForConditionalGeneration.from_pretrained(args.model, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager').eval().npu()
    class ImagePrefill(torch.nn.Module):
        def __init__(self, actual):
            super().__init__(); self.actual = actual
        def forward(self, payload):
            return self.actual(**payload, use_cache=False, return_dict=False)[0]
    payload = {key: value.npu() if isinstance(value,torch.Tensor) else value for key,value in encoded.items()}
    return ImagePrefill(network).eval(), (payload,), provenance
