"""Bounded byte evidence from an actual tensor; no qualification from hashes."""
import hashlib
import json
from pathlib import Path
import os

import torch
from torch.utils._pytree import tree_flatten

from .postwrite import PostwriteRefused


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def process_start():
    path = Path('/proc/self/stat')
    if path.is_file():return path.read_text().rsplit(')', 1)[1].split()[19]
    # CPU contract lifetime only; never accepted as a natural NPU Worker.
    import time
    return f'cpu-contract:{os.getpid()}:{time.monotonic_ns()}'


def snapshot(transport, tensor, offset=0, elements=None):
    count = tensor.numel() if elements is None else elements
    if getattr(transport, 'transport_kind', None) == 'cpu_test':
        if tensor.device.type != 'cpu':raise PostwriteRefused('CPU_EVIDENCE_DEVICE_MISMATCH')
        copied = tensor.detach().contiguous().view(-1).narrow(0, offset, count)
        return copied.contiguous().view(torch.uint8).numpy().tobytes()
    return transport.snapshot(tensor, offset, count)


def tensor_digest(transport, tensor, offset=0, elements=None):
    return digest(dict(shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype),
        offset=offset, elements=tensor.numel() if elements is None else elements,
        bytes_sha256=hashlib.sha256(snapshot(transport, tensor, offset, elements)).hexdigest()))


def output_digest(transport, output):
    leaves, spec = tree_flatten(output)
    values = []
    for item in leaves:
        if isinstance(item, torch.Tensor):values.append(('tensor', tensor_digest(transport, item)))
        elif isinstance(item, (str, int, float, bool, type(None))):values.append(('value', item))
        else:raise PostwriteRefused('POSTWRITE_OUTPUT_EVIDENCE_TYPE_UNADAPTED')
    return digest(dict(spec=str(spec), values=values))
