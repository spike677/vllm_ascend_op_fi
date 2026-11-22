"""FP32 online attention for one real query row, GQA or absorbed MLA.

Sites own causal/sparse key selection, cache decoding and the real consumer.
This numerical engine does not infer sequence boundaries from tensor shapes.
The replay variant deliberately counts selected blocks twice in the denominator.
"""
import math
import torch


def recompute_attention_row(query, keys, values, *, scale, block_size,
                            key_indices=None, replay_blocks=(), sinks=None):
    """Shapes Q[Hq,D], K[N,Hkv,D], V[N,Hkv,Dv]; sparse indices are unique.

MLA sites must supply the absorbed Q/K and latent V with the exact model scale.
No hidden assumption that QK and V head dimensions are equal.
"""
    if (query.ndim != 2 or keys.ndim != 3 or values.ndim != 3
            or keys.shape[:2] != values.shape[:2]
            or query.shape[-1] != keys.shape[-1]
            or min(*query.shape, *keys.shape, *values.shape) <= 0
            or query.shape[0] % keys.shape[1]):
        raise ValueError('ATTENTION_LAYOUT_NOT_SUPPORTED')
    if (not all(t.is_floating_point() and t.device == query.device for t in (query, keys, values))
            or type(block_size) is not int or block_size <= 0
            or not isinstance(scale, (float, int)) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('EXPLICIT_ATTENTION_NUMERIC_CONTRACT_REQUIRED')
    indices = tuple(range(keys.shape[0])) if key_indices is None else tuple(key_indices)
    if (not indices or any(type(i) is not int or i < 0 or i >= keys.shape[0] for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('UNIQUE_NONEMPTY_LEGAL_KEY_INDICES_REQUIRED')
    replay = tuple(replay_blocks)
    blocks = (len(indices) + block_size - 1) // block_size
    if (len(set(replay)) != len(replay)
            or any(type(i) is not int or not 0 <= i < blocks for i in replay)):
        raise ValueError('REPLAY_BLOCK_OUT_OF_RANGE_OR_DUPLICATE')
    if sinks is not None:
        if (not isinstance(sinks, torch.Tensor) or sinks.shape != (query.shape[0],)
                or not sinks.is_floating_point() or sinks.device != query.device
                or not bool(torch.isfinite(sinks).all())):
            raise ValueError('FINITE_PER_HEAD_SINKS_REQUIRED')
    q = query.float()
    groups = query.shape[0] // keys.shape[1]
    maximum = torch.full((query.shape[0],), -torch.inf, device=q.device)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros((query.shape[0], values.shape[-1]), dtype=torch.float32, device=q.device)
    for number, start in enumerate(range(0, len(indices), block_size)):
        ids = torch.tensor(indices[start:start + block_size], device=q.device, dtype=torch.long)
        k = keys.index_select(0, ids).float().repeat_interleave(groups, dim=1)
        v = values.index_select(0, ids).float().repeat_interleave(groups, dim=1)
        scores = torch.einsum('hd,khd->hk', q, k) * scale
        updated = torch.maximum(maximum, scores.max(dim=-1).values)
        rescale = torch.exp(maximum - updated)
        probability = torch.exp(scores - updated[:, None])
        numerator = numerator * rescale[:, None] + torch.einsum('hk,khd->hd', probability, v)
        denominator = denominator * rescale + probability.sum(dim=-1) * (2 if number in replay else 1)
        maximum = updated
    if sinks is not None:
        # A caller-owned sink is one zero-value softmax term, never a replayed key.
        updated = torch.maximum(maximum, sinks.float())
        rescale = torch.exp(maximum - updated)
        numerator = numerator * rescale[:, None]
        denominator = denominator * rescale + torch.exp(sinks.float() - updated)
    return (numerator / denominator[:, None]).to(query.dtype)
