"""Metadata-derived CPU structure fixture, not a runtime site binding or grant.

Dummy scalars deliberately cannot run a model forward. Quantization metadata
cannot prove which fused operator executes, stream ordering, or consumers.
"""
from collections import Counter
from copy import deepcopy
import torch


def inspect_metadata(config, quant):
    if config.get('model_type') != 'deepseek_v41':
        raise ValueError('DEEPSEEK_V41_CONFIG_REQUIRED')
    text = config['text_config']
    if len(text['compress_ratios']) < text['num_hidden_layers']:
        raise ValueError('INCOMPLETE_COMPRESSION_METADATA')
    def group(predicate):
        values = {k: v for k, v in quant.items() if predicate(k) and isinstance(v, str)}
        return dict(count=len(values), types=dict(Counter(values.values())))
    return dict(model_type=config['model_type'], layers=text['num_hidden_layers'],
        heads=text['num_attention_heads'], head_dim=text['head_dim'],
        experts=text['n_routed_experts'], top_k=text['num_experts_per_tok'],
        vision=group(lambda k: k.startswith('vision.') or k in ('image_start','image_end','image_newline')),
        main_experts=group(lambda k: k.startswith('layers.') and '.ffn.experts.' in k),
        main_wq_b=group(lambda k: k.startswith('layers.') and '.attn.wq_b.' in k),
        mtp_wq_b=group(lambda k: k.startswith('mtp.') and '.attn.wq_b.' in k),
        norm=group(lambda k: 'norm' in k), embed=group(lambda k: 'embed' in k),
        exceptions=[dict(path=k,actual=v) for k,v in quant.items()
                    if (('wq_b' in k and v != 'W8A8_DYNAMIC')
                        or ('embed' in k and isinstance(v,str) and v != 'FLOAT'))],
        qualification_granted=False)


class DummyLeaf(torch.nn.Module):
    def __init__(self, actual_path, quant_types):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.actual_path = actual_path
        self.quant_types = tuple(sorted(quant_types))

    def forward(self, *args, **kwargs):
        raise RuntimeError('STRUCTURE_FIXTURE_IS_NOT_MODEL_INFERENCE')


class StructureFixture(torch.nn.Module):
    """Preserve real layer IDs; scalar dummy weights, never materialize experts."""
    def __init__(self, config, quant, layer_ids=(0, 2, 20)):
        super().__init__()
        inspect_metadata(config, quant)
        self.config = deepcopy(config)
        self.original_layer_ids = tuple(layer_ids)
        count = config['text_config']['num_hidden_layers']
        if not layer_ids or len(set(layer_ids)) != len(layer_ids) or any(type(i)!=int or i<0 or i>=count for i in layer_ids):
            raise ValueError('INVALID_ORIGINAL_LAYER_IDS')
        self.layers = torch.nn.ModuleDict()
        for i in layer_ids:
            layer = torch.nn.Module()
            layer.compression_ratio = config['text_config']['compress_ratios'][i]
            # Actual paths are discovered from the downloaded quantization keys.
            groups = {}
            for key, value in quant.items():
                if key.startswith(f'layers.{i}.') and isinstance(value, str):
                    path = key.rsplit('.', 1)[0]
                    if '.experts.' in path:
                        # Keep one real expert for structural tests, not all 384.
                        if '.experts.0.' not in path:
                            continue
                    groups.setdefault(path, set()).add(value)
            layer.leaves = torch.nn.ModuleList([DummyLeaf(path, types) for path, types in sorted(groups.items())])
            self.layers[str(i)] = layer

    def forward(self, *args, **kwargs):
        raise RuntimeError('STRUCTURE_FIXTURE_IS_NOT_MODEL_INFERENCE')


def site_requirements():
    """Operator names are investigation targets, not observations from config."""
    return {
        '28': ['SparseFlashMla call', 'LightningIndexer actual unique key indices', 'causal sequence bounds', 'consumer'],
        '8': ['MoeDistributeDispatchV2/CombineV2 calls', 'dynamic token/request map', 'actual expert placement'],
        'Batch': ['separate DynamicQuant call', 'at least two actual requests', 'original QMM consumption; exclude fused GroupedMatmul quant'],
        '18': ['RmsNorm/RmsNormCast call', 'actual dtype', 'consumer'],
        '22': ['InplacePartialRotaryMul call', 'position dimensions', 'rotary range and consumer'],
        '39': ['MatMulV2/QuantMatmulWeightNz output', 'actual BF16 ND', 'consumer'],
        '41': ['actual MLA cache layout', 'live object byte interval', 'writer/read stream order', 'safe restore boundary'],
    }
