"""Read-only, bounded model identity. No served-name architecture inference."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path

from .surface import stored, type_name, ModuleSurfaceAdapter


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def field(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    value = stored(obj, name, default)
    if value is not default:
        return value
    # HF model_type is often a class attribute. Never execute a property getter.
    value = inspect.getattr_static(obj, name, default)
    return default if isinstance(value, property) else value


CONFIG_FIELDS = ('model_type', 'architectures', 'num_hidden_layers', 'hidden_size',
                 'intermediate_size', 'num_attention_heads', 'num_key_value_heads',
                 'n_routed_experts', 'n_shared_experts', 'num_experts',
                 'first_k_dense_replace', 'moe_layer_freq', 'q_lora_rank', 'kv_lora_rank')


def config_view(config):
    # Only the language backbone's documented config, never arbitrary properties.
    nested = field(config, 'text_config')
    source = 'text_config' if nested is not None else 'config'
    config = nested if nested is not None else config
    return {k: field(config, k) for k in CONFIG_FIELDS
            if type(field(config, k)) in (str, int, float, bool, list)}, source


def implementation_record(cls):
    try:
        path = Path(inspect.getfile(cls))
        if path.stat().st_size > 4 * 1024**2:
            raise ValueError('SOURCE_BUDGET')
        return {'class': cls.__module__ + '.' + cls.__qualname__,
                'source_file': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    except (OSError, TypeError, ValueError):
        return {'class': cls.__module__ + '.' + cls.__qualname__,
                'source_file': None, 'source_sha256': None}


def module_contract(module, source_cache=None):
    """Explicit signature/partition metadata; no weight reads or tensor mutation."""
    try:
        signature = inspect.signature(type(module).forward)
        args = [{'name': p.name, 'kind': p.kind.name} for p in signature.parameters.values()
                if p.name != 'self']
    except (TypeError, ValueError, AttributeError):
        args = []
    feature_size = (field(module, 'input_size') if field(module, 'input_is_parallel') is False
                    else field(module, 'input_size_per_partition', field(module, 'input_size')))
    forward = getattr(type(module), 'forward', None)
    implementation = None
    if forward is not None and source_cache is not None:
        if forward not in source_cache:
            source_cache[forward] = implementation_record(forward)
        implementation = source_cache[forward]
    return {'forward_parameters': args,
            'forward_implementation': implementation,
            'class_mro': [c.__module__ + '.' + c.__qualname__ for c in type(module).__mro__],
            'input_feature_size': feature_size,
            'input_is_parallel': field(module, 'input_is_parallel'),
            'packed_mapping': getattr(type(module), 'packed_modules_mapping', None)}


def annotate_call(module, args, kwargs, parameters):
    """Augment metadata with signature binding and transient object aliases only."""
    signature = module_contract(module)['forward_parameters']
    positional = [p['name'] for p in signature if p['kind'] in
                  ('POSITIONAL_ONLY', 'POSITIONAL_OR_KEYWORD')]
    identities = {}
    for parameter in parameters:
        path = parameter['path']
        parameter['semantic_argument'] = None
        parameter['object_alias_paths'] = []
        if len(path) != 2:
            continue
        if path[0] == 'args':
            value = args[path[1]]
            if path[1] < len(positional):
                parameter['semantic_argument'] = positional[path[1]]
        else:
            value = kwargs[path[1]]
            if path[1] in {p['name'] for p in signature if p['kind'] != 'VAR_KEYWORD'}:
                parameter['semantic_argument'] = path[1]
        identities.setdefault(id(value), []).append(parameter)
        try:
            parameter['metadata']['contiguous'] = value.is_contiguous()
        except (AttributeError, RuntimeError):
            parameter['metadata']['contiguous'] = None
    for group in identities.values():
        for p in group:
            p['object_alias_paths'] = [q['path'] for q in group if q is not p]
            p['alias_scope'] = 'OBJECT_IDENTITY; storage-view alias not inferred'
    return parameters


def describe_model(session, worker=None):
    model = session.surface.model
    current_surface = ModuleSurfaceAdapter(model, session.surface.ranks, session.surface.predicate)
    if (set(current_surface.modules) != set(session.surface.modules) or
            any(m is not session.surface.modules[n] for n, m in current_surface.modules.items())):
        raise ValueError('MODEL_STRUCTURE_REPLACED; start a fresh Inspect epoch')
    runner = stored(worker, 'model_runner') or stored(stored(worker, 'worker'), 'model_runner')
    vc = stored(worker, 'vllm_config') or stored(runner, 'vllm_config')
    mc = field(vc, 'model_config') or stored(worker, 'model_config') or stored(runner, 'model_config')
    parallel = field(vc, 'parallel_config') or stored(worker, 'parallel_config')
    config = field(mc, 'hf_text_config') or field(mc, 'hf_config')
    source = 'worker.model_config'
    if config is None:
        config = stored(model, 'config')
        if config is None:
            config = stored(stored(model, '_modules', {}).get('model'), 'config')
        source = 'model.config'
    view, subsource = config_view(config)
    classes = {type(m) for m in session.surface.modules.values()}
    for m in session.surface.modules.values():
        for attr in ('quant_method', 'custom_op'):
            implementation = stored(m, attr)
            if implementation is not None:
                classes.add(type(implementation))
    sources = sorted((implementation_record(c) for c in classes), key=lambda r: r['class'])
    reference = sys.modules.get('fake_requant_reference_extension')
    reference_class = getattr(reference, 'FakeRequantReferenceExtension', None)
    backend = {'reference': implementation_record(reference_class) if reference_class else None,
               'worker_extensions': [implementation_record(c) for c in type(worker).__mro__
                   if c.__module__.startswith('ascend_fi.inspect')] if worker else []}
    rows = []
    source_cache = {}
    for record in current_surface.records:
        name = record['module_name']
        parent = name.rpartition('.')[0]
        rows.append({**record, **module_contract(session.surface.modules[name], source_cache),
                     'parent_name': parent,
                     'parent_class': session.surface.by_name.get(parent, {}).get('module_class')})
    value = {'schema': 'MODEL_DESCRIPTOR_V1', 'origin': 'LIVE', 'service_epoch': session.epoch,
             'runtime_fingerprint': session.fingerprint, 'model_class': type_name(model),
             'config': view, 'config_source': source + '.' + subsource,
             'runtime': session.runtime, 'ranks': session.surface.ranks,
             'backend': backend,
             'topology': {k: field(parallel, k) for k in
                          ('tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size',
                           'enable_expert_parallel')},
             'execution': {'enforce_eager': field(mc, 'enforce_eager'),
                           'phase_mapping': 'UNKNOWN', 'scheduler_step_coverage': 'UNPROVEN'},
             'implementations': sources,
             'enumeration': {'complete': True, 'module_count': len(rows), 'truncated': False},
             'owned_layers': session.surface.owned_layers,
             'packed_mapping': getattr(type(model), 'packed_modules_mapping', None)}
    # Structure hash excludes transient observed flags, includes actual classes and dimensions.
    value['structure_sha256'] = digest([{k: r.get(k) for k in
        ('module_name', 'module_class', 'parent_class', 'layer_index', 'aliases',
         'input_feature_size', 'forward_parameters', 'forward_implementation', 'quant_method')} for r in rows])
    value['model_fingerprint'] = digest(value)
    return value, rows
