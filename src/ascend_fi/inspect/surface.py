"""Module Surface Adapter: registry metadata, aliases and nested tensor paths."""
from __future__ import annotations

import re


def type_name(value):
    return type(value).__module__ + '.' + type(value).__qualname__


def stored(value, name, default=None):
    try:
        return vars(value).get(name, default)
    except TypeError:
        return default


def layer_index(name, module):
    explicit = stored(module, 'layer_idx')
    if type(explicit) is int and explicit >= 0:
        return explicit, 'stored:layer_idx'
    match = re.search(r'(?:^|\.)(?:layers|h|blocks)\.(\d+)(?:\.|$)', name)
    return (int(match[1]), 'numeric_path_component') if match else (None, 'UNKNOWN')


def natural_key(name):
    return tuple((0, int(s)) if s.isdigit() else (1, s) for s in re.split(r'(\d+)', name))


def logical_role(name, cls):
    roles = {'qkv_proj': 'ATTN_QKV_FUSED', 'q_proj': 'ATTN_Q', 'k_proj': 'ATTN_K',
             'v_proj': 'ATTN_V', 'o_proj': 'ATTN_OUTPUT', 'gate_up_proj': 'MLP_GATE_UP_FUSED',
             'down_proj': 'MLP_DOWN', 'up_proj': 'MLP_UP', 'gate_proj': 'MLP_GATE'}
    if 'fusedmoe' in cls.lower() or name.split('.')[-1] == 'experts':
        return 'ROUTED_EXPERTS', 'outer_fused_boundary; internal projections UNKNOWN'
    leaf = name.split('.')[-1]
    if leaf in roles:
        return roles[leaf], 'name_alias:' + leaf
    return 'UNKNOWN', 'no verified role adapter'


def is_tensor(value):
    import torch
    return isinstance(value, torch.Tensor)


def tensor_metadata(value):
    """Documented Tensor metadata only; no data_ptr, CPU copy, item or hash."""
    result = {}
    getters = {'dtype': lambda: str(value.dtype), 'shape': lambda: list(value.shape),
               'stride': lambda: list(value.stride()), 'device': lambda: str(value.device),
               'numel': lambda: value.numel(), 'layout': lambda: str(value.layout),
               'is_meta': lambda: bool(value.is_meta), 'floating': lambda: value.is_floating_point()}
    for key, fn in getters.items():
        try:
            result[key] = fn()
        except Exception:
            result[key] = None
    result['metadata_status'] = 'COMPLETE' if all(result[k] is not None for k in
                                                  ('dtype', 'shape', 'stride', 'device', 'numel', 'layout')) else 'PARTIAL'
    return result


def tensor_paths(args, kwargs, predicate=is_tensor, *, max_nodes=256, max_depth=8):
    """JSON selector segments; no arbitrary member traversal or tensor retention."""
    records, visited, count = [], set(), 0
    def visit(value, path, depth):
        nonlocal count
        count += 1
        if count > max_nodes or depth > max_depth:
            raise ValueError('INPUT_METADATA_BUDGET_EXCEEDED')
        if predicate(value):
            records.append({'path': path, 'metadata': tensor_metadata(value)})
        elif type(value) in (tuple, list, dict):
            if id(value) in visited:
                return
            visited.add(id(value))
            items = value.items() if type(value) is dict else enumerate(value)
            for key, child in items:
                if type(key) in (int, str):
                    visit(child, path + [key], depth + 1)
            visited.remove(id(value))
    visit(args, ['args'], 0)
    visit(kwargs, ['kwargs'], 0)
    return records


class ModuleSurfaceAdapter:
    def __init__(self, model, ranks, predicate=is_tensor, max_modules=10000, max_parameters=50000):
        self.model, self.ranks, self.predicate = model, dict(ranks), predicate
        self.modules, self.records = {}, []
        module_aliases, tensor_aliases, params = {}, {}, 0
        def walk(module, name, ancestors):
            nonlocal params
            if len(self.modules) >= max_modules or len(ancestors) > 64:
                raise ValueError('MODULE_ENUMERATION_BUDGET_EXCEEDED')
            if id(module) in ancestors:
                return
            self.modules[name] = module
            module_aliases.setdefault(id(module), []).append(name)
            cls = type_name(module)
            layer, source = layer_index(name, module)
            role, reason = logical_role(name, cls)
            quant = stored(module, 'quant_method')
            record = {'module_name': name, 'module_class': cls, 'layer_index': layer,
                      'layer_source': source, 'logical_role': role, 'role_evidence': reason,
                      'ranks': dict(ranks), 'quant_method': type_name(quant) if quant is not None else None,
                      'parameters': [], 'observed': False, 'aliases': [],
                      'internal_operator_mapping': 'UNKNOWN'}
            tp = stored(module, 'tp_rank')
            if type(tp) is int:
                record['ranks']['tp'] = {'value': tp, 'source': 'stored_module.tp_rank'}
            for registry, kind in (('_parameters', 'parameter'), ('_buffers', 'buffer')):
                for field, value in stored(module, registry, {}).items():
                    params += 1
                    if params > max_parameters:
                        raise ValueError('PARAMETER_ENUMERATION_BUDGET_EXCEEDED')
                    meta = tensor_metadata(value) if self.predicate(value) else {'metadata_status': 'NONE_OR_UNAVAILABLE'}
                    item = {'name': field, 'kind': kind, 'metadata': meta, 'aliases': []}
                    record['parameters'].append(item)
                    if value is not None:
                        tensor_aliases.setdefault(id(value), []).append((name + '.' + field, item))
            self.records.append(record)
            for key, child in stored(module, '_modules', {}).items():
                if child is not None:
                    walk(child, name + '.' + key if name else key, ancestors | {id(module)})
        walk(model, '', set())
        for r in self.records:
            r['aliases'] = module_aliases[id(self.modules[r['module_name']])]
        for items in tensor_aliases.values():
            for _, item in items:
                item['aliases'] = [name for name, _ in items]
                item['alias_scope'] = 'OBJECT_IDENTITY; storage-view alias UNKNOWN'
        self.records.sort(key=lambda r: natural_key(r['module_name']))
        self.by_name = {r['module_name']: r for r in self.records}
        self.owned_layers = sorted({r['layer_index'] for r in self.records if r['layer_index'] is not None})

    def selected(self, names):
        if not names or len(names) != len(set(names)) or any(n not in self.modules for n in names):
            raise ValueError('SELECT_EXACT_EXISTING_MODULE_NAMES')
        ids = {id(self.modules[n]) for n in names}
        expanded = {n for n, m in self.modules.items() if id(m) in ids}
        if expanded != set(names):
            raise ValueError('SHARED_MODULE_ALIAS_SELECTION_INCOMPLETE')
        return [(n, self.modules[n]) for n in names if n == self.by_name[n]['aliases'][0]]
