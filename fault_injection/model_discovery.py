"""Model identity and structural roles shared by native mechanism adapters."""
from collections.abc import Mapping
import ast
import hashlib
import inspect
import math
from pathlib import Path
import textwrap


def _class(value):
    cls = type(value)
    return cls.__module__ + '.' + cls.__qualname__


def _positive(value):
    return value if type(value) is int and value > 0 else None


def _value(config, name):
    value = config.get(name) if isinstance(config, Mapping) else getattr(config, name, None)
    return value if isinstance(value, str) or type(value) in (int, bool) else None


def _shape(tensor):
    shape = getattr(tensor, 'shape', None)
    return list(shape) if shape is not None and all(type(dim) is int for dim in shape) else None


def model_identity(model, config):
    """Diagnostic architecture identity; never select a provider by model name.

    A checkpoint title (including a Distill-DeepSeek title) is not architecture
    evidence. Keep actual implementation and config families separate so a
    mislabeled or custom implementation cannot inherit a native certificate.
    """
    def family(value):
        if not isinstance(value, str):
            return None
        token = value.lower().replace('-', '_')
        for prefix, result in (('qwen', 'Qwen'), ('glm', 'GLM'), ('chatglm', 'GLM'),
                               ('deepseek', 'DeepSeek'), ('llama', 'Llama')):
            if token.startswith(prefix):
                return result
        return None
    module = type(model).__module__
    leaf = None
    for prefix in ('transformers.models.', 'vllm.model_executor.models.'):
        if module.startswith(prefix):
            leaf = module[len(prefix):].split('.', 1)[0]
            break
    actual = family(leaf)
    configured = family(_value(config, 'model_type'))
    status = ('FAMILY_METADATA_CONFLICT' if actual and configured and actual != configured
              else 'IMPLEMENTATION_FAMILY_OBSERVED' if actual else 'CUSTOM_IMPLEMENTATION_UNVERIFIED')
    source = None
    try:
        path = Path(inspect.getsourcefile(type(model)))
        source = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    except (OSError, TypeError):
        pass
    return dict(status=status, implementation_family=actual, configured_family=configured,
                model_class=_class(model), implementation_module=module,
                implementation_source=source, config_class=_class(config),
                configured_model_type=_value(config, 'model_type'),
                native_provider_verified=False,
                note='Family is diagnostic only; actual operator/ABI/storage gates remain required')


def projection_geometry(module):
    """Return logical dimensions supported by the actual projection structure."""
    import torch

    weight = getattr(module, 'weight', None)
    if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
        return None
    if isinstance(module, torch.nn.Linear):
        dimensions = (_positive(module.in_features), _positive(module.out_features))
        if tuple(weight.shape) == dimensions[::-1]:
            return dimensions
        return None
    # Parallel/quantized modules may store the physical weight transposed.
    # Dimensions go in diagnostics only. Slot proof stays with the adapter.
    method = getattr(getattr(module, 'quant_method', None), 'apply', None)
    inputs = _positive(getattr(module, 'input_size_per_partition', None))
    outputs = _positive(getattr(module, 'output_size_per_partition', None))
    if callable(method) and inputs and outputs:
        return inputs, outputs
    return None


def _pick(children, names):
    matches = [name for name in names if name in children]
    return matches[0] if len(matches) == 1 else None


def _attention(children, dimensions, config, shared):
    q = _pick(children, ('q_proj', 'wq', 'query'))
    k = _pick(children, ('k_proj', 'wk', 'key'))
    v = _pick(children, ('v_proj', 'wv', 'value'))
    o = _pick(children, ('o_proj', 'wo', 'out_proj', 'dense'))
    fused = _pick(children, ('qkv_proj', 'query_key_value', 'wqkv', 'W_pack'))
    if o is None or o not in dimensions:
        return {}
    output_in, hidden = dimensions[o]
    roles = ('ATTENTION_Q_PROJ', 'ATTENTION_K_PROJ', 'ATTENTION_V_PROJ')
    q_heads = _positive(_value(config, 'num_attention_heads'))
    kv_rank = _positive(_value(config, 'kv_lora_rank'))
    rope_dim = _positive(_value(config, 'qk_rope_head_dim'))
    nope_dim = _positive(_value(config, 'qk_nope_head_dim'))
    value_dim = _positive(_value(config, 'v_head_dim'))
    if shared and all((q_heads, kv_rank, rope_dim, nope_dim, value_dim)):
        # An MLA projection may represent Q, K and V with different widths.
        # These relationships identify roles, while the actual split belongs
        # to the observed operator arguments.
        q_rank = _positive(_value(config, 'q_lora_rank'))
        q_width = q_heads * (rope_dim + nope_dim)
        q_name = 'q_b_proj' if q_rank else q
        latent_q_valid = not q_rank or dimensions.get('q_a_proj') == (hidden, q_rank)
        if (latent_q_valid and dimensions.get(q_name) == (q_rank or hidden, q_width)
                and dimensions.get('kv_a_proj_with_mqa') == (hidden, kv_rank + rope_dim)
                and dimensions.get('kv_b_proj') == (kv_rank, q_heads * (nope_dim + value_dim))
                and output_in == q_heads * value_dim):
            result = {q_name: (roles[0],), 'kv_b_proj': roles[1:], o: ('ATTENTION_O_PROJ',),
                      'kv_a_proj_with_mqa': ('ATTENTION_KV_LATENT_PROJ',)}
            if q_rank:
                result['q_a_proj'] = ('ATTENTION_Q_LATENT_PROJ',)
            return result
    if all(name in dimensions for name in (q, k, v)) and fused is None:
        geometries = [dimensions[name] for name in (q, k, v)]
        if (all(inputs == hidden for inputs, _ in geometries)
                and geometries[0][1] == output_in and geometries[1][1] == geometries[2][1]
                and output_in % geometries[1][1] == 0):
            return {**{name: (role,) for name, role in zip((q, k, v), roles)}, o: ('ATTENTION_O_PROJ',)}
    if fused not in dimensions or any(name is not None for name in (q, k, v)):
        return {}
    kv_heads = _positive(_value(config, 'num_key_value_heads'))
    if kv_heads is None and shared:
        kv_heads = q_heads  # absence of GQA metadata only supports the equal-head shape
    if q_heads and kv_heads and q_heads % kv_heads == 0 and output_in % q_heads == 0:
        head_dim = output_in // q_heads
        if dimensions[fused] == (hidden, (q_heads + 2 * kv_heads) * head_dim):
            return {fused: roles, o: ('ATTENTION_O_PROJ',)}
    return {}


def _mlp(children, dimensions):
    down = _pick(children, ('down_proj', 'w2', 'fc2', 'dense_4h_to_h'))
    if down not in dimensions:
        return {}
    intermediate, hidden = dimensions[down]
    up = _pick(children, ('up_proj', 'w3', 'fc1', 'dense_h_to_4h'))
    gate = _pick(children, ('gate_proj', 'w1'))
    fused = _pick(children, ('gate_up_proj', 'w13'))
    normal_width = (hidden, intermediate)
    result = {down: ('MLP_DOWN_PROJ',)}
    if fused in dimensions and up is None and gate is None and dimensions[fused] == (hidden, 2 * intermediate):
        result[fused] = ('MLP_GATE_UP_PROJ',)
    elif fused is None and dimensions.get(up) == normal_width:
        result[up] = ('MLP_UP_PROJ',)
        if gate is not None:
            if dimensions.get(gate) != normal_width:
                return {}
            result[gate] = ('MLP_GATE_PROJ',)
    else:
        return {}
    return result


def _experts_layout(experts):
    import torch

    if isinstance(experts, (torch.nn.ModuleList, torch.nn.ModuleDict)):
        return len(experts), None
    down = _shape(getattr(experts, 'down_proj', None))
    gate_up = _shape(getattr(experts, 'gate_up_proj', None))
    if down and len(down) == 3 and gate_up == [down[0], 2 * down[2], down[1]]:
        if getattr(experts, 'num_experts', None) == down[0]:
            return down[0], down[1]
    return 0, None


def _normalizer(module, normalizer_types):
    if isinstance(module, normalizer_types):
        return 'decoder_block_and_normalizer_type'
    weight = _shape(getattr(module, 'weight', None))
    epsilon = getattr(module, 'variance_epsilon', None)
    if not weight or len(weight) != 1 or type(epsilon) not in (int, float) or not 0 < epsilon < math.inf:
        return None
    try:
        function = inspect.unwrap(module.forward)
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    except (TypeError, OSError, ValueError, SyntaxError):
        return None
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    if {'mean', 'rsqrt'} <= calls or 'rms_norm' in calls:
        return 'decoder_block_and_normalizer_forward'
    return None


def inspect_structure(model, *, attention_paths=(), runner=None):
    """Read object identity and structural role candidates, without executing forward."""
    import torch

    named_modules = model.named_modules
    try:
        aliases_available = 'remove_duplicate' in inspect.signature(named_modules).parameters
    except (TypeError, ValueError):
        aliases_available = False
    modules = dict(named_modules(remove_duplicate=False) if aliases_available else named_modules())
    by_identity = {}
    child_paths = {}
    for path, module in modules.items():
        by_identity.setdefault(id(module), []).append(path)
        if path:
            parent, _, leaf = path.rpartition('.')
            child_paths.setdefault(parent, {})[leaf] = path

    config = getattr(model, 'config', None) or getattr(getattr(model, 'model', None), 'config', None)
    if attention_paths is None:
        from .attention import _resolve
        attention_paths = [path for path, module in modules.items() if _resolve(module)[0] is not None]
    attention_objects = {id(modules[path]) for path in attention_paths if path in modules}
    dimensions = {owner: projection_geometry(modules[paths[0]]) for owner, paths in by_identity.items()}
    roles = {}
    groups = {}

    def add(path, names, evidence):
        roles.setdefault(id(modules[path]), []).append(dict(path=path or '<root>', roles=list(names),
                                                           evidence=evidence))

    for parent, children in child_paths.items():
        local = {name: dimensions[id(modules[path])] for name, path in children.items()
                 if dimensions[id(modules[path])] is not None}
        module = modules.get(parent)
        shared = id(module) in attention_objects
        attention = _attention(children, local, getattr(module, 'config', None) or config, shared)
        mlp = _mlp(children, local)
        if attention or shared:
            groups[parent] = 'ATTENTION'
        elif mlp:
            groups[parent] = 'MLP'
        for name, values in attention.items():
            add(children[name], values, 'attention_projection_geometry')
        for name, values in mlp.items():
            add(children[name], values, 'mlp_projection_geometry')

        experts_path = children.get('experts')
        experts = modules.get(experts_path)
        router = _pick(children, ('gate', 'router', 'gate_proj'))
        # Router remains distinct from a dense MLP gate projection.
        expert_count, expert_width = _experts_layout(experts)
        if router is not None and expert_count > 0 and not mlp:
            router_shape = _shape(getattr(modules[children[router]], 'weight', None))
            router_in, router_out = local.get(router, (None, None))
            if router_shape and len(router_shape) == 2 and router_out is None:
                router_out, router_in = router_shape
            if router_out == expert_count and (expert_width is None or router_in == expert_width):
                groups[parent] = 'MOE'
                add(children[router], ('MLP_GATE',), 'expert_container_and_router_width')
                if expert_width is not None:
                    add(experts_path, ('MLP_EXPERTS',), 'packed_expert_weight_geometry')

    norm_types = (torch.nn.LayerNorm,) + ((torch.nn.RMSNorm,) if hasattr(torch.nn, 'RMSNorm') else ())
    for children in child_paths.values():
        kinds = {groups.get(path) for path in children.values()}
        if 'ATTENTION' not in kinds or not kinds.intersection(('MLP', 'MOE')):
            continue
        for names, role in ((('input_layernorm', 'attention_norm', 'ln1'), 'ATTENTION_NORM'),
                            (('post_attention_layernorm', 'ffn_norm', 'ln2'), 'MLP_NORM')):
            leaf = _pick(children, names)
            if leaf:
                evidence = _normalizer(modules[children[leaf]], norm_types)
                if evidence:
                    add(children[leaf], (role,), evidence)

    inventory = []
    for owner, paths in by_identity.items():
        module = modules[paths[0]]
        hypotheses = roles.get(owner, [])
        role_sets = {tuple(hypothesis['roles']) for hypothesis in hypotheses}
        weight = getattr(module, 'weight', None)
        method = getattr(module, 'quant_method', None)
        inventory.append(dict(
            path=paths[0] or '<root>', aliases=[path or '<root>' for path in paths],
            object_identity=owner, module_type=_class(module),
            weight_shape=_shape(weight), weight_dtype=str(weight.dtype) if isinstance(weight, torch.Tensor) else None,
            projection_dimensions=dimensions[owner], quant_method=_class(method) if method is not None else None,
            canonical_roles=list(next(iter(role_sets))) if len(role_sets) == 1 else [],
            role_status='AMBIGUOUS' if len(role_sets) > 1 else 'STRUCTURAL_CANDIDATE' if role_sets else 'UNRESOLVED',
            role_evidence=hypotheses,
            fused_segments=None, actual_operator_observed=False,
        ))
    architectures = getattr(config, 'architectures', None)
    runner_config = getattr(runner, 'vllm_config', None)
    parallel = getattr(runner_config, 'parallel_config', None)
    quantization = getattr(config, 'quantization_config', None)
    return dict(
        schema='demoncat.model_structure.v1', object_identity=id(model),
        architecture_identity=model_identity(model, config),
        model_type=_value(config, 'model_type'), architectures=list(architectures)
        if isinstance(architectures, (list, tuple)) and all(isinstance(name, str) for name in architectures) else [],
        model_class=_class(model), mro=[cls.__module__ + '.' + cls.__qualname__ for cls in type(model).__mro__],
        attention_backend=_value(config, '_attn_implementation'),
        attention_head_counts={name: _value(config, name) for name in ('num_attention_heads', 'num_key_value_heads', 'head_dim')},
        quantization={name: _value(quantization, name) for name in ('quant_method', 'bits')},
        moe={name: _value(config, name) for name in ('num_experts', 'num_local_experts', 'n_routed_experts', 'num_experts_per_tok')},
        runner=dict(type=_class(runner), tp=_value(parallel, 'tensor_parallel_size'),
                    pp=_value(parallel, 'pipeline_parallel_size')) if runner is not None else None,
        alias_paths_complete=aliases_available, modules=inventory,
    )
