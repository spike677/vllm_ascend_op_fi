"""Read-only natural prefill boundary for the two CANN 9.1 BNSD FIA strategies.

This module observes the existing API and returns its object unchanged. It does
not register a Provider, load a native library, set a policy environment, repack
KV or certify a Worker. The TND/page16 adapter for fault33 is a different route.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import threading

from .call_binding import ForwardBinding
from .serving_context import require, require_serial_eager, snapshot_fresh_prefill, tensor_metadata

MECHANISMS = ('28.fia.legal_tiling', '28.fia.denominator_replay')
ORIGINAL_CANN91_CONTRACT_SHA = '4b709903ea4ba1a8dc60a0608c4f443df7cc6e05669167391bec43b27077a91d'
APIS = ('npu_fused_infer_attention_score', 'npu_fused_infer_attention_score_v2',
        'npu_fusion_attention', 'npu_incre_flash_attention', '_npu_paged_attention')
OPTIONALS = ('pse_shift', 'dequant_scale1', 'quant_scale1', 'dequant_scale2', 'quant_scale2',
    'quant_offset2', 'antiquant_scale', 'antiquant_offset', 'block_table',
    'query_padding_size', 'kv_padding_size', 'key_antiquant_scale', 'key_antiquant_offset',
    'value_antiquant_scale', 'value_antiquant_offset', 'key_shared_prefix', 'value_shared_prefix',
    'actual_shared_prefix_len', 'query_rope', 'key_rope', 'learnable_sink')
SCALARS = {'num_heads', 'num_key_value_heads', 'scale', 'input_layout', 'pre_tokens',
           'next_tokens', 'sparse_mode', 'inner_precise', 'block_size', 'softmax_lse_flag'}
ALLOWED = set(OPTIONALS) | SCALARS | {'query', 'key', 'value', 'atten_mask', 'actual_seq_lengths', 'actual_seq_lengths_kv'}
_ACTIVE = None


def strategy_request(mechanism, arm):
    """An explicit request for later native work, never evidence of execution."""
    require(mechanism in MECHANISMS and arm in ('baseline', 'fault', 'off'), 'FIA28_UNKNOWN_MECHANISM_OR_ARM')
    if arm == 'off':mode, m, key = 'off', 512, 103000000000000000
    elif mechanism == MECHANISMS[0] and arm == 'baseline':mode, m, key = 'coupled_base512', 512, 109000000000000000
    elif mechanism == MECHANISMS[1] and arm == 'fault':mode, m, key = 'coupled_replay_m128', 128, 119000000000400000
    else:mode, m, key = 'coupled_m128', 128, 109000000000400000
    return dict(mechanism=mechanism, arm=arm, requested_mode=mode, requested_m=m, requested_s2=m,
                requested_key=key, native_scope_entered=False, worker_qualified=False)


def bind_actual_api(args, kwargs):
    require(len(args) <= 3, 'FIA28_POSITIONAL_OPTIONALS_UNADAPTED')
    actual = dict(kwargs)
    for name, value in zip(('query', 'key', 'value'), args):
        require(name not in actual, 'FIA28_DUPLICATE_ACTUAL_ARGUMENT')
        actual[name] = value
    return actual


def describe_actual_inputs(actual):
    """Retain geometry of rejected routes too, without copying their values."""
    result = {}
    for name in ('query', 'key', 'value', 'atten_mask', 'block_table'):
        value = actual.get(name)
        if value is None:
            result[name] = None
        elif isinstance(value, (list, tuple)):
            require(len(value) <= 8, 'FIA28_TENSOR_LIST_DESCRIPTION_BOUND')
            result[name] = [tensor_metadata(tensor) for tensor in value]
        else:
            result[name] = tensor_metadata(value)
    return result


def inspect_prefill_input(operator, actual, *, metadata, scheduler, allow_cpu_test=False):
    """Match only original BNSD input metadata; no device values are read.

    A match remains a candidate. A valid causal mask's *values*, original QKV,
    native core budget, native route and launch still require separate evidence.
    The existing scheduler reader supports one fresh request, so B8 remains a
    standalone-only contract until a genuine batch reader is implemented.
    """
    require(operator == 'npu_fused_infer_attention_score', 'FIA28_ACTUAL_API_UNADAPTED:' + operator)
    require(actual.get('input_layout') == 'BNSD', 'FIA28_BNSD_PREFILL_REQUIRED_TND_PAGE16_IS_SEPARATE')
    require(not (set(actual) - ALLOWED), 'FIA28_UNKNOWN_OPTIONAL_ARGUMENT')
    require(scheduler.get('status') == 'READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE'
            and scheduler.get('num_requests') == 1 and scheduler.get('num_computed_tokens') == 0
            and scheduler.get('num_prompt_tokens') == scheduler.get('num_scheduled_tokens') == 2048,
            'FIA28_BOUND_SINGLE_FRESH_FULL_PREFILL_REQUIRED')
    require(metadata.num_prefills == 1 and metadata.num_decodes == 0 and metadata.num_actual_tokens == 2048
            and metadata.causal is True and list(metadata.seq_lens_list) == [2048]
            and list(metadata.actual_seq_lengths_q) == [2048], 'FIA28_LIVE_PREFILL_METADATA_DIFFERS')
    require(actual.get('actual_seq_lengths') == [2048] and actual.get('actual_seq_lengths_kv') == [2048],
            'FIA28_ACTUAL_NONACCUMULATED_LENGTHS_REQUIRED')
    require(actual.get('num_heads') == 16 and actual.get('num_key_value_heads') == 8
            and type(actual.get('scale')) in (int, float)
            and math.isclose(float(actual['scale']), 128**-.5, rel_tol=0, abs_tol=1e-12)
            and actual.get('inner_precise', 0) == 0, 'FIA28_HEAD_SCALE_PRECISION_DIFFERS')
    require(actual.get('sparse_mode') == 2 and actual.get('pre_tokens') == 2147483647
            and actual.get('next_tokens') == 0 and actual.get('block_size', 0) == 0
            and actual.get('softmax_lse_flag', False) is False, 'FIA28_CAUSAL_NONPAGED_OPTIONS_DIFFER')
    for name in OPTIONALS:
        require(actual.get(name) is None, 'FIA28_OPTIONAL_FEATURE_UNADAPTED:' + name)
    tensors = {}
    for name, heads in (('query', 16), ('key', 8), ('value', 8)):
        tensor = actual[name]
        meta = tensor_metadata(tensor)
        require(meta['shape'] == [1, heads, 2048, 128] and tensor.is_contiguous()
                and meta['dtype'] == 'torch.float16', 'FIA28_ACTUAL_QKV_GEOMETRY_OR_DTYPE_DIFFERS:' + name)
        require(str(tensor.device).split(':')[0] == 'npu' or (allow_cpu_test and str(tensor.device) == 'cpu'),
                'FIA28_NPU_REQUIRED')
        tensors[name] = meta
    require(len({row['device'] for row in tensors.values()}) == 1, 'FIA28_QKV_DEVICES_DIFFER')
    mask = actual.get('atten_mask')
    require(mask is metadata.attn_mask, 'FIA28_MASK_NOT_ACTUAL_METADATA_OBJECT')
    mask_meta = tensor_metadata(mask)
    require(mask_meta['shape'] == [2048, 2048] and mask_meta['dtype'] == 'torch.bool'
            and mask_meta['device'] == tensors['query']['device'] and mask.is_contiguous(),
            'FIA28_CAUSAL_MASK_GEOMETRY_DIFFERS')
    return dict(status='INPUT_METADATA_CANDIDATE_ONLY', input_contract_matches=True,
        qualification_scope='CPU_CONTRACT_ONLY' if allow_cpu_test else 'NATURAL_INPUT_METADATA_ONLY',
        layout='BNSD', query=tensors['query'], key=tensors['key'], value=tensors['value'], mask=mask_meta,
        api_arguments_changed=False, tensor_values_read=False, mask_values_verified=False,
        native_route_verified=False, public_worker_qualified=False, native_launch_verified=False,
        mechanism_candidates=list(MECHANISMS))


def verify_sdk_binding(completion, completion_sha256, contract_path, sdk_root):
    """Read actual SDK files; CANN 8.5 cannot inherit a 9.1 artifact's ABI.

    This deliberately verifies only SDK identity. Actual mapped stock/private
    library coexistence after full Worker initialization remains another gate.
    """
    digest = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    require(digest(completion) == completion_sha256, 'FIA28_COMPLETION_HASH_DIFFERS')
    data = json.loads(Path(completion).read_bytes())
    require(data.get('schema') == 'demoncat.fia.fresh_build.v1'
            and data.get('status') == 'CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION'
            and data.get('variant') in ('legal_tiling', 'denominator_replay')
            and data.get('source_commit') == 'c77c0829f1f0f0b8f882db1d9b051a758b605472'
            and data.get('loading_allowed') is False and data.get('npu_executed') is False
            and data.get('previous_experiment_objects_used') is False, 'FIA28_INDEPENDENT_SOURCE_BUILD_REQUIRED')
    require(digest(contract_path) == data.get('contract_sha256'), 'FIA28_BUILD_CONTRACT_HASH_DIFFERS')
    contract = json.loads(Path(contract_path).read_bytes())
    pinned = [row for row in contract['files'] if row['kind'] == 'sdk']
    require(pinned and data.get('sdk_interface_manifest_sha256') is None,
            'FIA28_ORIGINAL_CANN91_INTERFACE_SET_REQUIRED')
    if 'sdk_interface_files' in data:
        require(data['sdk_interface_files'] == pinned, 'FIA28_ORIGINAL_CANN91_INTERFACE_SET_REQUIRED')
        pin_source = 'EXPLICIT_BUILD_INTERFACE_FILES_AND_CONTRACT'
    else:
        # The independently inspected initial FIA receipts predate the optional
        # SDK compile-probe field. Their exact original contract already pins
        # all five 9.1 interfaces; absence is not permission to use another SDK.
        require(data['contract_sha256'] == ORIGINAL_CANN91_CONTRACT_SHA,
                'FIA28_LEGACY_BUILD_REQUIRES_EXACT_ORIGINAL_CONTRACT')
        pin_source = 'EXACT_ORIGINAL_CANN91_BUILD_CONTRACT'
    sdk = Path(sdk_root).resolve(strict=True)
    for row in pinned:
        path = sdk / row['path']
        require(path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                'FIA28_RUNTIME_SDK_ABI_DIFFERS:' + row['path'])
    return dict(status='ACTUAL_SDK_FILE_IDENTITY_VERIFIED', sdk=str(sdk), interface_files=pinned, pin_source=pin_source,
        loading_allowed=False, native_worker_initialized=False, library_coexistence_verified=False,
        public_worker_qualified=False)


class FiaPrefillAudit:
    """Install after collector start; close before collector teardown.

    One ordinary driver request at a time. All mismatches are recorded without
    changing the invoked API. Input/output bytes and native policy are untouched.
    """
    def __init__(self, worker, *, max_calls=224):
        import torch_npu
        from vllm.model_executor.layers.attention.attention import Attention
        from vllm_ascend.attention import attention_v1
        from .paged33_audit import ATTENTION_SOURCE_SHA256, LAYER_SOURCE_SHA256
        self.worker, self.runner = worker, worker.model_runner
        self.model = self.runner.model
        require_serial_eager(self.runner)
        require(not self.model.training and type(max_calls) is int and 1 <= max_calls <= 224,
                'FIA28_BOUNDED_EVAL_AUDIT_REQUIRED')
        self.sources = {}
        # Reuse only reviewed Python source identities; no paged metadata or ABI.
        for label, cls, expected in [('backend', attention_v1.AscendAttentionBackendImpl, ATTENTION_SOURCE_SHA256),
                                     ('layer', Attention, LAYER_SOURCE_SHA256)]:
            source = Path(inspect.getsourcefile(cls)).resolve(strict=True)
            require(hashlib.sha256(source.read_bytes()).hexdigest() == expected, 'FIA28_UNREVIEWED_WORKER_SOURCE:' + label)
            self.sources[label] = dict(path=str(source), sha256=expected)
        self.candidates = [(path, module) for path, module in self.model.named_modules()
            if type(module) is Attention and type(getattr(module, 'impl', None)) is attention_v1.AscendAttentionBackendImpl]
        require(self.candidates, 'FIA28_NO_ACTUAL_REVIEWED_ATTENTION_MODULE')
        self.binding = ForwardBinding(self.model)
        self.binding.original = type(self.model).forward.__get__(self.model, type(self.model))
        self.binding.signature = inspect.signature(self.binding.original)
        self.api_module = torch_npu
        self.owner, self.max_calls = threading.get_ident(), max_calls
        self.handles, self.patches, self.rows, self.errors, self.seen, self.stack = [], [], [], [], [], []
        self.expected = self.current = None
        self.forward_count = self.overflow_calls = 0
        self.installed = False

    def _inside(self):
        return threading.get_ident() == self.owner and self.current is not None

    def expect(self, request_id, prompt_token_ids):
        require(self.installed and self.expected is None and self.current is None
                and isinstance(request_id, str) and request_id and request_id not in self.seen,
                'FIA28_FRESH_REAL_REQUEST_ID_REQUIRED')
        require(isinstance(prompt_token_ids, (list, tuple)) and len(prompt_token_ids) == 2048
                and all(type(x) is int and x >= 0 for x in prompt_token_ids), 'FIA28_FULL_2048_TOKEN_PROMPT_REQUIRED')
        self.expected = (request_id, list(prompt_token_ids))

    def _root_pre(self, module, args, kwargs):
        if threading.get_ident() != self.owner or self.expected is None:return
        require(self.current is None and not self.stack, 'FIA28_CONCURRENT_ROOT_FORWARD_REFUSED')
        self.current = dict(request_id=self.expected[0], scheduler=None, failure=None)
        try:
            from vllm.forward_context import get_forward_context
            require(self.worker.model_runner is self.runner and self.runner.model is module is self.model,
                    'FIA28_LOADED_MODEL_CHANGED')
            actual = self.binding.bind(module, args, kwargs)
            self.current['scheduler'] = snapshot_fresh_prefill(self.runner, self.model, actual, get_forward_context(),
                expected_prompt_token_ids=self.expected[1], expected_internal_request_id=self.expected[0],
                seen_internal_request_ids=self.seen)
        except BaseException as error:
            self.current['failure'] = repr(error)
            self.errors.append(dict(stage='scheduler', error=repr(error)))

    def _root_post(self, module, args, kwargs, output):
        if not self._inside():return
        if self.stack:self.errors.append(dict(stage='root_return', error='FIA28_ATTENTION_STACK_LEAK'))
        if output is None:self.errors.append(dict(stage='root_return', error='FIA28_ROOT_DID_NOT_RETURN_OUTPUT'))
        self.stack.clear();self.seen.append(self.current['request_id']);self.forward_count += 1
        self.current = self.expected = None

    def _observe(self, name, args, kwargs):
        if len(self.rows) >= self.max_calls:
            self.overflow_calls += 1
            return None
        row = dict(actual_operator='torch_npu.' + name, metadata_binding='UNPROVEN',
            status='OBSERVED_INELIGIBLE_OR_UNPROVEN', input_contract_matches=False,
            public_worker_qualified=False, native_launch_verified=False, consumer_returned=False)
        self.rows.append(row)
        try:
            require(self.stack and self.current['scheduler'] is not None, 'FIA28_ACTUAL_PREFILL_SCHEDULER_OR_MODULE_UNPROVEN')
            path, module = self.stack[-1]
            from vllm.forward_context import get_forward_context
            context = get_forward_context()
            require(context.model_instance is self.model and context.no_compile_layers.get(module.layer_name) is module,
                    'FIA28_ACTUAL_CONTEXT_MODULE_OWNER_DIFFERS')
            metadata = context.attn_metadata[module.layer_name]
            actual = bind_actual_api(args, kwargs)
            row.update(module_path=path, module_object_id=id(module), impl_object_id=id(module.impl),
                actual_layout=actual.get('input_layout'), actual_block_size=actual.get('block_size'),
                actual_argument_names=sorted(actual), scheduler=self.current['scheduler'],
                context_layer_name=module.layer_name, metadata_object_id=id(metadata), metadata_binding='VERIFIED')
            row['actual_input_metadata'] = describe_actual_inputs(actual)
            row['actual_scalar_options'] = {key: value for key, value in actual.items()
                if key in SCALARS and type(value) in (int, float, bool, str)}
            row['actual_host_lengths'] = {key: list(actual[key]) for key in
                ('actual_seq_lengths', 'actual_seq_lengths_kv') if isinstance(actual.get(key), (list, tuple))
                and len(actual[key]) <= 8 and all(type(value) is int for value in actual[key])}
            row.update(inspect_prefill_input(name, actual, metadata=metadata, scheduler=self.current['scheduler']))
        except BaseException as error:
            row['refusal'] = repr(error)
        return row

    def install(self):
        global _ACTIVE
        require(_ACTIVE is None and not self.installed, 'FIA28_AUDIT_OWNER_ALREADY_PRESENT')
        _ACTIVE = self
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._root_pre, with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._root_post, with_kwargs=True, always_call=True))
            for path, module in self.candidates:
                def pre(m, args, kwargs, path=path):
                    if self._inside():self.stack.append((path, m))
                def post(m, args, kwargs, output):
                    if self._inside():
                        require(self.stack and self.stack[-1][1] is m, 'FIA28_ATTENTION_STACK_OWNER_DIFFERS')
                        self.stack.pop()
                self.handles.append(module.register_forward_pre_hook(pre, with_kwargs=True))
                self.handles.append(module.register_forward_hook(post, with_kwargs=True, always_call=True))
            for name in APIS:
                if not hasattr(self.api_module, name):continue
                original = getattr(self.api_module, name)
                def wrapper(*args, _name=name, _original=original, **kwargs):
                    row = self._observe(_name, args, kwargs) if self._inside() else None
                    result = _original(*args, **kwargs)
                    if row is not None:row['consumer_returned'] = True
                    return result
                setattr(self.api_module, name, wrapper)
                self.patches.append((name, original, wrapper))
            self.installed = True
        except BaseException:
            self.close()
            raise

    def report(self):
        return dict(schema='demoncat.attention28.prefill_boundary.v1', read_only=True, source=self.sources,
            forward_count=self.forward_count, candidate_module_count=len(self.candidates), rows=self.rows,
            errors=self.errors, overflow_calls=self.overflow_calls, complete_census=False,
            input_candidates=sum(r['input_contract_matches'] for r in self.rows), native_scope_entered=False,
            worker_qualified=False, c2=False, c3=False, injection_enabled=False,
            missing=['complete natural eligible call census', 'mask and input value evidence',
                     'actual SDK and stock/private library coexistence', 'fresh native host and final launch',
                     'full payload bridge, B/F/OFF and independent raw statistics'])

    def close(self):
        global _ACTIVE
        require(threading.get_ident() == self.owner and self.current is None, 'FIA28_ACTIVE_OR_FOREIGN_AUDIT_CLOSE')
        for name, original, wrapper in reversed(self.patches):
            require(inspect.getattr_static(self.api_module, name) is wrapper, 'FIA28_API_WRAPPER_OWNERSHIP_CHANGED:' + name)
            setattr(self.api_module, name, original)
        self.patches.clear()
        for handle in reversed(self.handles):handle.remove()
        self.handles.clear();self.installed = False;self.expected = None
        if _ACTIVE is self:_ACTIVE = None
