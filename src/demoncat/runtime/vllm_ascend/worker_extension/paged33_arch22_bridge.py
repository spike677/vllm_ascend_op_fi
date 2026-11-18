"""Normal-only fresh arch22 route candidate; no fault or bridge certificate.

This reads the original Worker call and preserves its arguments/return object.
It reuses Native scope recovery and final-argument verification, while retaining
a distinct build, public serializer and normal-route contract from old FAI FD.
"""
import json
import os
from pathlib import Path
import threading
import uuid

from .attention_native import _NativeCall
from .paged33_tnd import _verify_bundle_artifacts, validate_tnd_call
from .serving_context import require
from fault_injection.native_adapters.attention28.launch import sha, bundle_kernel_abi, verify_launch
from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_bridge as recipe

NORMAL_PHASES = ('baseline', 'baseline_repeat', 'off', 'recovery_off')
OPTIONAL = ('pse_shift', 'dequant_scale1', 'quant_scale1', 'dequant_scale2', 'quant_scale2',
    'quant_offset2', 'antiquant_scale', 'antiquant_offset', 'query_padding_size', 'kv_padding_size',
    'key_antiquant_scale', 'key_antiquant_offset', 'value_antiquant_scale', 'value_antiquant_offset',
    'key_shared_prefix', 'value_shared_prefix', 'actual_shared_prefix_len', 'query_rope', 'key_rope',
    'key_rope_antiquant_scale', 'dequant_scale_query', 'learnable_sink', 'q_start_idx', 'kv_start_idx')
SCALARS = ('input_layout', 'block_size', 'sparse_mode', 'num_heads', 'num_key_value_heads', 'scale',
    'inner_precise', 'pre_tokens', 'next_tokens', 'softmax_lse_flag', 'antiquant_mode',
    'key_antiquant_mode', 'value_antiquant_mode', 'query_quant_mode', 'pse_type')
CANONICAL_TOKEN_SOURCE_SHA256 = {
    'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.cpp':
        '380248471fef863d9c7ae5f77f00aafb71185330d350388652c6f01557a9f271',
    'attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling_info_parser.h':
        '54753b541981eb176636bc0243c728830f5076a39d4e69721816d0d4eb6d44f0',
    'attention/common/op_host/fia_tiling_info.h':
        'd201e4ed6ca93fb90a7a60c9da5fbfbb5727810768ffd6cd74e8c67115febda4',
}


def verify_normal_token_sources(completion):
    """Pin the actual built attention parser; never use experimental/attention."""
    root = Path(completion).resolve(strict=True).parent.parent/'source'
    observed = {}
    for relative, expected in CANONICAL_TOKEN_SOURCE_SHA256.items():
        path = root/relative
        require(not path.is_symlink() and path.resolve(strict=True) == path
            and path.stat().st_size <= 256*1024 and sha(path) == expected,
            'ARCH22_NORMAL_TOKEN_PARSER_SOURCE_DIFFERS:'+relative)
        observed[relative] = expected
    return observed


def normal_token_window(actual):
    # The pinned FiaInfoParser.cpp GetPreNextToken calls
    # UpdatePreNextTokenBySparseMode before GenerateInfo. RIGHT_DOWN=3 sets
    # (INT_MAX, 0), irrespective of the supplied pre/next attributes. Its
    # later ANTI_QUANT + S1=1 override is excluded by the actual FP16 Q/K/V.
    require(actual.get('sparse_mode') == 3
        and all(str(actual[name].dtype) == 'torch.float16' for name in ('query','key','value'))
        and all(actual.get(name) is None for name in OPTIONAL)
        and all(actual.get(name, 0) == 0 for name in
                ('antiquant_mode','key_antiquant_mode','value_antiquant_mode','query_quant_mode')),
        'ARCH22_NONQUANT_CAUSAL_TOKEN_SCOPE_REQUIRED')
    require(all(type(actual[name]) is int for name in ('pre_tokens','next_tokens') if name in actual),
            'ARCH22_TOKEN_ATTRIBUTE_TYPE_DIFFERS')
    return dict(rule='PINNED_FIA_RIGHT_DOWN_NONQUANT',
        supplied_python_attributes={name:actual[name] for name in ('pre_tokens','next_tokens') if name in actual},
        omitted_python_defaults={name:2147483647 for name in ('pre_tokens','next_tokens') if name not in actual},
        effective_native_pre_token=2147483647, effective_native_next_token=0,
        parser_source_sha256=dict(CANONICAL_TOKEN_SOURCE_SHA256))


def verify_bundle(completion, completion_sha256, vendor):
    bundle = _verify_bundle_artifacts(completion, completion_sha256, vendor,
                                     variant=recipe.VARIANT, keys=recipe.KEYS)
    data = bundle['completion']
    require(data.get('normal_bridge_recipe_sha256') == sha(recipe.__file__)
        and data.get('normal_bridge_header_sha256') == sha(Path(recipe.__file__).with_name('paged33_arch22_bridge.h'))
        and data.get('fault_execution_ready') is False and data.get('stock_private_exact_proven') is False,
        'ARCH22_NORMAL_ONLY_SOURCE_BINDING_REQUIRED')
    folder = Path(bundle['completion_path']).parent/'arch22_schema'
    require(sha(folder/'compile_receipt.json') == data.get('arch22_schema_compile_sha256')
        and sha(folder/'fia_abi_schema.json') == data.get('arch22_public_schema_sha256'),
        'ARCH22_FRESH_PUBLIC_SCHEMA_EVIDENCE_DIFFERS')
    compiled = json.loads((folder/'compile_receipt.json').read_bytes())
    require(compiled.get('schema') == 'demoncat.fia.fresh_abi_compile.v1'
        and compiled.get('schema_sha256') == data['arch22_public_schema_sha256']
        and compiled.get('npu_executed') is False and compiled.get('loading_allowed') is False,
        'ARCH22_SCHEMA_IS_NOT_A_NATIVE_QUALIFICATION')
    bundle['canonical_token_sources'] = verify_normal_token_sources(completion)
    return bundle


def validate_normal_call(actual, boundary, *, allow_cpu_test=False):
    shape = validate_tnd_call(actual, boundary, allow_cpu_test=allow_cpu_test)
    known = {'query', 'key', 'value', 'atten_mask', 'block_table',
             'actual_seq_lengths', 'actual_seq_lengths_kv', *OPTIONAL, *SCALARS}
    require(set(actual).issubset(known), 'ARCH22_UNKNOWN_ACTUAL_ARGUMENT')
    require(all(actual.get(name) is None for name in OPTIONAL), 'ARCH22_OPTIONAL_FEATURE_UNADAPTED')
    mask = actual.get('atten_mask')
    require(mask is not None and str(mask.dtype) == 'torch.int8' and list(mask.shape) == [2048, 2048]
        and mask.is_contiguous() and str(mask.device) == str(actual['query'].device),
        'ARCH22_ACTUAL_INT8_MASK_REQUIRED')
    table = actual['block_table']
    require(str(table.dtype) == 'torch.int32' and table.ndim == 2 and table.shape[0] == 1
        and 65 <= table.shape[1] <= 128 and table.is_contiguous(), 'ARCH22_ACTUAL_PAGE_TABLE_DIFFERS')
    require(actual.get('softmax_lse_flag', False) is False, 'ARCH22_LSE_OUTPUT_UNADAPTED')
    for name in ('antiquant_mode', 'key_antiquant_mode', 'value_antiquant_mode', 'query_quant_mode'):
        require(actual.get(name, 0) == 0, 'ARCH22_QUANT_ATTRIBUTE_UNADAPTED')
    require(actual.get('pse_type', 1) == 1, 'ARCH22_PSE_ATTRIBUTE_UNADAPTED')
    # Defaults are declarations for omitted Python arguments. Native pre/next
    # are checked independently below and labeled, never filled into the call.
    defaults = {name: 2147483647 for name in ('pre_tokens', 'next_tokens') if name not in actual}
    for name in ('pre_tokens', 'next_tokens'):
        require(type(actual.get(name, 2147483647)) is int, 'ARCH22_TOKEN_ATTRIBUTE_TYPE_DIFFERS')
    normal_token_window(actual)
    return dict(shape, native_route='FiaTilingNonQuant', fault_s2=None,
                normal_only=True, omitted_argument_defaults=defaults)


def validate_native(launch, host, actual, boundary):
    require(host.get('schema') == 'demoncat.paged33.arch22_normal_host.v1'
        and host.get('route') == 'FiaTilingNonQuant' and host.get('mode') == 'arch22_normal'
        and host.get('normal_route_preference_changed') is True and host.get('eligible') is True
        and host.get('policy_applied') is False and host.get('native_s2') == host.get('selected_s2') == 512,
        'ARCH22_ACTUAL_NORMAL_HOST_ROUTE_DIFFERS')
    require(launch['tiling_key'] == 103000000010300003 and launch['block_dim'] == 16
        and launch['schedule_mode'] == 1, 'ARCH22_OBSERVED_STOCK_ROUTE_ENTRY_DIFFERS')
    typed = launch['typed']
    tokens = normal_token_window(actual)
    expected = {'baseParams.bSize': 1, 'baseParams.n2Size': 8, 'baseParams.gSize': 2,
        'baseParams.headDim': 128, 'baseParams.headDimRope': 0, 'baseParams.softmaxLseFlag': 0,
        'baseParams.isLegacyIfa': 0,
        'baseParams.actualSeqS1Dims': 1, 'baseParams.actualSeqS2Dims': 1,
        'innerSplitParams.s2BaseSize': 512, 'pageAttenParams.blockSize': 16,
        'pageAttenParams.maxBlockNumPerBatch': int(actual['block_table'].shape[1]),
        'maskParams.attenMaskFlag': 1, 'maskParams.sparseMode': 3,
        'maskParams.preToken': tokens['effective_native_pre_token'],
        'maskParams.nextToken': tokens['effective_native_next_token'],
        'leftPaddingParams.qPaddingFlag': 0, 'leftPaddingParams.kvPaddingFlag': 0,
        'prefixParams.prefixFlag': 0, 'pseParams.pseShiftFlag': 0}
    require(all(typed.get(name) == value for name, value in expected.items()),
            'ARCH22_FINAL_PUBLIC_ARGUMENT_GEOMETRY_DIFFERS')
    import struct
    require(typed['baseParams.scaleValue'] == struct.unpack('<f', struct.pack('<f', actual['scale']))[0],
            'ARCH22_FINAL_SCALE_DIFFERS')
    require(typed['baseParams.s1Size'] == 1
        and boundary['sequence_lengths'][0] <= typed['baseParams.s2Size'] <= int(actual['key'].shape[0])*16,
        'ARCH22_FINAL_SEQUENCE_CAPACITY_DIFFERS')
    return dict(route='FiaTilingNonQuant', selected_s2=512, normal_only=True,
        actual_pre_tokens=typed['maskParams.preToken'], actual_next_tokens=typed['maskParams.nextToken'],
        public_to_native_token_window=tokens,
        scalar_source='INDEPENDENT_FINAL_ARGUMENT_POD_MATCHED_TO_PUBLIC_HOST_SERIALIZER',
        stock_route_entry_matches=True, stock_abi_decoded=False, stock_private_exact_proven=False)


class Arch22NormalNativeCall(_NativeCall):
    mechanism = '33.paged_decode'
    bundle_verifier = staticmethod(verify_bundle)

    def __init__(self, *, completion, completion_sha256, vendor, schema, schema_sha256,
                 observer_receipt, observer_receipt_sha256, evidence_dir):
        self.build = self.bundle_verifier(completion, completion_sha256, vendor)
        data = self.build['completion']
        require(sha(schema) == schema_sha256 == data['arch22_public_schema_sha256'],
                'ARCH22_PUBLIC_SCHEMA_HASH_DIFFERS')
        self.schema = json.loads(Path(schema).read_bytes())
        require(self.schema.get('schema') == 'fault28.fia_graph.public_serializer_layout.v1'
            and self.schema.get('type') == 'FusedInferAttentionScoreTilingData' and self.schema.get('size') == 1576,
            'ARCH22_ACTUAL_PUBLIC_SCHEMA_REQUIRED')
        self.metadata, self.abi = bundle_kernel_abi(self.build['vendor'], self.build['artifacts'], data['kernel_binary_sha256'])
        from .paged33_loading import verify_runtime
        self.library_coexistence = verify_runtime(self.build)
        require(sha(observer_receipt) == observer_receipt_sha256, 'ARCH22_OBSERVER_RECEIPT_CHANGED')
        receipt = json.loads(Path(observer_receipt).read_bytes())
        contract = Path(recipe.__file__).with_name('build_contract.json')
        require(sha(contract) == data['contract_sha256'], 'ARCH22_BUILD_CONTRACT_DIFFERS')
        expected = {row['path'] for row in json.loads(contract.read_bytes())['files'] if row['kind'] == 'sdk'}
        interfaces = data.get('sdk_interface_files', [])
        require(len(interfaces) == len(expected) and {row['path'] for row in interfaces} == expected
            and all(sha(Path(receipt['sdk'])/row['path']) == row['sha256'] for row in interfaces),
            'ARCH22_FRESH_RUNTIME_SDK_ABI_DIFFERS')
        self.fingerprint = dict(completion_sha256=completion_sha256, kernel_sha256=data['kernel_binary_sha256'],
            mechanism=self.mechanism, normal_only=True, stock_private_exact_proven=False,
            canonical_token_sources=dict(self.build['canonical_token_sources']))
        self._initialize(self.mechanism, observer_receipt, observer_receipt_sha256, evidence_dir)
        self.policy_prefix = 'DCAT33_TND_'

    def execute(self, actual, boundary, phase, invoke_once):
        require(phase in NORMAL_PHASES, 'ARCH22_NORMAL_BRIDGE_NEVER_AUTHORIZES_FAULT')
        scope = validate_normal_call(actual, boundary)
        from .paged33_loading import validate_environment
        validate_environment(self.build['vendor'], platform_ready=True)
        token, log = uuid.uuid4().hex, self.folder/'host.jsonl'
        environment = dict(DCAT33_TND_MODE='arch22_normal', DCAT33_TND_TOKEN=token,
            DCAT33_TND_PID=str(os.getpid()), DCAT33_TND_TID=str(threading.get_native_id()), DCAT33_TND_LOG=str(log))
        def verify():
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            hosts = [row for row in rows if row.get('token') == token]
            require(len(hosts) == 1, 'ARCH22_EXACT_ONE_HOST_CALLBACK_REQUIRED')
            host = hosts[0]
            launch = verify_launch(self.runtime_dir, token, schema=self.schema, host=host,
                kernel_sha256=self.fingerprint['kernel_sha256'], kernel_metadata=self.metadata, kernel_abi=self.abi)
            geometry = validate_native(launch, host, actual, boundary)
            return dict(call_id=token, token=token, phase=phase, module_path=boundary['module_path'],
                module_object_id=boundary['module_object_id'], scheduler=boundary['scheduler'],
                virtual_engine=boundary.get('virtual_engine'), native=launch, host=host, scope=scope,
                geometry=geometry, original_calls=1, selection_applied=False, cache_repacked=False,
                api_arguments_changed=False, output_mutated=False, mechanism=self.mechanism,
                target_key=boundary.get('target_key'), stock_private_exact_proven=False)
        return self._run(token, environment, invoke_once, verify)


class NormalBridgeProbeAdapter:
    """Use the same full-output probe with the new normal-only Native strategy."""
    def __init__(self, *, arm, experimental_qualification=False, max_calls=224, **kwargs):
        require(experimental_qualification is True and arm in ('baseline', 'off')
            and type(max_calls) is int and 1 <= max_calls <= 224, 'ARCH22_NORMAL_ONLY_PROBE_REQUIRED')
        self.native = Arch22NormalNativeCall(**kwargs)
        self.arm, self.max_calls, self.enabled = arm, max_calls, True
        self.calls, self.runtime_dir = self.native.calls, self.native.runtime_dir

    def execute(self, actual, boundary, invoke_once):
        require(self.enabled and len(self.calls)+len(self.native.failures) < self.max_calls,
                'ARCH22_NORMAL_PROBE_BUDGET_OR_DISABLED')
        result, _ = self.native.execute(actual, boundary, self.arm, invoke_once)
        return result

    def disable(self):
        self.enabled = False
        state = self.native.stop()
        return dict(state, scope_cleared=not state['scope_owned'] and state['environment_owned'] == 0,
            local_streams_drained=True, library_coexistence=self.native.library_coexistence,
            public_worker_qualified=False, c1=False, c2=False, c3=False, stock_private_exact_proven=False)
