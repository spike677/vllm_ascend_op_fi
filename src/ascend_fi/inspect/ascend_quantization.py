"""Read-only contract for one audited Ascend W8A8 dynamic linear implementation.

File hashes identify the audited source, not a general claim about all CANN builds.
No scale values are read, no device operation is performed, and no provider is added.
"""
import hashlib
import inspect
from pathlib import Path

from .surface import stored, type_name

WRAPPER = 'vllm_ascend.quantization.method_adapters.AscendLinearMethod'
SCHEME = 'vllm_ascend.quantization.methods.w8a8_dynamic.AscendW8A8DynamicLinearMethod'
STATIC = 'vllm_ascend.quantization.methods.w8a8_static.AscendW8A8LinearMethod'
UNQUANTIZED = 'vllm_ascend.ops.linear.AscendUnquantizedLinearMethod'
SOURCE_HASHES = {
    WRAPPER: 'af15ada5b09e99cd2bb59ce07dd02459db1f3e1cb9dedf646531de469d8db932',
    SCHEME: '07b8ef64840d4bd29b19351945132e80cc4c1c7e46729b1a96413bb6e2b252b8',
    STATIC: 'dd3dff8c9c7b1fd5d48920417cda55561102aa285e03afce4834783cecb35459',
    UNQUANTIZED: 'fcf2520f8818cb02e6b27a7b9a926745c5c36f1af08a01d9be0627ce2c87316e',
}


def audited_ascend_contract(method, runtime):
    method_type = type_name(method)
    if method_type not in (WRAPPER, UNQUANTIZED):
        return None
    scheme = stored(method, 'quant_method')
    if method_type == WRAPPER and type_name(scheme) not in (SCHEME, STATIC):
        return None
    evidence = []
    for obj in ((method,) if method_type == UNQUANTIZED else (method, scheme)):
        try:
            path = Path(inspect.getfile(type(obj)))
            if path.stat().st_size > 512 * 1024:
                return None
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except (OSError, TypeError):
            return None
        if digest != SOURCE_HASHES[type_name(obj)]:
            return None
        evidence.append({'class': type_name(obj), 'source_file': path.name, 'sha256': digest})
    if method_type == UNQUANTIZED:
        return {'quant_method': method_type, 'runtime_fingerprint': runtime,
                'source_sha256': SOURCE_HASHES[UNQUANTIZED], 'source_evidence': evidence,
                'scheme': 'UNQUANTIZED', 'fields': {},
                'activation_scale': {'visibility': 'NOT_APPLICABLE_UNQUANTIZED_METHOD',
                    'providers': [], 'model_scale_fake_requant': 'NOT_APPLICABLE',
                    'inherited_apply_native_mapping': 'NOT_AUDITED'}}
    weight = {'target': 'weight', 'granularity': 'per_output_channel',
              'axis': 'output channel; flattened parameter axis 0 after loading',
              'dynamic': False, 'encoding': 'floating scale', 'providers': []}
    if type_name(scheme) == STATIC:
        per_tensor = {'target': 'activation', 'granularity': 'per_tensor',
                      'dynamic': False, 'encoding': 'floating scale', 'providers': []}
        return {'quant_method': WRAPPER, 'runtime_fingerprint': runtime,
                'source_sha256': SOURCE_HASHES[STATIC], 'source_evidence': evidence, 'scheme': STATIC,
                'fields': {
                    'input_scale': {**per_tensor, 'role': 'activation_scale'},
                    'input_offset': {**per_tensor, 'role': 'activation_offset', 'encoding': 'int8 offset'},
                    'aclnn_input_scale': {**per_tensor, 'role': 'activation_scale_repeated_over_input_channels'},
                    'aclnn_input_scale_reciprocal': {**per_tensor, 'role': 'reciprocal_activation_scale_repeated',
                                                    'source': 'stored_attribute'},
                    'aclnn_input_offset': {**per_tensor, 'role': 'activation_offset_repeated',
                                          'source': 'stored_attribute', 'encoding': 'floating offset'},
                    'weight_scale': {**weight, 'role': 'weight_scale_not_direct_apply_scale'},
                    'weight_offset': {**weight, 'role': 'weight_offset_not_used_by_audited_apply', 'encoding': 'floating offset'},
                    'deq_scale': {**weight, 'role': 'QMM_dequantization_scale', 'target': 'dequantization',
                                  'encoding': 'BF16 output: FP32; FP16 output: INT64 encoded scale'},
                    'quant_bias': {**weight, 'role': 'quantized_bias_tp_rank0', 'target': 'bias', 'encoding': 'int32'},
                },
                'activation_scale': {'role': 'activation_scale', 'dynamic': False, 'granularity': 'per_tensor',
                    'visibility': 'module.input_scale; repeated derived tensors consumed by torch.ops.vllm.quantize',
                    'quantization_bypassed_if': 'input dtype already INT8', 'numeric_value': 'NOT_CAPTURED',
                    'providers': [], 'model_scale_fake_requant': 'UNSUPPORTED; no verified runtime provider',
                    'deq_scale_warning': 'not interchangeable with input_scale; general checkpoint relation not proved'}}
    return {'quant_method': WRAPPER, 'runtime_fingerprint': runtime,
            'source_sha256': SOURCE_HASHES[SCHEME], 'source_evidence': evidence,
            'scheme': SCHEME,
            'fields': {
                'weight_scale': {**weight, 'role': 'weight_scale'},
                'weight_scale_fp32': {**weight, 'role': 'weight_scale_fp32_copy',
                                      'source': 'stored_attribute'},
                'weight_offset': {**weight, 'role': 'weight_offset_not_used_by_audited_apply',
                                  'encoding': 'floating offset; values not inspected'},
            },
            'activation_scale': {
                'role': 'activation_scale', 'dynamic': True, 'granularity': 'per_token',
                'producer': 'torch_npu.npu_dynamic_quant(x)',
                'consumer_argument': 'torch_npu.npu_quant_matmul(..., pertoken_scale=...)',
                'visibility': 'LOCAL_TEMPORARY_IN_SCHEME_APPLY; not module metadata',
                'numeric_value': 'NOT_CAPTURED', 'providers': [],
                'model_scale_fake_requant': 'UNSUPPORTED; no verified runtime provider',
            }}
