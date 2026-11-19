"""Shared read-only prerequisite reports; no report grants execution authority.

Actual loaders/typed bindings retain their identity checks. Missing facts stay
pending and mismatches stay refused, with deployment errors before call errors.
"""
from copy import deepcopy

LAYERS = ('deployment_identity', 'mechanism_compatibility', 'actual_call')
_MISSING = object()


def observed_field(root, path):
    value = root
    for name in path.split('.'):
        value = getattr(value, name, _MISSING)
        if value is _MISSING:
            return _MISSING
    return value


def expectation(layer, field, expected, actual, code):
    if layer not in LAYERS:
        raise ValueError('UNKNOWN_PREREQUISITE_LAYER')
    missing = actual is _MISSING
    matched = not missing and type(actual) is type(expected) and actual == expected
    return dict(layer=layer, field=field, expected=deepcopy(expected),
                actual={'missing': True} if missing else deepcopy(actual),
                status='pending' if missing else 'matched' if matched else 'mismatch', code=code)


def assessment(checks):
    ordered = sorted((deepcopy(row) for row in checks), key=lambda row: LAYERS.index(row['layer']))
    unmet = [row for row in ordered if row['status'] != 'matched']
    return dict(schema='demoncat.prerequisites.v1', checks=ordered, unmet=unmet,
                status='refused' if any(row['status']=='mismatch' for row in unmet)
                       else 'pending' if unmet else 'matched',
                primary=unmet[0] if unmet else None, execution_authorized=False)


def serial_eager_assessment(runner, *, expected_max_seqs=1):
    if type(expected_max_seqs) is not int or expected_max_seqs not in (1, 8):
        raise ValueError('UNREVIEWED_SEQUENCE_CAPACITY')
    specs = [
        ('vllm_config.parallel_config.'+name, 1, 'SERIAL_TP1_PP1_DP1_REQUIRED:'+name)
        for name in ('tensor_parallel_size','pipeline_parallel_size','data_parallel_size')]
    specs += [(name, 1, 'CONTEXT_PARALLELISM_UNSUPPORTED:'+name) for name in ('pcp_size','dcp_size')]
    specs += [
        ('use_cp', False, 'CONTEXT_PARALLELISM_UNSUPPORTED'),
        ('vllm_config.parallel_config.enable_dbo', False, 'DBO_UNSUPPORTED'),
        ('speculative_config', None, 'SPECULATIVE_DECODING_UNSUPPORTED'),
        ('use_async_scheduling', False, 'ASYNC_SCHEDULING_UNSUPPORTED'),
        ('vllm_config.scheduler_config.max_num_seqs', expected_max_seqs,
         'MAX_NUM_SEQS_MUST_BE_ONE' if expected_max_seqs==1 else 'MAX_NUM_SEQS_CONTRACT_MISMATCH'),
        ('vllm_config.scheduler_config.enable_chunked_prefill', False, 'CHUNKED_PREFILL_UNSUPPORTED'),
        ('vllm_config.cache_config.enable_prefix_caching', False, 'PREFIX_CACHE_MUST_BE_DISABLED'),
        ('vllm_config.cache_config.kv_sharing_fast_prefill', False, 'KV_SHARING_FAST_PREFILL_UNSUPPORTED'),
        ('vllm_config.model_config.enforce_eager', True, 'EAGER_REQUIRED'),
        ('is_multimodal_model', False, 'MULTIMODAL_MAPPING_UNADAPTED'),
        ('vllm_config.model_config.is_encoder_decoder', False, 'ENCODER_DECODER_UNADAPTED')]
    return assessment([expectation('mechanism_compatibility', path, expected,
                                  observed_field(runner, path), code) for path,expected,code in specs])
