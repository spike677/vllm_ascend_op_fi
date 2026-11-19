"""Shared read-only prerequisite reports; no report grants execution authority.

Actual loaders/typed bindings retain their identity checks. Missing facts stay
pending and mismatches stay refused, with deployment errors before call errors.
"""
from fault_injection.prerequisites import (
    LAYERS, _MISSING, observed_field, expectation, assessment, require_assessment)


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
