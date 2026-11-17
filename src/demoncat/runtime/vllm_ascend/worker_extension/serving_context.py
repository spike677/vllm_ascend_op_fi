"""Read-only integration draft for the inspected vLLM-Ascend 0.18.0 v1 runner.

This module imports no torch/vLLM/NPU runtime and changes no model/runner state.
Call snapshot_fresh_prefill ONLY from an owned model pre-hook during a normal
driver-submitted request, after verifying actual runtime class/source identity.
It reads CPU buffers and device tensor metadata; it never copies device values,
launches an operator, creates a request ID, installs a hook or enters a scope.
CPU fixtures verify refusal behavior only; they do not certify real scheduling.
"""
import ast
import hashlib
import inspect
import json
import math
import textwrap


class ContextRefused(RuntimeError):
    def __init__(self, reason):
        self.code = str(reason).split(':', 1)[0]
        super().__init__(reason)


def require(condition, reason):
    if not condition:
        raise ContextRefused(reason)


def field(obj, name):
    if not hasattr(obj, name):
        raise ContextRefused("MISSING_FIELD:" + type(obj).__name__ + "." + name)
    return getattr(obj, name)


def ints(values, reason):
    require(isinstance(values, (list, tuple)), reason)
    require(all(isinstance(value, int) and not isinstance(value, bool) for value in values), reason)
    return list(values)


def cpu_list(value):
    """Only CPU torch/NumPy buffers or ordinary sequences, never .cpu()/.to()."""
    if isinstance(value, (list, tuple)):
        return list(value)
    module = type(value).__module__
    if module.startswith("numpy"):
        return value.tolist()
    if module.startswith("torch"):
        require(str(field(value, "device")) == "cpu", "DEVICE_VALUE_READ_FORBIDDEN")
        return value.tolist()
    raise ContextRefused("UNSUPPORTED_CPU_BUFFER:" + module)


def tensor_metadata(value):
    """Host-only metadata; never reads tensor contents or synchronizes a device."""
    require(value is not None, "MISSING_ACTUAL_TENSOR")
    storage = value.untyped_storage()
    return {"object_id": id(value), "storage_identity": int(storage._cdata),
            "storage_offset": int(value.storage_offset()), "shape": list(value.shape),
            "stride": list(value.stride()), "dtype": str(value.dtype), "device": str(value.device)}


def same_buffer_prefix(actual, buffer, length, dtype):
    left, right = tensor_metadata(actual), tensor_metadata(buffer)
    require(left["device"] == right["device"] and left["device"].split(":")[0] == "npu",
            "ACTUAL_INPUT_NOT_RUNNER_NPU_BUFFER")
    require(left["storage_identity"] == right["storage_identity"]
            and left["storage_offset"] == right["storage_offset"], "ACTUAL_INPUT_STORAGE_MISMATCH")
    require(left["shape"] == [length] and left["stride"] == [1]
            and left["dtype"] == right["dtype"] == dtype, "ACTUAL_INPUT_GEOMETRY_MISMATCH")
    return left


def source_identity(method):
    function = getattr(method, "__func__", method)
    source = textwrap.dedent(inspect.getsource(function))
    return {"source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "file": inspect.getsourcefile(function), "first_line": inspect.getsourcelines(function)[1]}, source


def require_serial_eager(runner):
    config = field(runner, "vllm_config")
    parallel = field(config, "parallel_config")
    for name in ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"):
        require(field(parallel, name) == 1, "SERIAL_TP1_PP1_DP1_REQUIRED:" + name)
    for name in ("pcp_size", "dcp_size"):
        require(field(runner, name) == 1, "CONTEXT_PARALLELISM_UNSUPPORTED:" + name)
    require(field(runner, "use_cp") is False, "CONTEXT_PARALLELISM_UNSUPPORTED")
    require(field(parallel, "enable_dbo") is False, "DBO_UNSUPPORTED")
    require(field(runner, "speculative_config") is None, "SPECULATIVE_DECODING_UNSUPPORTED")
    require(field(runner, "use_async_scheduling") is False, "ASYNC_SCHEDULING_UNSUPPORTED")
    scheduler = field(config, "scheduler_config")
    require(field(scheduler, "max_num_seqs") == 1, "MAX_NUM_SEQS_MUST_BE_ONE")
    require(field(scheduler, "enable_chunked_prefill") is False, "CHUNKED_PREFILL_UNSUPPORTED")
    cache = field(config, "cache_config")
    require(field(cache, "enable_prefix_caching") is False, "PREFIX_CACHE_MUST_BE_DISABLED")
    require(field(cache, "kv_sharing_fast_prefill") is False, "KV_SHARING_FAST_PREFILL_UNSUPPORTED")
    require(field(field(config, "model_config"), "enforce_eager") is True, "EAGER_REQUIRED")
    require(field(runner, "is_multimodal_model") is False, "MULTIMODAL_MAPPING_UNADAPTED")
    require(field(field(config, "model_config"), "is_encoder_decoder") is False, "ENCODER_DECODER_UNADAPTED")


def snapshot_fresh_prefill(runner, model, hook_kwargs, context, *, expected_prompt_token_ids,
                           expected_internal_request_id,
                           seen_internal_request_ids=()):
    """Corroborate the actual scheduler state without inventing request identity.

    Scope: max_seqs=1, TP/PP/DP/PCP/DCP=1, ordinary text, full fresh prefill,
    one non-hybrid KV group, no prefix cache/chunk/spec/async/graph/DBO.
    The CPU token mirror and storage association do not prove device byte equality.
    That separate check belongs to a real NPU validation outside this reader.
    """
    require_serial_eager(runner)
    require(model is field(runner, "model"), "HOOK_MODEL_NOT_RUNNER_MODEL")
    require(context is not None and field(context, "model_instance") is model, "FOREIGN_FORWARD_CONTEXT")
    require(field(context, "in_profile_run") is False, "PROFILE_RUN_REFUSED")
    require(field(context, "capturing") is False, "GRAPH_CAPTURE_REFUSED")
    require(field(context, "is_draft_model") is False, "DRAFTER_CONTEXT_REFUSED")
    require(field(field(context, "cudagraph_runtime_mode"), "name") == "NONE", "GRAPH_RUNTIME_REFUSED")
    require(field(context, "ubatch_slices") is None, "MICROBATCH_MAPPING_UNADAPTED")
    require(field(context, "dp_metadata") is None, "DP_METADATA_UNADAPTED")
    require(field(context, "flash_comm_v1_enabled") is False
            and field(context, "flashcomm_v2_enabled") is False, "SEQUENCE_PARALLELISM_UNADAPTED")
    require(hook_kwargs.get("inputs_embeds") is None and hook_kwargs.get("intermediate_tensors") is None,
            "EMBEDDING_OR_PIPELINE_INPUT_UNADAPTED")
    batch = field(runner, "input_batch")
    req_ids = list(field(batch, "req_ids"))  # Snapshot mutable persistent storage.
    require(field(batch, "num_reqs") == 1 and len(req_ids) == 1, "EXACTLY_ONE_REAL_REQUEST_REQUIRED")
    req_id = req_ids[0]
    require(isinstance(req_id, str) and bool(req_id), "INVALID_REAL_REQUEST_ID")
    require(isinstance(expected_internal_request_id, str) and bool(expected_internal_request_id)
            and req_id == expected_internal_request_id, "ADVANCE_EXPECTED_ID_NOT_ACTUAL_REQUEST")
    require(req_id not in set(seen_internal_request_ids), "REQUEST_ID_REUSED_ACROSS_PHASES")
    require(field(batch, "req_id_to_index") == {req_id: 0}, "REQ_ID_INDEX_MISMATCH")
    requests = field(runner, "requests")
    require(set(requests) == {req_id}, "RUNNER_REQUEST_SET_MISMATCH")
    request = requests[req_id]
    require(field(request, "req_id") == req_id, "CACHED_REQUEST_ID_MISMATCH")
    require(field(request, "num_computed_tokens") == 0, "CACHED_REQUEST_NOT_FRESH")
    require(cpu_list(field(batch, "num_computed_tokens_cpu"))[:1] == [0], "BATCH_REQUEST_NOT_FRESH")
    require(field(request, "output_token_ids") == [], "PREVIOUS_GENERATION_ALREADY_PRESENT")
    require(not field(request, "mm_features") and field(request, "prompt_embeds") is None,
            "CACHED_MULTIMODAL_OR_EMBEDDING_INPUT_UNADAPTED")
    require(field(request, "lora_request") is None, "LORA_MAPPING_UNADAPTED")
    sampling = field(request, "sampling_params")
    require(sampling is not None and field(sampling, "n") == 1, "SAMPLING_FANOUT_UNSUPPORTED")
    prompt = ints(field(request, "prompt_token_ids"), "TOKEN_ID_PROMPT_REQUIRED")
    expected = ints(expected_prompt_token_ids, "FROZEN_TOKEN_ID_PROMPT_REQUIRED")
    length = len(prompt)
    require(length > 1 and prompt == expected, "SCHEDULER_PROMPT_NOT_FROZEN_INPUT")
    require(field(request, "num_prompt_tokens") == length, "REQUEST_PROMPT_LENGTH_MISMATCH")
    request_num_tokens = field(request, "num_tokens")
    require(type(request_num_tokens) is int and request_num_tokens == length,
            "REQUEST_TOTAL_TOKEN_COUNT_MISMATCH")
    require(cpu_list(field(batch, "num_prompt_tokens"))[:1] == [length], "BATCH_PROMPT_LENGTH_MISMATCH")
    batch_num_tokens_no_spec = cpu_list(field(batch, "num_tokens_no_spec"))[:1]
    require(batch_num_tokens_no_spec == [length], "SPEC_OR_DECODE_TOKEN_PRESENT")
    # In this inspected Ascend v1 path, NPUInputBatch initializes num_tokens
    # to zero but inherits InputBatch.add_request, which updates only
    # num_tokens_no_spec=request.num_tokens. The Ascend post-sampling path
    # subsequently updates both arrays. num_tokens can therefore still contain
    # zero or the previous request's count at this fresh-prefill hook. Keep it
    # as diagnostic evidence, never as the current scheduled-input authority.
    legacy_num_tokens = cpu_list(batch.num_tokens)[:1] if hasattr(batch, "num_tokens") else None
    batch_tokens = cpu_list(field(batch, "token_ids_cpu"))[0][:length]
    host_tokens = cpu_list(field(field(runner, "input_ids"), "cpu"))[:length]
    require(batch_tokens == host_tokens == prompt, "CPU_INPUT_MIRROR_MISMATCH")
    positions = cpu_list(field(field(runner, "positions"), "cpu"))[:length]
    require(positions == list(range(length)), "FRESH_PREFILL_POSITIONS_MISMATCH")
    require(cpu_list(field(field(runner, "query_start_loc"), "cpu"))[:2] == [0, length],
            "QUERY_START_LOC_MISMATCH")
    require(cpu_list(field(field(runner, "seq_lens"), "cpu"))[:1] == [length], "SEQ_LEN_MISMATCH")
    require(cpu_list(field(runner, "query_lens")) == [length], "SCHEDULED_TOKEN_COUNT_MISMATCH")
    require(field(context, "num_tokens") == length and field(context, "pad_size") == 0,
            "PADDED_OR_TRUNCATED_FORWARD_UNADAPTED")
    actual_inputs = {
        "input_ids": same_buffer_prefix(hook_kwargs.get("input_ids"), field(field(runner, "input_ids"), "gpu"),
                                         length, "torch.int32"),
        "positions": same_buffer_prefix(hook_kwargs.get("positions"), field(field(runner, "positions"), "gpu"),
                                         length, "torch.int64")}
    tables = field(field(batch, "block_table"), "block_tables")
    require(len(tables) == 1 and len(field(request, "block_ids")) == 1, "KV_GROUP_LAYOUT_UNADAPTED")
    table = tables[0]
    require(field(table, "use_hybrid_blocks") is False and field(table, "blocks_per_phys_block") == 1,
            "HYBRID_BLOCK_LAYOUT_UNADAPTED")
    block_size = int(field(table, "block_size"))
    require(block_size > 0 and field(table, "physical_block_size") == block_size, "INVALID_BLOCK_SIZE")
    num_blocks = int(cpu_list(field(table, "num_blocks_per_row"))[0])
    blocks = cpu_list(field(field(table, "block_table"), "cpu"))[0][:num_blocks]
    request_blocks = list(field(request, "block_ids")[0])
    require(blocks == request_blocks and len(blocks) >= math.ceil(length / block_size), "REAL_REQUEST_BLOCKS_MISMATCH")
    require(all(isinstance(x, int) and x >= 0 for x in blocks) and len(set(blocks)) == len(blocks),
            "INVALID_OR_DUPLICATED_BLOCK_ID")
    slots = cpu_list(field(field(table, "slot_mapping"), "cpu"))[:length]
    expected_slots = [blocks[pos // block_size] * block_size + pos % block_size for pos in positions]
    require(slots == expected_slots and len(set(slots)) == length, "ACTUAL_SLOT_MAPPING_MISMATCH")
    attention = field(context, "attn_metadata")
    require(isinstance(attention, dict) and bool(attention), "ATTENTION_METADATA_REQUIRED")
    observed_layers = []
    for name, metadata in attention.items():
        require(field(metadata, "num_actual_tokens") == length
                and field(metadata, "num_prefills") == 1 and field(metadata, "num_decodes") == 0,
                "ATTENTION_NOT_SINGLE_FRESH_PREFILL:" + name)
        require(field(metadata, "seq_lens_list") == [length]
                and field(metadata, "actual_seq_lengths_q") == [length], "ATTENTION_LENGTHS_MISMATCH:" + name)
        require(field(metadata, "causal") is True, "NON_CAUSAL_ATTENTION_UNADAPTED:" + name)
        slot_meta = tensor_metadata(field(metadata, "slot_mapping"))
        slot_buffer = tensor_metadata(field(field(table, "slot_mapping"), "gpu"))
        block_meta = tensor_metadata(field(metadata, "block_tables"))
        block_buffer = tensor_metadata(field(field(table, "block_table"), "gpu"))
        require(slot_meta["storage_identity"] == slot_buffer["storage_identity"]
                and slot_meta["storage_offset"] == slot_buffer["storage_offset"]
                and slot_meta["shape"] == [length], "ATTENTION_SLOT_BUFFER_UNALIGNED:" + name)
        require(block_meta["storage_identity"] == block_buffer["storage_identity"]
                and block_meta["storage_offset"] == block_buffer["storage_offset"]
                and block_meta["shape"][0] == 1, "ATTENTION_BLOCK_BUFFER_UNALIGNED:" + name)
        observed_layers.append(name)
    return {"status": "READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE",
            "internal_request_id": req_id, "external_request_id": None,
            "external_mapping_status": "REQUIRES_REAL_DRIVER_OUTPUT_PROCESSOR_MAP",
            "request_object_id": id(request), "request_type": type(request).__module__ + "." + type(request).__name__,
            "num_requests": 1, "num_prompt_tokens": length, "num_computed_tokens": 0,
            "num_scheduled_tokens": length, "token_ids": prompt, "positions": positions,
            "token_count_evidence": {"request_num_tokens": request_num_tokens,
                                     "batch_num_tokens_no_spec": batch_num_tokens_no_spec[0],
                                     "batch_num_tokens_legacy": legacy_num_tokens,
                                     "legacy_field_role": "POST_SAMPLING_COUNTER_NOT_PREFILL_AUTHORITY"},
            "token_ids_sha256": hashlib.sha256(json.dumps(prompt, separators=(",", ":")).encode()).hexdigest(),
            "input_tensor_metadata": actual_inputs, "block_ids": blocks, "block_size": block_size,
            "slot_mapping": slots, "attention_metadata_keys": observed_layers,
            "tensor_values_copied_from_npu": False, "device_input_bytes_equality": "NOT_ESTABLISHED_BY_CPU_MIRROR"}


def discover_logits_boundary(model, *, actual_logits_processor_type, actual_vocab_size):
    """Identify the processor/head that the actual compute_logits calls.

    No configured layer name, model family allowlist or compute_logits execution.
    Requires a directly inspectable self.processor(self.head, hidden_states)
    call; models with other contracts are explicitly unadapted.
    """
    require(actual_logits_processor_type.__module__ == "vllm.model_executor.layers.logits_processor"
            and actual_logits_processor_type.__name__ == "LogitsProcessor", "ACTUAL_VLLM_LOGITS_PROCESSOR_TYPE_REQUIRED")
    compute = field(model, "compute_logits")
    identity, source = source_identity(compute)
    tree = ast.parse(source)
    candidates = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if not (isinstance(call.func.value, ast.Name) and call.func.value.id == "self" and len(call.args) >= 2):
            continue
        head_expr, hidden = call.args[:2]
        if not (isinstance(head_expr, ast.Attribute) and isinstance(head_expr.value, ast.Name)
                and head_expr.value.id == "self" and isinstance(hidden, ast.Name)):
            continue
        processor = getattr(model, call.func.attr, None)
        if isinstance(processor, actual_logits_processor_type):
            candidates.append((call.func.attr, head_expr.attr, processor, getattr(model, head_expr.attr, None)))
    require(len(candidates) == 1, "LOGITS_CALL_CONTRACT_ABSENT_OR_AMBIGUOUS")
    processor_attr, head_attr, processor, head = candidates[0]
    modules = list(model.named_modules())
    processor_paths = [name for name, module in modules if module is processor]
    head_paths = [name for name, module in modules if module is head]
    require(len(processor_paths) == len(head_paths) == 1, "LOGITS_BOUNDARY_NOT_UNIQUE_MODULE")
    require(field(processor, "logits_as_input") is False, "LOGITS_AS_INPUT_CONTRACT_UNADAPTED")
    require(isinstance(actual_vocab_size, int) and actual_vocab_size > 0, "ACTUAL_VOCAB_SIZE_REQUIRED")
    require(field(processor, "org_vocab_size") == actual_vocab_size
            and field(processor, "vocab_size") == actual_vocab_size, "PROCESSOR_VOCAB_MISMATCH")
    require(field(head, "num_embeddings") >= actual_vocab_size, "LM_HEAD_VOCAB_TOO_SMALL")
    apply = field(field(head, "quant_method"), "apply")
    require(callable(apply) and len(inspect.signature(apply).parameters) >= 2, "LM_HEAD_APPLY_CONTRACT_UNAVAILABLE")
    get_logits = field(processor, "_get_logits")
    get_identity, get_source = source_identity(get_logits)
    get_tree = ast.parse(get_source)
    branch_guard = None
    dispatch_identity = None
    # Ascend's registered subclass has an explicit lmhead-TP dispatcher. Only
    # its real serial branch is adapted here; do not assume the base method runs.
    if (type(processor).__module__ == 'vllm_ascend.ops.vocab_parallel_embedding'
            and type(processor).__name__ == 'AscendLogitsProcessor'):
        body = get_tree.body[0].body
        require(len(body) == 1 and isinstance(body[0], ast.If), 'ASCEND_LOGITS_DISPATCH_UNADAPTED')
        branch = body[0]
        expected = ast.parse('if lmhead_tp_enable():\n    return self._get_logits_lmheadtp(hidden_states, lm_head, embedding_bias)\nelse:\n    return self._get_logits_normal(hidden_states, lm_head, embedding_bias)').body[0]
        require(ast.dump(branch) == ast.dump(expected), 'ASCEND_LOGITS_DISPATCH_UNADAPTED')
        dispatch_function = get_logits.__func__
        gate = dispatch_function.__globals__.get('lmhead_tp_enable')
        require(callable(gate), 'ASCEND_LMHEAD_TP_GATE_UNAVAILABLE')
        normal_logits = field(processor, '_get_logits_normal')
        normal_function = getattr(normal_logits, '__func__', None)
        require(normal_function is not None, 'ASCEND_NORMAL_LOGITS_METHOD_UNADAPTED')
        def branch_guard():
            require(getattr(processor._get_logits, '__func__', None) is dispatch_function
                    and getattr(processor._get_logits_normal, '__func__', None) is normal_function
                    and dispatch_function.__globals__.get('lmhead_tp_enable') is gate
                    and gate() is False, 'ASCEND_LMHEAD_TP_OR_DISPATCH_CHANGED')
            return True
        branch_guard()
        dispatch_identity = get_identity
        get_logits = normal_logits
        get_identity, get_source = source_identity(get_logits)
        get_tree = ast.parse(get_source)
    quant_calls = [node for node in ast.walk(get_tree) if isinstance(node, ast.Call)
                   and isinstance(node.func, ast.Attribute) and node.func.attr == "apply"
                   and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "quant_method"]
    require(len(quant_calls) == 1, "LOGITS_QUANT_APPLY_CONTRACT_UNADAPTED")
    quant_call = quant_calls[0]
    require(isinstance(quant_call.func.value.value, ast.Name) and quant_call.func.value.value.id == "lm_head"
            and len(quant_call.args) >= 2
            and isinstance(quant_call.args[0], ast.Name) and quant_call.args[0].id == "lm_head"
            and isinstance(quant_call.args[1], ast.Name) and quant_call.args[1].id == "hidden_states",
            "LOGITS_QUANT_APPLY_ARGUMENTS_UNADAPTED")
    return {"processor": processor, "lm_head": head, "branch_guard": branch_guard,
            "report": {"status": "STRUCTURAL_LOGITS_BOUNDARY_ONLY_NOT_EXECUTION_ACCEPTANCE",
                       "processor_path": processor_paths[0], "lm_head_path": head_paths[0],
                       "compute_processor_attribute": processor_attr, "compute_head_attribute": head_attr,
                       "compute_logits_source": identity, "get_logits_source": get_identity,
                       "ascend_dispatch_source": dispatch_identity,
                       "vocab_size": actual_vocab_size, "gather_contract": "Observe returned full-vocab tensor; None is not success",
                       "output_contract": [1, actual_vocab_size]}}


def actual_driver_request_map(output_processor):
    """Read while requests still exist; never reconstruct IDs from suffixes."""
    states = field(output_processor, "request_states")
    result = []
    for key, value in states.items():
        require(field(value, "request_id") == key, "DRIVER_REQUEST_STATE_ID_MISMATCH")
        result.append({"internal_request_id": key, "external_request_id": field(value, "external_req_id"),
                       "request_state_object_id": id(value)})
    return result


def require_host_driven_core(llm):
    """Default background EngineCoreProc can execute before advance; reject it."""
    core = field(field(llm, "llm_engine"), "engine_core")
    require(type(core).__module__ == "vllm.v1.engine.core_client"
            and type(core).__name__ == "InprocClient", "INPROC_ENGINE_CORE_REQUIRED_FOR_ENQUEUE_ADVANCE_ORDER")
    return core


def read_enqueued_identity(llm, enqueued_ids, *, seen_internal_request_ids=()):
    """After public enqueue, before advance/wait: actual IDs, no fabricated IDs.

    The host must call this on the supported InprocClient, after creating the
    one-request lease. Returned IDs remain assertions that the model pre-hook
    must corroborate against the actual input_batch/CachedRequestState.
    """
    require_host_driven_core(llm)
    require(isinstance(enqueued_ids, list) and len(enqueued_ids) == 1,
            "EXACTLY_ONE_ENQUEUED_REQUEST_REQUIRED")
    req_id = enqueued_ids[0]
    require(isinstance(req_id, str) and req_id and req_id not in set(seen_internal_request_ids),
            "MISSING_OR_REUSED_ENQUEUED_ID")
    mapping = actual_driver_request_map(field(field(llm, "llm_engine"), "output_processor"))
    require(len(mapping) == 1 and mapping[0]["internal_request_id"] == req_id,
            "ENQUEUE_DRIVER_REQUEST_MAP_MISMATCH")
    return dict(mapping[0], source="ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR",
                proof_scope="DRIVER_MAPPING_ONLY_REQUIRES_REAL_MODEL_PREHOOK_CORROBORATION")
