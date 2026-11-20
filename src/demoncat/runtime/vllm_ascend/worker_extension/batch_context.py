"""Read actual first-decode rows from the reviewed synchronous v1 runner.

No request ID construction, model execution or NPU tensor reads here. Device
bytes and projection input lineage must be corroborated separately at runtime.
"""
from .serving_context import require,field,cpu_list,same_buffer_prefix,require_serial_eager


def snapshot_first_decode(runner,model,actual,context,expected):
    """Retain the already reviewed B8-only quantization Provider contract."""
    result = snapshot_first_decode_rows(runner,model,actual,context,expected,expected_max_seqs=8)
    require(result['request_count']==8,'ACTUAL_REQUEST_SET_MISMATCH')
    return result


def snapshot_first_decode_rows(runner,model,actual,context,expected,*,expected_max_seqs=8):
    """Read the actual active rows within the reviewed serial capacity of 1 or 8.

    Capacity is a deployment condition, not request count. No shape[0]-based
    request inference occurs, and this reader does not widen a Provider's own
    B8, model, ABI, or Native qualification. Mixed/chunked/padded batches still
    refuse until a separate scheduler adapter can prove their row mapping.
    """
    require_serial_eager(runner,expected_max_seqs=expected_max_seqs)
    return _snapshot_first_decode_rows(runner,model,actual,context,expected,expected_max_seqs=expected_max_seqs)


def _snapshot_first_decode_rows(runner,model,actual,context,expected,*,expected_max_seqs):
    # Caller must supply the separately reviewed deployment gate.
    require(model is runner.model and context.model_instance is model,'FOREIGN_MODEL_CONTEXT')
    for name in ('in_profile_run','capturing','is_draft_model','flash_comm_v1_enabled','flashcomm_v2_enabled'):
        require(field(context,name) is False,'UNADAPTED_CONTEXT:'+name)
    require(context.cudagraph_runtime_mode.name=='NONE' and context.ubatch_slices is None
        and context.dp_metadata is None,'GRAPH_MICROBATCH_DP_UNADAPTED')
    require(actual.get('inputs_embeds') is None and actual.get('intermediate_tensors') is None,'NON_TOKEN_INPUT')
    batch=runner.input_batch;ids=list(batch.req_ids)
    count=len(ids)
    require(type(batch.num_reqs) is int and 1<=count==batch.num_reqs<=expected_max_seqs
        and all(isinstance(key,str) and bool(key) for key in ids)
        and len(set(ids))==count and set(ids)==set(expected),'ACTUAL_REQUEST_SET_MISMATCH')
    require(batch.req_id_to_index=={key:i for i,key in enumerate(ids)},'REQUEST_INDEX_MISMATCH')
    require(set(runner.requests)==set(ids),'RUNNER_REQUEST_SET_MISMATCH')
    counts=cpu_list(runner.query_lens)
    require(counts==[1]*count and all(type(n) is int for n in counts)
        and runner.with_prefill is False,'FIRST_PURE_DECODE_REQUIRED')
    require(cpu_list(runner.query_start_loc.cpu)[:count+1]==list(range(count+1)),'ROW_BOUNDARIES_MISMATCH')
    require(type(context.num_tokens) is int and context.num_tokens==count and context.pad_size==0,'PADDED_DECODE_UNADAPTED')
    positions=cpu_list(runner.positions.cpu)[:count];host=cpu_list(runner.input_ids.cpu)[:count]
    computed=cpu_list(batch.num_computed_tokens_cpu)[:count];seq=cpu_list(runner.seq_lens.cpu)[:count]
    tokens=cpu_list(batch.token_ids_cpu);logical=[];prefixes=[]
    for i,key in enumerate(ids):
        req=runner.requests[key];binding=expected[key]
        require(req.req_id==key and req.prompt_token_ids==binding['prompt'],'FROZEN_PROMPT_MISMATCH')
        require(not req.mm_features and req.prompt_embeds is None and req.lora_request is None,'UNADAPTED_REQUEST')
        require(req.sampling_params.n==1,'FANOUT_UNADAPTED')
        output=list(req.output_token_ids);n=len(req.prompt_token_ids)
        require(len(output)==1 and req.num_computed_tokens==computed[i]==positions[i]==n,'NOT_FIRST_DECODE')
        require(req.num_tokens==n+1 and seq[i]==n+1,'SEQUENCE_LENGTH_MISMATCH')
        require(tokens[i][:n+1]==req.prompt_token_ids+output and host[i]==output[0],'CPU_INPUT_LINEAGE_MISMATCH')
        logical.append(binding['logical_id']);prefixes.append(dict(prompt=list(req.prompt_token_ids),output=output))
    require(len(set(logical))==count,'LOGICAL_REQUEST_COLLISION')
    input_meta=same_buffer_prefix(actual['input_ids'],runner.input_ids.gpu,count,'torch.int32')
    position_meta=same_buffer_prefix(actual['positions'],runner.positions.gpu,count,'torch.int64')
    return dict(request_ids=ids,counts=counts,row_request_indices=list(range(count)),
        request_count=count,token_row_count=sum(counts),scheduler_capacity=expected_max_seqs,
        row_token_positions=positions,logical_ids=logical,prefixes=prefixes,pure_decode=True,
        input_ids=host,actual_input_metadata=input_meta,actual_position_metadata=position_meta)
