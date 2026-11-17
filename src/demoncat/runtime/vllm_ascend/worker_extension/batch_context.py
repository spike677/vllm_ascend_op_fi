"""Read actual first-decode rows from the reviewed synchronous v1 runner.

No request ID construction, model execution or NPU tensor reads here. Device
bytes and projection input lineage must be corroborated separately at runtime.
"""
from .serving_context import require,field,cpu_list,same_buffer_prefix,require_serial_eager


def snapshot_first_decode(runner,model,actual,context,expected):
    require_serial_eager(runner,expected_max_seqs=8)
    require(model is runner.model and context.model_instance is model,'FOREIGN_MODEL_CONTEXT')
    for name in ('in_profile_run','capturing','is_draft_model','flash_comm_v1_enabled','flashcomm_v2_enabled'):
        require(field(context,name) is False,'UNADAPTED_CONTEXT:'+name)
    require(context.cudagraph_runtime_mode.name=='NONE' and context.ubatch_slices is None
        and context.dp_metadata is None,'GRAPH_MICROBATCH_DP_UNADAPTED')
    require(actual.get('inputs_embeds') is None and actual.get('intermediate_tensors') is None,'NON_TOKEN_INPUT')
    batch=runner.input_batch;ids=list(batch.req_ids)
    require(len(ids)==batch.num_reqs==8 and len(set(ids))==8 and set(ids)==set(expected),'ACTUAL_REQUEST_SET_MISMATCH')
    require(batch.req_id_to_index=={key:i for i,key in enumerate(ids)},'REQUEST_INDEX_MISMATCH')
    require(set(runner.requests)==set(ids),'RUNNER_REQUEST_SET_MISMATCH')
    counts=cpu_list(runner.query_lens)
    require(counts==[1]*8 and runner.with_prefill is False,'FIRST_PURE_DECODE_REQUIRED')
    require(cpu_list(runner.query_start_loc.cpu)[:9]==list(range(9)),'ROW_BOUNDARIES_MISMATCH')
    require(context.num_tokens==8 and context.pad_size==0,'PADDED_DECODE_UNADAPTED')
    positions=cpu_list(runner.positions.cpu)[:8];host=cpu_list(runner.input_ids.cpu)[:8]
    computed=cpu_list(batch.num_computed_tokens_cpu)[:8];seq=cpu_list(runner.seq_lens.cpu)[:8]
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
    require(len(set(logical))==8,'LOGICAL_REQUEST_COLLISION')
    input_meta=same_buffer_prefix(actual['input_ids'],runner.input_ids.gpu,8,'torch.int32')
    position_meta=same_buffer_prefix(actual['positions'],runner.positions.gpu,8,'torch.int64')
    return dict(request_ids=ids,counts=counts,row_request_indices=list(range(8)),
        row_token_positions=positions,logical_ids=logical,prefixes=prefixes,pure_decode=True,
        input_ids=host,actual_input_metadata=input_meta,actual_position_metadata=position_meta)
