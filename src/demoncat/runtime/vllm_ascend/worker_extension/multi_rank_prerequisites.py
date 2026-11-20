"""Read-only TP2 development preflight; never changes TP1 admission."""
from copy import deepcopy
from .prerequisites import serial_eager_assessment, assessment, expectation, observed_field


def tp2_eager_assessment(runner, actual_tp_group, rank, *, expected_max_seqs=1):
    # Reuse every other refusal, including missing CP fields and multimodal.
    checks=deepcopy(serial_eager_assessment(runner, expected_max_seqs=expected_max_seqs)['checks'])
    path='vllm_config.parallel_config.tensor_parallel_size'
    checks=[expectation('mechanism_compatibility',path,2,observed_field(runner,path),
                        'M1_EXACT_TP2_REQUIRED') if row['field']==path else row for row in checks]
    checks += [expectation('deployment_identity','actual_tp_group.'+name,expected,
                          observed_field(actual_tp_group,name),'M1_ACTUAL_TP_GROUP_REQUIRED:'+name)
               for name,expected in (('world_size',2),('ranks',[0,1]),('rank_in_group',rank))]
    checks.append(expectation('deployment_identity','rank_in_m1_group',True,
                              type(rank) is int and rank in (0,1),'M1_RANK_ZERO_OR_ONE_REQUIRED'))
    report=assessment(checks)
    report.update(scope='M1_A2_EXISTING_018_TP2_PREFLIGHT_ONLY',
                  public_admission_changed=False,qualification_granted=False)
    return report


def snapshot_tp2_fresh_prefill(runner, model, hook_kwargs, context, *, actual_tp_group, rank,
                               expected_prompt_token_ids, expected_internal_request_id,
                               seen_internal_request_ids=()):
    """Read rank-local scheduling only; no group or mechanism admission."""
    from .serving_context import ContextRefused, _snapshot_serial_text_prefill
    report=tp2_eager_assessment(runner,actual_tp_group,rank)
    if report['unmet']:
        error=ContextRefused(report['primary']['code']);error.prerequisites=report
        raise error
    row=_snapshot_serial_text_prefill(runner,model,hook_kwargs,context,
        expected_prompt_token_ids=expected_prompt_token_ids,
        expected_internal_request_id=expected_internal_request_id,
        seen_internal_request_ids=seen_internal_request_ids)
    row['tensor_parallel']=dict(world_size=2,rank_in_group=rank,ranks=[0,1])
    row['qualification_granted']=False
    return row
