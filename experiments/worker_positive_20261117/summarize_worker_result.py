"""Public summary only: no raw statistics, tensors, native addresses or vendors."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(case):
    result_path = case/'result.json'
    report = json.loads(result_path.read_text())
    receipt_path = case/'serving/runtime/serving_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    rows = []
    for row in receipt['native_phases']:
        state = row.get('native', {})
        if isinstance(state, list):
            value = {key: row.get(key) for key in ('phase', 'model_output_sha256', 'model_output_finite',
                     'model_output_changed', 'native_calls', 'paused_after', 'baseline_control')}
            value['protected_logical_input_sha256'] = row.get('protected')
            value['native'] = []
            for event in state:
                current = event['current_typed_call']
                evidence = event.get('provider_observation')
                value['native'].append(dict(operator=event['operator'], status=event['status'],
                    actual_original_calls=event['actual_original_calls'],
                    current_typed_formals_verified=(current['executor_bound'] and current['formal_params_current']
                                                  and not current['cached_layout_used']),
                    layout=event['native_facts']['layout'],
                    receipt=evidence,
                    original_storage_preserved=(evidence['input_sha256_before']==evidence['input_sha256_after'])
                                               if evidence else None))
            rows.append(value)
            continue
        observation = state.get('last_observation', {})
        proof = observation.get('proof', {})
        before = observation.get('original_storage_sha256_before')
        after = observation.get('original_storage_sha256_after')
        value = {key: row.get(key) for key in ('phase','input_sha256',
                    'model_output_sha256','model_output_finite','native_verdict','model_output_changed')}
        value.update(actual_live_inputs_verified=observation.get('serving_live_inputs_verified'),
                     original_storage_preserved=(before == after) if before and all(before) else None,
                     original_storage_sha256=before, output_storage_sha256=observation.get('output_storage_sha256'),
                     actual_api=proof.get('api'), mode=state.get('mode'), hits=state.get('hits'),
                     input_mask=proof.get('input_mask'), changed_elements=observation.get('changed'),
                     statistics={name:observation[name] for name in
                         ('original_self','original_mat2','temporary_self','temporary_mat2','output')
                         if name in observation},
                     native_idle=state.get('inflight') == 0 and state.get('leases') == 0)
        rows.append(value)
    runtime_requests = {row['phase']: row for row in receipt['requests']}
    phases = [dict(phase=row['phase'],
                output_tokens=[output['tokens'] for output in row.get('response',{}).get('outputs',[])],
                output_text=[output['text'] for output in row.get('response',{}).get('outputs',[])],
                actual_external_request_mapping_matches=(row.get('mapping',{}).get('external_request_id')
                    == row.get('response',{}).get('external_request_id')),
                actual_internal_request_id_sha256=hashlib.sha256(
                    row['advance']['native']['serving']['expected_request_id'].encode()).hexdigest(),
                actual_model_request_matches_enqueue=(
                    runtime_requests[row['phase']]['actual_internal_request_id']
                    == row['mapping']['internal_request_id']
                    == row['advance']['native']['serving']['expected_request_id']),
                advance_status=row.get('advance',{}).get('status'),
                query_status=row.get('query',{}).get('status'))
              for row in report['phases']]
    qualification = {key: receipt['qualification'].get(key) for key in (
        'status','recovery_verified','failure','drain_status','completed_phases','model_output_changed',
        'first_live_validation','manifest_sha256','baseline_control','coverage_scope','transaction_scope')}
    profile = receipt['qualification'].get('profile')
    if profile:
        qualification['profile'] = {key:profile[key] for key in (
            'validation_level','model_class','model_type','operator','boundary','soc','dtype','format',
            'execution_mode','tp','rank','logical_device','build_profile_sha256','environment_fingerprint')}
    return dict(schema='demoncat.worker_positive.summary.v1', source=str(case),
                source_hashes={'result.json':digest(result_path),'serving_receipt.json':digest(receipt_path)},
                status=report['status'], qualification=qualification,
                boundary=receipt['boundary'], phases=phases, native_phases=rows,
                installed_modules=report.get('setup',{}).get('modules'),
                deployment_preparation=report.get('deployment_preparation'),
                healthy_before=report.get('healthy_before'), healthy_after=report.get('healthy_after'),
                cleanup={key:report.get(key,{}).get('status') for key in ('finish','unregister','detach','unbind')},
                limitations=['TP1, single-request fresh eager prefill; one automatic logits target',
                             'Not multi-target/all_eligible, multi-token decode or text-quality acceptance',
                             'Native observations are auxiliary; statistics first difference is independently computed',
                             'Only this concrete loaded CANN and wheel deployment is covered'])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--case',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    args.out.write_text(json.dumps(summarize(args.case),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
