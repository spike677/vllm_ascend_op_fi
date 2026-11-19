"""Bounded review export from actual Host receipts; never writes acceptance records.

Receipts are operator-supplied evidence, not signed attestations. Successful
projection is NOT scientific acceptance or deployment registration.
"""
from copy import deepcopy
import hashlib
import json

MAX_SUMMARY_BYTES=64*1024
MAX_RECEIPT_BYTES=16*1024*1024
_FIELDS=('torch','torch_npu','vllm','vllm_ascend','cann','dtype','tp','pp','dp','eager',
         'model_class','model_config_sha256','live_config_matches_file','hardware_family')
_RESOURCES=('inflight','leases','registered','enabled','native_registered','owned_hooks',
            'owned_api_patches','owned_method_patches','owned_normal_readback_bytes','owned_row_modes')
_EFFECT=('phase','request_id','request_ids','actual_internal_request_id','input_sha256',
    'hidden_logits_sha256','model_output_sha256','output','effect','last_fault_effect','fault_effect',
    'hidden_changed','logits_changed','output_changed','applied','changed_elements','max_abs_diff_finite',
    'native_changed_quant_bytes','native_changed_scale_bytes','quant','scale','selected_site',
    'matched_off_and_health_confirmed','recovery_proven','recovered','complete','failure','failures',
    'error','status','mechanism','report','events','normal','fault','off','health',
    'decode','prefill','logical_ids','row_request_indices','row_token_positions','request_count',
    'hidden_sha256','logits_sha256','hidden_finite','logits_finite','logits_returned','returned',
    'top1','projection','functional_effect_observed','fault_attempted','actual_fault_hits',
    'batch','batch_phase','selected_site','original_calls','target','elements','dtype','shape')

def _bytes(x):
    return json.dumps(x,sort_keys=True,allow_nan=False,separators=(',',':')).encode()

def _compact(x,depth=0):
    if depth>10:raise ValueError('SUMMARY_DEPTH_LIMIT')
    if x is None or type(x) in (bool,int,float):return x
    if isinstance(x,str):
        if len(x)>1024:raise ValueError('SUMMARY_TEXT_LIMIT')
        return x
    if isinstance(x,list):
        if len(x)>128:raise ValueError('SUMMARY_ROW_LIMIT')
        return [_compact(v,depth+1) for v in x]
    if isinstance(x,dict):
        # Explicit projection: no tensor lists, input text, native raw records.
        return {k:_compact(v,depth+1) for k,v in x.items() if k in _EFFECT}
    raise ValueError('SUMMARY_NON_JSON_VALUE')

def summarize_scope(receipt):
    raw=_bytes(receipt)
    if len(raw)>MAX_RECEIPT_BYTES:raise ValueError('HOST_RECEIPT_LIMIT')
    history=receipt.get('history',[])
    states=[h['response']['provider_state'] for h in history if isinstance(h,dict)
        and isinstance(h.get('response'),dict) and isinstance(h['response'].get('provider_state'),dict)]
    errors=[]
    if not states:errors.append('NO_PROVIDER_STATE')
    state=states[-1] if states else {}
    use=state.get('deployment_acceptance_use',{})
    actual=use.get('actual') or {}
    if any(k not in actual or actual[k] is None for k in _FIELDS):errors.append('ACTUAL_DEPLOYMENT_FIELDS_MISSING')
    if actual.get('hardware_family') not in ('A2','A3'):errors.append('HARDWARE_FAMILY_UNOBSERVED')
    if not use or use.get('scope_valid') is not True:errors.append('VALID_ACCEPTANCE_USE_MISSING')
    if any(s.get('evidence_complete') is False for s in states):errors.append('INCOMPLETE_EVIDENCE')
    ev=state.get('evidence',state.get('serving',{}))
    serving=ev.get('serving',ev)
    rows=serving.get('actual_requests',serving.get('rows',[]))
    if not rows:rows=ev.get('actual_requests',[])
    if not rows:errors.append('ACTUAL_REQUEST_ROWS_MISSING')
    recovery=ev.get('matched_off_and_health_confirmed',serving.get('matched_off_and_health_confirmed',serving.get('recovery_proven')))
    if recovery is not True:errors.append('OFF_AND_HEALTH_RECOVERY_UNCONFIRMED')
    if receipt.get('mechanism')=='quant.batch_shared_scale' and not any(r.get('phase')=='health' for r in rows):
        errors.append('BATCH_EXTERNAL_HEALTH_RECEIPT_REQUIRES_SEPARATE_REVIEW')
    resources={k:state[k] for k in _RESOURCES if k in state}
    resources.update({k:serving[k] for k in _RESOURCES if k in serving})
    if any(k not in resources for k in ('inflight','leases','registered','enabled','native_registered')):
        errors.append('RESOURCE_FIELDS_MISSING')
    if any(v not in (False,0) for v in resources.values()):errors.append('RESOURCES_REMAIN')
    if receipt.get('ownership_retained') is not False or receipt.get('status')!='DETACHED':errors.append('HOST_NOT_DETACHED')
    # Preserve all request rows (not just a successful phase), but project values.
    return dict(schema='demoncat.deployment_scope_summary.v1',mechanism=receipt.get('mechanism'),
        owner=receipt.get('owner'),worker=deepcopy(receipt.get('worker')),source_receipt_sha256=hashlib.sha256(raw).hexdigest(),
        actual={k:actual.get(k) for k in _FIELDS},registration_source=use.get('registration_source'),
        permit=deepcopy(use.get('permit')),requests=_compact(rows),effects=_compact(ev),
        recovery_confirmed=recovery,resources=resources,ownership_retained=receipt.get('ownership_retained'),
        evidence_complete=not errors,missing_or_failed=errors,review_required=True,acceptance_granted=False)

def summarize_deployment(receipts):
    if not isinstance(receipts,(list,tuple)) or len(receipts)!=2:raise ValueError('TWO_INDEPENDENT_SCOPES_REQUIRED')
    scopes=[summarize_scope(r) for r in receipts]
    errors=[]
    if not scopes[0]['owner'] or scopes[0]['owner']==scopes[1]['owner']:errors.append('INDEPENDENT_OWNER_REQUIRED')
    for k in ('mechanism','actual','registration_source'):
        if scopes[0][k]!=scopes[1][k]:errors.append('SCOPE_MISMATCH:'+k)
    if any(not s['evidence_complete'] for s in scopes):errors.append('SCOPE_EVIDENCE_INCOMPLETE')
    result=dict(schema='demoncat.deployment_summary.v1',scopes=scopes,missing_or_failed=errors,
        review_ready=not errors,review_required=True,acceptance_granted=False,
        registration_written=False,source_kind='OPERATOR_HOST_RECEIPTS_NOT_SIGNED_ATTESTATION')
    if len(_bytes(result))>MAX_SUMMARY_BYTES:
        return dict(schema=result['schema'],review_ready=False,review_required=True,acceptance_granted=False,
            registration_written=False,missing_or_failed=['SUMMARY_LIMIT_EXCEEDED'],
            source_receipt_sha256=[s['source_receipt_sha256'] for s in scopes],summary_payload_omitted=True)
    return result
