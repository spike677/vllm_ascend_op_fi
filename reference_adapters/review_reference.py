"""Read-only independent reference-adapter review. Does not register a deployment."""
import argparse
import hashlib
import json
from pathlib import Path
from tools.review_product_mode import review

KINDS={'row7_mlp_last_row_zero':'MlpContributionLost','row8_decoder_last_row_zero':'DecoderPaddingRow',
    'row12_logits_columns_rotate':'LogitsColumnsRotate','row3_gate_fp16_round':'GateFp16Round',
    'row16_conv3d_compute_fp16':'VisionPrecisionSite','row32_conv3d_input_fp16':'ConvInputFp16RoundTrip',
    'row19_bf16_reverse_splitk':'ReverseSplitK','row34_bf16_before_bias':'EarlyBiasRound',
    'row32_conv3d_input_e4m3':'ConvInputE4M3','row32_conv3d_input_e5m2':'ConvInputE5M2',
    'row34_projection_mantissa5':'ProjectionMantissa5','row34_projection_mantissa3':'ProjectionMantissa3'}

def need(value,code):
    if not value:raise ValueError(code)

def run(directory,receipt,negative_file):
    root=Path(directory);path=root/receipt;data=json.loads(path.read_bytes())
    public=review(data);variant=data['mechanism'].removeprefix('external.');kind=KINDS[variant]
    neg=json.loads((root/negative_file).read_bytes())
    need(neg['missing_permit_cli']['exit_code']==4 and
        'EXTERNAL_CURRENT_ACCEPTANCE_PERMIT_REQUIRED' in neg['missing_permit_cli']['refusal_codes'],'REFERENCE_MISSING_PERMIT')
    need(data['scopes'][0]['duplicate_enable']['exit_code']==4,'REFERENCE_DUPLICATE_ENABLE')
    raw={};hashes={}
    for rank in range(2):
        p=root/f'external-input-rank{rank}.jsonl';content=p.read_bytes()
        hashes[p.name]=hashlib.sha256(content).hexdigest()
        for line in content.splitlines():
            chunk=json.loads(line)
            if chunk['kind']!=kind:continue
            need(chunk['rank']==rank,'REFERENCE_SIDECAR_RANK')
            for e in chunk['events']:
                need(len(e['request_ids'])==1,'REFERENCE_AMBIGUOUS_REQUEST')
                raw.setdefault((rank,e['request_ids'][0]),[]).append(e)
    totals={r:dict(calls=0,changed_calls=0,changed_elements=0,consumers=set(),distribution_l1=0.,
        rule_changed_elements=0,control_changed_elements=0,rule_effect_lower_bound=0) for r in range(2)}
    seen=set();scope_rule_effects=[]
    for scope in data['scopes']:
        scope_rule_effect=0
        for request in scope['requests'][:3]:
            for rank in request['result']['ranks']:
                key=(rank['rank'],rank['request_id']);seen.add(key);events=raw[key]
                need(len(events)==len(rank['sites'])==rank['coverage']['applied'],'REFERENCE_CALL_COUNT')
                for e,s in zip(events,rank['sites']):
                    need(all(e[k]==s[k] for k in ('site','changed','changed_elements','before_sha256',
                        'restored_sha256','consumer_operator','consumed','restored')),'REFERENCE_PUBLIC_SIDECAR')
                    need(e['consumed'] and e['restored'] and e['before_sha256']==e['restored_sha256'],'REFERENCE_RESTORE')
                    need(e['consumer_operator']!='vllm::maybe_chunk_residual',
                        'REFERENCE_SHAPE_ONLY_WRAPPER_IS_NOT_VALUE_CONSUMER')
                    if variant.startswith('row12_'):
                        need(e['consumer_operator']=='aten::_softmax' and e['consumer_sha256']==e['after_sha256']
                            and e['producer_sha256'],'REFERENCE_REAL_SAMPLER')
                    if variant.startswith('row3_'):
                        need(any(x in e['native_operator'] for x in ('swiglu','silu_and_mul')),'REFERENCE_NATIVE_GATE')
                    if variant.startswith(('row16_','row32_')):
                        need(e['readonly_inputs_weights_unchanged'] and e['native_return_identity'],'REFERENCE_READONLY_CONV')
                        if variant.startswith('row32_'):need(e['encoder_consumer_operator'],'REFERENCE_ENCODER')
                    if variant.startswith('row34_projection_'):
                        need(e['consumer_operator']=='_C_ascend::npu_add_rms_norm_bias'
                             and e['actual_projection_return_identity']
                             and e['fraction_bits']==(5 if variant.endswith('5') else 3),
                             'REFERENCE_ACTUAL_FUSED_PROJECTION_CONSUMER')
                    if variant.startswith(('row19_','row34_bf16_')):
                        need(e['control_passed'] and e['bf16_operands'] and not e['quantized_operand_recomputed'],'REFERENCE_BF16_CONTROL')
                        need(type(e.get('rule_changed_elements')) is int and type(e.get('control_changed_elements')) is int,
                            'REFERENCE_RULE_CONTROL_SEPARATION_REQUIRED')
                        rule=e['rule_changed_elements'];control=e['control_changed_elements'];actual=e['changed_elements']
                        need(rule>=0 and control>=0 and abs(rule-control)<=actual<=rule+control,
                            'REFERENCE_HAMMING_COUNTS_INCONSISTENT')
                        # At least rule-control positions differ from BOTH the
                        # forward reference and native output. This is a lower
                        # bound, not an invented exact intersection count.
                        bound=max(0,rule-control)
                        scope_rule_effect+=bound
                        totals[rank['rank']]['rule_effect_lower_bound']+=bound
                        if variant.startswith('row34_'):need(e['early_bf16_before_real_bias'],'REFERENCE_REAL_BIAS')
                    t=totals[rank['rank']];t['calls']+=1;t['changed_calls']+=e['changed']
                    t['changed_elements']+=e['changed_elements'];t['consumers'].add(e['consumer_operator'])
                    t['distribution_l1']+=e.get('distribution_l1',0.)
                    t['rule_changed_elements']+=e.get('rule_changed_elements',0)
                    t['control_changed_elements']+=e.get('control_changed_elements',0)
        scope_rule_effects.append(dict(scope=scope.get('scope'),rule_effect_lower_bound=scope_rule_effect))
    need(seen==set(raw),'REFERENCE_ORPHAN_REQUEST')
    changed=all(t['changed_calls']>0 for t in totals.values())
    if variant.startswith(('row19_','row34_bf16_')):
        # A native-vs-reference discrepancy alone does not establish that the
        # named change in accumulation order/rounding point made a difference.
        # All ranks must apply/consume/restore, but unchanged values are not
        # failures. Sharded operands can expose the numerical effect on only
        # one rank. Require attributable effect in EACH independent owner.
        changed=(all(t['calls']>0 for t in totals.values()) and len(scope_rule_effects)==2
            and all(s['rule_effect_lower_bound']>0 for s in scope_rule_effects))
    for t in totals.values():t['consumers']=sorted(t['consumers'])
    return dict(status='REFERENCE_A2_TP2_TWO_OWNERS_REVIEWED' if changed else 'EXECUTION_RECOVERY_PASSED_NO_NUMERIC_PHENOMENON',
        mechanism=data['mechanism'],semantics='PHENOMENON_PROXY_NOT_ORIGINAL_MECHANISM',qualification_granted=False,
        public_review=public,effects_by_rank=totals,scope_rule_effects=scope_rule_effects,receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),sidecar_sha256=hashes)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--receipt',required=True)
    p.add_argument('--negative',default='product_control_negative.json');p.add_argument('--output',required=True)
    a=p.parse_args();result=run(a.directory,a.receipt,a.negative)
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','mechanism','effects_by_rank')}))
