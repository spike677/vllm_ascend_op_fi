"""Additional checks for new r7.4.1 routes; no registration authority."""
import argparse
import json
import math
from pathlib import Path
from tools.review_product_mode import review


def route_checks(data):
    mechanism=data['mechanism']
    if mechanism not in ('18.variance_over4','18.variance_under4','external.conv3d_output_gain'):
        raise ValueError('R741_ROUTE_NOT_SUPPORTED')
    def need(ok,code):
        if not ok:raise ValueError(code)
    for scope in data['scopes']:
        for request in scope['requests'][:3]:
            for rank in request['result']['ranks']:
                if mechanism in ('18.variance_over4','18.variance_under4'):
                    multiplier,gain=(4,.5) if mechanism=='18.variance_over4' else (.25,2.)
                    calls=rank.get('norm_variance_effect',{}).get('calls',[])
                    need(len(calls)==rank['coverage']['applied']>0,'VARIANCE_EFFECT_CALLS_MISSING')
                    for c in calls:
                        need(c.get('named_variant')==mechanism and c.get('mean_square_multiplier')==multiplier
                             and c.get('gain')==gain and c.get('epsilon_exact') is False,'VARIANCE_PROXY_RULE_CHANGED')
                        shape=c.get('shape')
                        need(isinstance(shape,list) and shape and all(type(n) is int and n>0 for n in shape)
                             and c.get('covered_elements')==math.prod(shape),'WHOLE_NORM_OUTPUT_NOT_PROVEN')
                        need(type(c.get('changed_elements')) is int and 0<=c['changed_elements']<=c['covered_elements']
                             and c.get('changed')==int(c['changed_elements']>0),'VARIANCE_CHANGE_COUNTS_INVALID')
                        need(c.get('request_ids')==[rank['request_id']] and c.get('consumer_operator')
                             and c.get('consumed') is True and c.get('restored') is True
                             and c.get('before_sha256') and c['before_sha256']==c.get('restored_sha256'),
                             'VARIANCE_REAL_CONSUMER_OR_RESTORE_MISSING')
                else:
                    proof=rank.get('external_provider',{})
                    need(proof.get('stage')=='vision' and proof.get('encoder_calls')==1
                         and proof.get('embedding_calls',0)>=1 and proof.get('readonly_inputs_weights_unchanged') is True,
                         'EXTERNAL_IMAGE_REAL_CONSUMER_OR_READONLY_UNPROVEN')
                    need(rank.get('restore_kind')=='LIVE_REGION_BYTES','EXTERNAL_IMAGE_BYTE_RESTORE_REQUIRED')
                    producers=proof.get('native_producers',[])
                    need(len(producers)==rank['coverage']['applied']>0 and all(
                         p.get('native_operator') in ('aten::conv3d','aten::convolution')
                         and p.get('native_return_identity') is True
                         and p.get('request_ids')==[rank['request_id']]
                         and len(p.get('input_shape') or [])==5
                         and len(p.get('weight_shape') or [])==5 for p in producers),
                         'EXTERNAL_ACTUAL_CONV_PRODUCER_UNPROVEN')
    if mechanism.startswith('external.'):
        need(len(set(data.get('image_pixel_sha256',[])))>=3,'EXTERNAL_CANARY_PLUS_TWO_IMAGES_REQUIRED')
    return dict(route=mechanism,additional_checks='PASSED',qualification_granted=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('--output');a=p.parse_args()
    raw=json.loads(Path(a.receipt).read_bytes())
    result=dict(base=review(raw),route=route_checks(raw))
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if a.output:Path(a.output).write_text(text,encoding='utf-8')
    print(text)


if __name__=='__main__':main()
