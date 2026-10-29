"""Targeted CPU regression: only the source-audited cross-B padding exception."""
import copy
import json
from pathlib import Path

from verify_causal_matrix import compare_input_params


def profile(batch,padding):
    return {'event':{'typed':{'b':batch}},'pod':{'inputParams.bSize':batch,
        'inputParams.InputParamsPH':padding,'inputParams.implMode':0,'inputParams.sparseType':3,
        'inputParams.scaleValue':0.0883883461356163,'inputParams.keepProb':1.0}}


def main():
    checks=[]
    a=profile(1,[0]*6);b=profile(8,[11,0,0,0,0,0])
    before=copy.deepcopy([a,b]);result=compare_input_params(a,b)
    assert result['semantic_fields_equal'] and not result['same_batch'] and not result['padding_equal']
    assert [a,b]==before
    checks.append({'name':'cross_batch_padding_only_allowed_and_original_bytes_preserved','status':'PASS'})
    assert compare_input_params(a,copy.deepcopy(a))['same_batch']
    checks.append({'name':'same_batch_same_padding_allowed','status':'PASS'})
    def rejects(name,left,right):
        try:compare_input_params(left,right)
        except ValueError:checks.append({'name':name,'status':'PASS_REJECTED'});return
        raise AssertionError('Unexpected acceptance: '+name)
    c=copy.deepcopy(a);c['pod']['inputParams.InputParamsPH']=[11,0,0,0,0,0]
    rejects('same_batch_padding_difference_rejected',a,c)
    for field,value in [('inputParams.implMode',2),('inputParams.sparseType',1),('inputParams.scaleValue',0.5),('inputParams.keepProb',0.9)]:
        c=copy.deepcopy(b);c['pod'][field]=value
        rejects('semantic_change_rejected_'+field,a,c)
    c=copy.deepcopy(b);c['pod']['inputParams.InputParamsPH']=[0]*5
    rejects('malformed_padding_width',a,c)
    c=copy.deepcopy(b);c['pod']['inputParams.InputParamsPH']=[256]*6
    rejects('malformed_padding_value',a,c)
    report={'status':'PASS_TARGETED_CPU_PADDING_REGRESSION','npu_calls':0,'checks':checks,
            'does_not_confirm_new_native_result':True,'semantic_exception_scope':'Only cross-batch InputParamsPH alignment bytes'}
    (Path(__file__).parent/'inputparams_padding_cpu_checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'checks':len(checks)}))


if __name__=='__main__':main()
