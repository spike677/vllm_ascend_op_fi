"""Compare full public tensor hashes, never process/object IDs or PASS labels.

This supplements (not replaces) the private native-borrow and launch review.
The stock observer's unresolved native function join remains explicitly open.
"""
import argparse,hashlib,json
from pathlib import Path

def need(value,reason):
    if not value:raise ValueError(reason)

def tensors(value):
    if isinstance(value,dict):
        if 'sha256' in value:
            need(set(('shape','dtype','sha256','finite'))<=set(value),'INCOMPLETE_TENSOR_HASH')
            need(value['finite'] is True and len(value['sha256'])==64,'INVALID_TENSOR_HASH')
            return 1
        return sum(tensors(v) for v in value.values())
    if isinstance(value,list):return sum(tensors(v) for v in value)
    return 0

def review(stock,private):
    need(stock['protocol']=='EXPLICIT_RESEARCH_STOCK_ONLY' and stock['fault_authorized'] is False,'STOCK_PROTOCOL')
    need(not stock['host']['ownership_retained'] and not stock.get('cleanup_errors'),'STOCK_RELEASE')
    need(not private['ownership_retained'] and not private.get('cleanup_errors'),'PRIVATE_RELEASE')
    sf=stock['request']['forwards'];pf=private['request']['worker']['forwards']
    need(len(sf)==len(pf)==1,'ONE_MATCHED_FORWARD')
    s,p=sf[0],pf[0];need(s['input']==p['input'],'MODEL_INPUT_CHANGED')
    need(len(s['calls'])==len(p['calls'])==28,'ACTUAL_LAYER_COUNT')
    rows=[]
    for i,(a,b) in enumerate(zip(s['calls'],p['calls'])):
        need(a['module_path']==b['module_path'] and a['ordinal']==b['ordinal']==i,'CALL_ALIGNMENT')
        for c in (a,b):
            need(c['actual_operator']=='torch_npu.npu_fused_infer_attention_score' and c['original_calls']==1 and c['original_returned'] and c['inputs_unchanged'],'ACTUAL_RETURN')
            need(tensors(c['input'])>=4 and tensors(c['output'])==2,'FULL_QKV_MASK_OUTPUT_HASHES')
            need(c['input']['input_layout']=='TND','REAL_TND_REQUIRED')
        need(a['input']==b['input'],'COMPLETE_FIA_INPUT_DIFFERENCE:'+str(i))
        need(a['output']==b['output'],'COMPLETE_FIA_OUTPUT_DIFFERENCE:'+str(i))
        rows.append(dict(path=a['module_path'],complete_inputs_exact=True,complete_outputs_exact=True))
    for key in ('hidden','logits'):
        need(tensors(s[key])==tensors(p[key])==1 and s[key]==p[key],'MODEL_OUTPUT_DIFFERENCE:'+key)
    return dict(status='STOCK_PRIVATE_FULL_PUBLIC_HASH_BRIDGE_REVIEWED',calls=rows,hidden=s['hidden'],logits=s['logits'],fault_authorized=False,qualification_granted=False,
        limits=['Private native borrow/ELF requires separate review.', 'Stock census native function join remains unresolved; public complete tensor comparison is independently observable.', 'No M128 control, fault, OFF, or second scope established by this bridge.'])

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('stock','private','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();v=review(json.loads(a.stock.read_bytes()),json.loads(a.private.read_bytes()))
    for n in ('stock','private'):v[n+'_result_sha256']=hashlib.sha256(getattr(a,n).read_bytes()).hexdigest()
    v['reader_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.out.write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8');print(v['status'])
