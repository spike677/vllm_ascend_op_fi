"""Map request_00 across B1/B8; never compare different whole-batch statistics.

This is separate from each batch's B/F causal comparison. Matching prompts do
not imply identical intermediate Q/K/V when a model's batch shape changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(folder):
    result=json.loads((folder/'result.json').read_bytes());calls={}
    for ref in result['calls']:
        path=folder/ref['file'];assert sha(path)==ref['sha256']
        call=json.loads(path.read_bytes());assert call['request_ids'][0]=='request_00'
        calls[call['step'],call['layer_index']]=call
    phases=json.loads((folder/'phase_manifest.json').read_bytes())['phases']
    assert len(calls)==280 and len(phases)==10
    return calls,phases


def logits(folder,phase):
    path=folder/phase['private_logits_file'];meta=phase['logits'];assert sha(path)==meta['full_sha256']
    values=np.frombuffer(path.read_bytes(),dtype='<f2').reshape(meta['shape'])
    assert hashlib.sha256(values[0].tobytes()).hexdigest()==meta['per_batch_sha256'][0]
    return values[0]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--b1',type=Path,required=True)
    p.add_argument('--b8',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    one,phases1=capture(a.b1);eight,phases8=capture(a.b8);rows=[]
    for step in range(10):
        phase=phases1[step];assert phase['stage']==phases8[step]['stage'] and phase['mode']==phases8[step]['mode']
        layers=[]
        for layer in range(28):
            left,right=one[step,layer],eight[step,layer]
            changed=[kind for kind in ('query','key','value') if
                     left['inputs_before'][kind]['per_batch_sha256'][0]!=right['inputs_before'][kind]['per_batch_sha256'][0]]
            output=left['output']['per_batch_sha256'][0]!=right['output']['per_batch_sha256'][0]
            layers.append({'layer_index':layer,'target_changed_input_names':changed,'target_output_changed':output})
        x,y=logits(a.b1,phase),logits(a.b8,phases8[step]);delta=np.abs(x.astype('float64')-y.astype('float64'))
        rows.append({'step':step,'stage':phase['stage'],'mode':phase['mode'],
                     'first_target_attention_boundary_difference':next((r for r in layers if r['target_changed_input_names'] or r['target_output_changed']),None),
                     'target_layer0_qkv_bitwise_equal':not layers[0]['target_changed_input_names'],
                     'target_layers':layers,'target_logits_changed_elements':int(np.count_nonzero(x!=y)),
                     'target_logits_max_abs':float(delta.max()),'B1_top1':int(x.argmax()),'B8_top1':int(y.argmax())})
    report={'schema':'fault28.m_only.cross_batch_target.v1','status':'CROSS_BATCH_TARGET_REVIEW_COMPLETE',
            'target_request':'request_00','phases':rows,'analysis_source_sha256':sha(Path(__file__)),
            'limits':['Whole-batch B1/B8 msprobe statistics are not aligned or used as a target-request inference.',
                      'Different B1/B8 intermediate inputs do not invalidate within-batch M-only contrasts, but cannot establish native batch-to-FA causality.',
                      'Only complete target logits are read from private bytes; per-layer target comparisons use sealed per-request logical SHA.']}
    with a.output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':report['status'],'native_prefill':{k:rows[2][k] for k in (
        'target_layer0_qkv_bitwise_equal','first_target_attention_boundary_difference','target_logits_changed_elements')}}))


if __name__=='__main__':main()
