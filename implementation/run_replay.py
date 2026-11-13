"""One frozen reduction-error candidate, bounded diagnostic/model/control phases."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'analysis'))
from run_search import Search,write,sha
from case_factory import create
from compare_statistics import compare
from accept_statistics import accept
from audit_runtime import audit
from review_formal import review,values,logits

CANDIDATE={'candidate_id':'denominator_replay_m128','input_id':'input_00',
           'B_M':128,'B_S2':128,'F_M':128,'F_S2':128,'variant':'online_softmax_denominator_replay'}
B_MODE='coupled_m128'
F_MODE='coupled_replay_m128'

def array(folder,kind='attention'):
    import numpy as np
    _,v=values(folder);r=v[None if kind=='logits' else 0,kind]
    p=folder/r['private_file'];assert sha(p)==r['full_sha256']
    return np.fromfile(p,dtype='<f2').reshape(r['shape'])

def same(a,b):
    return a.keys()==b.keys() and all(a[k]['full_sha256']==b[k]['full_sha256'] and
             a[k]['request_sha256']==b[k]['request_sha256'] for k in a)

def existing(path,fn):
    if path.exists():return json.loads(path.read_bytes())
    result=fn();write(path,result);return result

def diagnostic(search):
    import numpy as np
    folders={}
    fixture=ROOT/'private/analytic_fixture/qkv_manifest.json'
    for arm,mode in [('B',B_MODE),('F',F_MODE)]:
        folders[arm]=search.run_case(create('analytic_'+arm,'input_00','calibration',mode,qkv=fixture,capture=False))
        assert folders[arm] is not None,'Analytic NPU execution failed'
    def check():
        b,f=map(array,(folders['B'],folders['F']))
        expected=np.ones(2048,dtype=np.float64)
        q=np.arange(128,2048,dtype=np.float64)+1
        expected[128:]=q/(q+128)
        rounded=expected.astype(np.float16)
        delta=np.abs(f.astype(np.float32)-rounded[None,None,:,None].astype(np.float32))
        ulp=np.spacing(rounded).astype(np.float32)
        bad=int(np.count_nonzero(delta>ulp[None,None,:,None]))
        assert np.all(b==1) and np.all(f[:,:,:128,:]==1) and bad==0
        native={k:audit(v) for k,v in folders.items()}
        assert all(x['actual_key']=='119000000000400000' for x in native['F']['census'])
        return {'status':'PASS_ANALYTIC_NATIVE_OPERATOR','elements_checked':int(f.size),
            'baseline_all_exactly_one':True,'q0_127_unchanged':True,'outside_one_fp16_ulp':bad,
            'max_abs_from_fp16_rounded_formula':float(delta.max()),'exact_rounded_formula_elements':int(np.count_nonzero(delta==0)),
            'formula':'q<128:1; else (q+1)/(q+1+128)',
            'samples':[{'q':q,'B':float(b[0,0,q,0]),'F':float(f[0,0,q,0]),'expected_fp16':float(rounded[q])} for q in (0,127,128,255,511,1023,2047)],
            'scope':'Synthetic diagnostic only, not natural-model acceptance',
            'native_runtime':{k:{x:v[x] for x in ('status','actual_FIA_executions','private_binary_sha256')} for k,v in native.items()}}
    return existing(ROOT/'reports/analytic.json',check)

def calibration(search):
    qkv=search.harvest('input_00');folders={}
    for arm,mode in [('B',B_MODE),('F',F_MODE)]:
        folders[arm]=search.run_case(create('real_'+arm,'input_00','calibration',mode,qkv=qkv))
        assert folders[arm] is not None
    def check():
        r=compare(*(folders[a]/'observation/step0/rank0/dump.json' for a in ('B','F')),rtol=1e-5,atol=1e-8)
        write(ROOT/'reports/calibration_independent.json',r);gate=accept(r)
        assert gate['accepted'],gate['reasons']
        for arm,folder in folders.items():write(ROOT/'reports'/('calibration_native_'+arm+'.json'),audit(folder))
        # Independent legal-tiling control from the previous immutable run.
        old=Path('/home/fault28-visible-r2-20261113')
        state=json.loads((old/'search_state.json').read_bytes())
        previous=old/state['cases']['stage3_input_00_m128_F']['folder']
        old_qkv=old/state['cases']['h_input_00']['folder']/'qkv_manifest.json'
        a=json.loads(qkv.read_bytes());z=json.loads(old_qkv.read_bytes())
        qkv_same=all(a[k]['sha256']==z[k]['sha256'] for k in ('query','key','value'))
        bridge=qkv_same and same(values(folders['B'])[1],values(previous)[1])
        assert bridge,'Healthy new compiled template differs from old legal M128 control'
        return dict(gate,healthy_old_vendor_bridge_exact=bridge,source_control=str(previous),
                    same_real_QKV=qkv_same,B=str(folders['B']),F=str(folders['F']))
    return existing(ROOT/'reports/calibration_gate.json',check)

def formal(search,label):
    folders=[]
    for arm,mode in [('B',B_MODE),('F',F_MODE)]:
        folder=search.run_case(create(label+'_'+arm,'input_00','model',mode,steps=3))
        assert folder is not None,'Full model execution failed'
        folders.append(folder)
    out=ROOT/'reports'/label
    if out.exists():r=json.loads((out/'review.json').read_bytes())
    else:r=review(*folders,out,CANDIDATE)
    assert r['accepted_before_controls'],'Full-model fixed statistical gate not met'
    return folders,r

def controls(search):
    import numpy as np
    primary,_=formal(search,'formal');restart,_=formal(search,'restart')
    br,bv=values(primary[0]);fr,fv=values(primary[1])
    assert same(bv,values(restart[0])[1]) and same(fv,values(restart[1])[1]),'Independent restart changed outputs'
    off=search.run_case(create('error_OFF','input_00','model',B_MODE,steps=3,capture=False))
    assert off is not None and same(bv,values(off)[1]),'Disabling replay did not restore baseline exactly'
    qkv=search.harvest('input_00');folders={}
    for batch in (1,8):
        for arm,mode in [('healthy',B_MODE),('fault',F_MODE),('auto','batch_coupled_replay_m128')]:
            name=arm+str(batch)
            folders[name]=search.run_case(create('ctrl_'+name,'input_00','calibration',mode,qkv=qkv,batch=batch,capture=False))
            assert folders[name] is not None
    reports={k:audit(v) for k,v in folders.items()}
    healthy_equal=np.array_equal(array(folders['healthy1'])[0],array(folders['healthy8'])[0])
    fault_equal=np.array_equal(array(folders['fault1'])[0],array(folders['fault8'])[0])
    auto8_equal=np.array_equal(array(folders['auto8']),array(folders['fault8']))
    assert healthy_equal and fault_equal and auto8_equal,'Fixed QKV/fixed strategy cross-batch control failed'
    assert {r['actual_key'] for r in reports['auto1']['census']}=={'109000000000000000'}
    assert {r['actual_m'] for r in reports['auto1']['census']}=={512}
    assert {r['actual_key'] for r in reports['auto8']['census']}=={'119000000000400000'}
    r={'status':'PASS_RESTART_OFF_AND_FIXED_QKV_BATCH_CONTROLS','restart_all_113_arrays_all_8_requests_exact':True,
       'OFF_all_113_arrays_all_8_requests_exact':True,'OFF_definition':'disable denominator replay; hold M128/S2=128',
       'same_QKV_fixed_healthy_strategy_B1_B8_exact':healthy_equal,
       'same_QKV_fixed_fault_strategy_B1_B8_exact':fault_equal,'auto_B8_equals_fixed_fault_exact':auto8_equal,
       'auto_B1_actual_M_S2':[512,512],'auto_B1_replay_enabled':False,
       'auto_B8_actual_M_S2':[128,128],'auto_B8_replay_enabled':True,
       'per_case':{k:{x:v[x] for x in ('status','actual_FIA_executions','layer_request_executions')} for k,v in reports.items()},
       'limits':['Controlled new arithmetic error; historical default FIA root cause not reproduced.',
                 'AUTO mixes M change and denominator error; fixed-M B/F isolates the arithmetic error.',
                 'Prefill only; no continuous decode/task accuracy acceptance.']}
    write(ROOT/'reports/controls.json',r);return r

def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['screen','formal','controls','all']);a=p.parse_args()
    client=ROOT/'build/inventory_client_a1/bin';client.mkdir(parents=True,exist_ok=True)
    link=client/'docker'
    if not link.exists():link.symlink_to(ROOT/'inventory_client.py')
    os.environ['PATH']=str(client)+os.pathsep+os.environ['PATH']
    s=Search(120);s.verify_freeze()
    if a.phase in ('screen','all'):
        print(json.dumps(diagnostic(s)),flush=True)
        r=calibration(s);print(json.dumps({k:r[k] for k in ('status','accepted','best_reliable_relative_change')}),flush=True)
    if a.phase in ('formal','all'):
        r=formal(s,'formal')[1];print(json.dumps({'status':r['status'],'first_FIA_layers':r['first_observed_FIA_layers']}),flush=True)
    if a.phase in ('controls','all'):print(json.dumps(controls(s)),flush=True)

if __name__=='__main__':main()
