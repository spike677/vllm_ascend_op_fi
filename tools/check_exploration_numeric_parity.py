"""CPU tensor check inside the verified torch environment; no NPU/model request."""
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from ascend_fi.inspect.combined import FakeRequantReferenceExtension as Reference
from ascend_fi.inspect.experiment_extension import ExplorationWorkerExtension as Observed


def main():
    os.environ['VLLM_SERVER_DEV_MODE']='1'
    original, observed=Reference(),Observed()
    rows=[]
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        for multiplier in (.25,1,2,5,10,20):
            x=torch.linspace(-127,127,1024,dtype=torch.float32).reshape(2,512).to(dtype)
            kw=dict(mode='REFERENCE_REQUANT',module_regex='x',layer_filter=None,rank_filter=[0],
                    tensor_selector='first_float',tensor_index=None,scale_provider='tensor_absmax',fixed_scale=None,
                    runtime_scale_attr=None,scale_multiplier=multiplier,max_hits=100,interval=1,request_generation=1,eps=1e-12)
            config=original._make_config(**kw)
            args=(object(),x,'x',0,0,'test',1,config)
            a,ea=original._requant_tensor(*args)
            b,eb=observed._requant_tensor(*args)
            assert torch.equal(a,b) and a.dtype==b.dtype and a.shape==b.shape
            assert all(eb[k]==v for k,v in ea.items())
            rows.append({'dtype':str(dtype),'multiplier':multiplier,'output_bitwise_equal':True,'original_stats_equal':True})
    print(json.dumps({'NUMERIC_PARITY':'PASS','cases':rows,'reference_module':importlib.import_module(Reference.__module__).__file__}))


if __name__=='__main__':main()
