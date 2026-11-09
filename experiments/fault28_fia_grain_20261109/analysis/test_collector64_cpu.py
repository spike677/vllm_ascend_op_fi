"""Exercise the actual staged msprobe method on CPU; never an NPU acceptance."""
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
from collector_stage import select_collector,verify_loaded


def main():
    case={'statistics_variant':'task_local_cpu_fp64_high_v1','capture_msprobe':True,
          'approved_collector_manifest_file':'/audit/collector_fp64_a1/stage_manifest.json',
          'approved_collector_manifest_sha256':'06ce36bae3463f44a143047d184f53915dbea4dd04b39d79960518e6878d8010'}
    selection=select_collector(case)
    import torch
    import numpy as np
    module=importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
    actual=verify_loaded(selection,module)
    class Handler:
        @staticmethod
        def is_empty_data(data):return data.numel()==0
        @staticmethod
        def is_batchedtensor(data):return False
        @staticmethod
        def is_gradtrackingtensor(data):return False
    observer=SimpleNamespace(tensor_handler=Handler(),config=SimpleNamespace(summary_mode='statistics'))
    rows=[]
    for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
        data=torch.tensor([1.,0.0001,-1.,0.125,-0.25],dtype=dtype)
        before=data.clone();version=data._version
        stat=module.PytorchDataProcessor.get_stat_info(observer,data,precision=module.Const.DUMP_PRECISION_HIGH)
        arr=before.double().numpy()
        expected=[float(arr.max()),float(arr.min()),float(arr.mean()),float(np.linalg.norm(arr))]
        got=[stat.max,stat.min,stat.mean,stat.norm]
        assert all(type(v) is float for v in got)
        assert all(abs(a-b)<1e-12 for a,b in zip(got,expected))
        assert torch.equal(data,before) and data.dtype==dtype and data._version==version
        rows.append({'dtype':str(dtype),'statistics':got,'input_unchanged':True})
    empty=module.PytorchDataProcessor.get_stat_info(observer,torch.empty(0),precision=module.Const.DUMP_PRECISION_HIGH)
    assert all(getattr(empty,name) is None for name in ('max','min','mean','norm'))
    report={'schema':'fault28.collector_cpu_fp64_test.v1','status':'PASS_ACTUAL_COLLECTOR_CPU_METHOD',
            'actual_loaded':actual,'cases':rows,'empty_remains_unobservable':True,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'devices_exposed':[],'npu_executed':False}
    out=Path('/evidence/collector_cpu_test.json')
    with out.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':report['status'],'npu_executed':False}))


if __name__=='__main__':main()
