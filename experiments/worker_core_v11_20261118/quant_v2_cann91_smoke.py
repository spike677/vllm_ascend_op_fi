"""CANN9.1 V2 native binding smoke, bounded tensors; not a model acceptance."""
import ctypes,hashlib,json,sys,traceback
from pathlib import Path

def main():
    result=dict(status='STARTED',qualification_granted=False,model_calls=0,scope='A2/CANN9.1/torch_npu2.10 V2-only B8x16 operator smoke',cases=[])
    control=None
    try:
        raw_build=Path('/quant/runtime/build.json').read_bytes()
        assert hashlib.sha256(raw_build).hexdigest()=='04f1c8af72231aec3d98ec071882e28c6960ccd7852381c9ea65e5fa212a3aa9'
        build=json.loads(raw_build)
        for name,key in [('backend_control.py','control_sha256'),('native_backend.cpp','source_sha256'),('batch_scale_inventory.h','inventory_source_sha256')]:
            assert hashlib.sha256(Path('/quant',name).read_bytes()).hexdigest()==build[key]
        assert hashlib.sha256(Path('/quant/runtime/vendor/op_api/lib/libcust_opapi.so').read_bytes()).hexdigest()==build['library_sha256']
        assert hashlib.sha256(Path(build['real_opapi']).read_bytes()).hexdigest()==build['real_opapi_sha256']
        result['build_manifest_sha256']=hashlib.sha256(Path('/quant/runtime/build.json').read_bytes()).hexdigest()
        import torch,torch_npu
        sys.path.insert(0,'/quant');from backend_control import BackendControl
        torch.npu.set_device(0);control=BackendControl();control.lib.dcat_dq_state_v1.argtypes=[ctypes.POINTER(ctypes.c_uint64),ctypes.c_size_t];result['resolver_identity']=control.resolver_identity
        raw=(torch.arange(1,129,dtype=torch.float32).reshape(8,16)/32).bfloat16();raw[-1]*=64
        x=raw.npu();baseline=None
        for phase in ('baseline','fault','off'):
            control.validate_input(x);control.arm_validated(x,fault=phase=='fault')
            q,scale=torch_npu.npu_dynamic_quant(x);torch.npu.synchronize();counts=control.finish()
            q=q.cpu().clone();scale=scale.cpu().clone()
            assert torch.equal(x.cpu(),raw) and bool(torch.isfinite(scale).all())
            if baseline is None:baseline=(q,scale)
            changed_q=int((q!=baseline[0]).sum());changed_s=int((scale!=baseline[1]).sum())
            assert counts['prepare_hits']==counts['execute_hits']==1 and counts['fault_executions']==int(phase=='fault') and counts['api_version']==2
            if phase=='fault':assert changed_q>0 and changed_s>0
            else:assert changed_q==changed_s==0
            values=(ctypes.c_uint64*7)();assert control.lib.dcat_dq_state_v1(values,7)==0 and list(values)==[1,0,0,0,0,0,0]
            result['cases'].append(dict(phase=phase,native=counts,q_changed_elements=changed_q,scale_changed_elements=changed_s,resources=list(values)))
        result['status']='V2_OPERATOR_B_F_OFF_PROVEN_NOT_MODEL_QUALIFIED'
    except BaseException as e:
        result.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
        if control:
            values=(ctypes.c_uint64*7)();rc=control.lib.dcat_dq_state_v1(values,7);result['failure_resources']=dict(rc=rc,state=list(values))
            # Native finish releases an unexecuted arm but cannot prove OFF.
            counters=(ctypes.c_uint64*6)();result['failure_finish_rc']=control.lib.dcat_dq_finish(counters,6)
    Path('/evidence/quant-v2-smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0 if result['status'].startswith('V2_OPERATOR') else 2
if __name__=='__main__':raise SystemExit(main())
