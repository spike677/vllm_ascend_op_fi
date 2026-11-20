"""No-device production ELF/ctypes/SDK binding, no arm or model execution."""
import hashlib,json,traceback
from pathlib import Path

def main():
    report=dict(status='STARTED',actual_npu_calls=0,qualification_granted=False)
    try:
        from fault_injection.quant_group_native import QuantGroupNative
        bundle=Path('/quant');sha=hashlib.sha256((bundle/'build.json').read_bytes()).hexdigest()
        assert sha=='05b1c1be096d5d80fd8972d38f23afbecb5138d58361ba1ef1c96c5f8b85e513'
        native=QuantGroupNative.load(bundle,approved_build_sha256=sha,approved_cann='9.1.0')
        state=native.query()
        assert not any(state['core'][k] for k in ('installed','pending_parent','device_bytes','host_bytes','owned_descriptors'))
        report.update(status='CANN91_GROUP_DSO_LOADED_IDLE_NOT_NPU_ACCEPTANCE',identity=native.identity,state=state)
    except BaseException as e:report.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
    Path('/evidence/load-result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','error') if k in report}))
    return 0 if report['status'].startswith('CANN91_GROUP') else 2
if __name__=='__main__':raise SystemExit(main())
