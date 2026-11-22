"""Isolated reference experiment, not a new public mechanism/qualification."""
import continuous_public_probe as original

def bootstrap(wrapper,action,mechanism,options=None):
    if action=='bind':
        import fault_injection.sites.moe8 as module
        from fault_injection.sites.allgather_output8 import AllGatherOutput8Site
        module.MoE8Site=AllGatherOutput8Site
    result=original_bootstrap(wrapper,action,mechanism,options)
    return result

original_bootstrap=original.bootstrap
if __name__=='__main__':
    import json,sys
    from pathlib import Path
    original.bootstrap=bootstrap
    code=original.main()
    path=Path('/evidence/continuous_public.json')
    data=json.loads(path.read_text());data['reference_variant']='allgather_output_row_proxy'
    data['acceptance_scope']='EXPERIMENT_REFERENCE_NOT_MC2_OR_PUBLIC_QUALIFICATION'
    path.write_text(json.dumps(data,indent=2)+'\n')

    raise SystemExit(code)
