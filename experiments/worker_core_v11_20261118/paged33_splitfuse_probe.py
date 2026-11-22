"""Direction A model experiment through Runtime; pending source review, no grant."""
import continuous_public_probe as original
original_bootstrap=original.bootstrap

def bootstrap(wrapper,action,mechanism,options=None):
    if action=='bind':
        import fault_injection.sites.paged33 as module
        from fault_injection.sites.paged33_splitfuse import SplitFuse33Site
        module.Paged33Site=type('ConfiguredSplitFuse33Site',(SplitFuse33Site,),{'reference_device':options['arithmetic_device'],'matmul_mode':'bf16_out_fp32' if options['arithmetic_device']=='input' else 'fp32'})
    return original_bootstrap(wrapper,action,mechanism,options)

if __name__=='__main__':
    import json
    from pathlib import Path
    original.bootstrap=bootstrap
    code=original.main()
    p=Path('/evidence/continuous_public.json');r=json.loads(p.read_text())
    r['reference_variant']='A2_BF16_SPLITFUSE_512_TO_256'
    r['acceptance_scope']='DIRECTION_A_EXPERIMENT_PENDING_LAUNCH_REVIEW'
    r['paged_deployment'].update(control_block=512,fault_block=256,control_checkpoint=.99)
    p.write_text(json.dumps(r,indent=2)+'\n')
    raise SystemExit(code)
