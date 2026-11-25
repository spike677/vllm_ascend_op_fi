"""Isolated A2 negative test; never install these diagnostic RPCs on production."""
import json

def control(worker,action,options):
    from fault_injection.sites.mla28 import fingerprint
    model=worker.model_runner.model
    if action=='reference_canary_start':
        assert 'compute_logits' not in vars(model)
        original=model.compute_logits;worker._reference_canary=[]
        def observe(hidden,*args,**kwargs):
            result=original(hidden,*args,**kwargs)
            worker._reference_canary.append(dict(hidden=fingerprint(hidden),logits=fingerprint(result)))
            return result
        model.compute_logits=observe;worker._reference_canary_hook=observe
        return dict(status='ARMED',rank=worker.rank)
    if action=='reference_canary_stop':
        assert vars(model)['compute_logits'] is worker._reference_canary_hook
        del model.compute_logits
        return dict(rank=worker.rank,steps=worker._reference_canary,removed=True)
    if action=='reference_identity_test':
        from dataclasses import replace
        from demoncat.runtime.vllm_ascend.worker_extension.external_provider import ATTRIBUTE
        sites=getattr(worker,ATTRIBUTE);key=options['mechanism'];spec,model=sites[key]
        if options['drift']:
            worker._reference_original_spec=(spec,model)
            sites[key]=(replace(spec,expected_deployment=dict(spec.expected_deployment,reference_negative_drift=True)),model)
        else:
            sites[key]=worker._reference_original_spec;del worker._reference_original_spec
        return dict(status='IDENTITY_TEST_SET',rank=worker.rank)
    raise ValueError('REFERENCE_DIAGNOSTIC_ACTION')


def run(ex,inference,root,variant):
    from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
    from demoncat.runtime.vllm_ascend.worker_extension.external_provider import ExternalFaultConfig
    from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_control_reply
    mechanism='external.'+variant
    client=ProductClient(ex,ExternalFaultConfig('external',variant),profile=mechanism,
        operator='product-acceptance',run_canary=inference,drain=lambda:None)
    client.attach()
    def exact_probe():
        ex.collective_rpc('demoncat_acceptance_bootstrap',args=('reference_canary_start',{}))
        api=inference()
        ranks=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('reference_canary_stop',{}))
        # HTTP ids/timing differ; compare actual choices and full hidden/logits hashes.
        return dict(api=api['choices'],ranks=ranks)
    before=exact_probe();report=dict(schema='dcat.reference.refusal.v1',before=before,checks=[])
    for kind in ['missing_site','identity_drift','range','missing_site_repeat']:
        if kind=='identity_drift':ex.collective_rpc('demoncat_acceptance_bootstrap',args=('reference_identity_test',dict(mechanism=mechanism,drift=True)))
        try:reply=product_control_reply(lambda:client.enable({'mode':'single'} if kind=='range' else None))
        finally:
            if kind=='identity_drift':ex.collective_rpc('demoncat_acceptance_bootstrap',args=('reference_identity_test',dict(mechanism=mechanism,drift=False)))
        after=exact_probe();audit=ex.collective_rpc('demoncat_acceptance_bootstrap',args=('audit',{}))
        report['checks'].append(dict(kind=kind,reply=reply,after=after,client_status=client.status(),audit=audit,
            canary_exact=after==before,engine_alive=True))
        (root/'reference_refusal.json').write_text(json.dumps(report,indent=2))
        assert reply['exit_code']==4 and before==after
    client.detach();report['status']='REFERENCE_CONTROL_REFUSAL_A2_PASSED'
    (root/'reference_refusal.json').write_text(json.dumps(report,indent=2))
