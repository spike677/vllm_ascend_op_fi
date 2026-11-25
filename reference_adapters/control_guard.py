"""Reference-only control preflight on frozen public RPC; never catches forward failures."""
from demoncat.runtime.vllm_ascend.worker_extension.external_provider import external_site
from demoncat.runtime.vllm_ascend.worker_extension.product_binding import ATTRIBUTE
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import ContextRefused
from fault_injection.engines.region_lease import RegionLeases

class ReferenceRefused(ValueError):pass


def snapshot(model):
    rows=[]
    for m in model.modules():
        rows.append((id(m),tuple(m._forward_hooks.items()),tuple(m._forward_pre_hooks.items()),
            tuple(m._backward_hooks.items()),vars(m).get('compute_logits')))
        scheme=getattr(getattr(m,'quant_method',None),'quant_method',None)
        if scheme is not None:rows.append(('scheme',id(scheme),vars(scheme).get('apply')))
    return rows


def preflight(factory,model,runner):
    # The abandoned transient row12 is explicitly refused. Its replacement is
    # a different persistent v2 factory; it must not borrow this admission.
    if factory.__module__=='reference_adapters.logits_mapping':
        raise ReferenceRefused('REFERENCE_ROW12_TRANSIENT_REPLACED_BY_PERSISTENT_V2')
    leases=RegionLeases('reference-control-preflight',1,stream=lambda:0,complete=lambda s:None)
    before=snapshot(model);site=None;failure=None
    try:
        site=factory(model,runner,fault=False,region_leases=leases)
        site.install()
        if not site.sites:raise ReferenceRefused('REFERENCE_NO_STRUCTURAL_SITE')
    except Exception as error:failure=error
    finally:
        if site is not None:site.close()
        if leases.count() or snapshot(model)!=before:
            raise RuntimeError('REFERENCE_PREFLIGHT_CLEANUP_UNPROVEN')
    if failure is not None:
        if isinstance(failure,RuntimeError) and str(failure).startswith(('REFERENCE_','EXTERNAL_')):
            raise ReferenceRefused(str(failure)) from None
        raise failure
    return dict(status='REFERENCE_STATIC_PREFLIGHT_PASSED',resources_zero=True,
                actual_consumers_proven=False,qualification_granted=False)


class ReferenceControlMixin:
    def demoncat_product_binding(self,action,payload):
        # Missing/changed identity and absent structural sites are rejected
        # before attaching a request observer. Actual kernel errors remain fatal.
        if action in ('attach','arm_canary','rollback') and isinstance(payload,dict):
            profile=getattr(self,ATTRIBUTE,{}).get(payload.get('profile'),{})
            mechanism=profile.get('mechanism','')
            spec=external_site(self,mechanism)
            if spec is not None and spec.factory.__module__.startswith('reference_adapters.'):
                try:
                    from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import authority_for
                    authority=authority_for(self)
                    if authority is None or authority.authorize_operator(payload.get('operator'),'product_binding_rpc') is not True:
                        raise ReferenceRefused('PRODUCT_RPC_OPERATOR_NOT_AUTHORIZED')
                    if action=='rollback':
                        from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import ATTRIBUTE as BINDINGS
                        runtime=getattr(self,'_demoncat_worker_fault_runtime',None)
                        bindings=getattr(self,BINDINGS,{})
                        # A control preflight refused before attach. There is
                        # nothing to restore; identity drift must not prevent
                        # acknowledgement of this observed empty ownership.
                        if mechanism not in bindings and mechanism not in getattr(runtime,'_leases',{}):
                            if not isinstance(payload.get('owner'),str) or not payload['owner']:
                                raise ReferenceRefused('PRODUCT_ROLLBACK_OWNER_REQUIRED')
                            return dict(status='PRODUCT_BINDING_ROLLED_BACK',rank=getattr(self,'rank',None),
                                owner=payload['owner'],resources_zero=True,preflight_never_attached=True)
                        return super().demoncat_product_binding(action,payload)
                    spec=external_site(self,mechanism,validate=True)
                    preflight(spec.factory,self.model_runner.model,self.model_runner)
                except (ReferenceRefused,ContextRefused,ValueError,PermissionError) as error:
                    return dict(status='PRODUCT_BINDING_REFUSED',rank=getattr(self,'rank',None),
                        reason=str(error),exit_code=4,refusal_stage='REFERENCE_CONTROL_PREFLIGHT')
                except Exception as error:
                    return dict(status='PRODUCT_BINDING_ERROR',rank=getattr(self,'rank',None),
                        reason=str(error),exit_code=5,recovery_required=True)
        return super().demoncat_product_binding(action,payload)
