"""Same read-only ordinary-route policy for query and construction.

A static catalog never admits C1/group/coverage through public configuration.
Unported mechanisms remain explicit pending entries; historical classes stay
available for direct research use, outside this public route.
"""
from .prerequisites import assessment, expectation, _MISSING

ORDINARY = {'41.tail_column','22.wrong_row','17.spatial_padding','8.restore_offset','18.api_half','39.output_proxy','quant.batch_shared_scale'}
NON_EXECUTABLE = {
    '15.alias_detach': 'FAULT15_CURRENT_DEPLOYMENT_NO_PRIVATE_VIEW_WRITE_CHAIN',
    '25.reduce_scatter': 'FAULT25_UNSUPPORTED_NO_SUBSTITUTE',
    '28.fa.legal_tiling': 'FAULT28_LEGAL_HISTORICAL_ONLY',
    '28.fia.legal_tiling': 'FAULT28_LEGAL_HISTORICAL_ONLY',
    '28.fia.denominator_replay': 'TND28_FAULT_OFF_RECOVERY_NOT_ACCEPTED',
}


def ordinary_route_report(mechanism, worker=None, model=None):
    code=NON_EXECUTABLE.get(mechanism)
    if code is None and mechanism not in ORDINARY:
        code='ORDINARY_TASK_ADAPTER_PENDING:'+mechanism
    checks=[expectation('mechanism_compatibility','ordinary_route_implemented',True,
                        code is None,code or 'ORDINARY_TASK_ROUTE')]
    if code is None:
        actual=_MISSING
        if worker is not None:
            if mechanism in ('8.restore_offset','18.api_half','22.wrong_row'):
                from .serving_tensor_binding import TensorServingBinding, _ATTRIBUTE
                bindings=getattr(worker,_ATTRIBUTE,None)
                binding=bindings.get(mechanism) if type(bindings) is dict else None
                if binding is not None:
                    actual=(type(binding) is TensorServingBinding and binding.runtime_only is True
                            and binding.worker is worker and binding.model is model)
            elif mechanism=='41.tail_column':
                from .serving41 import KV41ServingBinding, _ATTRIBUTE
                binding=getattr(worker,_ATTRIBUTE,None)
                if binding is not None:
                    actual=(type(binding) is KV41ServingBinding and binding.runtime_only is True
                            and binding.worker is worker and binding.model is model)
            elif mechanism=='17.spatial_padding':
                from .providers.vision17_task import Vision17TaskBinding, ATTRIBUTE
                binding=getattr(worker,ATTRIBUTE,None)
                if binding is not None:
                    actual=(type(binding) is Vision17TaskBinding and binding.worker is worker and binding.model is model
                        and not hasattr(worker,'_demoncat_vision17_group_binding')
                        and not hasattr(worker,'_demoncat_vision17_observation_binding'))
            elif mechanism=='39.output_proxy':
                from .serving39_output import Output39ServingBinding, _ATTRIBUTE
                binding=getattr(worker,_ATTRIBUTE,None)
                if binding is not None:
                    actual=(type(binding) is Output39ServingBinding and binding.runtime_only is True
                            and binding.stage=='c1' and binding.max_live_leases==1
                            and binding.worker is worker and binding.model is model)
            else:
                # The identity of the registered factory is checked again by
                # QuantProvider, before invoking any session/native loader.
                marker=getattr(worker,'_dcat_quant_ordinary_model',None)
                if hasattr(worker,'_dcat_quant_binding'):
                    actual=(marker is model and model is not None)
        checks.append(expectation('deployment_identity','typed_ordinary_binding',True,actual,
                      {'41.tail_column':'KV41_ORDINARY_TASK_BINDING_REQUIRED',
                       '22.wrong_row':'ROPE22_ORDINARY_BINDING_REQUIRED',
                       '17.spatial_padding':'VISION17_ORDINARY_BINDING_REQUIRED',
                       '8.restore_offset':'MOE8_ORDINARY_BINDING_REQUIRED',
                       '18.api_half':'RMS18_ORDINARY_BINDING_REQUIRED',
                       '39.output_proxy':'OUTPUT39_ORDINARY_TASK_BINDING_REQUIRED',
                       'quant.batch_shared_scale':'QUANT_ORDINARY_PROJECTION_BINDING_REQUIRED'}[mechanism]))
    from .deployment_acceptance import deployment_acceptance_report
    accepted=deployment_acceptance_report(mechanism,worker,model,implemented=code is None)
    if code is None:
        checks.append(expectation('deployment_identity','accepted_deployment_or_explicit_blue_validation',
            True,accepted['can_prepare'],'ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT'))
    checks.append(expectation('actual_call','source_bound_live_call',True,_MISSING,
                  'ACTUAL_CALL_NOT_YET_OBSERVED'))
    report=assessment(checks)
    # Real request observation is deliberately pending until normal inference.
    report['can_prepare']=not any(row['status']!='matched' and row['layer']!='actual_call'
                                  for row in report['checks'])
    report['deployment_acceptance']=accepted
    report['mechanism']=mechanism
    report['public_route']='ordinary_task_only'
    report['historical_validation_inherited']=False
    return report


def require_ordinary_route(mechanism,worker,model):
    report=ordinary_route_report(mechanism,worker,model)
    if not report['can_prepare']:
        primary=next(row for row in report['unmet'] if row['layer']!='actual_call')
        error=RuntimeError(primary['code'])
        error.code=primary['code'].split(':',1)[0]
        error.prerequisites=report
        raise error
    return report
