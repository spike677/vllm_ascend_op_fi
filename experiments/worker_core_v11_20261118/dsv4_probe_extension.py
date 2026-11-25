"""Mixin for an already authenticated deployment RPC; no auth or server creation.

The deployment MUST wrap demoncat_dsv4_probe with its existing authorization.
Do not publish vLLM's development collective_rpc endpoint on an untrusted network.
"""
from experiments.worker_core_v11_20261118.dsv4_observe_probe import observe
from experiments.worker_core_v11_20261118.dsv4_noise_capture import capture
from experiments.worker_core_v11_20261118.dsv4_probe_support import deployment_identity, ProbeRefused


class Dsv4ProbeExtension:
    def demoncat_dsv4_probe(self, *, tool, action, options=None):
        try:
            options=dict(options or {})
            if 'approved_baselines_by_rank' in options:
                options['approved_baselines']=options.pop('approved_baselines_by_rank').get(str(self.rank),{})
            if tool=='identity' and action=='read':result=deployment_identity(self)
            elif tool=='observe':result=observe(self,action,options)
            elif tool=='capture':
                options=options or {}
                result=capture(self,action,phase=options.get('phase','calibration'),options=options)
            else:raise ProbeRefused('PROBE_TOOL_ACTION_UNKNOWN')
            return dict(schema='dcat.probe.rpc.v2',rank=getattr(self,'rank',None),status='PROBE_OK',exit_code=0,result=result)
        except Exception as error:
            import traceback
            return dict(schema='dcat.probe.rpc.v2',rank=getattr(self,'rank',None),
                        status='PROBE_REFUSED' if isinstance(error,ValueError) else 'PROBE_ERROR',
                        exit_code=4 if isinstance(error,ValueError) else 5,
                        code=str(error).split(':',1)[0],error_type=type(error).__name__,reason=str(error),
                        traceback=traceback.format_exc(),qualification_granted=False)
