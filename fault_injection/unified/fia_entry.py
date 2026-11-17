"""FIA model-entry contract; preserves the exact existing prefill admission gate."""
from .protocol import AdapterError


class FIAModelEntry:
    def __init__(self,mechanism,*,mock_executor=None):
        if mechanism not in ('28.fia.legal_tiling','28.fia.denominator_replay'):
            raise ValueError('FIA_MECHANISM_REQUIRED')
        self.mechanism=mechanism
        self.mock_executor=mock_executor
        self.events=[]

    def invoke(self,operator,actual,*,metadata,scheduler,phase='baseline'):
        from demoncat.runtime.vllm_ascend.worker_extension.attention28_prefill import inspect_prefill_input,strategy_request
        if phase not in ('baseline','fault','off'):raise ValueError('UNKNOWN_PHASE')
        request=strategy_request(self.mechanism,phase)
        admitted=inspect_prefill_input(operator,actual,metadata=metadata,scheduler=scheduler,
                                      allow_cpu_test=self.mock_executor is not None)
        if self.mock_executor is None:
            raise AdapterError('NOT_IMPLEMENTED','Unified FIA native launcher/lease bridge pending; no kernel executed')
        # A fixture may only handle CPU tensors; it cannot confer native authority.
        for name in ('query','key','value'):
            if actual[name].device.type!='cpu':raise AdapterError('MOCK_CPU_ONLY','Mock entry refuses device tensors')
        result=self.mock_executor(actual,request)
        self.events.append(dict(phase=phase,admission=admitted,strategy=request,source='MOCK',npu_qualified=False))
        return result
