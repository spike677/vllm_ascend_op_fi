"""One actual multimodal Worker request with independent upper/lower evidence.

No forward is invoked here. The public serving driver owns enqueue/completion.
Torch dispatcher arguments are observed; ACLNN/device payload stays unknown.
"""
from .vision17_worker import Vision17WorkerCensus
from .vision17_trace import Vision17Trace,vision17_diagnostics,require
from .vision17_padding import Vision17PaddingSession


class ActualVision17Request(Vision17WorkerCensus):
    def __init__(self,worker,*,engine,phase,arm,**kwargs):
        super().__init__(worker,**kwargs)
        self.engine,self.phase,self.arm=engine,phase,arm
        self.adapter=self.upper=None
        engine.begin_request(self,phase,arm)

    def _open_encoder(self,args,kwargs):
        self.trace=Vision17Trace(self.visual,simulation=self.simulation,
                                diagnostics_context=vision17_diagnostics)
        self._enter_encoder_scope(self.trace,args,kwargs)
        if self.arm is not None:
            self.adapter=Vision17PaddingSession(self.visual,simulation=self.simulation,arm=self.arm,
                references=self.engine.references,admission=self.engine.admit,
                diagnostics_context=vision17_diagnostics)
            self._enter_encoder_scope(self.adapter,args,kwargs)
        self.upper=Vision17Trace(self.visual,simulation=self.simulation,
            diagnostics_context=vision17_diagnostics,on_call=self.engine.observe)
        self._enter_encoder_scope(self.upper,args,kwargs)

    def _finish_encoder(self):
        self.upper.close();self.current['initial_observation']=self.upper.report();self.upper=None
        if self.adapter is not None:
            self.adapter.close()
            self.current['mechanism']=self.adapter.report()
            self.current['adapter_cleanup']=self.adapter.runtime_state()
            if self.arm=='baseline' and self.engine.references is None:
                self.engine.references=dict(self.adapter.healthy_calls)
            self.adapter=None
        else:self.current['mechanism']=None
        super()._finish_encoder()
        self.current['phase']=self.phase
        self.current['fault_injections']=sum(bool(e.get('applied'))
            for e in (self.current['mechanism'] or {}).get('events',()))

    def _abort_encoder(self):
        if self.upper is not None:self.upper.close();self.upper=None
        if self.adapter is not None:self.adapter.close();self.adapter=None
        super()._abort_encoder()

    def collect(self):
        row=super().collect()
        require(row['encoder_observed'],'GROUP_EACH_REQUEST_REQUIRES_ACTUAL_ENCODER')
        return row
