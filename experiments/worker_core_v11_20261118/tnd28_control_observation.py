"""Explicit research M128 control, no denominator replay or public route.

The policy is installed only inside the existing matched live input window,
removed after native return, and refuses any foreign policy entry.
"""
import os
from pathlib import Path
from demoncat.runtime.vllm_ascend.worker_extension.attention28_prepared_input import PreparedInputCapture
from demoncat.runtime.vllm_ascend.worker_extension.serving_context import require

class ControlInputCapture(PreparedInputCapture):
    def __init__(self,*args,**kwargs):
        self.control_env={}
        super().__init__(*args,**kwargs)
        require(self.borrow_query is not None,'CONTROL_LIVE_TYPED_BORROW_REQUIRED')
    def _check_policy(self):
        actual={k:v for k,v in os.environ.items() if k.startswith('DCAT28_TND_')}
        require(actual==self.control_env,'TND_CONTROL_FOREIGN_POLICY')
    def enter(self,token,row,actual,api):
        receipt=super().enter(token,row,actual,api)
        values=dict(MODE='denominator_control',TOKEN=token,OWNER=receipt['owner'],GENERATION=str(receipt['generation']),
            PID=str(self.owner['pid']),TID=str(self.owner['native_tid']),PROCESS_START=self.owner['process_start_ticks'],
            PHASE='control',TARGET=receipt['module_path'],OBJECT=str(receipt['module_object_id']),
            REQUEST_SHA256=receipt['request_sha256'],INPUT_SHA256=receipt['input_sha256'],LOG=str(self.root/(token+'.control-host.jsonl')))
        self._check_policy()
        for k,v in values.items():
            key='DCAT28_TND_'+k
            try:os.environ[key]=v
            finally:
                if os.environ.get(key)==v:self.control_env[key]=v
        self.receipt['control_policy']=dict(values=values,denominator_replay=False,public_request_phase='observe',host_policy_phase='control')
        return self.receipt
    def stop(self):
        # Parent cleanup proves actual Execute drained and releases borrowed
        # references first. Uncertain return retains the environment/owner.
        super().stop()
        self._check_policy()
        for k,v in list(self.control_env.items()):
            require(os.environ.get(k)==v,'TND_CONTROL_OWNER_CHANGED')
            del os.environ[k];del self.control_env[k]
        if self.receipt is not None:self.receipt['control_policy_released']=True

from demoncat.runtime.vllm_ascend.worker_extension.attention28_stock_census import StockTndPrefillCensus

class ControlTndPrefillCensus(StockTndPrefillCensus):
    """Explicit research construction; absent from public registry/config."""
    def _validate_prepared_capture(self,capture,journal):
        require(type(capture) is ControlInputCapture and journal is None and capture.borrow_query is not None,
                'TND_CONTROL_EXACT_LIVE_CAPTURE_REQUIRED')
