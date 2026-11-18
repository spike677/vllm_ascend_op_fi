"""Probe scheduling only; fake RPCs never stand in for Native qualification."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/c2c3_closure_20261117'))
import natural33_arch22_s2_worker_core_probe as probe
from demoncat.runtime.vllm_ascend.worker_extension import serving_context


class Client:
    def __init__(self):
        self.history=[];self.index=0;self.pending=None;self.rows=[];self.owned=True
        self.effect=False;self.off=False;self.fail_finish=False;self.actions=[]
    def snapshot(self):return dict(history=deepcopy(self.history),ownership_retained=self.owned,
        ticket={'fixture_only':True} if self.owned else None)
    def state(self):
        return dict(serving=dict(next_phase=probe.PHASES[self.index] if self.index<5 else 'complete',pending_request=self.pending,
            resource_state={'needs_off':self.effect and not self.off,'owned_input_references':0}),
            evidence=dict(actual_requests_sha256=str(len(self.rows)),requests=deepcopy(self.rows)),
            native_state=dict(effect_attempted=self.effect,off_confirmed=self.off,retained_input_objects=0),
            C1=True,C2=False,C3=False,leases=0,inflight=0,owned_hooks=0,owned_api_wrappers=0)
    def __getattr__(self,name):
        def action(*args):
            self.actions.append(name);status='QUERY_OK' if name=='query' else 'DETACHED' if name=='detach' else 'ACTION_COMPLETED'
            if name=='advance':self.pending=[args[0],probe.PHASES[self.index]]
            elif name=='collect_replay':
                self.rows.append(dict(request_id=self.pending[0],phase=self.pending[1]));self.index+=1;self.pending=None
            elif name=='recover':self.index=4;self.pending=None
            elif name=='finish' and self.fail_finish:status='PROVIDER_ACTION_FAILED'
            elif name=='detach':self.owned=False
            reply=dict(status=status,provider_state=self.state())
            self.history.append(dict(action='advance' if name=='collect_replay' else name,response=deepcopy(reply)))
            return dict(status=status,response=reply,history=deepcopy(self.history))
        return action


class LLM:
    def __init__(self,client,fail_fault=False):
        self.client=client;self.fail_fault=fail_fault;self.sequence=0;self.current=None;self.exc_contexts=[]
        self.llm_engine=NS(has_unfinished_requests=lambda:False,abort_request=lambda *a,**k:None)
    def reset_prefix_cache(self,**kwargs):return True
    def enqueue(self,*args,**kwargs):
        self.sequence+=1;self.current=str(self.sequence);return [self.current]
    def wait_for_completion(self,**kwargs):
        self.exc_contexts.append(sys.exc_info()[1])
        pending=self.client.pending
        if pending:
            if pending[1]=='fault':
                self.client.effect=True
                if self.fail_fault:raise RuntimeError('original uncertain native return')
            if pending[1]=='recovery_off':self.client.off=True
        return [NS(request_id=self.current,prompt_token_ids=[1],outputs=[NS(token_ids=[2,3],text='value',finish_reason='length')])]


class ProbeScheduleContracts(unittest.TestCase):
    def execute(self,*,fail_fault=False,fail_finish=False):
        client=Client();client.fail_finish=fail_finish;llm=LLM(client,fail_fault)
        report={'warm_healthy':{'outputs':[dict(tokens=[2,3],text='value',finish_reason='length')]}}
        def mapping(llm,ids,**kwargs):return dict(internal_request_id=ids[0],external_request_id=ids[0])
        with tempfile.TemporaryDirectory() as folder,patch.object(serving_context,'read_enqueued_identity',side_effect=mapping):
            probe.drive(llm,client,prompt={},params=None,out=Path(folder),report=report)
        return llm,client,report
    def test_five_requests_then_cleanup_and_ordinary_health(self):
        llm,client,report=self.execute()
        self.assertEqual(report['status'],probe.SUCCESS);self.assertEqual(llm.sequence,6)
        self.assertEqual([r['phase'] for r in report['requests']],list(probe.PHASES))
        self.assertEqual(client.actions[-4:],['finish','disable','unregister','detach'])
        self.assertFalse(report['worker_core']['ownership_retained'])
        self.assertTrue(all(error is None for error in llm.exc_contexts))
    def test_uncertain_fault_is_not_retried_and_off_leaves_original_exception_context(self):
        llm,client,report=self.execute(fail_fault=True)
        self.assertEqual(report['status'],'FAILED_WITH_ORIGINAL_ERROR_AND_RECOVERED_OFF')
        self.assertIn('original uncertain native return',report['error']);self.assertEqual(llm.sequence,6)
        self.assertEqual(client.actions.count('recover'),1)
        self.assertEqual([r['phase'] for r in report['requests']].count('fault'),1)
        self.assertFalse(report['requests'][3]['completed']);self.assertTrue(report['requests'][4]['completed'])
        self.assertTrue(all(error is None for error in llm.exc_contexts));self.assertFalse(client.owned)
    def test_uncertain_finish_retains_owner_and_does_not_run_health_or_blind_retry(self):
        llm,client,report=self.execute(fail_finish=True)
        self.assertEqual(report['status'],'FAILED_WITH_OWNER_OR_RECOVERY_EVIDENCE')
        self.assertEqual(llm.sequence,5);self.assertTrue(client.owned)
        self.assertNotIn('unregister',client.actions);self.assertNotIn('post_unregister_health',report)


if __name__=='__main__':unittest.main()
