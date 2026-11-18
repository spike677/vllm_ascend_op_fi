"""Root probe scheduling only; fake replies cannot establish device C2."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments/c2c3_closure_20261117'))
import natural33_arch22_group_worker_core_probe as group
from demoncat.runtime.vllm_ascend.worker_extension import serving_context


class Client:
    def __init__(self):
        self.index=0;self.pending=None;self.rows=[];self.history=[];self.actions=[];self.owned=True;self.needs_off=False;self.recovery=None
    def phase(self):return self.recovery or (group.PHASES[self.index] if self.index<len(group.PHASES) else 'complete')
    def snapshot(self):return dict(history=deepcopy(self.history),ownership_retained=self.owned,ticket={'cpu':True} if self.owned else None)
    def state(self):return dict(serving=dict(next_phase=self.phase(),pending_request=self.pending,
        resource_state=dict(needs_off=self.needs_off,owned_input_references=0)),
        evidence=dict(actual_requests_sha256=str(len(self.rows)),requests=deepcopy(self.rows)),
        native_state=dict(effect_attempted=True,off_confirmed=not self.needs_off,retained_input_objects=0),
        C1=False,C2=True,C3=False,leases=0,inflight=0,owned_hooks=0,owned_api_wrappers=0)
    def __getattr__(self,name):
        def action(*args):
            self.actions.append(name)
            if name=='advance':self.pending=[args[0],self.phase()]
            elif name=='collect_replay':
                self.rows.append(dict(request_id=self.pending[0],phase=self.pending[1]));self.pending=None
                if self.recovery:self.recovery='recovery_health' if self.recovery=='recovery_off' else None
                else:self.index+=1
            elif name=='recover':self.pending=None;self.recovery='recovery_off'
            elif name=='detach':self.owned=False
            status='QUERY_OK' if name=='query' else 'DETACHED' if name=='detach' else 'ACTION_COMPLETED'
            reply=dict(status=status,provider_state=self.state());self.history.append(dict(action=name,response=deepcopy(reply)))
            return dict(status=status,response=reply,history=deepcopy(self.history))
        return action


class LLM:
    def __init__(self,client,fail=None):
        self.client=client;self.fail=fail;self.count=0;self.exceptions=[]
        self.llm_engine=NS(has_unfinished_requests=lambda:False,abort_request=lambda *a,**k:None)
    def reset_prefix_cache(self,**kwargs):return True
    def enqueue(self,*a,**k):self.count+=1;return [str(self.count)]
    def wait_for_completion(self,**kwargs):
        self.exceptions.append(sys.exc_info()[1]);phase=self.client.pending[1] if self.client.pending else None
        if phase and phase.startswith('fault_'):
            self.client.needs_off=True
            if phase==self.fail:raise RuntimeError('CPU_UNCERTAIN_ACTUAL_REPLY')
        if phase and (phase.startswith('off_') or phase=='recovery_off'):self.client.needs_off=False
        return [NS(request_id=str(self.count),prompt_token_ids=[1],outputs=[NS(token_ids=[2,3],text='cpu',finish_reason='length')])]


class GroupProbeContracts(unittest.TestCase):
    def run_probe(self,fail=None):
        client=Client();llm=LLM(client,fail);report={'warm_healthy':{'outputs':[dict(tokens=[2,3],text='cpu',finish_reason='length')]}}
        def mapping(llm,ids,**kwargs):return dict(internal_request_id=ids[0],external_request_id=ids[0])
        with tempfile.TemporaryDirectory() as tmp,patch.object(serving_context,'read_enqueued_identity',side_effect=mapping):
            group.common.drive(llm,client,prompt={},params=None,out=Path(tmp),report=report,phases=group.PHASES,group=True)
        return client,llm,report
    def test_all_60_phases_and_only_one_post_unregister_ordinary_health(self):
        client,llm,report=self.run_probe()
        self.assertEqual([r['phase'] for r in report['requests']],list(group.PHASES));self.assertEqual(llm.count,61)
        self.assertEqual(client.actions[-4:],['finish','disable','unregister','detach'])
        self.assertFalse(client.owned);self.assertEqual(report['status'],'PUBLIC_ALL_TARGET_S2_C2_RECORDED_PENDING_INDEPENDENT_REVIEW')
    def test_uncertain_target_is_never_repeated_and_off_health_are_outside_exception(self):
        client,llm,report=self.run_probe('fault_05');phases=[r['phase'] for r in report['requests']]
        self.assertEqual(phases.count('fault_05'),1);self.assertNotIn('fault_06',phases)
        self.assertEqual(phases[-2:],['recovery_off','recovery_health']);self.assertTrue(all(x is None for x in llm.exceptions))
        self.assertFalse(client.owned);self.assertIn('CPU_UNCERTAIN_ACTUAL_REPLY',report['error'])
        self.assertEqual(report['status'],'FAILED_WITH_ORIGINAL_ERROR_AND_RECOVERED_OFF')


if __name__=='__main__':unittest.main()
