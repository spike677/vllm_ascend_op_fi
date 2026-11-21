"""Actual public factory/Runtime/group lifecycle; only request execution is fake."""
from copy import deepcopy
import os,threading,unittest
from unittest.mock import patch
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
from demoncat.runtime.vllm_ascend.worker_extension import identity
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.multi_rank import MultiRankTaskClient
from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import ContinuousBinding,ATTRIBUTE
from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider


class Request:
    pending=[]
    def __init__(self,b,ids,phase):self.ids=ids;self.phase=phase;self.samples=0;self.inside=0;self.hooks=[];self.patches=[];self.site=None
    def install(self):self.pending.append(self)
    def close(self):pass
    def collect(self):
        return dict(phase=self.phase,request_ids=self.ids,all_finite=True,
            output=['fault' if self.phase=='fault' else 'normal']*4,coverage={},resources=dict(leases=0,hooks=0,inflight=0))


class ContinuousRuntimeTests(unittest.TestCase):
    def test_two_scopes_through_actual_public_factory_and_runtime(self):
        for size in (1,2):
            workers=[_extended_worker(rank=i) for i in range(size)]
            for w in workers:w.load_model()
            executor=ExistingExecutor(workers)
            with patch.object(identity,'_current_device',return_value=0), \
                 patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.deployment_acceptance_report',
                       return_value=dict(can_prepare=True,accepted=False)), \
                 patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization.acceptance_use',return_value=None), \
                 patch('demoncat.runtime.vllm_ascend.worker_extension.providers.continuous.ContinuousRequest',Request):
                for scope in range(2):
                    def bind(actual,config,contract):
                        w=workers[actual['rank']]
                        b=ContinuousBinding(w,w.model_runner.model,config.mechanism,((1,2),),deepcopy(contract),actual,
                            'input',os.getpid(),threading.get_ident())
                        b.warmups=[dict(all_finite=True,output=['stable']) for _ in range(2)]
                        setattr(w,ATTRIBUTE,{config.mechanism:b});return {}
                    c=MultiRankTaskClient(executor,FaultConfig(28,'fia.denominator_replay'),
                        prepare_binding=bind,continuous=True,world_size=size)
                    for action in ('prepare','register','check','enable'):
                        r=getattr(c,action)();self.assertEqual(r['status'],'ACTION_COMPLETED',r)
                    for phase in ('normal','fault','off','health'):
                        r=c.advance(str(scope)+phase);self.assertEqual(r['status'],'ACTION_COMPLETED',r)
                        for request in Request.pending:request.samples=4
                        Request.pending.clear()
                        r=c.query();self.assertEqual(r['status'],'QUERY_OK',r)
                    self.assertEqual(c.advance('extra')['status'],'GROUP_PHASES_EXHAUSTED')
                    for action in ('finish','unregister'):
                        r=getattr(c,action)();self.assertEqual(r['status'],'ACTION_COMPLETED',r)
                    self.assertEqual(c.detach()['status'],'DETACHED')
                    self.assertTrue(all(not getattr(w,ATTRIBUTE)[c.config.mechanism].leases for w in workers))

    def test_unbound_native_research_route_stays_closed(self):
        w=_extended_worker();w.load_model()
        with self.assertRaises(RuntimeError):
            create_worker_provider('28.fia.denominator_replay','1',w,w.model_runner.model,{})
