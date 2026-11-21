"""Same existing Worker RPC lifecycle for one, two and eight simulated ranks."""
from copy import deepcopy
import unittest
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.multi_rank import MultiRankTaskClient
from fault_injection.continuous_contract import continuous_request_plan
from test_extension import _extended_worker
from test_core_host import ExistingExecutor
from test_multi_rank import ScheduledProvider


class ContinuousProvider(ScheduledProvider):
    def complete_actual(self):
        super().complete_actual()
        self.state['request_plan']=continuous_request_plan(self.contract['mechanism'])
        self.state['multi_rank'].update(injection_policy='ALL_RANK_EVERY_ELIGIBLE_CALL',
            role='injector',selected_rank=None,eligible_target=self.rank == 0)


class ContinuousGroupTests(unittest.TestCase):
    def test_same_worker_lifecycle_two_scopes_all_world_sizes(self):
        for size in (1,2,8):
            with self.subTest(world_size=size):
                workers=[_extended_worker(rank=i) for i in range(size)]
                for worker in workers: worker.load_model()
                executor=ExistingExecutor(workers)
                contracts={}; providers={}; decisions=[]
                def bind(actual,config,contract):
                    contracts[actual['rank']]=deepcopy(contract)
                    return {}
                def resolve(mechanism,version,worker,model,actual):
                    p=ContinuousProvider(worker.rank,contracts[worker.rank])
                    p.state['request_plan']=continuous_request_plan(mechanism)
                    providers[worker.rank]=p
                    return p
                with patch.object(identity,'_current_device',return_value=0), \
                     patch.object(runtime,'_resolve_provider',side_effect=resolve):
                    owners=[]
                    for scope in range(2):
                        client=MultiRankTaskClient(executor,FaultConfig(18,'api_half'),prepare_binding=bind,
                            commit_roles=lambda d:decisions.append(d),continuous=True,world_size=size)
                        for action in ('prepare','register','check','enable'):
                            self.assertEqual(getattr(client,action)()['status'],'ACTION_COMPLETED')
                        owners.append(client.contract()['owner'])
                        for phase in ('normal','fault','off','health'):
                            self.assertEqual(client.advance(f'{scope}-{phase}')['status'],'ACTION_COMPLETED')
                            for p in providers.values(): p.complete_actual()
                            self.assertEqual(client.query()['status'],'QUERY_OK')
                        self.assertEqual(decisions[-1]['roles'],{i:'injector' for i in range(size)})
                        self.assertIsNone(decisions[-1]['selected_rank'])
                        self.assertEqual(client.unregister()['status'],'ACTION_COMPLETED')
                        self.assertEqual(client.detach()['status'],'DETACHED')
                        self.assertFalse(client.snapshot()['ownership_retained'])
                    self.assertNotEqual(owners[0],owners[1])

if __name__=='__main__': unittest.main()
