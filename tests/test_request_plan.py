import unittest
from types import SimpleNamespace
from fault_injection.request_plan import request_plan,phases,task_state
from fault_injection.delivery_scope import describe_delivery
from demoncat.runtime.vllm_ascend.worker_extension.registry import rpc_provider_capabilities
from demoncat.runtime.vllm_ascend.worker_extension.providers.rms18_task import Rms18TaskProvider
from demoncat.runtime.vllm_ascend.worker_extension.providers.output39_task import Output39TaskProvider
from demoncat.runtime.vllm_ascend.worker_extension.providers.vision17_task import PHASES as VISION
from demoncat.runtime.vllm_ascend.worker_extension.providers.kv41_task import PHASES as KV
from demoncat.runtime.vllm_ascend.worker_extension.serving_quant_projection import PHASES as QUANT

class PlanTests(unittest.TestCase):
    def test_actual_provider_phases_match_shared_plan(self):
        for provider,mechanism in ((Rms18TaskProvider,'18.api_half'),(Output39TaskProvider,'39.output_proxy')):
            for i,phase in enumerate(phases(mechanism)):
                self.assertEqual(provider.phase.fget(SimpleNamespace(index=i,recovering=None)),phase)
        self.assertEqual(VISION,phases('17.spatial_padding'))
        self.assertEqual(KV,phases('41.tail_column'))
        self.assertEqual(QUANT,phases('quant.batch_shared_scale'))
        self.assertEqual(phases('8.restore_offset'),phases('22.wrong_row'))
    def test_query_and_task_projection_share_one_plan(self):
        for row in rpc_provider_capabilities():
            name=row['mechanism'];plan=request_plan(name)
            self.assertEqual(row['request_plan'],plan)
            self.assertEqual(describe_delivery(name)['request_plan'],plan)
            if plan:
                actual=task_state(name,{'serving':{'next_phase':'normal'}})
                self.assertEqual(actual['request_plan'],plan)
                self.assertFalse(actual['complete'])
    def test_copies_and_unimplemented_have_no_plan(self):
        first=request_plan('18.api_half');first['phases'].clear()
        self.assertEqual(len(request_plan('18.api_half')['phases']),4)
        for name in ('20.buffer_substitute','26.fia_destination','33.paged_decode','28.fia.denominator_replay'):
            self.assertIsNone(request_plan(name))
    def test_external_health_is_two_b8_batches(self):
        plan=request_plan('quant.batch_shared_scale')
        self.assertEqual((plan['max_batches'],plan['max_requests'],plan['max_total_requests_including_external']),(6,48,64))
        self.assertEqual([x['request_count'] for x in plan['external_health']],[8,8])
    def test_failed_or_missing_phase_not_complete(self):
        for state in ({},{'evidence_limit_error':True},{'next_phase':'complete','replay_failed':True}):
            self.assertFalse(task_state('18.api_half',state)['complete'])
