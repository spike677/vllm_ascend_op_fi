import unittest
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
import test_core_host as fixtures


class ProductHostLatestTests(unittest.TestCase):
    setUp=fixtures.CoreHostTests.setUp
    _resolve=fixtures.CoreHostTests._resolve

    def test_latest_receipt_preserves_owner_and_cleanup_without_accumulating_history(self):
        client=WorkerFaultClient(HostAdapter(self.executor),FaultConfig(20,'buffer_substitute'),receipt_mode='latest')
        owner=client.owner
        for action in ('prepare','register','check','enable','disable','unregister','detach'):
            result=getattr(client,action)()
            self.assertEqual(result['status'],'DETACHED' if action=='detach' else 'ACTION_COMPLETED')
            self.assertEqual(result['owner'],owner)
            self.assertEqual(len(result['history']),1)
            self.assertEqual(result['history'][0]['action'],action)
        self.assertFalse(client.snapshot()['ownership_retained'])
        self.assertFalse(self.worker._demoncat_worker_fault_runtime._leases)
