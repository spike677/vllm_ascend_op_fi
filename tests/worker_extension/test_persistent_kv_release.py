import unittest
from types import SimpleNamespace as NS
from demoncat.runtime.vllm_ascend.worker_extension.persistent_kv_release import KVReleaseBarrier


class Manager:
    def __init__(self,trace):self.trace=trace;self.block=NS(block_id=4,ref_cnt=1,is_null=False)
    def get_blocks(self,rid):return NS(blocks=([self.block],))
    def free(self,request):self.trace.append('free')
    def pop_blocks_for_free(self,request):self.trace.append('pop');return [self.block]


class Executor:
    def __init__(self,trace):self.trace=trace
    def execute_model(self,output,**kwargs):self.trace.append('execute');return 7


class KVReleaseTests(unittest.TestCase):
    def setUp(self):
        self.trace=[];self.request=NS(request_id='actual')
        self.engine=NS(async_scheduling=False,batch_queue=None,
            vllm_config=NS(cache_config=NS(enable_prefix_caching=False),kv_transfer_config=None,
                ec_transfer_config=None,speculative_config=None,parallel_config=NS(pipeline_parallel_size=1)),
            scheduler=NS(requests={'actual':self.request},kv_cache_manager=Manager(self.trace)),
            model_executor=Executor(self.trace))
        self.bad_rank=False
        def notify(action,payload):
            self.trace.append(action)
            return [dict(rank=r,sequence=payload['sequence']-(1 if self.bad_rank and r==1 else 0),
                request_id=payload['request_id'],request_generation=payload['request_generation'],acknowledged=True,
                device_readers_complete=True,bytes_restored=True,request_leases=0) for r in range(2)]
        self.barrier=KVReleaseBarrier(self.engine,notify=notify,world_size=2,group_ids=(0,),source_check=lambda e:{'fixture':True})
        self.output=NS(num_scheduled_tokens={'actual':1})
    def test_announce_before_execute_restore_before_normal_free(self):
        self.barrier.install();self.engine.model_executor.execute_model(self.output,non_block=True)
        self.engine.scheduler.kv_cache_manager.free(self.request)
        self.assertEqual(self.trace,['before_execute','execute','before_free','free'])
        self.assertFalse(self.barrier.tickets);self.barrier.close()
        self.assertNotIn('free',vars(self.engine.scheduler.kv_cache_manager))
    def test_deferred_or_abort_pop_waits_for_same_restore_barrier(self):
        self.barrier.install();self.engine.model_executor.execute_model(self.output)
        result=self.engine.scheduler.kv_cache_manager.pop_blocks_for_free(self.request)
        self.assertEqual(len(result),1);self.assertEqual(self.trace[-2:],['before_free','pop'])
        self.assertFalse(self.barrier.tickets)
    def test_lagged_rank_ack_never_returns_page_to_pool(self):
        self.barrier.install();self.engine.model_executor.execute_model(self.output);self.bad_rank=True
        with self.assertRaisesRegex(RuntimeError,'STALE_OR_REFUSED'):
            self.engine.scheduler.kv_cache_manager.free(self.request)
        self.assertNotIn('free',self.trace);self.assertTrue(self.barrier.tickets)
        with self.assertRaisesRegex(RuntimeError,'ACTIVE_REQUEST'):self.barrier.close()
    def test_unshared_page_and_request_generation_are_required(self):
        self.barrier.install();self.engine.scheduler.kv_cache_manager.block.ref_cnt=2
        with self.assertRaisesRegex(RuntimeError,'NOT_EXCLUSIVE'):self.engine.model_executor.execute_model(self.output)
        self.assertNotIn('execute',self.trace)
        self.engine.scheduler.kv_cache_manager.block.ref_cnt=1
        self.engine.scheduler.requests['actual']=NS(request_id='actual')
        with self.assertRaisesRegex(RuntimeError,'ID_REUSED'):self.engine.model_executor.execute_model(self.output)
    def test_prefix_cache_and_async_queue_are_not_guessed_safe(self):
        for name,value in [('async_scheduling',True),('batch_queue',[])]:
            old=getattr(self.engine,name);setattr(self.engine,name,value)
            with self.assertRaisesRegex(RuntimeError,'SYNCHRONOUS'):
                KVReleaseBarrier(self.engine,notify=lambda *a:[],world_size=2,group_ids=(0,),source_check=lambda e:{})
            setattr(self.engine,name,old)
        self.engine.vllm_config.cache_config.enable_prefix_caching=True
        with self.assertRaisesRegex(RuntimeError,'PRIVATE_NONCONNECTOR'):
            KVReleaseBarrier(self.engine,notify=lambda *a:[],world_size=2,group_ids=(0,),source_check=lambda e:{})
    def test_free_failure_does_not_forget_ticket(self):
        class Failing(Manager):
            def free(self,request):raise RuntimeError('pool failure')
        self.engine.scheduler.kv_cache_manager=Failing(self.trace)
        self.barrier.install();self.engine.model_executor.execute_model(self.output)
        with self.assertRaisesRegex(RuntimeError,'pool failure'):self.engine.scheduler.kv_cache_manager.free(self.request)
        self.assertTrue(self.barrier.tickets)

    def test_page_reassignment_before_release_is_not_accepted(self):
        self.barrier.install();self.engine.model_executor.execute_model(self.output)
        self.engine.scheduler.kv_cache_manager.block.block_id=5
        with self.assertRaisesRegex(RuntimeError,'PAGE_SET_CHANGED'):
            self.engine.scheduler.kv_cache_manager.free(self.request)
        self.assertNotIn('free',self.trace)
        with self.assertRaisesRegex(RuntimeError,'REMOVED_WITHOUT'):
            self.engine.model_executor.execute_model(self.output)

    def test_empty_page_inventory_cannot_qualify_a_writer(self):
        self.engine.scheduler.kv_cache_manager.get_blocks=lambda rid:NS(blocks=([],))
        self.barrier.install()
        with self.assertRaisesRegex(RuntimeError,'NO_OWNED'):
            self.engine.model_executor.execute_model(self.output)
        self.assertNotIn('execute',self.trace)


if __name__=='__main__':unittest.main()
