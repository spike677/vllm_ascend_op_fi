import sys,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/worker_core_v11_20261118'))
from external_kv_v2_example import KVProvider
from fault_injection.engines.region_lease import RegionLeases
from demoncat.runtime.vllm_ascend.worker_extension.persistent_kv_regions import RequestKVRegions


class KVExampleTests(unittest.TestCase):
    def setup_provider(self):
        p=object.__new__(KVProvider);p.current_request='r';p.slot_by_name={}
        notice=dict(request_id='r',request_generation='g',pages=[[0,3]],before_free=False,allocation_block_sizes={'0':8})
        p.owner_observation=lambda:notice
        table=SimpleNamespace(physical_block_size=8,block_size=4,blocks_per_phys_block=2,dcp_world_size=1)
        p.runner=SimpleNamespace(input_batch=SimpleNamespace(block_table=SimpleNamespace(block_tables=[table])))
        p.regions=RequestKVRegions(RegionLeases('o',1,stream=lambda:0,complete=lambda s:None),owner_observation=p.owner_observation,maximum_bytes=32)
        return p,notice,SimpleNamespace(value_cache=torch.zeros(10,4,1,2,dtype=torch.bfloat16))
    def test_kernel_subblock_maps_back_to_owned_allocation(self):
        p,n,impl=self.setup_provider();p.after_writer('layer',impl,{'slot_mapping':torch.tensor([24,25])})
        e=p.regions.entries['layer'];self.assertEqual(e['page'],[0,3]);self.assertEqual(e['kernel_page'],6)
        self.assertEqual(impl.value_cache[6,0,0,0],1)
        p.after_writer('layer',impl,{'slot_mapping':torch.tensor([26])})
        self.assertEqual(len(p.regions.events),1)
    def test_foreign_allocation_refused_before_write(self):
        p,n,impl=self.setup_provider()
        with self.assertRaisesRegex(RuntimeError,'NOT_OWNED'):p.after_writer('layer',impl,{'slot_mapping':torch.tensor([16])})
        self.assertEqual(p.regions.leases.count(),0);self.assertEqual(impl.value_cache.count_nonzero(),0)
    def test_inconsistent_actual_block_geometry_refused(self):
        p,n,impl=self.setup_provider();n['allocation_block_sizes']['0']=16
        with self.assertRaisesRegex(RuntimeError,'MAPPING'):p.after_writer('layer',impl,{'slot_mapping':torch.tensor([24])})
