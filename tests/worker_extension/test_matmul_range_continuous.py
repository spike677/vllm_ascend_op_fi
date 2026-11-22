import unittest
import sys
from types import SimpleNamespace
from unittest.mock import patch
import torch
from fault_injection.engines.matmul_range import recompute_cell
from fault_injection.engines.region_lease import RegionLeases
from fault_injection.sites.projection39 import ProjectionConsumerMode


class MatmulTests(unittest.TestCase):
    def test_unqualified_and_post_collective_projections_stay_closed(self):
        from fault_injection.sites.matmul39 import Matmul39Site
        module=torch.nn.Linear(4,4,bias=False);module.gather_output=False
        module.quant_method=type('UnquantizedLinearMethod',(),{})()
        site=object.__new__(Matmul39Site);x=torch.ones(2,4,dtype=torch.bfloat16)
        with patch.dict(sys.modules,{'vllm.model_executor.layers.linear':SimpleNamespace(ColumnParallelLinear=torch.nn.Linear),
                                    'torch_npu':SimpleNamespace(get_npu_format=lambda t:2)}):
            self.assertEqual(site.classify(module,(x,),x),(False,'UNQUANTIZED_REFERENCE_CPU_ONLY_NOT_NPU_QUALIFIED'))
            module.gather_output=True
            self.assertEqual(site.classify(module,(x,),x),(False,'ONLY_LOCAL_COLUMN_SHARDS_NO_COLLECTIVE'))

    def test_actual_integer_operands_and_scales(self):
        q=torch.tensor([127,-128,20,2],dtype=torch.int8)
        w=torch.tensor([127,127,-10,30],dtype=torch.int8)
        full,fault,info=recompute_cell(q,w,scale=.125,bias=3.)
        self.assertEqual(float(full),float((q.long()*w.long()).sum())*.125+3.)
        self.assertEqual(float(full-fault),7.5)
        self.assertEqual(info['omitted_k'],1)

    def test_consumer_reads_fault_then_original_bytes_restore(self):
        x=torch.tensor([2.,4.,6.,8.]);w=torch.ones(4)
        full,fault,_=recompute_cell(x,w)
        output=full.float().reshape(1);before=output.clone();event={}
        leases=RegionLeases('test',1,stream=lambda:0,complete=lambda s:None)
        mode=ProjectionConsumerMode(leases)
        mode.watch(output,event,replacement=lambda r:fault.float().reshape(1))
        with mode:consumed=output*2
        self.assertEqual(float(consumed),24.)
        self.assertTrue(torch.equal(output,before));self.assertEqual(leases.count(),0)
        self.assertTrue(event['consumed'] and event['restored'])

    def test_nonfinite_or_misaligned_refused(self):
        for x,y in [(torch.ones(4),torch.ones(5)),(torch.full((4,),float('nan')),torch.ones(4))]:
            with self.assertRaises(ValueError):recompute_cell(x,y)
