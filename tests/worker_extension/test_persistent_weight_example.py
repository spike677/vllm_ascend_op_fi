import hashlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'experiments/worker_core_v11_20261118'))
import external_weight_scale_v2_example as example


class AscendW8A8DynamicLinearMethod:pass


class WeightExampleTests(unittest.TestCase):
    def setUp(self):
        self.model=torch.nn.Module();layer=torch.nn.Module();self.model.add_module('projection',layer)
        layer.weight_scale=torch.nn.Parameter(torch.ones(4,dtype=torch.bfloat16),requires_grad=False)
        layer.weight_scale_fp32=layer.weight_scale.float()
        layer.quant_method=SimpleNamespace(quant_method=AscendW8A8DynamicLinearMethod())
        self.layer=layer
        self.census={'scale_readers':{'projection.weight_scale':{'operator':'npu::npu_quant_matmul','argument':'scale','calls':4}}}
        self.worker=SimpleNamespace(model_runner=SimpleNamespace(model=self.model),
            _v2_weight_example_context={'census':self.census,'boundary':lambda:None})
        file=Path(__file__);sha=hashlib.sha256(file.read_bytes()).hexdigest()
        for item in (patch.object(example.inspect,'getsourcefile',return_value=str(file)),
                     patch.object(example,'SCHEME_SHA',sha),patch.object(example,'WRAPPER_SHA',sha)):
            item.start();self.addCleanup(item.stop)
    def make(self):return example.WeightScaleProvider(self.worker,self.model,{'rank':0},
        mechanism='external.weight_scale_half',region_kind='persistent_weight')
    def test_actual_census_selects_tensor_without_identity_name_shadow(self):
        p=self.make();self.assertIs(p.scale,self.layer.weight_scale)
        self.assertIs(p.derived,self.layer.weight_scale_fp32)
        self.assertEqual(p.allocation,example.identity(self.layer.weight_scale))
        self.assertFalse(p.enabled);self.assertFalse(p.invalidated)
    def test_unobserved_or_chunked_route_not_selected(self):
        self.layer._chunk_size=2
        with self.assertRaisesRegex(RuntimeError,'ACTUALLY_OBSERVED'):self.make()
        self.layer._chunk_size=0;self.census['scale_readers'].clear()
        with self.assertRaisesRegex(RuntimeError,'ACTUALLY_OBSERVED'):self.make()
    def test_unknown_derived_bytes_refused(self):
        self.layer.weight_scale_fp32.add_(1)
        with self.assertRaisesRegex(RuntimeError,'DERIVED_SCALE_IDENTITY'):self.make()


if __name__=='__main__':unittest.main()
