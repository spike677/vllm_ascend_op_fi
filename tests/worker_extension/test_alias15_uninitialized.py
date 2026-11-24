"""Risk characterization only: synthetic CPU allocator, not NPU evidence."""
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch
import test_alias_output_census as fixture
from fault_injection.engines.region_lease import RegionLeases


class UninitializedAliasTests(unittest.TestCase):
    def test_skipped_real_writer_can_propagate_old_nonfinite_bytes(self):
        from fault_injection.sites.alias15 import Alias15Site
        model=torch.nn.Sequential(fixture.GatedModel())
        x=torch.arange(8,dtype=torch.bfloat16).reshape(2,4)
        baseline=model(x);self.assertTrue(torch.isfinite(baseline).all())
        real_empty=torch.empty
        def poisoned_allocator(*args,**kwargs):
            # Deterministic stand-in for arbitrary old allocation contents.
            # Never installed by product code or proposed as an NPU fix.
            return real_empty(*args,**kwargs).fill_(float('nan'))
        leases=RegionLeases('risk',1,stream=lambda:0,complete=lambda stream:None)
        with patch.dict(sys.modules,torch_npu=NS(get_npu_format=lambda t:2)):
            site=Alias15Site(model,NS(input_batch=NS(req_ids=['risk-request'])),fault=True,region_leases=leases)
        site.source_sha=hashlib.sha256(Path(fixture.__file__).read_bytes()).hexdigest()
        with patch.object(torch,'empty',poisoned_allocator):
            site.install()
            try:
                result=model(x)
                self.assertTrue(torch.isnan(result).all())
                event=site.events[0]
                self.assertFalse(event['initial_all_finite'])
                self.assertTrue(event['projection_reads_exact_produced_value'])
                self.assertEqual(event['before_sha256'],event['restored_sha256'])
                self.assertEqual(leases.count(),0)
            finally:site.close()
        self.assertTrue(torch.equal(model(x),baseline))
