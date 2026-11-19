"""Public config cannot select research/group protocols or borrow acceptance."""
from types import SimpleNamespace as NS
from unittest.mock import patch
import unittest
from demoncat.runtime.vllm_ascend.worker_extension.ordinary_admission import ordinary_route_report
from demoncat.runtime.vllm_ascend.worker_extension.registry import create_worker_provider
from demoncat.runtime.vllm_ascend.worker_extension.serving_tensor_binding import TensorServingBinding
from demoncat.runtime.vllm_ascend.worker_extension.runtime import _provider_error

class OrdinaryAdmissionTests(unittest.TestCase):
    def test_unbound_query_and_create_share_diagnostic_without_side_effect(self):
        worker=NS();model=object()
        report=ordinary_route_report('18.api_half',worker,model)
        self.assertFalse(report['can_prepare'])
        with self.assertRaises(RuntimeError) as caught:
            create_worker_provider('18.api_half','1',worker,model,{})
        self.assertEqual(caught.exception.prerequisites,report)
        self.assertEqual(_provider_error(caught.exception)['prerequisites'],report)
        self.assertEqual(vars(worker),{})

    def test_pending_and_excluded_do_not_fall_back_to_legacy_provider(self):
        for mechanism in ('15.alias_detach','25.reduce_scatter','28.fa.legal_tiling',
                          '28.fia.legal_tiling','28.fia.denominator_replay','17.spatial_padding',
                          '20.buffer_substitute','22.wrong_row','26.residual_stomp','26.fia_destination',
                          '33.paged_decode','41.tail_column','39.matmul_range'):
            with self.subTest(mechanism=mechanism):
                report=ordinary_route_report(mechanism,NS(),object())
                self.assertFalse(report['can_prepare'])
                with patch('importlib.import_module',side_effect=AssertionError('legacy factory import')):
                    with self.assertRaises(RuntimeError) as caught:
                        create_worker_provider(mechanism,'1',NS(),object(),{})
                self.assertEqual(caught.exception.prerequisites['primary'],report['primary'])

    def test_research_tensor_binding_rejected_in_public_registry(self):
        model=object();worker=NS();runner=NS(model=model)
        binding=TensorServingBinding('18.api_half',worker,runner,model,(1,2),'{}','a'*64,'b'*64,1,1)
        worker._demoncat_natural_tensor_serving_bindings={'18.api_half':binding}
        with self.assertRaisesRegex(RuntimeError,'RMS18_ORDINARY_BINDING_REQUIRED'):
            create_worker_provider('18.api_half','1',worker,model,{})
        self.assertFalse(binding.leases)

if __name__=='__main__':unittest.main()
