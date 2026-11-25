import types,unittest,torch
from reference_adapters.control_guard import preflight,snapshot,ReferenceRefused
class GuardTests(unittest.TestCase):
    def test_all_missing_sites_fail_on_control_and_leave_hooks_clean(self):
        from reference_adapters import hidden_rows,swiglu,vision_precision,matmul_order,quant_scale,logits_mapping
        for module in (hidden_rows,swiglu,vision_precision,matmul_order,quant_scale,logits_mapping):
            for factory in module.FACTORIES.values():
                model=torch.nn.Module();model.visual=torch.nn.Module();before=snapshot(model)
                with self.subTest(factory=factory.__name__),self.assertRaises(ReferenceRefused):
                    preflight(factory,model,types.SimpleNamespace())
                self.assertEqual(snapshot(model),before)
    def test_known_site_preflight_is_inert_and_repeatable(self):
        from reference_adapters.matmul_order import ReverseSplitK
        model=torch.nn.Sequential(torch.nn.Linear(8,8).to(torch.bfloat16));before=snapshot(model)
        x=torch.ones(1,8,dtype=torch.bfloat16);output=model(x)
        for i in range(2):
            result=preflight(ReverseSplitK,model,types.SimpleNamespace())
            self.assertTrue(result['resources_zero']);self.assertFalse(result['actual_consumers_proven'])
            self.assertEqual(snapshot(model),before);self.assertTrue(torch.equal(model(x),output))
