import copy
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

from ascend_fi.inspect.surface import ModuleSurfaceAdapter, tensor_paths
from ascend_fi.inspect.quantization import QuantizationResolver
from ascend_fi.inspect.session import InspectSession, page
from ascend_fi.inspect.extension import InspectWorkerExtension
from ascend_fi.inspect.operators import OperatorEvidenceAdapter
from ascend_fi.inspect.plans import module_candidates, export_plan, validate_plan
from ascend_fi.inspect.client import InspectClient
from ascend_fi.discovery.adapters import TraceContext
from ascend_fi.discovery.resolver import load_capabilities
from ascend_fi.validation import validate_target
from ascend_fi.validation.evidence import RecordedLosslessExecutor
from ascend_fi.contracts import Candidate

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    dtype, device, layout, is_meta = 'torch.bfloat16', 'cpu', 'torch.strided', False
    def __init__(self, floating=True):
        self.shape = (1, 8)
        self.floating = floating
    def stride(self): return (8, 1)
    def numel(self): return 8
    def is_floating_point(self): return self.floating
    def cpu(self): raise AssertionError('no CPU copy')
    def data_ptr(self): raise AssertionError('no pointer reading')


class Handle:
    def __init__(self, module, hook): self.module, self.hook = module, hook
    def remove(self):
        if self.hook in self.module.hooks: self.module.hooks.remove(self.hook)


class Module:
    def __init__(self, children=None):
        self._modules = children or {}
        self._parameters = {'weight': Tensor(False)}
        self._buffers = {}
        self.hooks = []
    def register_forward_pre_hook(self, hook, with_kwargs=False):
        assert with_kwargs
        self.hooks.append(hook)
        return Handle(self, hook)
    def forward(self, *args, **kwargs):
        for hook in list(self.hooks):
            assert hook(self, args, kwargs) is None
        return args[0] if args else kwargs


def make_session(children=None, rank=0):
    model = Module(children or {'layers.10.mlp.down_proj': Module(),
                                'layers.1.mlp.down_proj': Module(), 'layers.2.self_attn.qkv_proj': Module()})
    return InspectSession(model, {'global': {'value': rank, 'source': 'fixture'}}, {'fixture': True},
                          epoch='epoch-fixture', predicate=lambda v: isinstance(v, Tensor))


ARM = {'mode': 'REFERENCE_REQUANT', 'scale_provider': 'tensor_absmax', 'scale_multiplier': .25,
       'rank_filter': [0], 'max_hits': 4096, 'interval': 1, 'tensor_selector': 'first_float'}


class InspectTests(unittest.TestCase):
    def setUp(self): self.s = make_session()
    def tearDown(self): self.s.stop()
    def observe(self, value=None, kwargs=None, **budget):
        name = 'layers.1.mlp.down_proj'
        self.s.observe_start(service_epoch=self.s.epoch, module_names=[name], request_window='one-user-request',
                             exclusive=True, **budget)
        m = self.s.surface.modules[name]
        m.forward(*(()) if kwargs else (value if value is not None else Tensor(),), **(kwargs or {}))
        self.s.stop()
        return module_candidates(self.s, True)
    def test_natural_layer_sort_not_depth64(self):
        self.assertEqual(self.s.surface.owned_layers, [1, 2, 10])
        rows = self.s.list_modules({'layer_index': 1})['items']
        self.assertEqual([r['layer_index'] for r in rows], [1])
        self.assertEqual(self.s.surface.records[-1]['layer_index'], 10)
    def test_module_and_parameter_alias(self):
        shared = Module()
        surface = ModuleSurfaceAdapter(Module({'a': shared, 'b': shared}), {}, lambda x: isinstance(x, Tensor))
        self.assertEqual(surface.by_name['a']['aliases'], ['a', 'b'])
        self.assertEqual(surface.by_name['a']['parameters'][0]['aliases'], ['a.weight', 'b.weight'])
        with self.assertRaisesRegex(ValueError, 'ALIAS'): surface.selected(['a'])
    def test_properties_not_executed(self):
        class Hazard(Module):
            @property
            def quant_method(self): raise AssertionError('property executed')
            @property
            def layer_idx(self): raise AssertionError('property executed')
        surface = ModuleSurfaceAdapter(Hazard(), {}, lambda v: isinstance(v, Tensor))
        self.assertIsNone(surface.records[0]['quant_method'])
    def test_nested_kwargs_and_no_tensor_retention(self):
        t = Tensor()
        records = tensor_paths(({'nested': [t]},), {'hidden_states': t}, lambda x: isinstance(x, Tensor))
        self.assertEqual(records[0]['path'], ['args', 0, 'nested', 0])
        self.assertEqual(records[1]['path'], ['kwargs', 'hidden_states'])
        json.dumps(records)
    def test_input_depth_budget(self):
        with self.assertRaisesRegex(ValueError, 'BUDGET'):
            tensor_paths(([[[Tensor()]]],), {}, lambda x: isinstance(x, Tensor), max_depth=2)
    def test_observer_returns_none_and_removes(self):
        c = self.observe()
        self.assertEqual(len(c), 1)
        self.assertFalse(self.s.active)
        self.assertEqual(self.s.handles, [])
        self.assertEqual(self.s.events[0]['phase'], 'UNKNOWN')
        self.assertFalse(c[0]['proof']['stages']['LOSSLESS_VALIDATED'])
    def test_kwargs_selector(self):
        c = self.observe(kwargs={'hidden_states': Tensor()})
        self.assertEqual(c[0]['binding']['parameter_selector']['tensor_selector'], 'kwarg:hidden_states')
    def test_nested_selector_not_supported(self):
        c = self.observe(value=[Tensor()])
        self.assertIn('EXISTING_BACKEND_CANNOT_SELECT_NESTED_PATH', c[0]['export_blockers'])
    def test_no_floating_input(self): self.assertEqual(self.observe(Tensor(False)), [])
    def test_unknown_layout(self):
        class Lazy(Tensor):
            @property
            def layout(self): raise RuntimeError('lazy')
        c = self.observe(Lazy())
        self.assertIn('METADATA_NOT_EXECUTABLE', c[0]['export_blockers'])
    def test_not_observed_is_not_pass(self):
        self.s.observe_start(service_epoch=self.s.epoch, module_names=['layers.1.mlp.down_proj'],
                             request_window='w', exclusive=True)
        self.assertEqual(self.s.status()['observation_status'], 'HOOK_INSTALLED_BUT_NOT_OBSERVED')
        self.assertEqual(module_candidates(self.s, True), [])
    def test_event_budget(self):
        self.observe(max_events=1)
        self.assertEqual(len(self.s.events), 1)
        self.assertFalse(self.s.handles)
    def test_byte_budget(self):
        self.observe(max_bytes=1)
        self.assertFalse(self.s.events)
        self.assertFalse(self.s.handles)
    def test_exception_removes_hooks(self):
        self.s.surface.predicate = lambda v: (_ for _ in ()).throw(RuntimeError('fixture failure'))
        self.observe()
        self.assertTrue(self.s.errors)
        self.assertFalse(self.s.handles)
    def test_ttl_cleanup_without_next_rpc(self):
        self.s.observe_start(service_epoch=self.s.epoch, module_names=['layers.1.mlp.down_proj'],
                             request_window='w', exclusive=True, ttl_seconds=1)
        time.sleep(1.15)
        self.assertFalse(self.s.handles)
        self.assertEqual(self.s.stop_reason, 'TTL_EXPIRED')
    def test_stale_start_and_exclusive_required(self):
        with self.assertRaisesRegex(ValueError, 'STALE'):
            self.s.observe_start(service_epoch='old', module_names=['layers.1.mlp.down_proj'], request_window='w', exclusive=True)
        with self.assertRaisesRegex(ValueError, 'EXCLUSIVE'):
            self.s.observe_start(service_epoch=self.s.epoch, module_names=['layers.1.mlp.down_proj'], request_window='w', exclusive=False)
    def test_page_budgets(self):
        self.assertEqual(page([{'n': n} for n in range(5)], 0, 2)['next_offset'], 2)
        with self.assertRaises(ValueError): page([], limit=201)
        with self.assertRaises(ValueError): page([{'x': 'long'}], max_bytes=2)
    def test_unknown_quant_no_scale_guess(self):
        m = self.s.surface.modules['layers.1.mlp.down_proj']
        m._buffers['weight_scale'] = Tensor()
        s = make_session({'layers.1.mlp.down_proj': m})
        q = s.describe_target('layers.1.mlp.down_proj')['quantization']
        self.assertEqual(q['model_fields'][0]['role'], 'unknown')
        self.assertFalse(q['reference_policies'][0]['model_scale'])
        s.stop()
    def test_quant_contract_role_dynamic_and_version(self):
        m = Module(); m.quant_method = object(); m._buffers['a_scale'] = Tensor()
        s = make_session({'layers.1.mlp.down_proj': m})
        contract = {'quant_method': 'builtins.object', 'runtime_fingerprint': s.fingerprint,
                    'source_sha256': 'fixture', 'fields': {'a_scale': {'role': 'activation_scale',
                    'granularity': 'per_token', 'dynamic': True, 'target': 'x1', 'providers': []}}}
        q = QuantizationResolver([contract])
        r = q.resolve(m, s.surface.by_name['layers.1.mlp.down_proj'], s.fingerprint)
        self.assertEqual(r['model_fields'][0]['rebind'], 'EVERY_INVOCATION')
        self.assertEqual(r['model_fields'][0]['provider_compatibility'], [])
        self.assertEqual(q.resolve(m, s.surface.records[-1], 'changed')['resolution'], 'UNKNOWN_IMPLEMENTATION')
        s.stop()
    def test_plan_exact_selector_and_draft(self):
        c = self.observe()
        p = export_plan(c, self.s.surface.modules, service_epoch=self.s.epoch,
                        runtime_fingerprint=self.s.fingerprint, frozen_arm=ARM)
        self.assertEqual(p['selected_modules'], ['layers.1.mlp.down_proj'])
        self.assertFalse(p['enabled'])
        self.assertTrue(p['unique_float_input_proven'])
        self.assertEqual(p['prospective_arm']['max_hits'], 4096)
        self.assertEqual(p['prospective_arm']['tensor_selector'], 'tensor_index')
        self.assertIsNone(__import__('re').search(p['prospective_arm']['module_regex'], 'layers.10.mlp.down_proj'))
    def test_plan_no_model_scale_fallback_or_stale(self):
        c = self.observe()
        for options in ({'scale_source': 'model'}, {'service_epoch': 'old'}):
            args = dict(service_epoch=self.s.epoch, runtime_fingerprint=self.s.fingerprint, frozen_arm=ARM)
            args.update(options)
            with self.assertRaises(ValueError): export_plan(c, self.s.surface.modules, **args)
    def test_absent_backend_blocks_export(self):
        self.observe()
        c = module_candidates(self.s, False)
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED'):
            export_plan(c, self.s.surface.modules, service_epoch=self.s.epoch, runtime_fingerprint=self.s.fingerprint, frozen_arm=ARM)
    def test_fused_outer_does_not_prove_internals(self):
        surface = ModuleSurfaceAdapter(Module({'layers.3.mlp.experts': Module()}), {}, lambda v: isinstance(v, Tensor))
        self.assertEqual(surface.records[-1]['logical_role'], 'ROUTED_EXPERTS')
        self.assertEqual(surface.records[-1]['internal_operator_mapping'], 'UNKNOWN')
    def test_validate_stale_never_calls_executor(self):
        class Never:
            def validate_identity(self, p): raise AssertionError('must not call')
        r = validate_plan({'service_epoch': 'old'}, epoch='new', fingerprint='f', executor=Never())
        self.assertFalse(r.lossless_attempted)


class OperatorTests(unittest.TestCase):
    def setUp(self):
        self.raw = json.loads((ROOT/'fixtures/qmm_run_trace.json').read_text().splitlines()[1])
        self.adapter = OperatorEvidenceAdapter(load_capabilities(ROOT.parent/'profiles/capabilities'))
    def test_cold_address_mismatch_preserves_slot_proof(self):
        r = self.adapter.native_run(self.raw, context=TraceContext(rank=0))
        self.assertTrue(r['proof']['semantic_slot_proven'])
        self.assertEqual(r['parameters'][0]['runtime_slot_proven'], 0)
        self.assertFalse(r['direct_gws_address_equality_required'])
        self.assertIsNone(r['module_name'])
        self.assertNotIn('0x2000', json.dumps(r))
    def test_physical_device_not_rank(self):
        r = self.adapter.native_run(self.raw, context=TraceContext())
        self.assertIsNone(r['rank'])
    def test_optional_none_and_unproven_slots(self):
        r = self.adapter.native_run(self.raw, context=TraceContext())
        by = {p['name']: p for p in r['parameters']}
        self.assertEqual(by['x1_scale']['presence'], 'NULL_OR_EMPTY_STORAGE')
        self.assertIsNone(by['x2']['runtime_slot_proven'])
    def test_incorrect_formal_ordinal_cannot_prove(self):
        self.raw['x1_param_ordinal'] = 1
        r = self.adapter.native_run(self.raw, context=TraceContext())
        self.assertFalse(r['proof']['semantic_slot_proven'])
    def test_no_live_promotion(self):
        with self.assertRaisesRegex(ValueError, 'NOT_WIRED'):
            self.adapter.native_run(self.raw, context=TraceContext(), origin='LIVE')
    def test_torch_args_not_native_slots(self):
        r = self.adapter.typed_probe({'event': 'typed_qmm_call_exit', 'tensors': {'x1': {'shape': [2,8]},
                                         'scale': {'dtype': 'float32'}, 'bias': None}}, source_ref='fixture')
        self.assertEqual(r['parameters'][1]['role'], 'unknown')
        self.assertIsNone(r['parameters'][0]['native_slot'])
        self.assertEqual(r['parameters'][2]['presence'], 'NONE')
    def test_recorded_candidate_sanitized_and_cannot_arm(self):
        p = ROOT.parent/'artifacts/evaluation/qwen_2card_fault_effect_v1/v4/candidate.json'
        r = self.adapter.recorded_candidate(json.loads(p.read_text()), source_ref=str(p))
        self.assertFalse(r['arm_allowed'])
        self.assertNotIn('address_metadata', r['binding'])
        self.assertIsNone(r['module_name'])
    def test_existing_native_lossless_contract_unchanged(self):
        r = self.adapter.native_run(self.raw, context=TraceContext(rank=0))
        c = next(c for c in r['candidates'] if c['capability_id'] == 'op_api.quant_matmul.input_buffer_transform')
        result = validate_target(Candidate.from_dict(c), RecordedLosslessExecutor(ROOT/'fixtures/qmm_lossless_validation.json'))
        self.assertTrue(result.lossless_passed)
    def test_native_weight_scale_is_not_activation_scale(self):
        r = self.adapter.native_parameter({'event':'qmm_x2_scale_observation', 'scale_role':'x2_scale',
            'scale_formal_ordinal':3, 'scale_slot':2, 'scale_slot_proven':True, 'view_shape':[8]}, source_ref='fixture')
        p = r['parameters'][0]
        self.assertEqual(p['role'],'weight_scale')
        self.assertEqual(p['runtime_slot_proven'],2)
        self.assertFalse(p['activation_scale_compatible'])
    def test_native_formal_ordinal_does_not_imply_runtime_slot(self):
        r = self.adapter.native_parameter({'event':'qmm_param_collection', 'formal_ordinals':{'x2':1}},source_ref='fixture')
        self.assertEqual(r['parameters'][1]['formal_ordinal'],1)
        self.assertIsNone(r['parameters'][1]['runtime_slot_proven'])


class ExtensionTests(unittest.TestCase):
    def setUp(self):
        self.worker = InspectWorkerExtension()
        self.worker.model_runner = types.SimpleNamespace(model=Module())
        self.worker.rank = 4
        self.env = patch.dict(os.environ, {'VLLM_SERVER_DEV_MODE':'1'})
        self.env.start()
        self.torch = patch.dict(sys.modules, {'torch': types.SimpleNamespace(Tensor=Tensor)})
        self.torch.start()
    def tearDown(self):
        self.worker.demoncat_inspect('observe_stop')
        self.env.stop(); self.torch.stop()
    def test_lazy_init_json_prefixed_and_allowlist(self):
        first = self.worker.demoncat_inspect()
        second = self.worker.demoncat_inspect()
        self.assertEqual(first['service_epoch'], second['service_epoch'])
        self.assertEqual(first['worker']['global']['value'],4)
        self.assertIsNone(first['worker']['tp']['value'])
        json.dumps(first)
        self.assertFalse(self.worker.demoncat_inspect('exec', code='bad')['ok'])
    def test_dev_gate(self):
        with patch.dict(os.environ, {'VLLM_SERVER_DEV_MODE':'0'}):
            self.assertFalse(self.worker.demoncat_inspect()['ok'])
    def test_other_worker_epoch_not_used(self):
        r = self.worker.demoncat_inspect('observe_start', target_service_epochs=['different-epoch'])
        self.assertTrue(r['skipped'])
        self.assertFalse(self.worker.demoncat_inspect()['result']['active'])
    def test_unknown_kwargs_fail_closed(self):
        self.assertFalse(self.worker.demoncat_inspect('status', unsafe='foo')['ok'])
    def test_in_process_model_replacement_invalidates_epoch(self):
        first=self.worker.demoncat_inspect()['service_epoch']
        self.worker.model_runner.model=Module()
        second=self.worker.demoncat_inspect()['service_epoch']
        self.assertNotEqual(first,second)
    def test_rpc_client_preserves_rank_not_list_index(self):
        class Transport:
            def _post(self, path, payload):
                assert payload['method'] == 'demoncat_inspect'
                return 200, {'results': [{'ok':True,'worker':{'global':{'value':9}},'result':{}}]}, None
        self.assertEqual(InspectClient('unused', Transport()).call('status')[0]['worker']['global']['value'],9)


if __name__ == '__main__': unittest.main()
