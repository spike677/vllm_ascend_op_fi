import copy
import types
import unittest
from unittest.mock import Mock, patch

from ascend_fi.inspect.model_descriptor import config_view, describe_model, digest, module_contract, annotate_call
from ascend_fi.inspect.model_adapters import select_adapter, structural_targets
from ascend_fi.inspect.target_resolver import resolve_targets
from ascend_fi.inspect.plan_builder import build_target_plan, verify_plan
from ascend_fi.inspect.plan_executor import TargetPlanExecutor, plan_window_gate
from ascend_fi.inspect.input_contract import check_input, PlanInputContractMixin
from ascend_fi.inspect.effect_stats import EffectStats
from ascend_fi.inspect.session import InspectSession
from runtime_inspect.test_inspect import Tensor, Module
from runtime_inspect.test_inspect import Handle


class Activation(Tensor):
    def is_contiguous(self): return True


def report_fixture(layers=3, rank=0, model_type='qwen3', alias_name=None):
    prefix, arch = {'qwen3': ('Qwen3', 'Qwen3ForCausalLM'),
                    'qwen2': ('Qwen2', 'Qwen2ForCausalLM'),
                    'deepseek_v4': ('DeepseekV4', 'DeepseekV4ForCausalLM')}[model_type]
    descriptor = {'schema': 'MODEL_DESCRIPTOR_V1', 'origin': 'LIVE', 'service_epoch': f'e{rank}',
        'runtime_fingerprint': f'r{rank}', 'model_fingerprint': f'm{rank}',
        'model_class': 'vllm.model_executor.models.model.' + arch,
        'config': {'model_type': model_type, 'architectures': [arch], 'num_hidden_layers': layers, 'hidden_size': 8},
        'served_model_name': alias_name or 'arbitrary', 'ranks': {'global': {'value': rank}},
        'backend': {'reference': {'source_sha256': 'a' * 64}},
        'enumeration': {'complete': True, 'truncated': False}, 'owned_layers': list(range(layers)),
        'topology': {'tensor_parallel_size': 2, 'pipeline_parallel_size': 1,
                     'data_parallel_size': 1, 'enable_expert_parallel': False},
        'execution': {'enforce_eager': True}}
    modules, events = [], []
    for i in range(layers):
        for parent, pc, leaves in [('self_attn', prefix + 'Attention',
                [('qkv_proj', 'QKVParallelLinear'), ('o_proj', 'RowParallelLinear')]),
                ('mlp', prefix + 'MLP', [('gate_up_proj', 'MergedColumnParallelLinear'), ('down_proj', 'RowParallelLinear')])]:
            root = f'model.layers.{i}.{parent}'
            modules.append({'module_name': root, 'module_class': 'vllm.' + pc, 'layer_index': i})
            for leaf, cls in leaves:
                name = root + '.' + leaf
                modules.append({'module_name': name, 'parent_name': root, 'module_class': 'vllm.' + cls,
                    'class_mro': ['vllm.' + cls], 'layer_index': i, 'aliases': [name],
                    'input_feature_size': 8, 'ranks': descriptor['ranks']})
                events.append({'module_name': name, 'origin': 'LIVE', 'service_epoch': f'e{rank}',
                    'runtime_fingerprint': f'r{rank}', 'session': 's', 'phase': 'UNKNOWN',
                    'parameters': [{'path': ['args', 0], 'semantic_argument': 'x', 'object_alias_paths': [],
                        'metadata': {'floating': True, 'dtype': 'torch.bfloat16', 'layout': 'torch.strided',
                            'is_meta': False, 'contiguous': True, 'metadata_status': 'COMPLETE', 'shape': [1, 8]}}]})
    status = {'session': 's', 'active': False, 'installed_handles': 0, 'errors': [],
              'terminal_stop_reason': None, 'module_call_counts': {e['module_name']: 3 for e in events}}
    return descriptor, modules, events, status


ROLES = ['ATTN_QKV', 'ATTN_OUT', 'MLP_GATE_UP', 'MLP_DOWN']
ARM = {'mode': 'REFERENCE_REQUANT', 'scale_provider': 'tensor_absmax', 'rank_filter': [0],
       'interval': 1, 'max_hits': 4096, 'scale_multiplier': .25, 'tensor_selector': 'first_float',
       'tensor_index': None, 'layer_filter': None, 'module_regex': '.*', 'eps': 1e-12}


def reports(layers=3):
    return [resolve_targets(*report_fixture(layers, rank)) for rank in (0, 1)]


def plan_for(rows=None, **kwargs):
    return build_target_plan(rows if rows is not None else reports(), roles=kwargs.pop('roles', ROLES),
        rank_filter=kwargs.pop('rank_filter', [0]), scale_multiplier=kwargs.pop('scale_multiplier', 20),
        reference_arm=kwargs.pop('reference_arm', ARM), max_tokens=kwargs.pop('max_tokens', 1024),
        max_prompt_tokens=kwargs.pop('max_prompt_tokens', 69), **kwargs)


def reseal_report(report):
    report.pop('report_sha256', None)
    report['report_sha256'] = digest(report)


def measured_fault(plan, identity=False):
    workers = []
    for target in plan['workers']:
        n = len(target['targets'])
        rank = target['rank']
        stats = EffectStats()
        per_module = {}
        matched = []
        for i, t in enumerate(target['targets'], 1):
            name = t['module_name']
            matched.append({'module': name, 'layer': t['layer'], 'tp_rank': rank})
            per_module[name] = {'hook_seen': 1, 'candidate_seen': int(target['targeted']), 'hit_count': int(target['targeted'])}
            if target['targeted']:
                stats.append({'module': name, 'layer': t['layer'], 'rank': rank,
                    'mode': 'IDENTITY' if identity else 'REFERENCE_REQUANT', 'scale_provider': 'tensor_absmax', 'scale_multiplier': 20,
                    'hit_index': i, 'changed_elements': 0 if identity else 8, 'total_elements': 16, 'clip_count': 0,
                    'mean_abs_error': 1., 'max_abs_error': 2., 'scale_min': .1, 'scale_max': .1, 'scale_mean': .1})
        workers.append({'pid': 100 + rank, 'enabled': True, 'installed': True, 'matched_modules': matched,
            'hook_seen': n, 'candidate_seen': n if target['targeted'] else 0, 'hit_count': n if target['targeted'] else 0,
            'reject_reasons': {} if target['targeted'] else {'rank_filter': n}, 'per_module': per_module,
            'effect_summary': stats.snapshot(), 'armed_config': {**plan['prospective_arm'], 'request_generation': 7,
                'mode': 'IDENTITY' if identity else 'REFERENCE_REQUANT'},
            'input_contract': {'installed': True, 'handles': n, 'plan_sha256': plan['plan_sha256'],
                               'calls_checked': n, 'violations': 0}})
    ack = {'http_status': 200, 'results': [{'ok': True} for _ in workers]}
    return {'http_status': 200, 'request_generation': 7, 'fault_disarm': ack,
            'request_window_evidence': {'single_request_completed': True},
            'fault_transaction': {'arm': ack, 'measurement_query': {'http_status': 200, 'results': workers}}}


class ModelTargetingTests(unittest.TestCase):
    def test_dynamic_layers(self):
        for count in (1, 4, 28, 64):
            plan = plan_for(reports(count))
            self.assertEqual(len(plan['selected_modules']), 4 * count)
            self.assertEqual(plan['prospective_arm']['layer_filter'], list(range(count)))
            self.assertEqual(plan['prospective_arm']['tensor_selector'], 'tensor_index')
            verify_plan(plan)

    def test_distill_name_uses_actual_architecture(self):
        d, *_ = report_fixture(model_type='qwen2', alias_name='DeepSeek-R1-Distill-Qwen-32B')
        self.assertEqual(select_adapter(d)['adapter_id'], 'qwen_dense_v1')

    def test_nested_text_config(self):
        value, source = config_view({'model_type': 'vision', 'text_config': {'model_type': 'qwen3', 'num_hidden_layers': 2}})
        self.assertEqual(value['model_type'], 'qwen3')
        self.assertEqual(source, 'text_config')

    def test_identity_conflict(self):
        d, *_ = report_fixture()
        d['model_class'] = 'runtime.DeepseekV4ForCausalLM'
        self.assertEqual(select_adapter(d)['status'], 'IDENTITY_IMPLEMENTATION_MISMATCH')

    def test_unknown_model(self):
        d, *_ = report_fixture()
        d['config']['model_type'] = 'future_unknown'
        self.assertEqual(select_adapter(d)['status'], 'UNSUPPORTED_ARCHITECTURE')

    def test_missing_layer_role_not_silently_ignored(self):
        rows = reports()
        rows[0]['targets'].pop()
        reseal_report(rows[0])
        with self.assertRaisesRegex(ValueError, 'COVERAGE_GAP'): plan_for(rows)

    def test_wrong_parent_not_main_attention(self):
        d, m, e, s = report_fixture()
        m[0]['module_class'] = 'vllm.Indexer'
        r = resolve_targets(d, m, e, s)
        self.assertNotIn('model.layers.0.self_attn.qkv_proj', [t['module_name'] for t in r['targets']])

    def test_keyword_binding(self):
        rows = []
        for rank in (0, 1):
            d, m, e, s = report_fixture(rank=rank)
            for event in e:
                event['parameters'][0].update(path=['kwargs', 'hidden_states'], semantic_argument='hidden_states')
            rows.append(resolve_targets(d, m, e, s))
        self.assertEqual(plan_for(rows)['prospective_arm']['tensor_selector'], 'kwarg:hidden_states')

    def test_mixed_selectors_blocked(self):
        d, m, e, s = report_fixture()
        e[0]['parameters'][0].update(path=['kwargs', 'x'])
        rows = [resolve_targets(d, m, e, s), reports()[1]]
        with self.assertRaisesRegex(ValueError, 'MIXED_SELECTORS'): plan_for(rows)

    def test_unobserved_keeps_structural_role(self):
        d, m, e, s = report_fixture()
        report = resolve_targets(d, m, [], s)
        self.assertEqual(len(report['targets']), 12)
        self.assertTrue(all(t['role_resolved'] and not t['observed'] and not t['bound'] for t in report['targets']))

    def test_unsafe_bindings_blocked(self):
        changes = [('dtype', 'torch.int8'), ('layout', 'packed'), ('contiguous', False),
                   ('is_meta', True), ('shape', [1, 7]), ('shape', [1, 2, 4])]
        for key, value in changes:
            d, m, e, s = report_fixture()
            e[0]['parameters'][0]['metadata'][key] = value
            r = resolve_targets(d, m, e, s)
            self.assertFalse(r['targets'][0]['bound'], (key, value))

    def test_alias_and_auxiliary_float_rejected(self):
        d, m, e, s = report_fixture()
        e[0]['parameters'].append(copy.deepcopy(e[0]['parameters'][0]))
        e[0]['parameters'][1].update(path=['args', 1], semantic_argument='router_logits')
        r = resolve_targets(d, m, e, s)
        self.assertFalse(r['targets'][0]['bound'])
        d, m, e, s = report_fixture()
        e[0]['parameters'][0]['object_alias_paths'] = [['args', 1]]
        self.assertFalse(resolve_targets(d, m, e, s)['targets'][0]['bound'])

    def test_residual_not_activation(self):
        d, m, e, s = report_fixture()
        e[0]['parameters'][0]['semantic_argument'] = 'residual'
        self.assertFalse(resolve_targets(d, m, e, s)['targets'][0]['bound'])

    def test_old_epoch_and_observation_session_rejected(self):
        for field in ('service_epoch', 'runtime_fingerprint', 'session', 'origin'):
            d, m, e, s = report_fixture()
            e[0][field] = 'wrong'
            self.assertFalse(resolve_targets(d, m, e, s)['targets'][0]['bound'])

    def test_unknown_phase_does_not_become_decode(self):
        with self.assertRaisesRegex(ValueError, 'PHASE_NOT_PROVEN'): plan_for(phase='decode')

    def test_observation_budget_is_partial(self):
        d, m, e, s = report_fixture()
        s['terminal_stop_reason'] = 'METADATA_BUDGET_REACHED'
        self.assertFalse(any(t['bound'] for t in resolve_targets(d, m, e, s)['targets']))

    def test_partial_enumeration_blocked(self):
        rows = reports()
        rows[0]['descriptor']['enumeration']['truncated'] = True
        reseal_report(rows[0])
        with self.assertRaisesRegex(ValueError, 'ENUMERATION_PARTIAL'): plan_for(rows)

    def test_graph_or_parallel_topology_not_silently_changed(self):
        for obj, key, value in [('execution', 'enforce_eager', False),
                                ('topology', 'pipeline_parallel_size', 2),
                                ('topology', 'enable_expert_parallel', True)]:
            rows = reports()
            rows[0]['descriptor'][obj][key] = value
            reseal_report(rows[0])
            with self.assertRaises(ValueError): plan_for(rows)

    def test_safety_cap_is_not_4096_and_log_budget_independent(self):
        p = plan_for(reports(64))
        self.assertEqual(p['request_window']['max_hits'], 567808)
        self.assertEqual(p['request_window']['log_limit'], 32)

    def test_invalid_budgets(self):
        for kw in ({'max_tokens': 1000000}, {'max_tokens': -1}, {'max_prompt_tokens': 0},
                   {'scale_multiplier': float('nan')}, {'scale_multiplier': 0}, {'rank_filter': [2]},
                   {'roles': ['UNKNOWN']}):
            with self.assertRaises(ValueError): plan_for(**kw)

    def test_no_runtime_scale_fallback(self):
        with self.assertRaisesRegex(ValueError, 'no fallback'):
            plan_for(reference_arm={**ARM, 'scale_provider': 'module_runtime_scale'})

    def test_plan_tamper(self):
        p = plan_for()
        p['prospective_arm']['max_hits'] = 0
        with self.assertRaisesRegex(ValueError, 'INTEGRITY'): verify_plan(p)

    def test_deepseek_is_not_distill_qualification(self):
        d, *_ = report_fixture(model_type='deepseek_v4')
        self.assertEqual(select_adapter(d)['adapter_id'], 'deepseek_v4_v1')
        self.assertEqual(select_adapter(d)['qualification'], 'READONLY_UNTIL_VARIANT_LIVE_VALIDATED')

    def test_v4_fused_projection_one_physical_target(self):
        d, m, e, s = report_fixture(model_type='deepseek_v4')
        m[1]['module_name'] = 'model.layers.0.self_attn.fused_wqa_wkv'
        adapter, targets, gaps = structural_targets(d, m)
        fused = next(t for t in targets if t['module_name'].endswith('fused_wqa_wkv'))
        self.assertEqual(fused['roles'], ['ATTN_Q_A', 'ATTN_KV_A'])
        self.assertEqual(len([t for t in targets if t['module_name'] == fused['module_name']]), 1)

    def test_v4_compressor_not_main_projection(self):
        d, m, e, s = report_fixture(model_type='deepseek_v4')
        m[1]['module_name'] = 'model.layers.0.self_attn.compressor.fused_wkv_wgate'
        m[1]['parent_name'] = 'model.layers.0.self_attn.compressor'
        _, targets, gaps = structural_targets(d, m)
        self.assertIn(m[1]['module_name'], [g['module_name'] for g in gaps])
        self.assertNotIn(m[1]['module_name'], [t['module_name'] for t in targets])

    def test_signature_and_partition_metadata(self):
        class Linear(Module):
            def forward(self, hidden_states, *, optional=None): return hidden_states
        m = Linear()
        m.input_size, m.input_size_per_partition, m.input_is_parallel = 8, 4, False
        self.assertEqual(module_contract(m)['input_feature_size'], 8)
        a = Activation()
        params = [{'path': ['args', 0], 'metadata': {}}]
        annotate_call(m, (a,), {}, params)
        self.assertEqual(params[0]['semantic_argument'], 'hidden_states')
        self.assertTrue(params[0]['metadata']['contiguous'])

    def test_qwen3_real_runtime_reuses_qwen2_mlp(self):
        d, m, e, s = report_fixture()
        for row in m:
            if row['module_name'].endswith('.mlp'):
                row['module_class'] = 'vllm.model_executor.models.qwen2.Qwen2MLP'
            if row['module_name'].endswith('down_proj'):
                row['module_class'] = 'vllm_ascend.ops.linear.AscendRowParallelLinear'
                row['class_mro'] = [row['module_class']]
        for event in e:
            event['parameters'][0]['semantic_argument'] = 'input_'
        result = resolve_targets(d, m, e, s)
        self.assertEqual(len(result['targets']), 12)
        self.assertTrue(all(t['bound'] for t in result['targets']))

    def test_model_submodule_replacement_invalidates_descriptor(self):
        m = Module({'child': Module()})
        s = InspectSession(m, {}, {}, predicate=lambda x: isinstance(x, Tensor))
        m._modules['child'] = Module()
        with self.assertRaisesRegex(ValueError, 'MODEL_STRUCTURE_REPLACED'):
            describe_model(s)

    def test_hf_class_config_attribute(self):
        class Config:
            model_type = 'qwen3'
            @property
            def hidden_size(self): raise AssertionError('do not execute config properties')
        value, _ = config_view(Config())
        self.assertEqual(value['model_type'], 'qwen3')
        self.assertNotIn('hidden_size', value)

    def test_model_descriptor_no_tensor_data_read(self):
        m = Module()
        m.config = types.SimpleNamespace(model_type='qwen3', architectures=['Qwen3ForCausalLM'], num_hidden_layers=1)
        s = InspectSession(m, {'global': {'value': 0}}, {'pid': 1}, predicate=lambda x: isinstance(x, Tensor))
        d, rows = describe_model(s)
        self.assertEqual(d['config']['num_hidden_layers'], 1)
        self.assertTrue(d['enumeration']['complete'])
        self.assertEqual(d['model_fingerprint'], describe_model(s)[0]['model_fingerprint'])

    def test_stop_does_not_hide_budget_exhaustion(self):
        s = InspectSession(Module(), {}, {}, predicate=lambda x: isinstance(x, Tensor))
        s.stop(reason='METADATA_BUDGET_REACHED')
        s.stop()
        self.assertEqual(s.status()['terminal_stop_reason'], 'METADATA_BUDGET_REACHED')

    def test_runtime_guard_is_identity_and_checks_alias(self):
        binding = reports()[0]['targets'][0]['binding']
        x = Activation()
        predicate = lambda v: isinstance(v, Tensor)
        self.assertIs(check_input(binding, (x,), {}, predicate), x)
        with self.assertRaises(ValueError): check_input(binding, (x, x), {}, predicate)
        x.shape = (1, 7)
        with self.assertRaisesRegex(ValueError, 'METADATA'): check_input(binding, (x,), {}, predicate)

    def test_executor_never_arms_without_identity(self):
        owner = Mock()
        runner = TargetPlanExecutor(owner, plan_for(), lambda: [])
        with self.assertRaisesRegex(ValueError, 'IDENTITY_REQUIRED'): runner.run_pair({})
        owner.fault_request.assert_not_called()

    def test_executor_rejects_stale_model_before_owner(self):
        owner = Mock()
        runner = TargetPlanExecutor(owner, plan_for(), lambda: [])
        with self.assertRaisesRegex(ValueError, 'STALE_MODEL'): runner.validate_identity({})
        owner.preflight.assert_not_called()

    def test_window_missing_mechanism_is_not_success(self):
        gate = plan_window_gate(plan_for(), {})
        self.assertEqual(gate['mechanism'], 'INVALID_FAULT_RUN')
        self.assertEqual(gate['full_request_window'], 'UNPROVEN')

    def test_all_hooks_do_not_certify_full_request_execution(self):
        p = plan_for()
        gate = plan_window_gate(p, measured_fault(p))
        self.assertEqual(gate['mechanism'], 'PASS', gate)
        self.assertEqual(gate['target_call_window'], 'PASS')
        self.assertEqual(gate['full_request_window'], 'UNPROVEN')

    def test_armed_strength_mismatch_invalidates_effect(self):
        p = plan_for()
        f = measured_fault(p)
        f['fault_transaction']['measurement_query']['results'][0]['armed_config']['scale_multiplier'] = .25
        self.assertEqual(plan_window_gate(p, f)['mechanism'], 'INVALID_FAULT_RUN')

    def test_budget_guard_rank_and_missing_target_rejections(self):
        p = plan_for()
        for case in ('cap', 'guard', 'rank', 'module', 'disarm'):
            f = measured_fault(p)
            w = f['fault_transaction']['measurement_query']['results']
            if case == 'cap': w[0]['reject_reasons'] = {'max_hits': 1}
            elif case == 'guard': w[0]['input_contract']['violations'] = 1
            elif case == 'rank': w[1]['hit_count'] = 1
            elif case == 'module': w[0]['per_module'].pop(next(iter(w[0]['per_module'])))
            else: f['fault_disarm'] = None
            self.assertEqual(plan_window_gate(p, f)['mechanism'], 'INVALID_FAULT_RUN', case)

    def test_executor_always_removes_when_fault_request_raises(self):
        p = plan_for()
        owner = Mock()
        owner.arm = p['prospective_arm']
        owner.install.return_value = types.SimpleNamespace(actual_module_names=p['selected_modules'])
        owner.fault_request.side_effect = RuntimeError('transport failed')
        runner = TargetPlanExecutor(owner, p, lambda: [])
        runner.identity_receipt = {'plan_sha256': p['plan_sha256']}
        with patch.object(runner, 'rebind'):
            with self.assertRaisesRegex(RuntimeError, 'transport failed'):
                runner.run_pair({'probe_id': 'p'})
        owner.disarm_and_remove.assert_called_once()

    def test_executor_stops_on_recovery_mismatch(self):
        p = plan_for()
        owner = Mock()
        owner.arm = p['prospective_arm']
        owner.install.return_value = types.SimpleNamespace(actual_module_names=p['selected_modules'])
        owner.fault_request.return_value = measured_fault(p)
        owner.request.side_effect = [{'http_status': 200, 'response_text': 'one', 'token_pieces': ['one']},
                                    {'http_status': 200, 'response_text': 'two', 'token_pieces': ['two']}]
        runner = TargetPlanExecutor(owner, p, lambda: [])
        runner.identity_receipt = {'plan_sha256': p['plan_sha256']}
        with patch.object(runner, 'rebind'), patch('ascend_fi.inspect.plan_executor.cleanup_gate', return_value={'status': 'PASS'}):
            pair = runner.run_pair({'probe_id': 'p'})
        self.assertFalse(pair['recovery_exact'])
        self.assertIsNone(runner.identity_receipt)

    def test_executor_no_lossy_when_token_pieces_identity_differs(self):
        p = plan_for()
        owner = Mock()
        result = Mock(lossless_passed=True, restore_exact=True)
        result.to_dict.return_value = {'raw': {'baseline': {'token_pieces': ['one']},
            'disabled': {'token_pieces': ['one']}, 'identity': {'token_pieces': ['two']},
            'recovery': {'token_pieces': ['one']}}}
        runner = TargetPlanExecutor(owner, p, lambda: [])
        with patch.object(runner, 'rebind'), patch('ascend_fi.inspect.plan_executor.ExistingOwnerIdentityValidator') as validator:
            validator.return_value.validate_identity.return_value = result
            with self.assertRaisesRegex(RuntimeError, 'IDENTITY_GATE_FAILED'): runner.validate_identity({})
        self.assertIsNone(runner.identity_receipt)
        owner.fault_request.assert_not_called()

    def contract_worker(self):
        class Linear(Module):
            def forward(self, x): return x
            def register_forward_pre_hook(self, hook, *, with_kwargs, prepend):
                self.hooks.insert(0, hook)
                return Handle(self, hook)
        class Reference:
            def fake_requant_query(self): return {'enabled': False, 'installed': False}
        class Worker(PlanInputContractMixin, Reference):
            pass
        plan = plan_for()
        model = Module({n: Linear() for n in plan['selected_modules']})
        session = InspectSession(model, {'global': {'value': 0}}, {}, epoch='e0', predicate=lambda x: isinstance(x, Tensor))
        worker = Worker()
        worker._demoncat_inspect_session_v1 = session
        d, _ = describe_model(session, worker)
        plan['workers'][0]['model_fingerprint'] = d['model_fingerprint']
        plan.pop('plan_sha256')
        plan['plan_sha256'] = digest(plan)
        return worker, model, plan

    def test_runtime_guard_install_and_exact_cleanup(self):
        worker, model, plan = self.contract_worker()
        with patch.dict('os.environ', {'VLLM_SERVER_DEV_MODE': '1'}):
            bound = worker.fake_requant_bind_target_plan(plan)
            self.assertTrue(bound['ok'], bound)
            self.assertEqual(bound['handles'], len(plan['selected_modules']))
            self.assertTrue(worker.fake_requant_query()['input_contract']['installed'])
            cleared = worker.fake_requant_bind_target_plan(clear=True)
        self.assertTrue(cleared['ok'], cleared)
        self.assertTrue(all(not m.hooks for m in model._modules.values()))

    def test_runtime_guard_blocks_changed_actual_argument_before_reference(self):
        worker, model, plan = self.contract_worker()
        with patch.dict('os.environ', {'VLLM_SERVER_DEV_MODE': '1'}):
            self.assertTrue(worker.fake_requant_bind_target_plan(plan)['ok'])
            guard = next(iter(model._modules.values())).hooks[0]
            def cpu_check(binding, args, kwargs):
                return check_input(binding, args, kwargs, lambda v: isinstance(v, Tensor))
            x = Activation()
            with patch('ascend_fi.inspect.input_contract.check_input', side_effect=cpu_check):
                self.assertIsNone(guard(None, (x,), {}))
                x.shape = (1, 7)
                with self.assertRaisesRegex(RuntimeError, 'INPUT_CONTRACT_FAILED'):
                    guard(None, (x,), {})
            self.assertEqual(worker.fake_requant_query()['input_contract']['violations'], 1)
            worker.fake_requant_bind_target_plan(clear=True)

    def test_plan_guard_requires_developer_mode(self):
        worker, _, plan = self.contract_worker()
        with patch.dict('os.environ', {'VLLM_SERVER_DEV_MODE': '0'}):
            self.assertEqual(worker.fake_requant_bind_target_plan(plan)['error'], 'DEVELOPER_MODE_REQUIRED')

    def test_resealed_unbounded_plan_is_still_rejected(self):
        p = plan_for()
        p['prospective_arm']['max_hits'] = 0
        p.pop('plan_sha256')
        p['plan_sha256'] = digest(p)
        with self.assertRaisesRegex(ValueError, 'WINDOW_BUDGET'): verify_plan(p)

    def test_generic_executor_complete_owner_lifecycle_mock_not_live(self):
        p = plan_for()
        cfg = types.SimpleNamespace(world_size=2, actual_module_names=p['selected_modules'],
                                    targeted_layers=p['prospective_arm']['layer_filter'])
        cfg.to_dict = lambda: {'actual_module_names': cfg.actual_module_names}
        owner = Mock()
        owner.arm = p['prospective_arm']
        owner.preflight.return_value = {'status': 'PASS', 'exclusive_lock_owned': True}
        owner.service_healthy.return_value = True
        owner.clean_query.return_value = {'http_status': 200, 'results': [
            {'pid': r + 100, 'enabled': False, 'installed': False} for r in (0, 1)]}
        owner.install.return_value = cfg
        owner.request.side_effect = lambda *args: {'http_status': 200, 'response_text': 'ok', 'token_pieces': ['ok']}
        def fault(*args, **kw):
            return {**measured_fault(p, identity=kw.get('identity', False)),
                    'response_text': 'ok' if kw.get('identity') else 'different',
                    'token_pieces': ['ok'] if kw.get('identity') else ['different']}
        owner.fault_request.side_effect = fault
        current = lambda: [{k: w[k] for k in ('service_epoch', 'runtime_fingerprint', 'model_fingerprint')}
                           for w in p['workers']]
        runner = TargetPlanExecutor(owner, p, current)
        identity = runner.validate_identity({'probe_id': 'official'})
        self.assertTrue(identity['identity_receipt']['token_pieces_exact'])
        pair = runner.run_pair({'probe_id': 'official'})
        self.assertEqual(pair['status'], 'PASS')
        self.assertTrue(pair['recovery_exact'])
        self.assertEqual(pair['window']['full_request_window'], 'UNPROVEN')
        self.assertEqual(owner.disarm_and_remove.call_count, 3)


if __name__ == '__main__':
    unittest.main()
