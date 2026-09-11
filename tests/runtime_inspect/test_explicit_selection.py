"""Offline CPU metadata / simulated RPC contracts, NOT Ascend multi-rank evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from ascend_fi.inspect.selection import resolve_selection, target_view, render_target_view, parse_ranks
from ascend_fi.inspect.plan_builder import verify_plan
from ascend_fi.inspect.plan_executor import plan_window_gate, TargetPlanExecutor
from ascend_fi.inspect.worker_confirmation import confirm_workers
from ascend_fi.inspect.model_descriptor import digest
from runtime_inspect.test_model_targeting import reports, plan_for, reseal_report, measured_fault
from runtime_inspect import test_model_targeting as model_fixtures

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
from run_normal_diagnostic_live import VerifiedOwner
from private_service_owner import PrivateServiceOwner
from inspect_model_targets import main as preview_main
from run_private_demo_once import run_once_host
from ascend_fi.inspect.single_effect import evaluate_single
from ascend_fi.evaluation.distortion.contracts import content_hash, FaultConfigurationRecord
from ascend_fi.evaluation.distortion.report import write_report
from evaluation.test_distortion import SUITE, config as base_config


def new_fault(plan, identity=False):
    f = measured_fault(plan, identity)
    q = f['fault_transaction']['measurement_query']
    for row, w in zip(q['results'], plan['workers']):
        row['input_contract']['worker_identity'] = {'service_epoch': w['service_epoch'],
            'global_rank': w['rank'], 'reference_rank': w['rank'], 'module_tp_ranks': [w['rank']]}
        row['developer_mode'] = True
        for m in row['matched_modules']:
            m['tp_size'] = len(plan['workers'])
    ack = {'http_status': 200, 'results': [{'ok': True, 'pid': r['pid'],
            'armed_config': r['armed_config']} for r in q['results']]}
    f['fault_transaction'].update(arm=ack, pre_request_query=copy.deepcopy(q), transaction_query=copy.deepcopy(q))
    f['fault_disarm'] = {'http_status': 200, 'results': [{'ok': True, 'pid': r['pid'], 'disarmed': True} for r in q['results']]}
    f['service_healthy'] = True
    return f


class ExplicitSelectionTests(unittest.TestCase):
    def test_manual_intersection_does_not_expand(self):
        p = plan_for(selection_mode='manual', roles=['MLP_DOWN'], layers=[1], rank_filter=[0])
        verify_plan(p)
        self.assertEqual(p['selected_modules'], ['model.layers.1.mlp.down_proj'])
        self.assertEqual(p['prospective_arm']['rank_filter'], [0])
        self.assertEqual(target_view(reports(), plan=p)['selected_worker_targets'], 1)

    def test_exact_name_not_regex(self):
        with self.assertRaisesRegex(ValueError, 'FILTER_CONFLICT'):
            plan_for(selection_mode='manual', roles=None, module_names=['.*'])
        p = plan_for(selection_mode='manual', roles=None, module_names=['model.layers.2.mlp.down_proj'])
        self.assertEqual(p['selected_modules'], ['model.layers.2.mlp.down_proj'])

    def test_explicit_rank_all_expands_complete_runtime_group(self):
        p = plan_for(selection_mode='all_bindable', roles=None, rank_filter='all')
        self.assertEqual(p['prospective_arm']['rank_filter'], [0, 1])
        self.assertEqual(target_view(reports(), plan=p)['selected_worker_targets'], 24)
        cap = p['request_window']['max_hits']
        self.assertEqual(p['request_window']['per_worker_max_hits'], {'0': cap, '1': cap})
        self.assertEqual(p['request_window']['theoretical_sum_of_targeted_worker_caps'], 2 * cap)

    def test_rank_all_is_not_hardcoded_two(self):
        rows = reports()[:1]
        rows[0]['descriptor']['topology']['tensor_parallel_size'] = 1
        reseal_report(rows[0])
        p = plan_for(rows, selection_mode='all_bindable', rank_filter='all')
        self.assertEqual(p['prospective_arm']['rank_filter'], [0])

    def test_partial_worker_or_rank_mapping_explicitly_rejected(self):
        with self.assertRaisesRegex(ValueError, 'PARTICIPATING_REQUEST'):
            plan_for(reports()[:1], selection_mode='all_bindable', rank_filter='all')
        for kind in ('global', 'tp', 'duplicate'):
            rows = reports()
            if kind == 'global': rows[0]['descriptor']['ranks']['global']['value'] = 3
            if kind == 'tp': rows[0]['descriptor']['ranks']['tp'] = {'value': 1}
            if kind == 'duplicate': rows[1] = copy.deepcopy(rows[0])
            for r in rows: reseal_report(r)
            with self.assertRaises(ValueError): plan_for(rows, selection_mode='all_bindable', rank_filter='all')

    def test_missing_explicit_scope_never_silently_shrunk(self):
        for kwargs in ({'layers': [0, 999]}, {'roles': ['MLP_DOWN', 'missing']}):
            with self.assertRaisesRegex(ValueError, 'REQUESTED_.*_MISSING'):
                plan_for(selection_mode='manual', **kwargs)

    def test_blocked_all_excluded_with_reason_manual_rejects(self):
        rows = reports()
        for r in rows:
            r['targets'][0].update(bound=False, blockers=['NOT_OBSERVED'])
            reseal_report(r)
        with self.assertRaisesRegex(ValueError, 'MANUALLY_SELECTED'): plan_for(rows, selection_mode='manual')
        p = plan_for(rows, selection_mode='all_bindable', rank_filter='all')
        self.assertEqual(len(p['selection']['excluded']), 2)
        self.assertEqual(len(p['selected_modules']), 11)

    def test_one_rank_binding_gap_does_not_downgrade(self):
        rows = reports()
        rows[1]['targets'][0].update(bound=False, blockers=['NOT_OBSERVED'])
        reseal_report(rows[1])
        with self.assertRaisesRegex(ValueError, 'WORKER_BINDING_GAP'):
            plan_for(rows, selection_mode='all_bindable', rank_filter='all')

    def test_duplicate_rows_and_multi_role_install_once_not_cross_rank(self):
        rows = reports()
        for r in rows:
            r['targets'].append(copy.deepcopy(r['targets'][0]))
            reseal_report(r)
        p = plan_for(rows, selection_mode='all_bindable', rank_filter='all')
        self.assertEqual(len(p['selection']['deduplicated']), 2)
        self.assertEqual(len(p['selected_modules']), 12)
        self.assertEqual(sum(len(w['targets']) for w in p['workers']), 24)

    def test_parent_child_still_rejected(self):
        rows = reports()
        for r in rows:
            t = copy.deepcopy(r['targets'][0]); t['module_name'] += '.child'
            r['targets'].append(t); reseal_report(r)
        with self.assertRaisesRegex(ValueError, 'PARENT_CHILD'):
            plan_for(rows, selection_mode='all_bindable', rank_filter='all')

    def test_resealed_selection_budget_mismatch_rejected(self):
        p = plan_for(selection_mode='all_bindable', rank_filter='all')
        p['selection']['targeted_global_ranks'] = [0]
        p.pop('plan_sha256'); p['plan_sha256'] = digest(p)
        with self.assertRaisesRegex(ValueError, 'CONTRACT_MISMATCH'): verify_plan(p)

    def test_preview_contains_evidence_no_arm(self):
        v = target_view(reports(), plan=plan_for(selection_mode='manual', roles=['MLP_DOWN']))
        self.assertFalse(v['arm'])
        self.assertTrue(all(r['observations'] and r['role_proof'] for r in v['rows']))
        self.assertIn('不做敏感度排名', render_target_view(v))

    def test_offline_cli_never_constructs_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'reports.json').write_text(json.dumps(reports()))
            argv = ['inspect_model_targets.py', '--reports', str(root / 'reports.json'), '--out', str(root / 'view')]
            with patch.object(sys, 'argv', argv), patch('inspect_model_targets.InspectClient') as client:
                preview_main(); client.assert_not_called()
            self.assertTrue((root / 'view/target_preview.md').exists())

    def test_rank_parser_does_not_mix_all_and_explicit(self):
        self.assertEqual(parse_ranks(['all']), 'all')
        self.assertEqual(parse_ranks(['0', '1']), [0, 1])
        for values in (['all', '0'], ['0', '0'], ['-1']):
            with self.assertRaises(ValueError): parse_ranks(values)


class WorkerConfirmationTests(unittest.TestCase):
    def test_identity_then_allrank_lossy_uses_existing_executor_mock(self):
        p = plan_for(selection_mode='all_bindable', rank_filter='all')
        owner = Mock(); owner.arm = p['prospective_arm']
        cfg = types.SimpleNamespace(world_size=2, actual_module_names=p['selected_modules'],
                                    targeted_layers=p['prospective_arm']['layer_filter'])
        cfg.to_dict = lambda: {'actual_module_names': cfg.actual_module_names}
        owner.install.return_value = cfg
        owner.preflight.return_value = {'status': 'PASS', 'exclusive_lock_owned': True}
        owner.service_healthy.return_value = True
        owner.clean_query.return_value = {'http_status': 200, 'results': [
            {'pid': 100+r, 'installed': False, 'enabled': False} for r in (0, 1)]}
        owner.request.side_effect = lambda *a: {'http_status': 200, 'response_text': 'ok', 'token_pieces': ['ok']}
        def fault(*a, **kw):
            return {**new_fault(p, identity=kw.get('identity', False)),
                    'response_text': 'ok' if kw.get('identity') else 'changed',
                    'token_pieces': ['ok'] if kw.get('identity') else ['changed']}
        owner.fault_request.side_effect = fault
        descriptors = [{k: w[k] for k in ('service_epoch', 'runtime_fingerprint', 'model_fingerprint')} for w in p['workers']]
        runner = TargetPlanExecutor(owner, p, lambda: descriptors)
        self.assertTrue(runner.validate_identity({'probe_id': 'p'})['identity_receipt'])
        result = runner.run_once({'probe_id': 'p'})
        self.assertEqual(result['window']['mechanism'], 'PASS')
        self.assertEqual(owner.fault_request.call_count, 2)  # one IDENTITY, one lossy, not two lossy

    def test_all_rank_arm_and_measurement_pass_mock_only(self):
        p = plan_for(selection_mode='all_bindable', rank_filter='all')
        f = new_fault(p)
        self.assertEqual(plan_window_gate(p, f)['mechanism'], 'PASS')
        self.assertEqual(plan_window_gate(p, f)['full_request_window'], 'UNPROVEN')

    def test_rank0_preserves_rank1_zero_hit_policy(self):
        p = plan_for(selection_mode='manual', roles=['MLP_DOWN'])
        f = new_fault(p)
        self.assertEqual(plan_window_gate(p, f)['mechanism'], 'PASS')
        self.assertEqual(f['fault_transaction']['measurement_query']['results'][1]['hit_count'], 0)

    def test_missing_arm_ack_duplicate_pid_wrong_generation(self):
        p = plan_for(selection_mode='all_bindable', rank_filter='all')
        for kind in ('missing', 'duplicate', 'generation', 'epoch', 'measurement'):
            f = new_fault(p); tx = f['fault_transaction']
            if kind == 'missing': tx['arm']['results'].pop()
            elif kind == 'duplicate': tx['arm']['results'][1]['pid'] = tx['arm']['results'][0]['pid']
            elif kind == 'generation': tx['pre_request_query']['results'][1]['armed_config']['request_generation'] = 99
            elif kind == 'epoch': tx['pre_request_query']['results'][1]['input_contract']['worker_identity']['service_epoch'] = 'old'
            else: tx['measurement_query']['results'][1]['pid'] = 999
            self.assertEqual(plan_window_gate(p, f)['mechanism'], 'INVALID_FAULT_RUN', kind)

    def test_owner_partial_arm_never_sends_request_and_disarms(self):
        p = plan_for(selection_mode='all_bindable', rank_filter='all')
        f = new_fault(p, identity=True)
        owner = VerifiedOwner.__new__(VerifiedOwner)
        owner.arm, owner.target_plan, owner.generation, owner.attempt = p['prospective_arm'], p, 6, 'fixture'
        owner.guard, owner.service_healthy, owner.persist, owner.request = Mock(), Mock(return_value=True), Mock(), Mock()
        arm = copy.deepcopy(f['fault_transaction']['arm']); arm['results'].pop()
        owner.rpc = Mock(side_effect=lambda method, *args: arm if method == 'fake_requant_arm' else
            f['fault_disarm'] if method == 'fake_requant_disarm' else f['fault_transaction']['pre_request_query'])
        row = owner.fault_request({'probe_id': 'p'}, None, 1, identity=True)
        owner.request.assert_not_called()
        self.assertIn('ACK_MISSING', row['error'])
        self.assertEqual(owner.rpc.call_args[0][0], 'fake_requant_disarm')

    def test_runtime_actual_reference_rank_mismatch_blocks_install(self):
        worker, model, old = model_fixtures.ModelTargetingTests().contract_worker()
        p = plan_for(selection_mode='all_bindable', rank_filter='all')
        p['workers'][0]['model_fingerprint'] = old['workers'][0]['model_fingerprint']
        p.pop('plan_sha256'); p['plan_sha256'] = digest(p)
        for m in model._modules.values(): m.tp_rank = 0
        reference = types.ModuleType('fake_requant_reference_extension')
        reference._rank_for = lambda m: (1, 'fixture_wrong_mapping')
        with patch.dict(sys.modules, {'fake_requant_reference_extension': reference}), patch.dict('os.environ', {'VLLM_SERVER_DEV_MODE': '1'}):
            result = worker.fake_requant_bind_target_plan(p)
        self.assertFalse(result['ok'])
        self.assertIn('MAPPING_UNSUPPORTED', result['error'])
        self.assertTrue(all(not m.hooks for m in model._modules.values()))


class OnceLifecycleTests(unittest.TestCase):
    def test_single_report_existing_contract_no_fabricated_recovery(self):
        p = plan_for(selection_mode='all_bindable', roles=['MLP_DOWN'], layers=[0, 1], rank_filter='all')
        cfg = base_config().to_dict()
        cfg.update(targeted_ranks=[0, 1], frozen_arm_config=p['prospective_arm'],
                   hit_policy={'max_hits': p['request_window']['max_hits'], 'budget_semantics': 'AT_MOST_NO_PADDING'})
        cfg = FaultConfigurationRecord.from_dict(cfg)
        suite = copy.deepcopy(SUITE)
        probe = suite['probes'][0]
        request = {'messages': [{'role': 'user', 'content': probe['prompt']}], **probe['generation_config'],
                   **{k: suite['request_protocol'][k] for k in ('chat_template_kwargs', 'logprobs', 'top_logprobs')}}
        baseline = {'sample_id': probe['probe_id'], 'http_status': 200, 'request': request,
                    'response_text': '<script>not code</script>', 'token_pieces': ['normal'], 'finish_reason': 'stop'}
        f = {**new_fault(p), **baseline, 'response_text': 'different', 'token_pieces': ['different']}
        run = {'pair': {'baseline': baseline, 'fault': f, 'recovery': None},
               'window': plan_window_gate(p, f), 'cleanup': {}, 'execution_policy': 'SINGLE_LOSSY_THEN_STOP_INSTANCE'}
        result = evaluate_single(suite, cfg, p, run, probe_id=probe['probe_id'])
        data = result.to_dict()
        self.assertTrue(data['single_request_effect']['eligible'], data['mechanism'])
        self.assertIsNone(data['recovery'])
        self.assertIsNone(data['reliability']['recovery_rate'])
        self.assertEqual(data['single_request_effect']['effect']['comparison_basis'], 'token_pieces')
        with tempfile.TemporaryDirectory() as tmp:
            write_report(tmp, result, {'fixture': True})
            page = (Path(tmp) / 'effect_dashboard.html').read_text(encoding='utf-8')
            self.assertIn('&lt;script&gt;not code&lt;/script&gt;', page)
            self.assertNotIn('<script>not code</script>', page)
            self.assertIn('NOT_MEASURED', page)
            with self.assertRaises(FileExistsError): write_report(tmp, result, {})

    def setup_runner(self):
        p = plan_for(selection_mode='all_bindable', rank_filter='all')
        owner = Mock(); owner.arm = p['prospective_arm']
        owner.request.return_value = {'http_status': 200, 'response_text': 'ok', 'token_pieces': ['ok']}
        owner.install.return_value = types.SimpleNamespace(actual_module_names=p['selected_modules'])
        owner.fault_request.return_value = new_fault(p)
        runner = TargetPlanExecutor(owner, p, lambda: [])
        runner.identity_receipt = {'plan_sha256': p['plan_sha256']}
        return runner, owner

    def test_one_lossy_no_recovery_and_no_second_instance_use(self):
        runner, owner = self.setup_runner()
        with patch.object(runner, 'rebind'):
            result = runner.run_once({'probe_id': 'p'})
            with self.assertRaisesRegex(ValueError, 'CONSUMED'): runner.run_once({})
            with self.assertRaisesRegex(ValueError, 'CONSUMED'): runner.validate_identity({})
            with self.assertRaisesRegex(ValueError, 'CONSUMED'): runner.run_pair({})
        owner.fault_request.assert_called_once()
        self.assertEqual(owner.request.call_count, 1)  # baseline only
        owner.disarm_and_remove.assert_called_once()
        self.assertIsNone(result['pair']['recovery'])
        self.assertFalse(result['stop_confirmed'])

    def test_lossy_exception_cleanup_failure_records_no_retry(self):
        runner, owner = self.setup_runner()
        owner.fault_request.side_effect = RuntimeError('failed request')
        owner.disarm_and_remove.side_effect = RuntimeError('cleanup unconfirmed')
        with patch.object(runner, 'rebind'): result = runner.run_once({'probe_id': 'p'})
        self.assertEqual(result['status'], 'PARTIAL')
        self.assertEqual(result['cleanup']['attempt'], 'ERROR')
        self.assertIn('failed request', result['error'])
        self.assertEqual(owner.request.call_count, 1)

    def test_new_driver_rejects_reserved_session_before_any_network(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'single_lossy_instance_reserved.json').write_text('{"reservation_id":"old"}')
            owner = PrivateServiceOwner.__new__(PrivateServiceOwner); owner.session = root
            with self.assertRaisesRegex(RuntimeError, 'RESERVED_OR_CONSUMED'): owner.guard()

    def test_host_finally_stops_own_instance_after_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            d = {'lease_id': 'ours', 'owner': 'test', 'lock_owner_path': str(root / 'owner.json'),
                 'container_name': 'ours-container', 'source_root': str(root)}
            (root / 'deployment.json').write_text(json.dumps(d))
            (root / 'owner.json').write_text(json.dumps({'lease_id': 'ours', 'owner': 'test', 'result_dir': str(root)}))
            for name in ('suite.json', 'plan.json'): (root / name).write_text('{}')
            info = [{'Config': {'Labels': {'ascend-fi-lease': 'ours'}}, 'State': {'Running': True}}]
            with patch('run_private_demo_once.command', return_value=json.dumps(info)), \
                 patch('run_private_demo_once.subprocess.run', side_effect=TimeoutError('fixture timeout')), \
                 patch('run_private_demo_once.stop') as stopped:
                with self.assertRaises(TimeoutError):
                    run_once_host(root, root / 'suite.json', root / 'plan.json', 'p', 'p', root / 'out')
                stopped.assert_called_once_with(root)
            final = json.loads((root / 'out/final_instance_state.json').read_text())
            self.assertTrue(final['stop_confirmed'])
            self.assertEqual(final['hot_recovery'], 'NOT_MEASURED')
            self.assertTrue((root / 'out/artifact_index.json').exists())

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch absent: mock-module tests are not torch/Ascend live evidence')
    def test_actual_cpu_module_hook_handle_removal(self):
        import torch
        module = torch.nn.Linear(4, 2)
        seen = []
        h = module.register_forward_pre_hook(lambda m, a: seen.append(a[0].dtype))
        module(torch.ones(1, 4)); h.remove(); module(torch.ones(1, 4))
        self.assertEqual(seen, [torch.float32])


if __name__ == '__main__':
    unittest.main()
