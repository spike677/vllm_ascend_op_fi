"""CPU protocol tests through real Runtime and targeted collective_rpc.

Fake providers/CPU workers are deliberately not a TP2 deployment qualification.
"""
from copy import deepcopy
import unittest
from unittest.mock import patch
from demoncat.runtime.vllm_ascend.worker_extension import identity, runtime
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.multi_rank import MultiRankTaskClient
from fault_injection.request_plan import request_plan
from test_extension import _extended_worker, ReadbackProvider
from test_core_host import ExistingExecutor


class ScheduledProvider(ReadbackProvider):
    def __init__(self, rank, contract):
        super().__init__()
        self.rank, self.contract = rank, deepcopy(contract)
        self.index = 0
        self.ids = []
        self.pending = None
        self.state.update(request_plan=request_plan(contract['mechanism']), next_phase='normal', complete=False)

    def dispatch(self, action, settings, *, owner, generation):
        reply = super().dispatch(action, settings, owner=owner, generation=generation)
        if action == 'advance':
            if self.pending is not None or settings['request_id'] in self.ids:
                raise RuntimeError('DUPLICATE_OR_INFLIGHT_REQUEST')
            self.pending = settings['request_id']
        if action == 'recover':
            self.pending = None
            self.index = 2
            self.state['next_phase'] = 'off'
        return reply

    def complete_actual(self):
        assert self.pending is not None
        self.ids.append(self.pending)
        self.pending = None
        self.index += 1
        self.state.update(next_phase=['normal','fault','off','health','complete'][self.index],
                          complete=self.index == 4)
        self.state['multi_rank'] = dict(owner=self.contract['owner'], group_generation=1,
            actual_normal_request_observed=True, eligible_target=True,
            normal_request_id=self.ids[0],members_sha256=self.contract['members_sha256'],
            worker=self.contract['members'][self.rank],request_fingerprint='a'*64,
            selected_rank=0, role='injector' if self.rank == 0 else 'guard')


class MultiRankTests(unittest.TestCase):
    def setUp(self):
        self.workers = [_extended_worker(rank=i) for i in range(2)]
        for worker in self.workers: worker.load_model()
        self.executor = ExistingExecutor(self.workers)
        self.contracts, self.providers = {}, {}
        self.enterContext(patch.object(identity, '_current_device', return_value=0))
        self.enterContext(patch.object(runtime, '_resolve_provider', side_effect=self.resolve))
        self.client = self.make_client()

    def resolve(self, mechanism, version, worker, model, actual):
        provider = ScheduledProvider(worker.rank, self.contracts[worker.rank])
        self.providers[worker.rank] = provider
        return provider

    def make_client(self, config=None):
        def bind(actual, config, contract):
            self.contracts[actual['rank']] = contract
            return {}
        return MultiRankTaskClient(self.executor, config or FaultConfig(18,'api_half'), prepare_binding=bind)

    def ok(self, action):
        row = getattr(self.client, action)()
        self.assertEqual(row['status'], 'QUERY_OK' if action == 'query' else 'ACTION_COMPLETED', row)
        return row

    def start(self):
        for action in ('prepare','register','check','enable'): self.ok(action)

    def request(self, name):
        row = self.client.advance(name)
        self.assertEqual(row['status'], 'ACTION_COMPLETED', row)
        # Only this test scheduler completes the request, never control RPC.
        for provider in self.providers.values(): provider.complete_actual()
        return self.ok('query')

    def cleanup(self):
        self.ok('unregister')
        self.assertEqual(self.client.detach()['status'], 'DETACHED')
        self.assertFalse(self.client.snapshot()['ownership_retained'])

    def test_two_scopes_same_workers_common_owner_local_generations_and_request_ids(self):
        previous = None
        for scope in range(2):
            if scope: self.client = self.make_client()
            self.start()
            snap = self.client.snapshot()
            owner = snap['group']['owner']
            self.assertNotEqual(owner, previous)
            self.assertEqual(set(snap['local_generations']), {0,1})
            for client in self.client.clients.values():
                self.assertEqual(client.snapshot()['ticket']['owner'], owner)
            for phase in ('normal','fault','off','health'):
                row = self.request(f's{scope}-{phase}')
            self.assertTrue(row['response']['provider_state']['complete'])
            self.assertEqual(self.providers[0].ids, self.providers[1].ids)
            self.assertEqual(self.client.advance('extra')['status'], 'GROUP_PHASES_EXHAUSTED')
            self.ok('finish'); self.cleanup(); previous = owner
        self.assertEqual([w.events for w in self.workers], [['load'],['load']])

    def test_fault_requires_all_rank_normal_roles(self):
        self.start(); self.request('normal')
        del self.providers[1].state['multi_rank']
        before = [len(p.actions) for p in self.providers.values()]
        self.assertEqual(self.client.advance('fault')['status'], 'GROUP_ACTION_FAILED')
        self.assertEqual(before, [len(p.actions) for p in self.providers.values()])
        self.assertIn('ROLE_ADAPTER', self.client.failed)

    def test_stale_group_or_different_actual_input_rejected(self):
        self.start(); self.request('normal')
        self.providers[1].state['multi_rank']['request_fingerprint']='b'*64
        self.assertEqual(self.client.advance('fault')['status'],'GROUP_ACTION_FAILED')
        self.assertIn('INPUTS_NOT_ALIGNED',self.client.failed)

    def test_old_normal_request_id_rejected(self):
        self.start(); self.request('normal')
        self.providers[1].state['multi_rank']['normal_request_id']='previous-scope'
        self.assertEqual(self.client.advance('fault')['status'],'GROUP_ACTION_FAILED')

    def test_role_commit_is_selected_from_actual_candidates_and_read_back(self):
        decisions=[]
        def commit(decision):
            decisions.append(decision)
            for rank,provider in self.providers.items():
                provider.state['multi_rank'].update(selected_rank=decision['selected_rank'],role=decision['roles'][rank])
        self.client.commit_roles=commit
        self.start();self.request('normal')
        for provider in self.providers.values():
            provider.state['multi_rank'].pop('role')
            provider.state['multi_rank'].pop('selected_rank')
        self.request('fault')
        self.assertEqual(len(decisions),1)
        self.assertEqual(decisions[0]['selected_rank'],0)
        self.assertEqual(decisions[0]['roles'],{0:'injector',1:'guard'})

    def test_role_commit_lost_reply_never_arms_fault_or_retries(self):
        calls=[]
        def commit(decision):
            calls.append(decision)
            raise TimeoutError('commit acknowledgement lost')
        self.client.commit_roles=commit
        self.start();self.request('normal')
        self.assertEqual(self.client.advance('fault')['status'],'GROUP_ACTION_FAILED')
        self.client.advance('retry')
        self.assertEqual(len(calls),1)
        self.assertEqual(self.providers[0].ids,['normal'])
        self.assertEqual(self.providers[1].ids,['normal'])

    def test_sharded_projection_selects_qualified_actual_rank_not_always_zero(self):
        self.client=self.make_client(FaultConfig(39,'output_proxy'))
        def commit(decision):
            for rank,provider in self.providers.items():
                provider.state['multi_rank'].update(selected_rank=decision['selected_rank'],role=decision['roles'][rank])
        self.client.commit_roles=commit
        self.start();self.request('normal')
        self.providers[0].state['multi_rank']['eligible_target']=False
        self.request('fault')
        self.assertEqual(self.client.snapshot()['role_decision']['selected_rank'],1)

    def test_rms_replica_does_not_fall_back_to_rank_one(self):
        self.start();self.request('normal')
        self.providers[0].state['multi_rank']['eligible_target']=False
        self.assertEqual(self.client.advance('fault')['status'],'GROUP_ACTION_FAILED')
        self.assertIn('NO_ACTUAL_AUTOMATIC_TARGET',self.client.failed)

    def test_two_injectors_rejected(self):
        self.start(); self.request('normal')
        self.providers[1].state['multi_rank']['role'] = 'injector'
        self.assertEqual(self.client.advance('fault')['status'], 'GROUP_ACTION_FAILED')
        self.assertIn('ROLE_AGREEMENT', self.client.failed)

    def test_phase_disagreement_blocks_global_advance(self):
        self.start()
        self.providers[1].state['next_phase'] = 'health'
        self.assertEqual(self.client.advance('never')['status'], 'GROUP_ACTION_FAILED')
        self.assertTrue(all(not p.ids and p.pending is None for p in self.providers.values()))

    def test_missing_provider_state_is_failure(self):
        self.start()
        def omit(method, args, rows):
            if method == 'demoncat_fault_query':
                for row in rows:
                    if row['status'] == 'QUERY_OK': row.pop('provider_state', None)
            return rows
        self.executor.after = omit
        self.assertEqual(self.client.query()['status'], 'GROUP_ACTION_FAILED')

    def test_actual_request_id_same_on_all_ranks_and_no_control_forward(self):
        self.start()
        self.assertEqual(self.client.advance('scheduler-internal-7')['status'], 'ACTION_COMPLETED')
        self.assertEqual([p.pending for p in self.providers.values()], ['scheduler-internal-7']*2)
        self.assertTrue(all(not p.ids for p in self.providers.values()))
        self.assertEqual(self.client.advance('another')['status'], 'GROUP_ACTION_FAILED')

    def test_lost_enable_response_is_not_retried_and_recovery_visits_both(self):
        for action in ('prepare','register','check'): self.ok(action)
        def lose(method, args, rows):
            if method == 'demoncat_fault_rpc' and args[0] == 'enable':
                self.executor.after = None
                raise TimeoutError('lost after mutation')
            return rows
        self.executor.after = lose
        self.assertEqual(self.client.enable()['status'], 'GROUP_ACTION_FAILED')
        self.assertEqual(self.client.enable()['status'], 'GROUP_RECOVERY_REQUIRED')
        self.assertEqual([a[0] for a in self.providers[0].actions].count('enable'),1)
        self.ok('recover')
        self.assertTrue(all(any(a[0]=='recover' for a in p.actions) for p in self.providers.values()))
        self.cleanup()

    def test_recovery_attempts_other_rank_after_first_failure(self):
        self.start()
        def fail(): raise RuntimeError('rank0 restore failure')
        self.providers[0].callbacks['recover'] = fail
        self.assertEqual(self.client.recover()['status'], 'GROUP_ACTION_FAILED')
        self.assertTrue(any(a[0]=='recover' for a in self.providers[1].actions))
        self.assertTrue(self.client.snapshot()['ownership_retained'])

    def test_failed_scope_allows_only_explicit_off_health_recovery(self):
        self.start(); self.request('normal')
        self.providers[1].state['multi_rank']['role']='injector'
        self.client.advance('rejected-fault')
        self.ok('recover')
        self.request('recovery-off'); final=self.request('recovery-health')
        self.assertFalse(final['response']['provider_state']['complete'])
        self.assertIsNotNone(final['response']['provider_state']['failure'])
        self.cleanup()

    def test_identity_drift_fails_before_new_mutation(self):
        self.start(); self.workers[1].model_runner.model=object()
        self.assertEqual(self.client.check()['status'], 'GROUP_ACTION_FAILED')
        self.assertIn(self.client.failed, ('WORKER_CONTEXT_NOT_READY','GROUP_WORKER_OR_MODEL_CHANGED'))
        self.assertEqual([a[0] for a in self.providers[0].actions].count('check'),1)

    def test_missing_rank_prevents_any_prepare(self):
        self.executor.workers.pop()
        self.assertEqual(self.client.prepare()['status'], 'GROUP_ACTION_FAILED')
        self.assertEqual(self.providers,{})

    def test_no_public_rank_and_no_batch_or_tp1_implicit_admission(self):
        with self.assertRaises(TypeError): FaultConfig.from_dict(dict(fault_id=18,variant='api_half',rank=0))
        with self.assertRaises(ValueError):
            MultiRankTaskClient(self.executor,FaultConfig(41,'tail_column'),prepare_binding=lambda *a:{})
        self.assertFalse(self.client.contract()['qualification_granted'])


class Tp2PrerequisiteTests(unittest.TestCase):
    def test_actual_group_required_and_tp1_gate_unchanged(self):
        from types import SimpleNamespace as NS
        from test_serving_context import fixture
        from demoncat.runtime.vllm_ascend.worker_extension.multi_rank_prerequisites import tp2_eager_assessment
        from demoncat.runtime.vllm_ascend.worker_extension.prerequisites import serial_eager_assessment
        runner=fixture()[0]
        runner.vllm_config.parallel_config.tensor_parallel_size=2
        report=tp2_eager_assessment(runner,NS(world_size=2,ranks=[0,1],rank_in_group=1),1)
        self.assertEqual(report['status'],'matched')
        self.assertFalse(report['execution_authorized'])
        self.assertEqual(serial_eager_assessment(runner)['status'],'refused')
        self.assertEqual(tp2_eager_assessment(runner,NS(),1)['status'],'pending')

    def test_context_parallel_multimodal_and_missing_fields_still_refused(self):
        from types import SimpleNamespace as NS
        from test_serving_context import fixture
        from demoncat.runtime.vllm_ascend.worker_extension.multi_rank_prerequisites import tp2_eager_assessment
        runner=fixture()[0];runner.vllm_config.parallel_config.tensor_parallel_size=2
        del runner.pcp_size
        runner.is_multimodal_model=True
        report=tp2_eager_assessment(runner,NS(world_size=2,ranks=[0,2],rank_in_group=0),0)
        self.assertEqual({x['field'] for x in report['unmet']},
                         {'actual_tp_group.ranks','pcp_size','is_multimodal_model'})
        self.assertFalse(report['execution_authorized'])


class Tp2MappingTests(unittest.TestCase):
    def test_exact_tp2_mapping_reuses_checks_without_loosening_tp1(self):
        from types import SimpleNamespace as NS
        from test_serving_context import fixture
        from demoncat.runtime.vllm_ascend.worker_extension.multi_rank_prerequisites import snapshot_tp2_fresh_prefill
        from demoncat.runtime.vllm_ascend.worker_extension.serving_context import snapshot_fresh_prefill,ContextRefused
        runner,model,kwargs,context,prompt,request=fixture()
        runner.vllm_config.parallel_config.tensor_parallel_size=2
        group=NS(world_size=2,ranks=[0,1],rank_in_group=1)
        params=dict(expected_prompt_token_ids=prompt,expected_internal_request_id=request)
        row=snapshot_tp2_fresh_prefill(runner,model,kwargs,context,actual_tp_group=group,rank=1,**params)
        self.assertEqual(row['internal_request_id'],request)
        self.assertEqual(row['tensor_parallel']['rank_in_group'],1)
        self.assertFalse(row['qualification_granted'])
        with self.assertRaises(ContextRefused):snapshot_fresh_prefill(runner,model,kwargs,context,**params)
        with self.assertRaises(ContextRefused):snapshot_tp2_fresh_prefill(runner,model,kwargs,context,
            actual_tp_group=group,rank=1,seen_internal_request_ids=[request],**params)
        runner.input_batch.req_ids=['foreign']
        with self.assertRaises(ContextRefused):snapshot_tp2_fresh_prefill(runner,model,kwargs,context,
            actual_tp_group=group,rank=1,**params)

    def test_real_ascend_subclass_is_included_by_probe(self):
        from types import SimpleNamespace as NS
        from test_serving_context import LogitsProcessor,AscendLogitsProcessor
        from experiments.worker_core_v11_20261118.m1_tp2_executor_probe import find_logits_processor
        actual=AscendLogitsProcessor()
        self.assertIs(find_logits_processor(NS(modules=lambda:[object(),actual]),LogitsProcessor),actual)
        with self.assertRaises(RuntimeError):find_logits_processor(NS(modules=lambda:[]),LogitsProcessor)
        with self.assertRaises(RuntimeError):find_logits_processor(NS(modules=lambda:[actual,LogitsProcessor()]),LogitsProcessor)


class BatchScheduledProvider(ScheduledProvider):
    def dispatch(self,action,settings,*,owner,generation):
        if action!='advance':return super().dispatch(action,settings,owner=owner,generation=generation)
        result=ReadbackProvider.dispatch(self,action,settings,owner=owner,generation=generation)
        ids=settings['request_ids']
        if self.pending is not None or any(x in sum(self.ids,[]) for x in ids):raise RuntimeError('DUPLICATE_BATCH')
        self.pending=list(ids);return result
    def complete_actual(self):
        assert self.pending is not None
        self.ids.append(self.pending);self.pending=None;self.index+=1
        phases=[p['phase'] for p in request_plan(self.contract['mechanism'])['phases']]+['complete']
        self.state.update(next_phase=phases[self.index],complete=self.index==6)
        self.state['multi_rank']=dict(owner=self.contract['owner'],group_generation=1,
            actual_normal_request_observed=True,eligible_target=True,normal_request_id=self.ids[0],
            members_sha256=self.contract['members_sha256'],worker=self.contract['members'][self.rank],
            request_fingerprint='a'*64,selected_rank=0,role='injector' if self.rank==0 else 'guard')

class BatchGroupTests(unittest.TestCase):
    def test_six_real_batches_hold_original_normal_ids_and_one_plan(self):
        f=MultiRankTests();f.setUp();self.addCleanup(f.doCleanups)
        def resolve(mechanism,version,worker,model,actual):
            p=BatchScheduledProvider(worker.rank,f.contracts[worker.rank]);f.providers[worker.rank]=p;return p
        self.enterContext(patch.object(runtime,'_resolve_provider',side_effect=resolve))
        f.client=f.make_client(FaultConfig('quant','batch_shared_scale'));f.start()
        with self.assertRaisesRegex(ValueError,'ADVANCE_BATCH'):f.client.advance('single')
        with self.assertRaisesRegex(ValueError,'EIGHT'):f.client.advance_batch(['duplicate']*8)
        for phase in ['normal','reference','lossless','calibration_off','fault','off']:
            ids=[phase+'-'+str(i) for i in range(8)]
            r=f.client.advance_batch(ids);self.assertEqual(r['status'],'ACTION_COMPLETED',r)
            for p in f.providers.values():p.complete_actual()
            r=f.client.query();self.assertEqual(r['status'],'QUERY_OK',r)
        self.assertTrue(r['response']['provider_state']['complete'])
        self.assertEqual(f.client.snapshot()['role_decision']['normal_request_id'],['normal-'+str(i) for i in range(8)])
        self.assertEqual(f.providers[0].ids,f.providers[1].ids)
        f.ok('finish');f.cleanup()

if __name__=='__main__': unittest.main()
