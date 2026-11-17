"""17 mechanisms share config, result and explicit lifecycle-refusal contracts."""
import copy
import json
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import torch
import demoncat
from fault_registry import definitions
from fault_registry.generate import generate
from fault_injection.unified.protocol import AdapterProtocol
from fault_injection.unified.trigger import CallIdentity,TriggerEngine


class RegistryConformance(unittest.TestCase):
    def test_exact_registry_and_generated_views(self):
        from fault_injection.catalog import CAPABILITIES
        from fault_injection.capabilities import list_mechanisms
        from engineering_closure.operator_adapters import OPERATOR_PATHS
        names={r['legacy_id'] for r in definitions()}
        self.assertEqual(len(names),17)
        self.assertEqual(names,{c.name for c in CAPABILITIES})
        self.assertEqual(names,{c['name'] for c in list_mechanisms()})
        self.assertEqual(names,set(OPERATOR_PATHS))
        self.assertEqual(generate(check=True)['mechanisms'],17)

    def test_ambiguous_legacy_trigger_never_silently_maps(self):
        with self.assertRaises(demoncat.ConfigError):demoncat.migrate_legacy({'mechanism':'18.api_half','occurrence':2})
        value=demoncat.migrate_legacy({'mechanism':'18.api_half'})
        self.assertEqual(value['trigger']['mode'],'every_match')
        self.assertIsNone(value['trigger']['max_hits'])


def register_case(row):
    def registration(self):
        adapter=demoncat.registry.resolve(row['canonical_id'])
        self.assertIsInstance(adapter,AdapterProtocol)
        self.assertEqual(adapter.contract_version,'1')
        self.assertTrue(adapter.evidence_schema)
        self.assertEqual(adapter.definition['legacy_id'],row['legacy_id'])
    def configuration(self):
        good=demoncat.example_config(row['legacy_id'])
        with self.assertRaises(demoncat.ConfigError):demoncat.registry.resolve(row['legacy_id'],{})
        self.assertEqual(demoncat.validate_config(good)['mechanism']['id'],row['legacy_id'])
        for mutate in (lambda x:x['mechanism'].update(version='2'),lambda x:x['fault']['parameters'].update(unknown=1),
                       lambda x:x['trigger'].update(ordinal=0),lambda x:x['target'].update(rank=[True]),
                       lambda x:x['fault'].pop('parameters'),lambda x:x.update(schema='future')):
            bad=copy.deepcopy(good);mutate(bad)
            with self.assertRaises(demoncat.ConfigError):demoncat.validate_config(bad)
        bad=copy.deepcopy(good);del bad['fault']['parameters'][next(iter(bad['fault']['parameters']))]
        with self.assertRaises(demoncat.ConfigError):demoncat.validate_config(bad)
    def lifecycle(self):
        session=demoncat.configure(row['legacy_id'],demoncat.example_config(row['legacy_id']))
        self.assertTrue(session.inspect().ok)
        self.assertTrue(session.configure(demoncat.example_config(row['legacy_id'])).ok)
        for operation in ('observe','verify','disarm','restore'):
            refused=getattr(session,operation)()
            self.assertFalse(refused.ok)
            self.assertEqual(refused.code,'INVALID_STATE')
        refused=session.inject()
        self.assertEqual(refused.code,'INVALID_STATE')
        arm=session.arm()
        self.assertFalse(arm.ok)
        self.assertEqual(arm.code,'INVALID_STATE' if row['adapter_status']=='CPU_EXECUTABLE' else 'NOT_IMPLEMENTED')
        self.assertFalse(arm.qualification_granted)
        self.assertTrue(session.status().ok)
        self.assertTrue(session.cleanup().ok)
        self.assertEqual(session.state,'CLOSED')
        json.dumps(session.audit().to_dict(),allow_nan=False)
    for label,fn in [('registration',registration),('configuration',configuration),('lifecycle',lifecycle)]:
        setattr(RegistryConformance,'test_'+row['legacy_id'].replace('.','_')+'_'+label,fn)


for definition in definitions():register_case(definition)


class TriggerConformance(unittest.TestCase):
    def config(self,mode='nth',scope='request',**kwargs):
        config=demoncat.example_config('18.api_half');config['target']['rank']=[0,1]
        config['trigger']=dict(mode=mode,scope=scope,max_hits=None,**kwargs)
        return config
    def call(self,n,**kwargs):
        values=dict(event_id=str(n),process=1,worker='w',rank=0,device=0,request_id='r',operator='RMSNorm',layer='l')
        values.update(kwargs);return CallIdentity(**values)
    def test_all_six_modes(self):
        cases=[('first',{},[1]),('nth',{'ordinal':3},[3]),('every_nth',{'ordinal':2},[2,4,6]),
               ('every_match',{},[1,2,3,4,5,6]),('probability',{'probability':1,'seed':8},[1,2,3,4,5,6]),
               ('condition',{'condition':{'field':'layer','op':'eq','value':'l'}},[1,2,3,4,5,6])]
        for mode,fields,expected in cases:
            with self.subTest(mode=mode):
                engine=TriggerEngine(self.config(mode,**fields))
                self.assertEqual([i for i in range(1,7) if engine.decide(self.call(i),eligible=True).selected],expected)
    def test_filters_precede_counts_duplicate_event_refuses_and_address_reuse_is_irrelevant(self):
        config=self.config(ordinal=2);config['target']['layer']='l'
        engine=TriggerEngine(config)
        self.assertIsNone(engine.decide(self.call(1),eligible=False).ordinal)
        self.assertIsNone(engine.decide(self.call(2,layer='other'),eligible=True).ordinal)
        self.assertFalse(engine.decide(self.call(3),eligible=True).selected)
        self.assertTrue(engine.decide(self.call(4),eligible=True).selected)
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):engine.decide(self.call(4),eligible=True)
    def test_request_rank_worker_isolation(self):
        engine=TriggerEngine(self.config(ordinal=2))
        for rank in (0,1):
            for request in ('a','b'):
                for worker in ('x','y'):
                    self.assertFalse(engine.decide(self.call(f'{rank}{request}{worker}1',rank=rank,request_id=request,worker=worker),eligible=True).selected)
                    self.assertTrue(engine.decide(self.call(f'{rank}{request}{worker}2',rank=rank,request_id=request,worker=worker),eligible=True).selected)
        self.assertEqual(engine.status()['reserved_hits'],8)
    def test_concurrent_counts_and_hit_limit_are_atomic(self):
        config=self.config('every_nth',ordinal=3);config['trigger']['max_hits']=7
        engine=TriggerEngine(config)
        with ThreadPoolExecutor(max_workers=12) as pool:
            decisions=list(pool.map(lambda i:engine.decide(self.call(i),eligible=True),range(300)))
        self.assertEqual(sum(d.selected for d in decisions),7)
        self.assertEqual(engine.status()['eligible_calls'],300)
        self.assertEqual(sorted(d.ordinal for d in decisions),list(range(1,301)))
    def test_probability_is_reproducible_and_condition_does_not_eval(self):
        config=self.config('probability',probability=.42,seed=12)
        a,b=TriggerEngine(config),TriggerEngine(config)
        self.assertEqual([a.decide(self.call(i),eligible=True) for i in range(70)],
                         [b.decide(self.call(i),eligible=True) for i in range(70)])
        bad=self.config('condition',condition={'field':'__import__','op':'eq','value':'os'})
        with self.assertRaises(demoncat.ConfigError):TriggerEngine(bad)
    def test_missing_request_and_bounded_memory_refuse(self):
        engine=TriggerEngine(self.config('first'),max_events=1)
        with self.assertRaisesRegex(ValueError,'IDENTITY_MISSING'):engine.decide(self.call(1,request_id=None),eligible=True)
        engine.decide(self.call(1),eligible=True)
        with self.assertRaisesRegex(RuntimeError,'BUDGET'):engine.decide(self.call(2),eligible=True)

    def test_all_scopes_have_explicit_sharing(self):
        for scope in ('process','worker','rank','request','operator','layer','device'):
            engine=TriggerEngine(self.config('first',scope=scope))
            self.assertTrue(engine.decide(self.call(1),eligible=True).selected)
            different_request=engine.decide(self.call(2,request_id='other'),eligible=True)
            self.assertEqual(different_request.selected,scope=='request')


class TwoCalls(torch.nn.Module):
    def __init__(self):super().__init__();self.norm=torch.nn.RMSNorm(4)
    def forward(self,x):return self.norm(x)+self.norm(x)


class ActualLifecycle(unittest.TestCase):
    def session(self):
        model=TwoCalls().eval();x=torch.tensor([[1.,2.,3.,4.]])
        config=demoncat.example_config('18.api_half');config['trigger'].update(mode='nth',ordinal=2)
        session=demoncat.configure('18.api_half',config)
        self.assertTrue(session.discover(model,demoncat.Request(args=(x,))).ok)
        return session,model,x
    def test_actual_nth_effect_off_and_cleanup(self):
        session,model,x=self.session();baseline=model(x).detach().clone()
        self.assertTrue(session.arm().ok)
        self.assertTrue(session.inject().ok)
        self.assertTrue(torch.allclose(session.output,baseline*.75))
        self.assertTrue(session.verify().data['effect_observed'])
        self.assertTrue(session.disarm().ok);self.assertEqual(session.state,'OFF')
        self.assertTrue(session.restore().ok);self.assertTrue(torch.equal(model(x),baseline))
        self.assertTrue(session.cleanup().ok)
        self.assertTrue(session.audit().data['baseline_fault']['effect_observed'])
        self.assertFalse(model._forward_hooks);self.assertFalse(model.norm._forward_hooks)
    def test_recovery_failure_retains_lease_and_owner_generation(self):
        session,model,x=self.session();session.arm();session.inject();session.disarm()
        lease=session.adapter.lease;identity=(session.owner,session.generation)
        with patch.object(lease,'close',side_effect=RuntimeError('temporary cleanup failure')):
            failed=session.restore()
        self.assertFalse(failed.ok);self.assertEqual(session.state,'RECOVERY_FAILED')
        self.assertIs(session.adapter.lease,lease)
        self.assertFalse(session.cleanup().ok)
        self.assertTrue(session.restore().ok);self.assertEqual(identity,(session.owner,session.generation))
        self.assertTrue(session.cleanup().ok)
    def test_foreign_thread_and_reentrant_control_are_rejected(self):
        session,model,x=self.session();session.arm()
        with ThreadPoolExecutor(1) as pool:
            self.assertEqual(pool.submit(session.restore).result().code,'OWNER_MISMATCH')
        seen=[]
        handle=model.norm.register_forward_hook(lambda *args:seen.append(session.disarm().code))
        try:self.assertTrue(session.inject().ok)
        finally:handle.remove()
        self.assertEqual(set(seen),{'BUSY'})
        session.disarm();session.restore();session.cleanup()
    def test_actual_conv_padding_route(self):
        model=torch.nn.Conv3d(3,2,(2,16,16),stride=(2,16,16)).eval()
        x=torch.arange(1536.).reshape(1,3,2,16,16)/1000
        session=demoncat.configure('17.spatial_padding',demoncat.example_config('17.spatial_padding'))
        self.assertTrue(session.discover(model,demoncat.Request(args=(x,))).ok)
        self.assertTrue(session.arm().ok);self.assertTrue(session.inject().ok)
        self.assertTrue(session.verify().data['effect_observed'])
        session.disarm();self.assertTrue(session.restore().ok);self.assertTrue(session.cleanup().ok)
    def test_model_labels_do_not_grant_eligibility(self):
        for name in ('Qwen','GLM','DeepSeek'):
            config=demoncat.example_config('18.api_half');config['execution']['model']=name
            session=demoncat.configure('18.api_half',config)
            self.assertEqual(session.discover(torch.nn.Identity().eval(),demoncat.Request()).code,'NO_TARGET')
            self.assertTrue(session.cleanup().ok)

    def test_repeated_fault_observations_are_retained_and_off_does_not_spend_budget(self):
        session,model,x=self.session();session.arm()
        self.assertTrue(session.inject().ok);self.assertTrue(session.inject().ok)
        self.assertEqual(session.trigger.status()['reserved_hits'],1)
        self.assertEqual(len(session.audit().data['phases']['fault']),2)
        session.disarm();self.assertTrue(session.restore().ok)
        self.assertEqual(session.trigger.status()['eligible_calls'],4)
        session.cleanup()
