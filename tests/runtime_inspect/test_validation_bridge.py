import copy
import types
import unittest

from ascend_fi.inspect.validation import ExistingOwnerIdentityValidator
from ascend_fi.contracts import ValidationStatus


class Config:
    world_size = 2
    def to_dict(self): return {'actual_module_names': ['layers.1.mlp.down_proj']}


class Owner:
    def __init__(self):
        self.arm = {'mode': 'REFERENCE_REQUANT', 'module_regex': r'.*down_proj$', 'rank_filter': [0],
                    'scale_provider': 'tensor_absmax', 'scale_multiplier': .25, 'max_hits': 4096,
                    'interval': 1, 'tensor_selector': 'first_float'}
        self.installed = False
        self.removes = 0
        self.fail = False
        self.identity_calls = []
        self.corrupt_rank = False
    def preflight(self): return {'status': 'PASS', 'exclusive_lock_owned': True}
    def clean_query(self): return {'http_status':200, 'results':[
        {'pid':11,'installed':self.installed,'enabled':False}, {'pid':12,'installed':self.installed,'enabled':False}]}
    def service_healthy(self): return True
    def request(self, probe, phase, index): return {'http_status':200,'response_text':'answer'}
    def install(self, config): self.installed=True; return config
    def disarm_and_remove(self): self.installed=False; self.removes+=1
    def fault_request(self, probe, config, index, identity=False):
        self.identity_calls.append(identity)
        if self.fail: raise RuntimeError('request failed')
        results=[]
        for rank in range(2):
            hit = rank == 0 or self.corrupt_rank
            results.append({'matched_modules':[{'module':'layers.1.mlp.down_proj','tp_rank':rank}],
                'hit_count':int(hit), 'tensor_effect_history': [
                    {'mode':'IDENTITY','changed_elements':0,'rank':rank,'module':'layers.1.mlp.down_proj'}] if hit else []})
        return {'http_status':200,'response_text':'answer',
                'fault_transaction':{'measurement_query':{'http_status':200,'results':results}}}


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.owner = Owner()
        self.plan = {'service_epoch':'e','runtime_fingerprint':'f','candidate_ids':['c'],
                     'selected_modules':['layers.1.mlp.down_proj'], 'ranks':[0], 'unique_float_input_proven':True,
                     'prospective_arm':{**self.owner.arm,'tensor_selector':'tensor_index','tensor_index':0}}
        self.executor = ExistingOwnerIdentityValidator(self.owner, Config(), {}, lambda: ('e','f'))
    def test_delegates_identity_only_and_recovers(self):
        r=self.executor.validate_identity(self.plan)
        self.assertEqual(r.status, ValidationStatus.LOSSLESS_VALIDATED)
        self.assertEqual(self.owner.identity_calls,[True])
        self.assertFalse(self.owner.installed)
        self.assertEqual(r.extra['validation_kind'],'MODULE_IDENTITY')
    def test_request_failure_finally_removes(self):
        self.owner.fail=True
        with self.assertRaises(RuntimeError): self.executor.validate_identity(self.plan)
        self.assertEqual(self.owner.removes,1)
        self.assertFalse(self.owner.installed)
    def test_never_expand_selected_set(self):
        self.plan['selected_modules']=['layers.10.mlp.down_proj']
        with self.assertRaisesRegex(ValueError,'NOT_EQUIVALENT'): self.executor.validate_identity(self.plan)
        self.assertFalse(self.owner.identity_calls)
        self.assertFalse(self.owner.installed)
    def test_no_selector_equivalence_guess(self):
        self.plan['unique_float_input_proven']=False
        with self.assertRaisesRegex(ValueError,'SELECTOR'): self.executor.validate_identity(self.plan)
        self.assertFalse(self.owner.installed)
    def test_wrong_rank_identity_does_not_pass(self):
        self.owner.corrupt_rank=True
        r=self.executor.validate_identity(self.plan)
        self.assertFalse(r.lossless_passed)
    def test_changed_epoch_rejects_before_owner(self):
        self.plan['service_epoch']='old'
        r=self.executor.validate_identity(self.plan)
        self.assertFalse(r.lossless_attempted)
        self.assertFalse(self.owner.identity_calls)
    def test_frozen_strength_no_override(self):
        self.plan['prospective_arm']['scale_multiplier']=.5
        with self.assertRaisesRegex(ValueError,'CONFIG_MISMATCH'): self.executor.validate_identity(self.plan)
        self.assertFalse(self.owner.identity_calls)


if __name__=='__main__': unittest.main()
