"""CPU-only negative checks for the independent phase1 evidence gate."""
import copy
import json
from pathlib import Path
import struct
import unittest
import validate_phase1 as v

class EvidenceChecks(unittest.TestCase):
    def test_fixed_88_statistics_pairs_cover_all_four_steps(self):
        plan=v.statistics_plan();self.assertEqual(len(plan),88)
        self.assertEqual(plan['controlled_joint_b8_r1_s0_vs_s3'],
            ('off_restore','controlled_joint_b8_r1',0,'controlled_joint_b8_r1',3))
        self.assertEqual(plan['controlled_query_tile_b1_r0_stock_vs_private_s0'],
            ('stock_vs_private_native','discovery_b1_r0',0,'controlled_query_tile_b1_r0',0))
        for target in v.TARGETS:
            for batch in (1,8):
                for step in range(4):self.assertIn(f'controlled_{target}_b{batch}_r0_vs_r1_s{step}',plan)

    def test_only_six_explicit_absent_optional_inputs(self):
        frozen={'scale':0.25,'keep_prob':1.,'inner_precise':0}
        actual={**frozen,**dict.fromkeys(('pse','padding_mask','prefix','actual_seq_qlen','actual_seq_kvlen','sink'))}
        v.native_kwargs_equal(actual,frozen)
        for modified in (dict(actual,inner_precise=1),dict(actual,pse=[1]),dict(actual,unapproved=None)):
            with self.assertRaises(ValueError):v.native_kwargs_equal(modified,frozen)

    def test_partial_attempt_rejected_even_when_case_completed(self):
        c={'status':'FAIL','card':5,'cases':[{'status':'CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE'}],
           'cleanup_errors':[],'owned_container_removed':True,'device_lock_acquired':True,
           'device_lock_released':True,'external_container_mutations':False,'owned_container_id':'owned'}
        with self.assertRaisesRegex(ValueError,'Whole attempt'):v.strict_completion(c)
        c['status']='CAPTURE_COMPLETED';v.strict_completion(c)
        c['owned_container_removed']=False
        with self.assertRaises(ValueError):v.strict_completion(c)

    def test_only_forward_format_normalization_allowed(self):
        a={'actual_format':0,'full_sha256':'same','object_id':3,'data_ptr':4,'stride':[1]}
        b=dict(a,actual_format=2);v.unchanged(a,b)
        for bad in (dict(b,full_sha256='other'),dict(b,object_id=7),dict(b,actual_format=29)):
            with self.assertRaises(ValueError):v.unchanged(a,bad)
        with self.assertRaises(ValueError):v.unchanged(b,a)

    def test_complete_output_includes_early_query(self):
        a=bytes(16*2048*128*2);b=bytearray(a);struct.pack_into('<e',b,64*128*2,1.0)
        d=v.binary_difference(a,bytes(b))
        self.assertEqual(d['changed_elements'],1);self.assertEqual(d['changed_queries'],1)
        self.assertEqual(d['first_changed'],{'head':0,'query':64,'channel':0});self.assertEqual(d['max_abs'],1.)
        self.assertEqual(a[-128*2:],b[-128*2:])

    def test_fp16_subnormal_has_no_tolerance(self):
        a=bytes(16*2048*128*2);b=b'\x01\x00'+a[2:]
        d=v.binary_difference(a,b);self.assertEqual(d['max_abs'],2**-24)
        self.assertFalse(d['bitwise_equal'])
        self.assertTrue(v.binary_difference(a,a)['bitwise_equal'])
        with self.assertRaises(ValueError):v.binary_difference(a[:2],b[:2])

    def test_exact_five_padding_fields_only(self):
        pod={str(i):i for i in range(200)};pod.update({k:0 for k in v.PADDING})
        other=copy.deepcopy(pod);other['inputParams.InputParamsPH']=11
        self.assertEqual(v.semantic(pod),v.semantic(other))
        other['1']=999;self.assertNotEqual(v.semantic(pod),v.semantic(other))
        pod.pop('2')
        with self.assertRaises(ValueError):v.semantic(pod)

    def test_three_opaque_sections_are_part_of_plan(self):
        schema=Path(__file__).resolve().parents[3]/'native_execution_strategy/adapters/fa_cann91_schema.json'
        # Path in repository: experiment/analysis -> repository is parents[3].
        if not schema.exists():schema=Path(__file__).resolve().parents[3]/'runtime/native_execution_strategy/adapters/fa_cann91_schema.json'
        data=json.loads(schema.read_bytes());fields={f['name']:f for f in data['fields']}
        obj=v.Validator.__new__(v.Validator)
        obj.plan_fields={n for n in fields if not n.startswith('inputParams.') and n not in v.PADDING}
        pod={n:0 for n in fields};raw=bytes(1464)
        left={'pod':pod,'native':{'raw_hex':raw.hex()}};p=obj.plan(left)
        self.assertEqual(len(p),169)
        for name,(offset,count) in v.OPAQUE.items():
            changed=bytearray(raw);changed[offset]=1
            right={'pod':pod,'native':{'raw_hex':changed.hex()}}
            differences=[key for key in p if p[key]!=obj.plan(right)[key]]
            self.assertEqual(differences,[name]);self.assertEqual(len(p[name]),count*2)
        covered=set()
        for f in fields.values():covered.update(range(f['offset'],f['offset']+f['width']*f['count']))
        for offset,count in v.OPAQUE.values():covered.update(range(offset,offset+count))
        self.assertEqual(covered,set(range(1464)))

    def test_other_batch_slots_not_assumed_equal(self):
        pod={str(i):i for i in range(200)};pod.update({k:0 for k in v.PADDING})
        left={'raw':b'x','pod':pod,'step':{'output':{'per_batch_sha256':['a','b']}}}
        right=copy.deepcopy(left);right['step']['output']['per_batch_sha256'][1]='c'
        with self.assertRaisesRegex(ValueError,'other batch slot'):v.check_outputs_equal(left,right,'repeat')
        right=copy.deepcopy(left);right['raw']=b'y'
        with self.assertRaisesRegex(ValueError,'complete private'):v.check_outputs_equal(left,right,'repeat')

if __name__=='__main__':unittest.main()
