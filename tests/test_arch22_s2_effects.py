"""CPU-only raw geometry and source-bound input gates for the S2 effect."""
import copy
import os
import pickle
import threading
import unittest

import torch
from demoncat.runtime.vllm_ascend.worker_extension import paged33_s2_evidence as proof
from demoncat.runtime.vllm_ascend.worker_extension.paged33_arch22_s2_effects import validate_s2_launch
from demoncat.runtime.vllm_ascend.worker_extension.paged33_logical_binding import LogicalPagedInput
from fault_injection.native_adapters.attention28.fia import arch22_s2_geometry as geometry
from test_paged33_logical_binding import Fixture,PATH
from test_paged33_arch22_s2_recipe import split_fixture
import test_paged33_arch22_bridge as base


class S2EffectsContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.threads=torch.get_num_threads();torch.set_num_threads(1)
    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_raw_final_fault_geometry_uses_full_matching_matrix_not_normal_block_count(self):
        actual,boundary,host,launch=base.Arch22NormalContracts().native_fixture()
        row=split_fixture(256);checked=geometry.validate_split(row);typed=launch['typed']
        typed.update({'baseParams.s2Size':1040,'baseParams.usedCoreNum':row['usedCoreNum'],
            'innerSplitParams.mBaseSize':512,'innerSplitParams.s2BaseSize':256,
            'fdParams.numOfFdHead':8,'fdParams.usedVecNumOfFd':8,'fdParams.gS1BaseSizeOfFd':8})
        for name in geometry.ARRAYS:
            prefix='outerSplitParams.' if name in ('bN2End','gS1End','s2End') else 'fdParams.'
            width=52 if name in geometry.ARRAYS[-2:] else 26
            typed[prefix+name]=row[name]+[0]*(width-len(row[name]))
        typed.update({'workspaceParams.'+key:value for key,value in checked['workspace_elements'].items()})
        host.update(schema='demoncat.paged33.arch22_s2_host.v1',mode='arch22_s2_256',phase='fault',
            target_path=PATH,owner='a'*32,generation=3,token='b'*32,pid=os.getpid(),tid=threading.get_native_id(),
            actual_observer_scope_matched=True,policy_applied=True,selected_s2=256,
            actual_kv_length=1025,real_s2_capacity=1040,s2_loops=5,s2_tail=16,s2_aligned=256,
            aic=20,aiv=40,block_dim=row['usedCoreNum'],libapi_workspace=16*1024**2,
            workspace=16*1024**2+checked['workspace_excluding_libapi'])
        kwargs=dict(token='b'*32,owner='a'*32,generation=3,phase='fault',rows={256:{20:row}})
        value=validate_s2_launch(launch,host,actual,boundary,**kwargs)
        self.assertEqual(value['selected_s2'],256);self.assertFalse(value['output_mutated_by_adapter'])
        for field,value in (('phase','off'),('mode','arch22_s2_off'),('generation',4),('target_path',PATH+'x'),
                            ('s2_loops',3),('workspace',0),('policy_applied',False)):
            changed=dict(host);changed[field]=value
            with self.subTest(field=field),self.assertRaises((RuntimeError,ValueError)):
                validate_s2_launch(launch,changed,actual,boundary,**kwargs)
        changed=copy.deepcopy(launch);changed['typed']['fdParams.s2SplitStartIdxOfCore'][1]=0
        with self.assertRaises(ValueError):validate_s2_launch(changed,host,actual,boundary,**kwargs)

    def test_only_proven_page_relocation_can_match_a_completed_normal_reference(self):
        fixture=Fixture();first=LogicalPagedInput(**fixture.arguments())
        normal=proof.VerifiedNormalEvidence(proof._MINT,dict(reference_input=first.before['signature']),simulation=True)
        second=LogicalPagedInput(**fixture.arguments('B',reversed(range(65))))
        result=normal.match(second)
        self.assertFalse(result['raw_input_exact']);self.assertTrue(result['logical_input_equivalent'])
        with self.assertRaises(TypeError):pickle.dumps(normal)
        with self.assertRaises(RuntimeError):proof.VerifiedNormalEvidence(object(),{})
        for kind in ('query','logical_key','unused_page','supplied_default','duplicate_page','wrong_slot'):
            changed=copy.deepcopy(first.before['signature'])
            if kind=='query':changed['tensors']['query']['sha256']='0'*64
            elif kind=='logical_key':changed['logical_kv']['key']['sha256']='0'*64
            elif kind=='unused_page':changed['block_table_values'][0][-1]=1
            elif kind=='supplied_default':changed['scalars']['pre_tokens']=2147483647
            elif kind=='duplicate_page':changed['block_table_values'][0][1]=0
            else:changed['writer_consumer_slot']=[1]
            bad=proof.VerifiedNormalEvidence(proof._MINT,dict(reference_input=changed),simulation=True)
            with self.subTest(kind=kind),self.assertRaises(RuntimeError):bad.match(second)


if __name__=='__main__':unittest.main()
