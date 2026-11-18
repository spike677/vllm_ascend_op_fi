"""Independent raw-reader contracts; CPU source/dispatch fixtures, no NPU proof."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch

import test_quant_route_frames as fixture
from experiments.worker_core_v11_20261118 import review_quant_routed_frames as reader


class RoutedFramesReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        case = fixture.RouteFrameTests()
        case.setUp()
        try:
            with torch.inference_mode():
                case.invoke()
                graph, boundary = case.graph.report(), case.source.report()
                record = case.record()
                integer = boundary['integer_routes']['records'][0]
            cls.original = dict(record=record, graph=graph, boundary=boundary, integer=integer)
            cls.reviewed = {}
            for row in boundary['boundaries']:
                cls.reviewed.setdefault(row['role'],set()).add((row['source_sha256'],row['function']))
            cls.schemas = {name:stage['registration']['actual_schema'] for name,stage in integer['stages'].items()}
        finally:
            case.doCleanups()

    def setUp(self):
        self.data = copy.deepcopy(self.original)

    def review(self):
        return reader.review_record(**self.data, reviewed=self.reviewed, schemas=self.schemas)

    def frame(self, role):
        return next(item for item in self.data['record']['frames'] if item['role'] == role)

    def event(self, sequence):
        return self.data['graph']['events'][sequence-1]

    def assertRefused(self, reason):
        result = self.review()
        self.assertFalse(result['independently_complete'],result)
        self.assertFalse(result['native_row_authority'])
        self.assertIn(reason,json.dumps(result['issues']))

    def rehash_integer(self):
        integer = self.data['integer']
        integer['token_route_proof']['evidence_sha256'] = reader.digest({k:v for k,v in integer.items() if k != 'token_route_proof'})
        value = reader.digest(integer)
        self.data['record']['integer_route_evidence']['sha256'] = value
        self.event(integer['parent_sequence'])['integer_route_evidence']['sha256'] = value

    def test_actual_cpu_frames_are_reconciled_but_never_native_authority(self):
        result = self.review()
        self.assertTrue(result['independently_complete'],result)
        self.assertEqual(len(result['checked_sections']),6)
        self.assertFalse(result['native_row_authority'])

    def test_completion_boolean_is_not_the_auditors_authority(self):
        self.data['record']['python_chain_complete'] = False
        self.data['record']['completed'] = False
        self.assertTrue(self.review()['independently_complete'])

    def test_true_completion_flag_cannot_hide_same_shape_replacement(self):
        self.frame('rchain_gmm2')['input']['hidden_states']['tensor']['object_id'] += 1
        self.assertRefused('RAW_CURRENT_NATIVE_INPUT:gmm2:hidden_states')

    def test_same_value_group_payload_cannot_replace_current_dispatch_object(self):
        self.frame('rchain_builder')['input']['token_dispatch_output']['object_identity'] += 1
        self.assertRefused('ACTUAL_DISPATCH_TO_BUILDER')

    def test_gmm2_weight_and_scale_join_is_not_only_activation_geometry(self):
        for name in ('weight','weight_scale'):
            with self.subTest(name=name):
                self.data = copy.deepcopy(self.original)
                self.frame('rchain_gmm2')['input'][name][0]['tensor']['object_id'] += 1
                self.assertRefused('CURRENT_GMM2_QUANT_ARGUMENT:' + name)

    def test_abs_indices_must_be_current_combine_metadata(self):
        self.frame('route_combine')['input']['combine_metadata']['object_identity'] += 1
        self.assertRefused('CURRENT_COMBINE_PAYLOAD')

    def test_stage_cannot_be_relabelled_to_another_source_frame(self):
        self.data['record']['actual_stage_frame_links']['gmm2']['source_boundary_sequence'] = self.frame('rchain_gmm1')['boundary_sequence']
        self.assertRefused('ACTUAL_STAGE_FRAME_REFERENCE:gmm2')

    def test_duplicate_role_or_missing_stage_remains_explicit(self):
        self.data['record']['frames'].append(copy.deepcopy(self.frame('rchain_gmm2')))
        self.assertRefused('EXACT_REQUIRED_FRAME_SET')

    def test_wrong_forward_or_request_scope_is_rejected(self):
        for key in ('forward_id','request_scope_sha256'):
            with self.subTest(key=key):
                self.data = copy.deepcopy(self.original)
                self.data['record'][key] = 'different-forward-or-request'
                self.assertRefused('CURRENT_FORWARD_REQUIRED' if key == 'forward_id' else 'EXACT_CURRENT_INTEGER_PARENT_DIGEST')

    def test_source_function_fingerprint_cannot_be_relabelled(self):
        self.frame('rchain_gmm1')['source_sha256'] = '0'*64
        self.assertRefused('PINNED_FRAME_SOURCE_OR_FUNCTION')

    def test_ancestor_cannot_point_to_the_frame_itself(self):
        sequence = self.frame('rchain_builder')['boundary_sequence']
        original = next(row for row in self.data['boundary']['boundaries'] if row['sequence'] == sequence)
        original['parent_source_sequence'] = sequence
        self.assertRefused('CURRENT_MODEL_FRAME_ANCESTRY')

    def test_actual_scalar_and_alpha_are_checked_without_default_guess(self):
        self.data['record']['consumers'][0]['formals']['other'] += 1
        self.assertRefused('ACTUAL_SCALAR_AND_CURRENT_CONFIG')
        self.data = copy.deepcopy(self.original)
        self.data['record']['consumers'][1]['formals']['alpha'] = 2
        self.assertRefused('ACTUAL_ADD_OPERATOR_AND_ALPHA')

    def test_alias_single_actual_return_is_required(self):
        alias = next(row for row in self.data['graph']['events'] if row['operator'] == 'aten.alias.default')
        alias['original_calls'] = 2
        self.assertRefused('ACTUAL_ALIAS_SINGLE_RETURN')

    def test_matching_address_does_not_replace_alias_object_identity(self):
        self.data['record']['consumers'][1]['formals']['other']['tensor']['object_id'] += 1
        self.assertRefused('ACTUAL_SHARED_ADDITION_OPERAND')

    def test_packet_epoch_is_recomputed_from_complete_raw_events(self):
        self.frame('rchain_gmm2')['input']['hidden_states']['write_epoch'] += 1
        self.assertRefused('NATIVE_INPUT_WRITE_EPOCH')

    def test_inplace_after_epoch_does_not_override_raw_invalidations(self):
        self.data['record']['consumers'][0]['target_after']['write_epoch'] += 1
        self.assertRefused('CONSUMER_AFTER_EPOCH')

    def test_mutation_raw_epoch_gap_is_a_global_ledger_error(self):
        event = next(row for row in self.data['graph']['events'] if row.get('invalidated_storages'))
        event['invalidated_storages'][0]['write_epoch'] += 1
        with self.assertRaisesRegex(ValueError,'RAW_WRITE_EPOCH_SEQUENCE_INVALID'):
            self.review()

    def test_snapshot_epoch_is_not_proved_only_by_equal_before_after(self):
        snap = self.data['integer']['snapshots']['expert_counts']
        snap['epoch_before'] += 1
        snap['epoch_after'] += 1
        self.rehash_integer()
        self.assertRefused('CURRENT_INTEGER_SNAPSHOT_EPOCH:expert_counts')

    def test_helper_omission_or_model_reclassification_is_not_hidden(self):
        self.data['boundary']['integer_routes']['helper_calls'].pop()
        self.assertRefused('ALL_FIVE_UNIQUE_INTEGER_HELPERS')
        self.data = copy.deepcopy(self.original)
        self.data['boundary']['integer_routes']['helper_calls'][0]['classification'] = 'MODEL_INVOCATION'
        self.assertRefused('EXACT_SEPARATE_INTEGER_HELPER')

    def test_final_view_is_joined_to_actual_model_return(self):
        self.frame('rchain_model')['output']['tensor']['object_id'] += 1
        self.assertRefused('ACTUAL_MODEL_RETURN_OBJECT')

    def test_original_failure_survives_other_complete_fields(self):
        self.data['record']['first_failure'] = 'OBSERVED_FAILURE'
        self.assertRefused('OBSERVED_FRAME_FAILURE_REMAINS_A_REFUSAL')

    def test_native_claim_cannot_be_smuggled_into_observation(self):
        self.data['record']['native_registration_proven'] = True
        self.assertRefused('OBSERVATION_CANNOT_GRANT_NATIVE_AUTHORITY')


class ReaderIsolationTests(unittest.TestCase):
    @staticmethod
    def native_fixture():
        parent = dict(token=1,thread=1,device=0,reserved=0)
        public = dict(operator='npu_dynamic_quant',original_calls=1,forward_id='f',request_ids=['r'],
                      module_object_identity=12,errors=[])
        for name,label,dtype,description,element in (('input','float_input',27,'torch.bfloat16',2),
                ('quant','quantized_output',2,'torch.int8',1),('scale','per_token_scale',0,'torch.float32',4)):
            shape,stride = ([8],[1]) if name == 'scale' else ([8,4],[4,1])
            address = {'input':1000,'quant':2000,'scale':3000}[name]
            size = (8 if name == 'scale' else 32)*element
            parent[name] = dict(descriptor=address+4,reserved=0,dtype=dtype,format=2,dimensions=len(shape),
                address=address,bytes=size,offset=0,rows=8,columns=0 if name == 'scale' else 4,
                stride0=stride[0],stride1=0 if name == 'scale' else 1)
            public[label] = dict(shape=shape,stride=stride,actual_format=2,device='npu:0',storage_addr=address,
                storage_bytes=size,storage_offset=0,element_size=element,dtype=description)
        events = [dict(kind=1,parent=parent)]
        for kind in range(1,5):
            raw = dict(sequence=kind+1,kind=kind,registration=2,thread=1,device=0,status=0,reserved=0,
                original_calls=int(kind%2 == 0),executor=0 if kind == 1 else 44,smooth_tensor=0,
                group_index_tensor=0,offset_tensor=0,destination_dtype=2,workspace_required=0,
                workspace_capacity=0,workspace=0,stream=55 if kind >= 3 else 0,
                **{name:copy.deepcopy(parent[name]) for name in ('input','quant','scale')})
            events.append(dict(kind=2,native=raw,binding={'parent_token':0,'child_lease':0,'owner_hi':0,'owner_lo':0,'generation':0,'role':0,'reserved':0}))
        public['independent_native'] = dict(errors=[],native_events=copy.deepcopy(events))
        before = dict(last_sequence=0,abi_exceptions=0,orphan_descriptors=0,pending=0,executing=0,quarantined=0,
            census=dict(errors=0,overflow=0,preparing=0,pending=0,executing=0,model_parents=0,native_prepares=0,native_executes=0))
        after = copy.deepcopy(before);after['last_sequence']=5
        after['census'].update(model_parents=1,native_prepares=1,native_executes=1)
        forward = dict(forward_id='f',before=before,after=after,native_events=events,unowned_native_events=[],
            row_provenance={'forward_id':'f'},candidates=[dict(native=copy.deepcopy(events[0]),public_api_index=0,module_object_identity=12)])
        projection = dict(forward_id='f',request_ids=['r'],projections=[dict(api_call_indices=[0,1],module_object_identity=12)])
        return forward,projection,[public,dict(operator='npu_quant_matmul',original_calls=1,errors=[],forward_id='f',module_object_identity=12)]

    def test_native_denominator_recounts_actual_parent_prepare_execute(self):
        result = reader.review_native_denominator(*self.native_fixture(),expected_calls=1)
        self.assertEqual(result,dict(raw_model_parents=1,raw_prepare_calls=1,raw_execute_calls=1,raw_public_calls=2,actual_native_events=5))

    def test_summary_counter_does_not_fill_missing_native_event(self):
        forward,public,calls = self.native_fixture();forward['native_events'].pop()
        with self.assertRaisesRegex(ValueError,'COMPLETE_RAW_NATIVE_DENOMINATOR'):
            reader.review_native_denominator(forward,public,calls,expected_calls=1)

    def test_swapped_native_registration_or_public_owner_refuses(self):
        for kind in ('registration','owner','view'):
            with self.subTest(kind=kind):
                forward,public,calls = self.native_fixture()
                if kind == 'registration':forward['native_events'][3]['native']['registration'] += 1
                elif kind == 'owner':calls[0]['module_object_identity'] += 1
                else:calls[0]['float_input']['storage_addr'] += 1
                with self.assertRaises(ValueError):
                    reader.review_native_denominator(forward,public,calls,expected_calls=1)

    def test_actual_nonzero_status_does_not_become_completed_pass_through(self):
        forward,public,calls = self.native_fixture();forward['native_events'][-1]['native']['status']=9
        with self.assertRaisesRegex(ValueError,'RAW_PASS_THROUGH_NATIVE_CALLS'):
            reader.review_native_denominator(forward,public,calls,expected_calls=1)

    def test_sparse_epoch_history_tracks_reuse_without_copying_full_maps(self):
        events = [dict(sequence=i,invalidated_storages=[dict(device='cpu',address=5,bytes=32,
            storage_identity=7,write_epoch=i)]) for i in range(1,10001)]
        ledger = reader.EpochLedger(events)
        self.assertEqual(len(ledger.history),1)
        self.assertEqual(ledger.at(8000,('cpu',5,32,7)),7999)
        self.assertEqual(ledger.at(8000,('cpu',5,32,7),after=True),8000)
        self.assertEqual(ledger.at(8000,('cpu',5,32,8)),0)

    def test_script_imports_with_stdlib_only_from_another_directory(self):
        script = Path(reader.__file__).resolve()
        command = ('import runpy,sys;sys.path.insert(0,sys.argv[1]);'
            'runpy.run_path(sys.argv[2],run_name="reader_only");'
            'assert "torch" not in sys.modules;assert "demoncat" not in sys.modules')
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run([sys.executable,'-I','-S','-c',command,str(script.parent),str(script)],
                cwd=directory,capture_output=True,text=True,timeout=20)
        self.assertEqual(completed.returncode,0,completed.stderr)

    def test_explicit_original_sha_is_required_before_parsing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'inventory.json').write_text('{}',encoding='utf-8')
            (root/'result.json').write_text('{}',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'EXPLICIT_ORIGINAL_SHA_MISMATCH'):
                reader.audit(root/'inventory.json','0'*64,root/'result.json','1'*64,root)


if __name__ == '__main__':
    unittest.main()
