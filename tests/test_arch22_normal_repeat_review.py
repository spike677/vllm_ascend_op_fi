"""CPU byte contracts for physical page relocation, never a device execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import test_attention33_first_boundary_review as fixtures
from experiments.c2c3_closure_20261117 import diagnose_attention33_baselines as differences
from experiments.c2c3_closure_20261117 import review_attention33_first_boundary_source53 as boundary
from experiments.c2c3_closure_20261117 import review_arch22_normal_bridge as review


def make_boundary(run):
    phase, _ = fixtures.boundary_fixture(run)
    saved = phase['finish']['capture']['first_decode_boundary']
    mask = saved['inputs']['atten_mask']['snapshot']
    mask.update(metadata=fixtures.metadata([2048,2048],'torch.int8',25),bytes=2048**2,
                sha256=hashlib.sha256(bytes(2048**2)).hexdigest())
    saved['inputs_after']['atten_mask'] = deepcopy(mask)
    del saved['inputs']['optional_tensor']
    for key,value in dict(num_heads=16,num_key_value_heads=8,scale=128**-.5,
        actual_seq_lengths=[1],actual_seq_lengths_kv=[1025],pse_shift=None).items():
        saved['inputs'][key] = dict(kind='scalar',value=value)
    saved['omitted_argument_defaults'].update(pre_tokens=2147483647,next_tokens=2147483647)
    bound = saved['boundary']
    bound.update(expected_slots=bound['slots'][:],consumer_returned=False)
    bound['scheduler'].update(counts=[1],pure_decode=True,row_token_positions=[1024])
    bound['writer'] = dict(returned=True,module_path=review.FIRST,slots=bound['slots'][:],
        key_cache=deepcopy(bound['live_key_cache']),value_cache=deepcopy(bound['live_value_cache']))
    return phase


def relocate(phase, pages):
    capture = phase['finish']['capture']; saved = capture['first_decode_boundary']; bound = saved['boundary']
    saved['block_table_values'] = [pages[:]]
    digest = hashlib.sha256(struct.pack('<80i',*pages)).hexdigest()
    saved['inputs']['block_table']['snapshot']['sha256'] = digest
    saved['inputs_after']['block_table']['sha256'] = digest
    bound['used_pages'] = [pages[:65]]
    bound['slots'] = bound['expected_slots'] = bound['writer']['slots'] = [pages[64]*16]
    for kind in ('key','value'):
        saved['logical_kv_before'][kind]['physical_pages'] = pages[:65]
        saved['logical_kv_after'][kind]['physical_pages'] = pages[:65]
        capture['attention'][review.FIRST][kind]['physical_pages'] = pages[:65]


def read_boundary(run, phase, *, proof=True):
    saved = phase['finish']['capture']['first_decode_boundary']
    saved['supplied_argument_names'] = sorted(saved['inputs'])
    saved['subsequent_checks'] = {key:True for key in set(saved['inputs']) | {
        'logical_key','logical_value','impl_key_same_object','impl_value_same_object','output_0','output_1'}}
    (run/'baseline_before/first_boundary/boundary.json').write_text(json.dumps(saved))
    final_bound = deepcopy(saved['boundary']); final_bound['consumer_returned'] = True
    result = boundary.first_boundary(run,phase,final_bound,difference_reader=differences)
    if proof: result['page_relocation'] = review.page_relocation_evidence(saved,result,final_bound)
    payloads = {name:result['outputs'][0] for name in ('hidden','logits_hidden','logits')}
    for path in review.LAYERS:
        payloads[path+'/output'] = result['outputs'][0]
        for kind in ('key','value'):
            row = result['summary']['logical_kv'][kind]
            payloads[path+'/'+kind] = dict(path=row['post_consumer_file'],
                **{key:row[key] for key in ('shape','dtype','bytes','sha256')})
    return dict(boundary=result,payloads=payloads)


class NormalRepeatContracts(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.left_run,self.right_run = self.root/'left',self.root/'right'
        self.left_phase,self.right_phase = make_boundary(self.left_run),make_boundary(self.right_run)
        self.pages = list(range(1,66))+[0]*15
        relocate(self.right_phase,self.pages)
        self.saved = self.right_phase['finish']['capture']['first_decode_boundary']
        self.helper = SimpleNamespace(boundary=boundary,differences=differences)

    def pair(self, *, allow=True):
        return review.compare_pair(read_boundary(self.left_run,self.left_phase),
            read_boundary(self.right_run,self.right_phase),self.helper,allow_page_relocation=allow)

    def test_relocation_has_raw_difference_and_explicit_logical_repeat_proof(self):
        result = self.pair()
        self.assertFalse(result['raw_input_exact'])
        self.assertFalse(result['exact'])
        self.assertFalse(result['repeated_raw_inputs_claimed_identical'])
        self.assertTrue(result['logical_input_equivalent'])
        self.assertTrue(result['normal_repeat_valid'])
        self.assertEqual(result['repeat_kind'],'ACTUAL_PAGE_RELOCATION')
        self.assertEqual(result['page_relocation']['changed_used_pages'],65)
        self.assertEqual(result['all_complete_payloads']['payload_pairs'],87)
        self.assertEqual(result['all_complete_payloads']['differing_payloads'],0)
        self.assertEqual(result['page_relocation']['right']['writer_consumer_slot'],1040)
        self.assertFalse(result['page_relocation']['right']['captured_boundary_consumer_returned'])
        self.assertTrue(result['page_relocation']['right']['final_consumer_returned'])

    def test_cross_arm_never_normalizes_even_a_proven_legal_relocation(self):
        result = self.pair(allow=False)
        self.assertFalse(result['exact'])
        self.assertFalse(result['raw_input_exact'])
        self.assertNotIn('normal_repeat_valid',result)
        self.assertEqual(result['comparison_scope'],'STRICT_CROSS_ARM_RAW_PAIR')

    def test_raw_identical_repeat_is_separately_reported(self):
        relocate(self.right_phase,list(range(65))+[0]*15)
        result = self.pair()
        self.assertTrue(result['raw_input_exact'])
        self.assertTrue(result['logical_input_equivalent'])
        self.assertTrue(result['exact'])
        self.assertEqual(result['repeat_kind'],'RAW_INPUT_EXACT')

    def test_duplicate_or_out_of_range_pages_reject_even_with_rehashed_table(self):
        for value in (self.pages[0],128,-1):
            with self.subTest(page=value):
                changed=self.pages[:];changed[1]=value;relocate(self.right_phase,changed)
                with self.assertRaisesRegex(ValueError,'PAGES_AMBIGUOUS|PAGE_ID_OUTSIDE'):
                    read_boundary(self.right_run,self.right_phase)

    def test_used_pages_slots_and_live_storage_are_bound_independently(self):
        original=deepcopy(self.saved)
        edits = (
            lambda s:s['boundary']['used_pages'][0].__setitem__(0,99),
            lambda s:s['boundary'].__setitem__('expected_slots',[1008]),
            lambda s:s['boundary']['writer'].__setitem__('slots',[1008]),
            lambda s:s['boundary']['writer']['key_cache'].__setitem__('storage_identity',999),
            lambda s:s['boundary']['live_value_cache'].__setitem__('storage_offset',1),
            lambda s:s['boundary']['scheduler'].__setitem__('row_token_positions',[1023]),
        )
        for index,edit in enumerate(edits):
            self.saved.clear();self.saved.update(deepcopy(original));edit(self.saved)
            with self.subTest(edit=index),self.assertRaises((ValueError,KeyError)):
                read_boundary(self.right_run,self.right_phase)

    def test_table_hash_missing_metadata_and_unknown_actual_arguments_reject(self):
        original=deepcopy(self.saved)
        edits = (
            lambda s:s['inputs']['block_table']['snapshot'].__setitem__('sha256','0'*64),
            lambda s:s['boundary'].pop('expected_slots'),
            lambda s:s['logical_kv_before']['key'].pop('physical_pages'),
            lambda s:s['inputs'].__setitem__('unknown_input',dict(kind='scalar',value=0)),
        )
        for index,edit in enumerate(edits):
            self.saved.clear();self.saved.update(deepcopy(original));edit(self.saved)
            with self.subTest(edit=index),self.assertRaises((ValueError,KeyError)):
                read_boundary(self.right_run,self.right_phase)

    def test_unused_tail_and_mask_are_never_normalized(self):
        self.pages[-1]=100;relocate(self.right_phase,self.pages)
        result=self.pair();self.assertFalse(result['logical_input_equivalent'])
        self.assertFalse(result['normal_repeat_valid'])
        self.pages[-1]=0;relocate(self.right_phase,self.pages)
        self.saved['inputs']['atten_mask']['snapshot']['sha256']='1'*64
        self.saved['inputs_after']['atten_mask']['sha256']='1'*64
        result=self.pair();self.assertFalse(result['logical_input_equivalent'])
        self.assertFalse(result['normal_repeat_valid'])

    def test_actual_logical_kv_difference_cannot_be_hidden_by_equal_outputs(self):
        capture=self.right_phase['finish']['capture']
        path=self.right_run/'baseline_before/private/layer-0-key.bin'
        with path.open('r+b') as stream:stream.write(struct.pack('<e',3.0))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        self.saved['logical_kv_before']['key']['sha256']=digest
        self.saved['logical_kv_after']['key']['sha256']=digest
        capture['attention'][review.FIRST]['key']['sha256']=digest
        result=self.pair()
        self.assertFalse(result['logical_input_equivalent'])
        self.assertFalse(result['normal_repeat_valid'])
        self.assertGreater(result['all_complete_payloads']['differing_payloads'],0)
        self.assertTrue(result['immediate_outputs'][0]['exact'])

    def test_missing_post_bin_and_unrehashed_logical_evidence_reject(self):
        value=read_boundary(self.right_run,self.right_phase,proof=False)['boundary']
        value['summary']['logical_kv']['key']['matching_post_file_rehashed']=False
        final_bound=deepcopy(self.saved['boundary']);final_bound['consumer_returned']=True
        with self.assertRaisesRegex(ValueError,'LOGICAL_KV_PROOF_MISSING'):
            review.page_relocation_evidence(self.saved,value,final_bound)
        (self.right_run/'baseline_before/private/layer-0-key.bin').unlink()
        with self.assertRaises(FileNotFoundError):read_boundary(self.right_run,self.right_phase)

    def test_missing_page_proof_cannot_fall_back_to_raw_equal_or_exact_outputs(self):
        value=read_boundary(self.left_run,self.left_phase,proof=False)
        result=review.compare_pair(value,value,self.helper,allow_page_relocation=True)
        self.assertTrue(result['exact'])
        self.assertFalse(result['logical_input_equivalent'])
        self.assertFalse(result['normal_repeat_valid'])

    def test_pre_call_returned_false_requires_the_actual_completed_audit_join(self):
        value=read_boundary(self.right_run,self.right_phase,proof=False)['boundary']
        final_bound=deepcopy(self.saved['boundary'])
        with self.assertRaisesRegex(ValueError,'FINAL_CONSUMER_BOUNDARY_NOT_JOINED'):
            review.page_relocation_evidence(self.saved,value,final_bound)
        final_bound['consumer_returned']=True
        self.assertTrue(review.page_relocation_evidence(self.saved,value,final_bound)['summary']['final_consumer_returned'])
        final_bound['module_object_id']+=1
        with self.assertRaisesRegex(ValueError,'FINAL_CONSUMER_BOUNDARY_NOT_JOINED'):
            review.page_relocation_evidence(self.saved,value,final_bound)


if __name__ == '__main__': unittest.main()
