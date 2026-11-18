"""CPU metadata/coordinate/source contracts, not an SDK or NPU acceptance."""
import copy
import hashlib
import math
from pathlib import Path
import unittest
from unittest.mock import patch

from fault_injection.native_adapters.attention28.fia import apply_tnd28_prefill_normal as recipe
from fault_injection.native_adapters.attention28.fia import tnd28_prefill_geometry as geometry
from experiments.worker_core_v11_20261118 import check_tnd28_normal_materials as materials


def fixture():
    values = {k: dict(kind='scalar', value=v) for k, v in dict(input_layout='TND',
        block_size=128, block_table=None, num_heads=16, num_key_value_heads=8,
        sparse_mode=3, scale=128**-.5, actual_seq_lengths=[2048],
        actual_seq_lengths_kv=[2048]).items()}
    for index, (name, (shape, stride, offset, storage, dtype, width)) in enumerate(geometry.PUBLIC_VIEWS.items()):
        values[name] = dict(kind='tensor', snapshot=dict(bytes=math.prod(shape)*width,
            finite=True, version=None, path=None, sha256=hashlib.sha256(name.encode()).hexdigest(),
            metadata=dict(shape=list(shape), stride=list(stride), storage_offset=offset,
                storage_nbytes=storage, dtype=dtype, device='npu:0',
                object_id=1000+index, storage_identity=2000+index)))
    captured = dict(inputs=values, supplied_argument_names=sorted(values),
        original_callable=dict(type='torch._ops.OpOverloadPacket', object_id=1),
        public_default_schema='explicit CPU schema fixture, no actual registration claim',
        omitted_argument_defaults=copy.deepcopy(geometry.DEFAULTS),
        live_metadata=dict(num_prefills=1, num_decodes=0, num_actual_tokens=2048,
            causal=True, seq_lens_list=[2048], actual_seq_lengths_q=[2048],
            mask_is_actual_metadata_object=True))
    row = dict(actual_operator='torch_npu.npu_fused_infer_attention_score',
        original_calls=1, original_returned=True, request_id='cpu-metadata-1',
        scheduler=dict(status='READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE',
            internal_request_id='cpu-metadata-1', num_requests=1, num_computed_tokens=0,
            num_prompt_tokens=2048, num_scheduled_tokens=2048))
    return captured, row


def source_fixture():
    return {recipe.PATHS[0]: 'namespace optiling {\nclass X {};\n}',
        recipe.PATHS[1]: 'void arithmetic(){ CalcInnerSize(); Split(); FillTiling(); GenTilingKey(); }\n'
                        '    return ge::GRAPH_SUCCESS;\n}\n\n// original final return',
        recipe.ROUTE: 'CheckRouteToFiaPrerequisites(context);\n'+recipe.ROUTE_ANCHOR+'\nreturn true;'}


class PublicViewContracts(unittest.TestCase):
    def test_real_strided_v_scope_is_preserved_and_not_a_native_certificate(self):
        captured, row = fixture()
        before = copy.deepcopy(captured)
        result = geometry.public_candidate(captured, row)
        self.assertEqual(captured, before)
        self.assertEqual(result['tensors']['value']['first_byte'], 6144)
        self.assertEqual(result['tensors']['value']['span_end_byte'], 16777216)
        self.assertEqual(result['tensors']['value']['snapshot']['metadata']['stride'], [4096,128,1])
        self.assertIsNone(result['tensors']['value']['snapshot']['version'])
        for name in ('native_attributes_observed','public_prepared_pointer_stride_join',
                     'mask_values_verified','stock_private_exact_proven','fault_execution_ready','c1','c2','c3'):
            self.assertIs(result[name], False)
        result['tensors']['value']['snapshot']['metadata']['stride'][0] = 1024
        result['declared_defaults']['inner_precise'] = 1
        self.assertEqual(captured, before)

    def test_contiguous_copy_of_v_cannot_replace_the_observed_public_view(self):
        captured, row = fixture()
        meta = captured['inputs']['value']['snapshot']['metadata']
        meta.update(stride=[1024,128,1], storage_offset=0, storage_nbytes=4194304)
        with self.assertRaisesRegex(ValueError, 'ACTUAL_PUBLIC_VIEW_DIFFERS:value'):
            geometry.public_candidate(captured, row)

    def test_each_actual_shape_stride_offset_extent_and_dtype_is_required(self):
        for role in geometry.PUBLIC_VIEWS:
            for field in ('shape','stride','storage_offset','storage_nbytes','dtype'):
                captured, row = fixture()
                meta = captured['inputs'][role]['snapshot']['metadata']
                if field in ('shape','stride'):
                    meta[field][-1] += 1
                elif field == 'dtype':
                    meta[field] = 'torch.bfloat16' if role != 'atten_mask' else 'torch.bool'
                else:
                    meta[field] += 1
                with self.subTest(role=role, field=field), self.assertRaisesRegex(ValueError, 'ACTUAL_PUBLIC_VIEW_DIFFERS'):
                    geometry.public_candidate(captured, row)

    def test_operand_current_identity_bytes_and_device_cannot_be_missing(self):
        for role in geometry.PUBLIC_VIEWS:
            for changed in ('object_id','storage_identity','bytes','finite','sha256','version','device','shape_bool'):
                captured, row = fixture()
                snap = captured['inputs'][role]['snapshot']
                if changed in ('object_id','storage_identity'):
                    snap['metadata'][changed] = 0
                elif changed == 'device':
                    snap['metadata']['device'] = 'npu:1'
                elif changed == 'shape_bool':
                    snap['metadata']['stride'][-1] = True
                elif changed == 'sha256':
                    snap[changed] = 'unobserved'
                elif changed == 'version':
                    snap.pop(changed)
                else:
                    snap[changed] = False
                with self.subTest(role=role, changed=changed), self.assertRaises(ValueError):
                    geometry.public_candidate(captured, row)
        captured, row = fixture()
        captured['inputs']['value']['snapshot']['metadata']['object_id'] = 1001
        with self.assertRaisesRegex(ValueError, 'OPERAND_OBJECT_ALIAS'):
            geometry.public_candidate(captured, row)

    def test_public_scalars_lengths_and_optional_features_cannot_change(self):
        variants = dict(input_layout='BNSD', block_size=16, block_table=[0],
            num_heads=8, num_key_value_heads=16, sparse_mode=2,
            actual_seq_lengths=[0,2048], actual_seq_lengths_kv=[2047],
            scale=.1)
        for name, value in variants.items():
            captured, row = fixture()
            captured['inputs'][name]['value'] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                geometry.public_candidate(captured, row)
        for name in ('query_rope','key_shared_prefix','antiquant_scale'):
            captured, row = fixture()
            captured['inputs'][name] = dict(kind='scalar', value=None)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'ARGUMENT_OR_CALLABLE'):
                geometry.public_candidate(captured, row)

    def test_declaration_is_not_a_current_native_attribute(self):
        captured, row = fixture()
        captured['actual_native_attributes_observed'] = True
        result = geometry.public_candidate(captured, row)
        self.assertFalse(result['native_attributes_observed'])
        for name in geometry.DEFAULTS:
            modified = copy.deepcopy(captured)
            modified['omitted_argument_defaults'][name] = object()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'DECLARED_DEFAULT_SCHEMA'):
                geometry.public_candidate(modified, row)

    def test_batch_decode_foreign_request_and_foreign_mask_are_rejected(self):
        for name, value in dict(num_requests=8, num_computed_tokens=1,
                                num_prompt_tokens=2047, num_scheduled_tokens=1,
                                internal_request_id='different').items():
            captured, row = fixture()
            row['scheduler'][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'FRESH_SINGLE_REQUEST'):
                geometry.public_candidate(captured, row)
        for name, value in dict(num_prefills=0, num_decodes=1, num_actual_tokens=1,
                causal=False, mask_is_actual_metadata_object=False, seq_lens_list=[1024],
                actual_seq_lengths_q=[1]).items():
            captured, row = fixture()
            captured['live_metadata'][name] = value
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'CURRENT_CAUSAL_METADATA'):
                geometry.public_candidate(captured, row)

    def test_not_returned_duplicate_and_no_public_invocation_are_rejected(self):
        for name, value in (('original_calls',0),('original_calls',2),('original_calls',True),
                             ('original_returned',False),('actual_operator','aten.linear')):
            captured, row = fixture()
            row[name] = value
            with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, 'ACTUAL_PUBLIC_CALL'):
                geometry.public_candidate(captured, row)


class Coordinates(unittest.TestCase):
    def test_all_flattened_rows_follow_sg_not_bnsd_gs(self):
        independent = [(q,g) for q in range(2048) for g in range(2)]
        actual = [geometry.tnd_coordinate(flat) for flat in range(4096)]
        self.assertEqual(actual, independent)
        self.assertEqual(actual[512], (256,0))
        self.assertNotEqual(actual[512], (512 % 2048, 512 // 2048))
        self.assertEqual(actual[-1], (2047,1))

    def test_every_row_and_key_block_is_retained_and_causal_pairs_are_complete(self):
        plan = geometry.normal_coordinate_matrix()
        self.assertEqual(len(plan['tiles']), 32)
        for flat in range(4096):
            selected = [t for t in plan['tiles'] if t['flat_start'] <= flat < t['flat_start']+t['rows']]
            self.assertEqual([(t['key_start'],t['key_end']) for t in selected],
                             [(0,512),(512,1024),(1024,1536),(1536,2048)])
        expected = sum((q+1)*2 for q in range(2048))
        self.assertEqual(plan['unmasked_pairs_per_kv_head'], expected)
        self.assertEqual(expected*8, 33570816)
        self.assertTrue(any(t['unmasked_pairs'] == 0 for t in plan['tiles']))
        self.assertFalse(plan['actual_split_core_traversal_verified'])
        self.assertFalse(plan['kernel_executed'])

    def test_normal_contract_has_no_m128_or_replay_geometry(self):
        for field, value in (('m',128),('s2_base',128),('query_tokens',1),
                ('key_tokens',2047),('group',1),('query_tokens',True)):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'NORMAL_ONLY'):
                geometry.normal_coordinate_matrix(**{field:value})
        for row in (-1,4096,True,1.0):
            with self.subTest(row=row), self.assertRaisesRegex(ValueError, 'FLATTENED_ROW'):
                geometry.tnd_coordinate(row)


class SourceRecipe(unittest.TestCase):
    def test_patch_is_exactly_route_preference_and_host_capture_only(self):
        original = source_fixture()
        modified = recipe.patched(original)
        self.assertEqual(modified[recipe.PATHS[0]].replace('#include "tnd28_prefill_normal.h"\n\n',''),
                         original[recipe.PATHS[0]])
        self.assertEqual(modified[recipe.PATHS[1]].replace(recipe.CAPTURE,''), original[recipe.PATHS[1]])
        self.assertEqual(modified[recipe.ROUTE].replace(
            'CheckSpecConditions(context) && !tnd28_prefill_normal::Matches(context)',
            'CheckSpecConditions(context)'), original[recipe.ROUTE])
        self.assertEqual(set(modified), set(recipe.PATHS))
        self.assertFalse(any('op_kernel/' in p for p in modified))

    def test_missing_duplicate_and_already_patched_anchors_refuse(self):
        original = source_fixture()
        for path in original:
            for text in ('unrelated source', original[path]+original[path]):
                changed = dict(original, **{path:text})
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Missing/ambiguous'):
                    recipe.patched(changed)
        with self.assertRaisesRegex(ValueError, 'Missing/ambiguous'):
            recipe.patched(recipe.patched(original))

    def test_every_source_pin_is_checked_before_patching(self):
        raw = {p: ('CPU fixture:'+p).encode() for p in recipe.SOURCE_SHA}
        pins = {p: hashlib.sha256(v).hexdigest() for p,v in raw.items()}
        with patch.dict(recipe.SOURCE_SHA, pins, clear=True):
            recipe.verify_source(raw)
            for name in raw:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, 'PRISTINE_SOURCE_DIFFERS'):
                    recipe.verify_source(dict(raw, **{name:raw[name]+b'changed'}))
            with self.assertRaises(ValueError):
                recipe.verify_source({**raw,'experimental/alternate.cpp':b'alternate'})

    def test_independent_variant_has_no_33_fault_or_replay_keys(self):
        self.assertEqual(recipe.VARIANT,'tnd_prefill_arch22_normal')
        self.assertEqual(recipe.VENDOR,'dcat_f28_tnd_normal')
        self.assertEqual(recipe.NORMAL_KEY,geometry.NORMAL_KEY)
        self.assertEqual(len(set(recipe.KEYS)),7)
        self.assertEqual(set(recipe.KEYS)-{geometry.NORMAL_KEY},set(recipe.FALLBACK_KEYS))
        self.assertFalse(any(str(k).startswith(('109','119','600')) for k in recipe.KEYS))

    def test_host_capture_requires_real_observer_scope_and_keeps_unknown_views_explicit(self):
        header = Path(recipe.__file__).with_name('tnd28_prefill_normal.h').read_text(encoding='utf-8')
        match = header.split('inline bool Matches(',1)[1].split('inline std::string Quote',1)[0]
        self.assertNotIn('Env(', match)
        self.assertNotIn('scope_matches', match)
        self.assertIn('dcat_attention_observer_scope_matches_v1(s.token.c_str())!=1', header)
        self.assertIn('!SameScope(scope,CurrentScope())', header)
        self.assertIn('DCAT28_TND_PROCESS_START', header)
        self.assertIn(r'\"public_prepared_pointer_stride_join\":false', header)
        self.assertIn(r'\"prepared_stride_values_verified\":false', header)
        self.assertNotIn('paged33', header)
        self.assertNotIn('SetTilingKey', header)


class MaterialDenominator(unittest.TestCase):
    def fixture(self):
        requests = []
        for request_no in range(2):
            calls = []
            for ordinal in range(28):
                captured, row = fixture()
                row['request_id'] = 'cpu-request-'+str(request_no)
                row['scheduler']['internal_request_id'] = row['request_id']
                row.update(ordinal=ordinal, module_path=f'model.layers.{ordinal}.self_attn.attn',
                           module_object_id=5000+ordinal, native_census=dict(api_inputs_after=captured))
                calls.append(row)
            requests.append(dict(forwards=[dict(calls=calls)]))
        return dict(final_cleanup=dict(evidence=dict(requests=requests)))

    def test_original_metadata_join_requires_every_site_in_both_requests(self):
        result = self.fixture()
        report = materials.audit_calls(result)
        self.assertEqual((report['actual_requests'], report['actual_public_calls'], report['distinct_sites']),
                         (2,56,28))
        result['final_cleanup']['evidence']['requests'][1]['forwards'][0]['calls'].pop()
        with self.assertRaisesRegex(ValueError, 'COMPLETE_28_SITE_DENOMINATOR'):
            materials.audit_calls(result)

    def test_relabelled_reordered_or_new_worker_module_cannot_reuse_the_join(self):
        for changed in ('ordinal','module_path','module_object_id'):
            result = self.fixture()
            row = result['final_cleanup']['evidence']['requests'][1]['forwards'][0]['calls'][3]
            row[changed] = 'another.path' if changed == 'module_path' else 9000
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                materials.audit_calls(result)
        result = self.fixture()
        calls = result['final_cleanup']['evidence']['requests'][1]['forwards'][0]['calls']
        for row in calls:
            row['request_id'] = row['scheduler']['internal_request_id'] = 'cpu-request-0'
        with self.assertRaisesRegex(ValueError, 'REPEAT_REQUEST_OR_WORKER'):
            materials.audit_calls(result)

    def test_one_unproven_view_is_not_dropped_to_obtain_a_smaller_complete_set(self):
        result = self.fixture()
        calls = result['final_cleanup']['evidence']['requests'][0]['forwards'][0]['calls']
        meta = calls[17]['native_census']['api_inputs_after']['inputs']['value']['snapshot']['metadata']
        meta['storage_offset'] = 0
        with self.assertRaisesRegex(ValueError, 'ACTUAL_PUBLIC_VIEW_DIFFERS:value'):
            materials.audit_calls(result)
        self.assertEqual(len(calls),28)


if __name__ == '__main__':
    unittest.main()
