"""Fresh-source and independent span checks. These do not compile an NPU binary."""
import copy
from pathlib import Path
import tempfile
import unittest

from fault_injection.native_adapters.attention28.fia import apply_tnd33_arch22_s2 as recipe
from fault_injection.native_adapters.attention28.fia import arch22_s2_geometry as geometry
from fault_injection.native_adapters.attention28.fia import binary_registration,build_fresh
from fault_injection.native_adapters.attention28.fia import compile_arch22_s2_matrix as matrix


def source_fixture():
    return {recipe.PATHS[0]:'namespace optiling {\n    bool splitKVFlag_ = false;\n}',
        recipe.PATHS[1]:'void CalcInnerSize(){\n'+recipe.SELECT_ANCHOR+'\n ORIGINAL_TAIL_AND_ALIGN();\n}\n'
            'void Split(){ CalcInnerSize(); SplitCore(); SetSplitOutput(); CalcMmResSize(); }\n'
            'void DoOpTiling(){ Split(); CalcWorkspaceSize(); GenTilingKey(); SaveToBuffer();\n'
            '    return ge::GRAPH_SUCCESS;\n}\n\n// remainder',
        recipe.normal.ROUTE:recipe.normal.ROUTE_ANCHOR}


def split_fixture(tile=512):
    # Synthetic CPU fixture, not a captured or compiled 256-tile result. Each
    # head is split over two cores and reduced once, including the one-token tail.
    row=dict(aic=20,kv_length=1025,s2=tile,usedCoreNum=16,numOfFdHead=8,
             usedVecNumOfFd=8,maxS2SplitNum=2)
    row.update({name:[0]*(40 if name in geometry.ARRAYS[-2:] else 20) for name in geometry.ARRAYS})
    for head in range(8):
        row['bN2End'][2*head:2*head+2]=[head,head+1]
        row['s2End'][2*head]=1
        row['s2SplitStartIdxOfCore'][2*head+1]=1
        row['bN2IdxOfFdHead'][head]=head
        row['s2SplitNumOfFdHead'][head]=2
        row['gS1SplitNumOfFdHead'][head]=1
        row['gS1LastPartSizeOfFdHead'][head]=2
        row['gS1IdxEndOfFdHead'][head]=head
    return row


class Arch22S2SourceContracts(unittest.TestCase):
    def test_only_s2_selection_is_inserted_before_original_dependent_arithmetic(self):
        original=source_fixture();after=recipe.patched(original)
        normal=recipe.normal.patched(original)
        restored=after[recipe.PATHS[1]].replace(recipe.SELECT,'').replace(recipe.CAPTURE,
            '    paged33_arch22_bridge::Capture(*fiaInfo_, context_, tilingData_, workspaceSize_, sInnerSize_);\n')
        self.assertEqual(restored,normal[recipe.PATHS[1]])
        self.assertLess(after[recipe.PATHS[1]].index(recipe.SELECT),after[recipe.PATHS[1]].index(recipe.SELECT_ANCHOR))
        self.assertEqual(after[recipe.normal.ROUTE],normal[recipe.normal.ROUTE])
        self.assertFalse(any('op_kernel' in key for key in after))
        for path in recipe.PATHS:
            bad=dict(original);bad[path]+=original[path]
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,'ambiguous'):recipe.patched(bad)
        with self.assertRaises(ValueError):recipe.patched(after)

    def test_source_closure_pins_actual_parser_split_and_both_kernel_consumers(self):
        self.assertEqual(len(recipe.SOURCE_SHA),11)
        for suffix in ('split_core.cpp','split_core.h','fia_kernel_nonquant.h','fia_block_vec_flashdecode.h',
                       'fused_infer_attention_score_tiling_info_parser.cpp'):
            self.assertEqual(sum(key.endswith('/'+suffix) for key in recipe.SOURCE_SHA),1)
        with self.assertRaisesRegex(RuntimeError,'PRISTINE_SOURCE_DIFFERS'):
            recipe.verify_source({key:b'changed' for key in recipe.SOURCE_SHA})
        header=Path(recipe.__file__).with_name('paged33_arch22_s2.h').read_text()
        # The CPU build below is the actual compiler gate. Here verify its
        # source closure contains the independent live observer ownership join.
        self.assertIn('dcat_attention_observer_scope_matches_v1(s.token.c_str())!=1',header)
        self.assertIn('s.phase=="fault"',header)
        self.assertIn('s.target=="model.layers.0.self_attn.attn"',header)

    def test_new_build_namespace_cannot_inherit_normal_qualification(self):
        builder=build_fresh.Builder.__new__(build_fresh.Builder);builder.report={}
        builder.receipt=dict(mechanism='33.paged_decode',variant=recipe.VARIANT,
            recipe_sha256=build_fresh.sha(recipe.__file__),
            normal_route_recipe_sha256=build_fresh.sha(recipe.normal.__file__),
            normal_route_preference_changed=True,selection_before_original_split=True,
            files=[dict(path='attention/common/op_host/arch22/'+name,
                after_sha256=build_fresh.sha(Path(recipe.__file__).with_name(name)))
                for name in ('paged33_arch22_s2.h','paged33_arch22_bridge.h')])
        for key in ('tensor_payload_mutated','kernel_source_modified','m_policy_changed',
                    'old_normal_certificate_reused','fault_execution_ready','stock_private_exact_proven','loading_allowed'):
            builder.receipt[key]=False
        builder.configure_policy(None)
        self.assertEqual(builder.vendor_name,'dcat_f33_arch22_s2')
        self.assertFalse(builder.report['fault_execution_ready'])
        for key in ('old_normal_certificate_reused','stock_private_exact_proven','loading_allowed'):
            builder.receipt[key]=True
            with self.subTest(key=key),self.assertRaisesRegex(RuntimeError,'FRESH_INDEPENDENT_SOURCE'):
                builder.configure_policy(None)
            builder.receipt[key]=False
        builder.receipt['variant']=recipe.normal.VARIANT
        with self.assertRaisesRegex(RuntimeError,'NORMAL_ONLY_REVIEWED_SOURCE'):builder.configure_policy(None)

    def test_exact_entry_inventory_retains_normal_warmups_but_not_old_fault_keys(self):
        self.assertEqual(set(recipe.KEYS),set(build_fresh.TND33_ARCH22_KEYS))
        self.assertEqual(set(recipe.KEYS),binary_registration.KEYS[recipe.VARIANT])
        self.assertFalse(set(recipe.KEYS)&set(build_fresh.TND33_FAI_FD_KEYS[6:]))
        with tempfile.TemporaryDirectory() as root:
            binary=Path(root)/'fixture.o';binary.write_bytes(b'CPU fixture only')
            metadata=dict(binFileName=binary_registration.BINARY,binFileSuffix='.o',
                sha256=binary_registration.sha(binary),coreType='MIX',taskRation='tilingKey',
                kernelList=[dict(tilingKey=key) for key in recipe.KEYS],supportInfo=dict(
                    simplifiedKeyMode=0,simplifiedKey=['test'],optionalInputMode='gen_placeholder',
                    optionalOutputMode='gen_placeholder',tilingKey=list(recipe.KEYS)))
            binary_registration.validate_metadata(metadata,binary,recipe.VARIANT)
            metadata['kernelList'].append(dict(tilingKey=6000000000010200103))
            with self.assertRaisesRegex(RuntimeError,'KEY_SET_DIFFERS'):
                binary_registration.validate_metadata(metadata,binary,recipe.VARIANT)

    def test_matrix_driver_uses_actual_split_source_and_not_a_python_selection_model(self):
        self.assertIn('SplitCore(cores,b,p,r)',matrix.PROGRAM)
        self.assertNotIn('Select(',matrix.PROGRAM)
        self.assertIn('for(int64_t length=1025;length<=2048;++length)',matrix.PROGRAM)
        self.assertIn('b.preToken=INT_MAX;b.nextToken=0',matrix.PROGRAM)

    def test_complete_producer_consumer_spans_and_workspace_are_recomputed(self):
        for tile in (512,256):
            checked=geometry.validate_split(split_fixture(tile))
            self.assertEqual(checked['covered_logical_kv'],1025)
            self.assertEqual(checked['covered_q_heads'],16)
            self.assertEqual(checked['fd_producer_parts'],16)
            self.assertEqual(checked['workspace_elements']['mm1ResSize'],2*tile)
            self.assertFalse(checked['qualification_granted'])

    def test_wrong_head_tail_fd_producer_slot_or_consumer_range_is_rejected(self):
        mutations=[('bN2End',15,7),('s2End',0,0),('gS1End',0,1),
            ('s2SplitNumOfFdHead',2,3),('s2SplitStartIdxOfCore',1,0),
            ('gS1LastPartSizeOfFdHead',2,1),('gS1IdxEndOfFdHead',4,3),
            ('gS1IdxEndOfFdHeadSplit',1,1),('bN2End',19,1)]
        for key,index,value in mutations:
            row=split_fixture(256);row[key][index]=value
            with self.subTest(key=key,index=index),self.assertRaises(ValueError):geometry.validate_split(row)
        row=split_fixture();row['maxS2SplitNum']=3
        with self.assertRaisesRegex(ValueError,'MAX_SPLITS'):geometry.validate_split(row)

    def test_missing_arrays_and_incomplete_cpu_platform_matrix_do_not_pass(self):
        row=split_fixture();row['s2End'].pop()
        with self.assertRaisesRegex(ValueError,'COMPLETE_ARRAY'):geometry.validate_split(row)
        with self.assertRaisesRegex(ValueError,'INCOMPLETE'):geometry.validate_matrix([split_fixture()])
        with self.assertRaisesRegex(ValueError,'DUPLICATE'):geometry.validate_matrix([split_fixture(),split_fixture()])


if __name__=='__main__':unittest.main()
