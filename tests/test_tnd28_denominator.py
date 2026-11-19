"""CPU algebra/source/build contracts. Synthetic source is labelled, never an SDK proof."""
import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fault_injection.native_adapters.attention28.fia import apply_tnd28_denominator as recipe
from fault_injection.native_adapters.attention28.fia import tnd28_denominator_geometry as geometry
from fault_injection.native_adapters.attention28.fia import tnd28_denominator_build as helper
from fault_injection.native_adapters.attention28.fia import build_fresh as build
from fault_injection.native_adapters.attention28.fia import binary_registration as registration
from test_tnd28_prefill_normal import source_fixture as normal_fixture


def fixture():
    d=normal_fixture();h,c,_=recipe.normal.PATHS
    d[h]=d[h].replace('class X {};','class X {\n    bool splitKVFlag_ = false;\n};')
    d[c]+='\n    CalcMBaseSize();\n}\n    softmaxWithBrcbFlag_ = (mBaseSize_ <= M_BASE_SIZE_128);\n'
    d[c]+='    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||\n'
    d[recipe.ROUTE]+='\nFIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)'
    d[recipe.TRAIT]='    static constexpr bool enableTree = ENABLE_TREE;'
    d[recipe.KERNEL]='using namespace matmul;\n    info.actualSingleProcessSInnerSizeAlign =\n        AttentionCommon::Align((uint32_t)info.actualSingleProcessSInnerSize, (uint32_t)AttentionCommon::BYTE_BLOCK);'
    d[recipe.VECTOR]='before previous sum selection\n    if constexpr (SOFTMAX_WITH_BRC) {\n        SoftmaxFlashV2<SOFTMAX_TYPE after original recurrence'
    d[recipe.DISPATCH]='''using namespace AscendC;
#define INVOKE_FIA_OP_GENERAL_IMPL(...) FIAType<__VA_ARGS__>
#define FIA_COPY_TILING_DATA(...)
    TILING_KEY_IS(103000000030000003);
#elif TILING_KEY_VAR == 103000000030000003
    INVOKE_FIA_OP_GENERAL_IMPL(FiaKernelNonQuant, FiaBlockCubeNonQuantGqa, FiaBlockVecNonQuant, FiaBlockVecFlashDecode,
                               half, half, half, half, false, false, FIA_LAYOUT::TND, false, false, FIA_LAYOUT::TND,
                               false);
#endif
'''
    return d


def split_fixture(cores=16):
    r=dict(aic=cores,m=128,s2=128,usedCoreNum=cores,numOfFdHead=0,usedVecNumOfFd=0,maxS2SplitNum=0)
    r.update({n:[0]*(2*cores if n in geometry.ARRAYS[-2:] else cores) for n in geometry.ARRAYS})
    for i in range(cores):
        end=256*(i+1)//cores;r['bN2End'][i],r['gS1End'][i]=divmod(end,32)
    return r


class TndDenominatorAlgebra(unittest.TestCase):
    def test_keys_come_from_host_trait_digits_and_are_separate_from_bnsd(self):
        self.assertEqual(geometry.derived_key(m=512),recipe.NATIVE_KEY)
        self.assertEqual(geometry.derived_key(m=128),recipe.NATIVE_BRC_KEY)
        self.assertEqual(geometry.derived_key(m=128,causal=True),recipe.CONTROL_KEY)
        self.assertEqual(geometry.derived_key(m=128,causal=True,replay=True),recipe.REPLAY_KEY)
        self.assertEqual(build.TND28_REPLAY_KEYS,list(recipe.KEYS))
        self.assertEqual(registration.KEYS[recipe.VARIANT],set(recipe.KEYS))
        self.assertNotIn(119000000000400000,recipe.KEYS)
        for kwargs in (dict(m=128,replay=True),dict(m=512,causal=True),dict(m=64)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):geometry.derived_key(**kwargs)

    def test_actual_second_block_sum_replay_is_not_output_scaling(self):
        ratios=[]
        for offset in (-8.,0.,8.):
            scores=[.1*math.sin(i) for i in range(128)]+[offset+.2*math.cos(i) for i in range(128)]
            values=[1.+i/300 for i in range(256)]
            normal=geometry.online_scalar(scores,values)
            replay=geometry.online_scalar(scores,values,replay=True)
            maximum=max(scores);numerator=sum(math.exp(x-maximum)*v for x,v in zip(scores,values))
            mass=sum(math.exp(x-maximum) for x in scores);first=sum(math.exp(x-maximum) for x in scores[:128])
            self.assertAlmostEqual(normal['value'],numerator/mass,places=12)
            self.assertAlmostEqual(replay['value'],numerator/(mass+first),places=12)
            self.assertEqual(normal['numerator'],replay['numerator'])
            self.assertEqual(normal['maximum'],replay['maximum'])
            self.assertEqual(replay['replays'],1);ratios.append(replay['value']/normal['value'])
        self.assertGreater(max(ratios)-min(ratios),.49)

    def test_all_causal_rows_and_changing_maxima_match_independent_prefix_oracle(self):
        scores=[.003*i+math.sin(i*.1) for i in range(2048)]
        values=[math.cos(i*.07) for i in range(2048)]
        # Independent full-prefix sums under one fixed global maximum;
        # every online row still computes its own successive block maxima.
        weights=[math.exp(x-max(scores)) for x in scores]
        first_mass=sum(weights[:128]);den=num=0.
        for length in range(1,2049):
            a=scores[:length];v=values[:length]
            den+=weights[length-1];num+=weights[length-1]*values[length-1]
            first=first_mass if length>128 else 0
            r=geometry.online_scalar(a,v,replay=True)
            self.assertAlmostEqual(r['value'],num/(den+first),places=12)
            self.assertEqual(r['replays'],int(length>128))
        self.assertEqual(geometry.inner_width(2048,128,8),64)
        self.assertEqual(geometry.inner_width(2048,128,0),128)

    def test_whole_row_partition_and_causal_pair_counts(self):
        r=geometry.validate_split(split_fixture())
        self.assertEqual(r['covered_unmasked_pairs'],33570816)
        self.assertEqual(r['replay_eligible_rows'],30720)
        self.assertFalse(r['qualification_granted'])
        rows=[split_fixture(c) for c in range(16,27)]
        rows[3]['numOfFdHead']=1
        result=geometry.validate_matrix(rows)
        self.assertEqual(result['refused_profiles'],[dict(aic=19,reason='TND28_SPLITKV_REFUSED')])
        self.assertNotIn(19,result['supported_core_profiles'])
        self.assertEqual(len(result['supported_core_profiles']),10)

    def test_partial_missing_fd_unused_and_repeated_core_rows_refuse(self):
        variants=[('numOfFdHead',1),('numOfFdHead',False),('usedVecNumOfFd',1),('usedVecNumOfFd',False),
            ('maxS2SplitNum',2),('usedCoreNum',17),('m',512),('s2',512)]
        for field,value in variants:
            row=split_fixture();row[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):geometry.validate_split(row)
        for name in geometry.ARRAYS:
            row=split_fixture();row[name][0]=-1
            with self.subTest(name=name),self.assertRaises(ValueError):geometry.validate_split(row)
        for name in ('s2End','s2SplitStartIdxOfCore','gS1IdxEndOfFdHead'):
            row=split_fixture();row[name][0]=1
            with self.subTest(name=name),self.assertRaises(ValueError):geometry.validate_split(row)
        row=split_fixture();row['gS1End'][3]=row['gS1End'][2];row['bN2End'][3]=row['bN2End'][2]
        with self.assertRaises(ValueError):geometry.validate_split(row)
        for rows in ([split_fixture()],[split_fixture(c) for c in range(16,27)]+[split_fixture()]):
            with self.assertRaises(ValueError):geometry.validate_matrix(rows)


class TndDenominatorSource(unittest.TestCase):
    def test_actual_recurrence_only_mutates_old_sum_not_output_or_numerator(self):
        original=fixture();changed=recipe.patched(original)
        self.assertEqual(changed[recipe.VECTOR].replace(recipe.REPLAY_CODE,''),original[recipe.VECTOR])
        self.assertIn('Muls(inSumTensor, inSumTensor, COMPUTE_T(2)',changed[recipe.VECTOR])
        self.assertIn('!info.isFirstSInnerLoop && info.s2Idx == 1',changed[recipe.VECTOR])
        self.assertNotIn('Muls(mmResUb',changed[recipe.VECTOR])
        self.assertIn('Tnd28ReplayType<FIAType<__VA_ARGS__>, false>',changed[recipe.DISPATCH])
        self.assertIn('Tnd28ReplayType<FIAType<__VA_ARGS__>, true>',changed[recipe.DISPATCH])
        self.assertEqual(changed[recipe.DISPATCH].count('FIA_LAYOUT::TND,\n                               true);'),3)
        for key in (recipe.NATIVE_BRC_KEY,recipe.CONTROL_KEY,recipe.REPLAY_KEY):
            self.assertIn(f'REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_{key},',changed[recipe.ROUTE])
            self.assertEqual(changed[recipe.DISPATCH].count(f'#elif TILING_KEY_VAR == {key}\n'),1)
        self.assertNotIn('BNSD',changed[recipe.DISPATCH])

    def test_host_selects_before_original_split_and_never_rewrites_fd(self):
        original=fixture();changed=recipe.patched(original)
        host=changed[recipe.normal.PATHS[1]]
        self.assertIn('tnd28_denominator::Select',host)
        self.assertIn('tnd28_denominator::ExtendKey(tnd28Replay_, tilingKey_, tilingData_)',host)
        self.assertNotIn('set_numOfFdHead',host)
        self.assertIn('!tnd28_denominator::UseRoute(context)',changed[recipe.ROUTE])
        contract=Path(recipe.__file__).with_name('tnd28_denominator_contract.h').read_text()
        self.assertIn('TND28_REPLAY_ACTUAL_PREPARED_INPUT_BINDING_MISSING',contract)
        self.assertIn('!dcat_tnd28_denominator_current_input_v1',contract)
        self.assertIn('consumed.insert(s.token).second',contract)

    def test_tnd_width_is_separate_and_every_other_kernel_byte_unchanged(self):
        original=fixture();changed=recipe.patched(original)
        restored=changed[recipe.KERNEL].replace('#include "tnd28_causal_tile.h"\n\n','').replace(recipe.CAUSAL_CODE,'')
        self.assertEqual(restored,original[recipe.KERNEL])
        self.assertIn('constInfo.gSize',recipe.CAUSAL_CODE)
        self.assertNotIn('% s1',Path(recipe.__file__).with_name('tnd28_causal_tile.h').read_text())

    def test_missing_ambiguous_or_repeated_patch_is_refused(self):
        original=fixture()
        for name in original:
            for value in ('unreviewed source',original[name]+original[name]):
                altered=dict(original);altered[name]=value
                with self.subTest(name=name),self.assertRaises((ValueError,KeyError)):recipe.patched(altered)
        with self.assertRaises(ValueError):recipe.patched(recipe.patched(original))


class TndDenominatorBuild(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.source=self.root/'source';self.source.mkdir();self.sdk=self.root/'cann-8.5.1';self.sdk.mkdir()
        self.interfaces=self.root/'interfaces.json';self.interfaces.write_bytes(b'CPU_INTERFACE_FIXTURE')
        self.raw={p:('CPU_SOURCE_FIXTURE:'+p).encode() for p in (*recipe.SOURCE_SHA,*recipe.EXTRA_UNCHANGED)}
        self.raw.update({p:s.encode() for p,s in fixture().items()})
        projection=json.loads((Path(__file__).parent/'fixtures/tnd28_source109_public_layout.json').read_bytes())['projection']
        self.raw[recipe.EXTRA_UNCHANGED[0]]=projection.encode()
        for owner,name,value in ((recipe,'SOURCE_SHA',{p:hashlib.sha256(self.raw[p]).hexdigest() for p in recipe.SOURCE_SHA}),
            (helper,'recipe',lambda here:recipe),(build,'git_blob',lambda source,p:self.raw[p]),
            (build,'TND28_SDK_INTERFACE_SHA256',build.sha(self.interfaces))):
            p=patch.object(owner,name,value);p.start();self.addCleanup(p.stop)
        self.final,_,projection,fields=recipe.materials(self.raw)
        for name,b in self.final.items():
            p=self.source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        self.receipt=recipe.receipt(self.raw,self.final,projection,fields)
        args=SimpleNamespace(work=self.root/'build',python=Path('python'),opdef_compat='ascend910b-cann851',
            sdk_interface_manifest=self.interfaces,sdk_interface_sha256=build.sha(self.interfaces))
        self.b=build.Builder(args);self.b.sdk=self.sdk;self.b.receipt=self.receipt

    def test_exact_new_source_reconstruction_does_not_grant_fault(self):
        self.b.configure_policy(self.source)
        self.assertEqual(self.b.variant,recipe.VARIANT);self.assertEqual(self.b.vendor_name,recipe.VENDOR)
        self.assertTrue(self.b.report['tnd28_source_closure_reconstructed'])
        for key in build.TND28_DENIED:self.assertIs(self.b.report[key],False)

    def test_receipt_assertions_and_old_normal_or_bnsd_scope_do_not_substitute(self):
        for field,value in (('fault_execution_ready',True),('stock_private_exact_proven',True),
                ('normal_key','103000000000000000'),('compile_keys',[str(recipe.REPLAY_KEY)]),
                ('vendor','dcat_f28_replay'),('recipe_sha256','0'*64)):
            self.b.receipt=copy.deepcopy(self.receipt);self.b.receipt[field]=value
            with self.subTest(field=field),self.assertRaises(RuntimeError):self.b.configure_policy(self.source)

    def test_source_change_even_with_new_forged_manifest_is_rejected(self):
        name=recipe.base.HOST+'tnd28_denominator_policy.h';p=self.source/name
        p.write_bytes(p.read_bytes()+b'// changed\n')
        with self.assertRaisesRegex(RuntimeError,'RECONSTRUCTED_BYTES_DIFFER'):self.b.configure_policy(self.source)

    def test_unknown_sdk_or_missing_interface_refuses_before_source_copy(self):
        self.b.sdk=self.root/'cann-9.1.0'
        with self.assertRaisesRegex(RuntimeError,'EXACT_CANN851'):self.b.configure_policy(self.source)


if __name__=='__main__':unittest.main()
