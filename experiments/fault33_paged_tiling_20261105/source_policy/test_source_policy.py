"""CPU-only patch anchor and ordering regression checks; no SDK/NPU needed."""
import json
from pathlib import Path
import sys
import unittest

import apply_policy
import offline_dependencies
from generation import layout


class PatchChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = (Path(__file__).parents[1] / "source_matching" / apply_policy.ENTRY).read_text(encoding="utf-8")

    def test_real_pinned_source_anchors_and_dependency_order(self):
        p = apply_policy.patch_text(self.original)
        begin = p.index("fault33pa::Begin(context)")
        dispatch = p.index("auto resultCode = FiaTilingRegistry::GetInstance().DoTilingImpl(context, nullptr)")
        end = p.index("fault33pa::End(context")
        self.assertLess(begin, dispatch)
        self.assertLess(dispatch, end)
        select = p.index("sInnerSize_ = fault33pa::Select")
        align = p.rindex("sInnerSize_ = (sInnerSize_ / blockSize_) * blockSize_;", 0, select)
        loop = p.index("sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;", select)
        ub = p.index("CheckUbSpace();", select)
        self.assertLess(align, select)
        self.assertLess(select, loop)
        self.assertLess(loop, ub)
        serialized = p.index("context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());")
        typed = p.index("fault33pa::Typed(obs.str()", serialized)
        self.assertLess(serialized, typed)
        self.assertNotIn("SetDataSize", p[select:loop])
        self.assertIn('obs << "\\"baseParams.batchSize\\":"', p)
        self.assertIn('obs << ",\\"bmm2TilingData.singleCoreK\\":"', p)

    def test_double_application_fails(self):
        with self.assertRaises(ValueError):
            apply_policy.patch_text(apply_policy.patch_text(self.original))

    def test_ambiguous_native_anchor_fails(self):
        a = "    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;\n"
        with self.assertRaises(ValueError):
            apply_policy.patch_text(self.original + a)

    def test_missing_dispatch_fails(self):
        with self.assertRaises(ValueError):
            apply_policy.patch_text(self.original.replace("DoTilingImpl(context, nullptr)", "DifferentDispatcher(context)"))

    def test_generation_paths_are_distinct_and_bounded(self):
        old=layout(Path('/audit'),'a1')
        new=layout(Path('/audit'),'a2')
        for key,path in new.items():
            self.assertNotEqual(old[key],path)
            self.assertTrue(path.is_relative_to(Path('/audit')))
        for bad in ('a0','a1000','../a2','a2;sh','a2/../a1',''):
            with self.assertRaises(ValueError):
                layout(Path('/audit'),bad)

    def test_offline_cmake_override_and_pin_are_explicit(self):
        build=Path(__file__).with_name('build_cpu.py').read_text(encoding='utf-8')
        self.assertIn("CANN_CMAKE_COMMIT = '7e5691ce110367662017d9d94114845a7d6a8d8a'",build)
        self.assertIn('-DFETCHCONTENT_SOURCE_DIR_CANN-CMAKE:PATH=/deps/cann-cmake',build)
        self.assertIn("'--network', 'none'",build)
        self.assertIn("'--no-hardlinks'",build)

    def test_scope_checks_every_actual_length_with_public_tensor_api(self):
        policy=Path(__file__).with_name('fault33_pa_policy.h').read_text(encoding='utf-8')
        self.assertIn('FixedActualLengths(const gert::Tensor* tensor',policy)
        self.assertIn('tensor->GetData<int64_t>()',policy)
        self.assertIn('tensor->GetShapeSize()',policy)
        self.assertIn('&&FixedActualLengths(c.actualSeqLengths.tensor,state.batch,state.length)',policy)
        selector=Path(__file__).with_name('fault33_pa_selector.h').read_text(encoding='utf-8')
        self.assertIn('count!=1&&count!=static_cast<int64_t>(batch)',selector)
        self.assertIn('for(int64_t i=0;i<count;++i)',selector)

    def test_reviewed_dependency_inventory_covers_dynamic_packager(self):
        fixtures=Path(__file__).parents[1]/'source_matching'
        ops=offline_dependencies.inventory(fixtures/'cmake_source')
        names={i['name'] for i in ops if i['command'].lower()=='fetchcontent_declare'}
        self.assertEqual(names,{'cann-cmake','opbase','ops-tensor'})
        cann=offline_dependencies.inventory(fixtures/'cann_cmake_source')
        self.assertTrue(any(i['command'].lower()=='fetchcontent_makeavailable' and i['name']=='${MAKESELF_NAME}' for i in cann))
        enabled={i['name'] for i in ops if i['command'].lower()=='add_cann_third_party'}
        self.assertTrue({'json','eigen','abseil-cpp','protobuf'}.issubset(enabled))

    def test_offline_configure_binds_native_externalproject_cache_interfaces(self):
        build=Path(__file__).with_name('build_cpu.py').read_text(encoding='utf-8')
        self.assertIn('-DCANN_3RD_LIB_PATH:PATH=/deps',build)
        self.assertIn('-DCANN_3RD_PKG_PATH:PATH=/deps/pkg',build)
        self.assertIn('-DFETCHCONTENT_SOURCE_DIR_OPBASE:PATH=/deps/opbase',build)
        self.assertIn('-DFETCHCONTENT_SOURCE_DIR_OPS-TENSOR:PATH=/deps/ops-tensor',build)
        self.assertIn('deps/\'abseil-cpp/abseil-cpp-20230802.1.tar.gz\'',build)
        self.assertIn('verify_offline_configure.py --generation @GEN@',build)

    def test_a4_guard_matches_ordinary_pa_internal_default_without_widening_shapes(self):
        patched=apply_policy.patch_text(self.original)
        self.assertIn('{"query_layout_bnsd", inputLayout_ == IfaLayout::BNSD}',patched)
        self.assertIn('{"kv_layout_ordinary_pa_default", inputKvLayout_ == IfaLayout::BSH_BSND}',patched)
        self.assertNotIn('&& inputKvLayout_ == IfaLayout::BNSD',patched)
        policy=Path(__file__).with_name('fault33_pa_policy.h').read_text(encoding='utf-8')
        self.assertIn('Shape(c.key.shape,{int64_t(state.batch)*count,8,128,128})',policy)
        self.assertIn('std::strcmp(c.layOut,"BNSD")==0',policy)
        self.assertIn('state.nativeChecksJson',policy)
        self.assertIn('state.nativeValuesJson',policy)
        impl=(Path(__file__).parents[1]/'source_matching/attention/incre_flash_attention/op_host/incre_flash_attention_tiling_impl.h').read_text(encoding='utf-8')
        self.assertIn('IfaLayout inputKvLayout_ = IfaLayout::BSH_BSND;',impl)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PatchChecks))
    print(json.dumps({"status": "PASS_CPU_SOURCE_PATCH_CHECK" if result.wasSuccessful() else "FAIL_CPU_SOURCE_PATCH_CHECK",
        "tests": result.testsRun, "npu_executed": False, "sdk_compilation_validated": False}))
    sys.exit(not result.wasSuccessful())
