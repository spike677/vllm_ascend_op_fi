#!/usr/bin/env python3
"""Patch a pristine pinned IFA private checkout; does not build or execute."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "c77c0829f1f0f0b8f882db1d9b051a758b605472"
ENTRY = "attention/incre_flash_attention/op_host/incre_flash_attention_tiling.cpp"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact anchor; got {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


def typed_capture():
    # Getter names come from this pinned public host tiling definition. This
    # projection is also decoded independently with the public serializer layout.
    parts = {
        "baseParams": "batchSize seqSize qSeqSize headSize headSizeV blockSize maxBlockNumPerBatch maxBlockNumPerSeq scaleValue kvHeadNum headNumRatio qHeadNum nNumOfQInOneGroup batchContinuousFlag actualLenDims attenMaskFlag totalBlockNum paKvShapeType sparseMode slidingFlag",
        "splitKVParams": "s2 sInnerLoopSize accumOutSize logSumExpSize",
        "increFlashAttentionSingleCoreParams": "sInnerLoopTimes singleProcessSInnerSize singleProcessSInnerSizeTail usedCoreNum formerCoreNum blockSplitBn2Range tailSplitedBatchRange groupSplitSize s1SplitSize",
        "increFlashAttentionSingleCoreTensorSize": "mmResUbSize bmm2ResUbSize",
        "bmm1TilingData": "M N Ka Kb singleCoreM singleCoreN singleCoreK baseM baseN baseK",
        "bmm2TilingData": "M N Ka Kb singleCoreM singleCoreN singleCoreK baseM baseN baseK",
        "softmaxFlashTilingData": "srcM srcK srcSize outMaxM outMaxK outMaxSize splitM splitK splitSize reduceM reduceK reduceSize rangeM tailM tailSplitSize tailReduceSize",
    }
    fields = []
    for section, names in parts.items():
        for name in names.split():
            fields.append(f'            obs << "{"," if fields else ""}\\"{section}.{name}\\":" << base.{section}.get_{name}();')
    return """        if (fault33pa::state.active) {
            try {
                auto &base = tilingData.tilingBase;
                std::ostringstream obs;
                obs << std::setprecision(17) << "{";
""" + "\n".join(fields) + """
                obs << ",\\"coreSidxEnd\\":[";
                auto *starts = base.increFlashAttentionCoreParams.get_coreSidxEnd();
                for (size_t i = 0; i < 50; ++i) { if (i) obs << ','; obs << starts[i]; }
                obs << "]}";
                fault33pa::Typed(obs.str(), ifaContext_->workSpaces ? ifaContext_->workSpaces[0] : 0, context);
            } catch (...) { std::fprintf(stderr, "FAULT33_PA_TYPED_ERROR\\n"); }
        }
"""


def patch_text(text):
    text = replace_once(text, '#include "../../common/op_host/fia_tiling_templates_registry.h"',
        '#include "../../common/op_host/fia_tiling_templates_registry.h"\n#include "fault33_pa_policy.h"')
    text = replace_once(text,
        '    auto resultCode = FiaTilingRegistry::GetInstance().DoTilingImpl(context, nullptr);\n    return resultCode;',
        '    fault33pa::Begin(context);\n'
        '    auto resultCode = FiaTilingRegistry::GetInstance().DoTilingImpl(context, nullptr);\n'
        '    fault33pa::End(context, static_cast<uint32_t>(resultCode));\n'
        '    return resultCode;')
    anchor = '    sInnerLoopTimes_ = (seqSize + sInnerSize_ - 1U) / sInnerSize_;\n'
    selection = """    // Fault 33 controlled policy: select before all dependent native tiling.
    // No tensor payload or serialized tiling is edited by this hook.
    // In this exact ordinary PA path ProcessPageAttentionFlag sets inputLayout_
    // from rank-4 KV but leaves inputKvLayout_ at its BSH_BSND default. Public
    // BNSD attr and rank-4 KV shapes are checked separately by FixedInputs.
    const bool fault33NativeScope = fault33pa::NativeScope({
        {"soc_910b", socVersion_ == IfaSocVersion::SOC_ASCEND_910B},
        {"page_attention", pageAttentionFlag_}, {"no_split_kv", !splitKVFlag_},
        {"runtime_tiling", !isWorkspace_}, {"no_antiquant", !antiQuantFlag_},
        {"no_quant", !quantFlag_}, {"no_rope", !ropeFlag_},
        {"no_gqa_kv_nz", !gqaKvNZFlag_}, {"no_gqa_mtp", !gqaMtpFlag_},
        {"no_sys_prefix", !sysPrefixFlag_}, {"no_sliding", !slidingFlag_},
        {"no_balance", !balanceModeFlag_},
        {"query_layout_bnsd", inputLayout_ == IfaLayout::BNSD},
        {"kv_layout_ordinary_pa_default", inputKvLayout_ == IfaLayout::BSH_BSND},
        {"query_fp16", inputQType_ == ge::DT_FLOAT16},
        {"kv_fp16", inputKvType_ == ge::DT_FLOAT16},
        {"output_fp16", outputType_ == ge::DT_FLOAT16},
        {"inner_precise_0", innerPrecise_ == 0}, {"query_seq_1", qSeqSize_ == 1},
        {"query_heads_16", numHeads_ == 16}, {"kv_heads_8", numKvHeads_ == 8},
        {"head_dim_128", headDim_ == 128}, {"head_dim_v_128", headDimV_ == 128},
        {"block_size_128", blockSize_ == 128}
    }, {
        {"soc_version", static_cast<int64_t>(socVersion_)},
        {"input_layout", static_cast<int64_t>(inputLayout_)},
        {"input_kv_layout", static_cast<int64_t>(inputKvLayout_)},
        {"query_dtype", static_cast<int64_t>(inputQType_)},
        {"kv_dtype", static_cast<int64_t>(inputKvType_)},
        {"output_dtype", static_cast<int64_t>(outputType_)},
        {"head_dim", headDim_}, {"head_dim_v", headDimV_},
        {"query_seq", qSeqSize_}, {"query_heads", numHeads_},
        {"kv_heads", numKvHeads_}, {"inner_precise", innerPrecise_}
    });
    sInnerSize_ = fault33pa::Select(sInnerSize_, seqSize, *ifaContext_, fault33NativeScope,
        batchSize_, maxActualseq_, blockSize_, static_cast<uint32_t>(perfMode_));

"""
    text = replace_once(text, anchor, selection + anchor)
    anchor = '        context.GetRawTilingData()->SetDataSize(tilingData.GetDataSize());\n'
    text = replace_once(text, anchor, anchor + typed_capture())
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    src = args.source.resolve()
    head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=all"], text=True)
    if head != COMMIT or diff.strip():
        raise ValueError("Requires pristine pinned source including no untracked files")
    dest = args.output.resolve()
    if dest == src or src in dest.parents:
        raise ValueError("Evidence output must be outside the pristine source checkout")
    if dest.exists():
        raise ValueError("Refuse an existing evidence generation")
    entry = src / ENTRY
    original = entry.read_bytes()
    patched = patch_text(original.decode("utf-8")).encode("utf-8")
    headers = ["fault33_pa_selector.h", "fault33_pa_policy.h"]
    materials = [(entry, original, patched)]
    for name in headers:
        target = entry.with_name(name)
        if target.exists():
            raise ValueError("Patch already applied")
        materials.append((target, None, Path(__file__).with_name(name).read_bytes()))
    # Every anchor and precondition is checked before the first mutation.
    for target, _, content in materials:
        target.write_bytes(content)
    patch = subprocess.check_output(["git", "-C", str(src), "diff", "--", ENTRY])
    dest.mkdir(parents=True, exist_ok=False)
    (dest / "official_sources.patch").write_bytes(patch)
    receipt = {
        "schema": "fault33.pa.policy_source.v1", "source_commit": head,
        "source_release": "v9.1.0", "status": "PATCHED_NOT_BUILT_NOT_NPU_VALIDATED",
        "policy_kind": "controlled_current_source_host_selector_not_historical_stock_version",
        "files": [{"path": str(p.relative_to(src)).replace("\\", "/"),
            "before_sha256": hashlib.sha256(before).hexdigest() if before is not None else None,
            "after_sha256": hashlib.sha256(after).hexdigest(), "bytes": len(after)}
            for p, before, after in materials],
        "patch_sha256": hashlib.sha256(patch).hexdigest(), "kernel_source_modified": False,
        "off_behavior": "Same binary; FAULT33_PA_MODE=off preserves native SInner selection",
        "raw_observation": "Final public host TilingContext raw; device kernel argument not directly observed",
    }
    (dest / "policy_source_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
