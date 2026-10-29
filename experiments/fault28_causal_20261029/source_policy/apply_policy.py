#!/usr/bin/env python3
"""Apply experiment patch once to a pristine pinned private checkout; no build/run."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "c77c0829f1f0f0b8f882db1d9b051a758b605472"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one exact patch anchor, found {text.count(old)}: {old[:100]}")
    return text.replace(old, new, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    src = args.source.resolve()
    head = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    if head != COMMIT or subprocess.check_output(["git", "-C", str(src), "diff", "--name-only"], text=True).strip():
        raise ValueError("Requires pristine pinned source")
    host = src / "attention/flash_attention_score/op_host"
    entry = host / "flash_attention_score_tiling.cpp"
    general = host / "arch22/flash_attention_score_tiling_general.cpp"
    header = host / "fault28_causal_policy.h"
    if header.exists():
        raise ValueError("Patch already applied")
    before = {str(p.relative_to(src)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (entry, general)}
    e = entry.read_text()
    e = replace_once(e, '#include <queue>', '#include <queue>\n#include "fault28_causal_policy.h"')
    e = replace_once(e, '    auto resultCode = TilingRegistryArch::GetInstance().DoTilingImpl(context);\n    return resultCode;',
                    '    fault28causal::Begin(context);\n    auto resultCode = TilingRegistryArch::GetInstance().DoTilingImpl(context);\n    fault28causal::End(context, static_cast<uint32_t>(resultCode));\n    return resultCode;')
    g = general.read_text()
    # Keep the shared header after source includes, before any namespace.
    anchor = 'namespace optiling {'
    g = replace_once(g, anchor, '#include "../fault28_causal_policy.h"\n\n' + anchor)
    cls_start = g.index('class FlashAttentionScoreTilingS1s2Bn2gs1 :')
    cls_end = g.index('class FlashAttentionScoreTilingS1s2Bn2gs1SameAB :', cls_start)
    part = g[cls_start:cls_end]
    part = replace_once(part,
        '        s2BasicBlock = std::min(128L, alignedS2);\n',
        '        s2BasicBlock = std::min(128L, alignedS2);\n'
        '        s1BasicBlock = fault28causal::SelectM(context_, s1BasicBlock);\n')
    anchor = '            nRatio = 6L;\n        }\n    }\n'
    part = replace_once(part, anchor,
        '            nRatio = 6L;\n        }\n'
        '        nRatio = fault28causal::SelectNRatio(context_, nRatio);\n'
        '    }\n')
    part = replace_once(part, '        SetEnableL1Reuse();\n',
        '        SetEnableL1Reuse();\n'
        '        fault28causal::L1Reuse(enableL1Reuse);\n')
    g = g[:cls_start] + part + g[cls_end:]
    # Observe from the existing workspace pointer after all native modifications.
    # Do not add a call to the mutating GetWorkspaceSizes API.
    anchor = '    OP_LOGD(opName, "[%s] tiling data:%s", templateName, GetTilingDataDebugStr().c_str());\n'
    typed = '''    if (fault28causal::state.active) {
        try {
            const auto &c = tilingData->coreParams;
            auto &m = tilingData->multiCoreParams;
            std::ostringstream obs;
            std::ostringstream starts;
            starts << "[";
            for (size_t i = 0; i < 48; ++i) { if (i) starts << ","; starts << m.get_sparseStartIdx()[i]; }
            starts << "]";
            obs << "{\\"template\\":" << fault28causal::Quote(templateName)
                << ",\\"m\\":" << c.get_s1BaseSize()
                << ",\\"s2_base\\":" << c.get_s2BaseSize()
                << ",\\"n_ratio\\":" << c.get_nRatio()
                << ",\\"s2_group\\":" << (c.get_s2BaseSize() * c.get_nRatio())
                << ",\\"s1_outer\\":" << c.get_s1OuterSize()
                << ",\\"s2_outer\\":" << c.get_s2OuterSize()
                << ",\\"s1_tail\\":" << c.get_s1BaseTailSize()
                << ",\\"s2_tail\\":" << c.get_s2BaseTailSize()
                << ",\\"core_num\\":" << m.get_coreNum()
                << ",\\"total_query_tasks\\":" << m.get_totalSize()
                << ",\\"split_factor\\":" << m.get_splitFactorSize()
                << ",\\"split_factor_tail\\":" << m.get_splitFactorTailSize()
                << ",\\\"sparse_start_idx\\\":" << starts.str()
                << ",\\\"enable_l1_reuse\\\":" << (fault28causal::state.l1Known ? (fault28causal::state.l1Reuse ? "true" : "false") : "null")
                << ",\\\"sparse_type\\\":" << static_cast<int>(tilingData->inputParams.get_sparseType())
                << ",\\\"s1_sparse_valid_size\\\":" << c.get_s1SparseValidSize()
                << ",\\\"s2_sparse_valid_size\\\":" << c.get_s2SparseValidSize()
                << ",\\"b\\":" << bSize << ",\\"n1\\":" << n1Size << ",\\"n2\\":" << n2Size
                << ",\\"g\\":" << gSize << ",\\"s1\\":" << s1Size << ",\\"s2\\":" << s2Size
                << ",\\"d\\":" << dSize << ",\\"dv\\":" << d2Size
                << ",\\"dtype_enum\\":" << static_cast<int>(inputDtype)
                << ",\\"layout\\":" << fault28causal::Quote(inputLayout)
                << ",\\"calc_bytes\\":" << calcTypeSize
                << ",\\"high_precision\\":" << (isHighPercision ? "true" : "false")
                << ",\\"impl_mode\\":" << static_cast<int>(implMode)
                << ",\\"sparse_mode\\":" << sparseMode
                << ",\\"has_mask\\":" << static_cast<int>(attenMaskExistFlag)
                << ",\\"has_dropout\\":" << static_cast<int>(dropMaskExistFlag)
                << ",\\"has_pse\\":" << static_cast<int>(pseExistFlag)
                << ",\\"keep_prob\\":" << keepProb
                << ",\\"scale\\":" << std::setprecision(10) << scaleValue
                << ",\\"bmm1\\":" << fault28causal::CubeJson(tilingData->bmm1TilingData)
                << ",\\"bmm2\\":" << fault28causal::CubeJson(tilingData->bmm2TilingData) << "}";
            fault28causal::Typed(obs.str(), workspaces[0]);
        } catch (...) { std::fprintf(stderr, "FAULT28_CAUSAL_TYPED_ERROR\\n"); }
    }
'''
    g = replace_once(g, anchor, anchor + typed)
    # All text edits validated before writing any target.
    entry.write_text(e)
    general.write_text(g)
    header.write_bytes(Path(__file__).with_name('fault28_causal_policy.h').read_bytes())
    patch = subprocess.check_output(["git", "-C", str(src), "diff", "--", str(entry.relative_to(src)), str(general.relative_to(src))])
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'official_sources.patch').write_bytes(patch)
    changed = []
    for p in (entry, general, header):
        changed.append({"path": str(p.relative_to(src)), "before_sha256": before.get(str(p.relative_to(src))),
                        "after_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size})
    receipt = {"schema": "fault28.causal.policy_source.v1", "source_commit": head,
               "status": "PATCHED_NOT_BUILT_NOT_NPU_VALIDATED", "files": changed,
               "patch_sha256": hashlib.sha256(patch).hexdigest(),
               "kernel_source_modified": False, "off_restoration": "FAULT28_CAUSAL_PROFILE=base or absent/mismatched scope preserves original selector"}
    (args.output / 'policy_source_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
