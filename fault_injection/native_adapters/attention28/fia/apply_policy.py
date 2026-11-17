"""Derive either legal coupled tiling or denominator replay from pristine source.

All host dependents must subsequently be rebuilt. This removes the historical
dependency on a previous experiment's object files. It does not compile, install
or qualify a vendor. Only the exact pinned source commit is currently admitted.
"""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

_spec = importlib.util.spec_from_file_location('fault28_base_policy', Path(__file__).with_name('base_policy.py'))
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

VARIANTS = ('legal_tiling', 'denominator_replay')
VECTOR = base.KERNEL + 'arch22/fia_block_vec_nonquant.h'
REPLAY_KEY = 119000000000400000
REPLAY_CODE = '''    // Controlled arithmetic-error variant, distinct from legal tiling.
    // Replay the old denominator mass only on the second in-core S2 block.
    if constexpr (FIAT::fault28DenominatorReplay) {
        static_assert(SOFTMAX_WITH_BRC && !FLASH_DECODE, "replay requires no-SplitKV BRC template");
        if (!info.isFirstSInnerLoop && info.s2Idx == 1) {
            const uint32_t replayCount = dealRowCount * this->brcbNum;
            PipeBarrier<PIPE_V>();
            Muls(inSumTensor, inSumTensor, COMPUTE_T(2), replayCount);
            PipeBarrier<PIPE_V>();
        }
    }
'''


def patched(original, variant):
    if variant not in VARIANTS:
        raise ValueError('Explicit legal_tiling or denominator_replay variant required')
    result = base.patched(original)
    host = base.HOST + 'fia_tiling_nonquant.cpp'
    result[host] = base.once(result[host], '    CalcMBaseSize();\n}', '''    CalcMBaseSize();
    // Recompute the dependent loop and tail before later native host stages.
    if (fault28_fia::Coupled(fault28State_)) {
        sInnerSize_ = fault28State_.selectedM;
        sInnerLoopTimes_ = (s2Size + sInnerSize_ - 1U) / sInnerSize_;
        sInnerSizeTail_ = s2Size - (sInnerLoopTimes_ - 1U) * sInnerSize_;
        sInnerSizeAlign_ = Align(sInnerSize_, BYTE_BLOCK);
    }
}''')
    if variant == 'legal_tiling':
        return result
    result[host] = base.once(result[host], '    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||', '''    if (fault28_fia::Replay(fault28State_)) {
        if (tilingKey_ != 109000000000400000ULL || mBaseSize_ != 128 || sInnerSize_ != 128) {
            OP_LOGE(fiaInfo_->opName, "Replay requires the M128/S2=128 BRC template.");
            return ge::GRAPH_FAILED;
        }
        tilingKey_ = 119000000000400000ULL;
    }
    if ((SetNumBlocks(numBlocks_) != ge::GRAPH_SUCCESS) ||''')
    trait = base.KERNEL + 'fia_public_define.h'
    result[trait] = base.once(result[trait], 'static constexpr bool fault28CausalPrune = false;',
        'static constexpr bool fault28CausalPrune = false;\n    static constexpr bool fault28DenominatorReplay = false;')
    anchor = '    if constexpr (SOFTMAX_WITH_BRC) {\n        SoftmaxFlashV2<SOFTMAX_TYPE'
    result[VECTOR] = base.once(result[VECTOR], anchor, REPLAY_CODE + anchor)
    dispatch = base.FIA + 'op_kernel/fused_infer_attention_score_v3.cpp'
    text = result[dispatch]
    trait_line = 'template <class Base> struct Fault28CausalFIAType : Base { static constexpr bool fault28CausalPrune = true; };'
    text = base.once(text, trait_line, trait_line + '''
template <class Base> struct Fault28ReplayFIAType : Fault28CausalFIAType<Base> {
    static constexpr bool fault28DenominatorReplay = true;
};''')
    start = text.index('#define INVOKE_FAULT28_FIA_PRUNED(')
    end = text.index('\n#define FIA_COPY_TILING_DATA', start)
    replay_macro = text[start:end].replace('INVOKE_FAULT28_FIA_PRUNED', 'INVOKE_FAULT28_FIA_REPLAY').replace('Fault28CausalFIAType', 'Fault28ReplayFIAType')
    text = text[:end] + '\n' + replay_macro + text[end:]
    text = base.once(text, '    TILING_KEY_IS(109000000000400000);',
                    '    TILING_KEY_IS(109000000000400000);\n    TILING_KEY_IS(119000000000400000);')
    start = text.index('#elif TILING_KEY_VAR == 109000000000400000')
    end = text.index('#elif ', start + 6)
    replay_branch = text[start:end].replace('109000000000400000', str(REPLAY_KEY)).replace('INVOKE_FAULT28_FIA_PRUNED', 'INVOKE_FAULT28_FIA_REPLAY')
    result[dispatch] = text[:end] + replay_branch + text[end:]
    register = base.FIA + 'op_host/arch22/fused_infer_attention_score_tiling_v3.cpp'
    anchor = 'FIA_EXTERN_C ge::graphStatus TilingFusedInferAttentionScoreV3(gert::TilingContext *context)'
    result[register] = base.once(result[register], anchor,
        f'REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_{REPLAY_KEY}, FusedInferAttentionScoreTilingData)\n' + anchor)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=VARIANTS, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(strict=True), args.output.resolve()
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=all'], text=True).strip()
    if commit != base.COMMIT or dirty or output.exists() or source == output or source in output.parents:
        raise ValueError('Pristine pinned source and fresh external evidence required')
    paths = list(base.PATHS) + ([VECTOR] if args.variant == 'denominator_replay' else [])
    raw = {name: (source / name).read_bytes() for name in paths}
    original = {name: value.decode('utf-8') for name, value in raw.items()}
    updated = patched(original, args.variant)
    typed, projection, fields = base.typed_materials((source / (base.FIA + 'op_host/fused_infer_attention_score_tiling.h')).read_text(encoding='utf-8'))
    policy = Path(__file__).with_name('fault28_fia_policy.h').read_text(encoding='utf-8')
    if args.variant == 'legal_tiling':
        policy = base.once(policy, ', "batch_coupled_replay_m128", "coupled_replay_m128"', '')
    additions = {base.HOST + 'fault28_fia_policy.h': policy.encode(),
                 base.HOST + 'fault28_fia_typed.h': typed.encode(),
                 base.KERNEL + 'arch22/fault28_causal_tile.h': Path(__file__).with_name('fault28_causal_tile.h').read_bytes()}
    if any((source / name).exists() for name in additions):
        raise ValueError('Refuse repeated patch')
    materials = {name: value.encode() for name, value in updated.items()}
    materials.update(additions)
    # Every source anchor is checked before any mutation. The source checkout
    # must be task-private; build all dependents after its State layout changes.
    for name, value in materials.items():
        (source / name).write_bytes(value)
    output.mkdir(parents=True, exist_ok=False)
    diff = ''.join(''.join(difflib.unified_diff(original[name].splitlines(True), updated[name].splitlines(True),
                    fromfile='a/' + name, tofile='b/' + name)) for name in paths)
    (output / 'official_sources.patch').write_text(diff, encoding='utf-8')
    (output / 'fia_schema_projection.h').write_text(projection, encoding='utf-8')
    (output / 'typed_fields.json').write_text(json.dumps(fields, indent=2) + '\n', encoding='utf-8')
    receipt = dict(schema='demoncat.fia_policy_source.v1', mechanism='28.fia.' + args.variant,
        source_commit=commit, status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
        actual_launch_verified=False, baseline_key='109000000000400000' if args.variant == 'denominator_replay' else '109000000000000000',
        fault_key=str(REPLAY_KEY) if args.variant == 'denominator_replay' else '109000000000400000',
        precision_types_changed=False, numerator_or_output_path_changed=False,
        historical_default_policy_reproduced=False, requires_complete_host_and_kernel_rebuild=True,
        files=[dict(path=name, before_sha256=hashlib.sha256(raw[name]).hexdigest() if name in raw else None,
                    after_sha256=hashlib.sha256(value).hexdigest(), bytes=len(value)) for name, value in materials.items()])
    (output / 'policy_source_manifest.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
