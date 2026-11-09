"""Synthetic CPU regressions; they establish parser behavior, not NPU coverage."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_launch_census import census, sha


def fixture(root, *, decode='FIA', unscoped=False, wrong_metadata=False, missing_key=False,
            extra_dispatch=False, missing_dispatch=False, wrong_pid=False, duplicate_layer=False,
            extra_sidecar=False, open_window=False, ignored_hits=False,
            outside_phase=False, unfamiliar_dispatch=False):
    root.mkdir(parents=True)
    out = root / 'case'
    lines = []
    case = 'fixture'
    def marker(event, **fields):
        lines.append(json.dumps({'event': event, 'pid': 12, 'thread_id': 13,
                                'epoch_ns': len(lines), 'monotonic_ns': len(lines), **fields}) + '\n')
    def fa(op, *, key=True, dispatch=True, pid=12):
        prefix = f'OP({pid},python) '
        if key:
            lines.append(prefix + f'[NNOP][Launch][14] OpName:[{op}] Tiling Key: 22619644976, len: 1464, numBlocks: 20, Workspace Num: 1.\n')
        if dispatch:
            lines.append(prefix + f'[NNOP][LaunchKernel][14] OpName:[{op}] Launch kernel engine type: 0, numBlocks: 20, scheduleMode: 0, blockDimOffset: 0\n')
            # A second RUNTIME representation is not an additional invocation.
            lines.append(f'RUNTIME({pid},python) StreamLaunchKernelV2: kernel_name=FlashAttentionScore, coreDim=20\n')
    if outside_phase:
        fa('aclnnFlashAttentionScore_998_FlashAttentionScore')
    for step, stage in enumerate(('prefill', 'decode')):
        phase = {'case_id': case, 'batch_size': 1, 'step': step, 'run_index': 0,
                 'mode': 'calibration', 'stage': stage}
        marker('model_phase_enter', **phase)
        if unscoped and step == 0:
            fa('aclnnFlashAttentionScore_999_FlashAttentionScore')
        for layer in range(28):
            index = step * 28 + layer
            call = {'case_id': case, 'call_id': 'call_' + str(index),
                    'api': 'torch.nn.functional.scaled_dot_product_attention', 'pid': 12, 'thread_id': 13,
                    'phase': 'operator_execution', 'step': step, 'mode': 'calibration', 'batch_size': 1,
                    'scope_token': 'native_' + str(index) if step == 0 else None,
                    'layer_index': 0 if duplicate_layer and layer == 1 else layer,
                    'model_stage': stage, 'run_index': 0}
            marker('native_api_enter', **call)
            if step == 0 or decode == 'FA':
                op = f'aclnnFlashAttentionScore_{index}_FlashAttentionScore'
                fa(op, key=not (missing_key and index == 0), dispatch=not (missing_dispatch and index == 0),
                   pid=99 if wrong_pid and index == 0 else 12)
                fa(f'aclnnFlashAttentionScore_{index}_TransposeAiCore')
                if extra_dispatch and index == 0: fa(op)
            elif decode == 'FIA':
                label = 'thread14' if unfamiliar_dispatch else '14'
                lines.append(f'OP(12,python) [NNOP][NnopbaseExecutorLaunchKernel][{label}] OpName:[FusedInferAttentionScore_{index}] '
                    'Launch dynamic kernel FusedInferAttentionScore task, tilingKey is 5000000000000200103, scheMode is 1, numBlocks is 20, stream is 0x1\n')
            elif decode == 'UNKNOWN':
                lines.append(f'OP(12,python) [NNOP][LaunchKernel][14] OpName:[NewAttention_{index}] '
                    'Launch kernel engine type: 0, numBlocks: 20, scheduleMode: 0, blockDimOffset: 0\n')
            if not (open_window and index == 55):
                marker('native_api_exit', **call, status='returned')
            metadata = {'query': {'shape': [1, 16, 2048 if step == 0 else 1, 128], 'dtype': 'torch.float16'},
                        'key': {'shape': [1, 8, 2048 if step == 0 else 2049, 128], 'dtype': 'torch.float16'},
                        'value': {'shape': [1, 8, 2048 if step == 0 else 2049, 128], 'dtype': 'torch.float16'},
                        'attn_mask': None, 'dropout_p': 0.0, 'is_causal': step == 0, 'scale': None, 'enable_gqa': True}
            if wrong_metadata and index == 55: metadata['query']['shape'][2] = 2048
            payload = {'native_call': {**call, 'status': 'returned'}, 'sdpa_metadata': metadata,
                       'eligibility': 'THIS_FIELD_MUST_NOT_BE_READ',
                       'session': {'hits': -999 if ignored_hits else 999}}
            file = out / 'native_calls' / f'call_{index:04d}' / 'call.json'
            file.parent.mkdir(parents=True)
            file.write_text(json.dumps(payload))
        marker('model_phase_exit', **phase, status='returned')
    if extra_sidecar:
        payload['native_call']['call_id'] = 'not_in_log'
        file = out / 'native_calls/call_9999/call.json'
        file.parent.mkdir(parents=True); file.write_text(json.dumps(payload))
    log = root / 'case.log'; log.write_text(''.join(lines))
    return log, out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    checks = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='census_cpu_', dir=args.output.parent) as temporary:
        root = Path(temporary)
        assert root.resolve().is_relative_to(args.output.parent.resolve())
        for name, options, issue in [
            ('valid_FA_prefill_FIA_decode_distinct_dispatches', {}, None),
            ('decode_FA_is_observed_without_assumed_FIA', {'decode': 'FA'}, None),
            ('decode_absence_remains_unknown_not_math', {'decode': 'NONE'}, None),
            ('ignored_injection_ledger_does_not_drive_census', {'ignored_hits': True}, None),
            ('unscoped_attention_blocks_coverage', {'unscoped': True}, 'UNSCOPED_ATTENTION_LAUNCH_INSIDE_MODEL_PHASE'),
            ('whole_case_same_PID_outside_phase_dispatch_blocks', {'outside_phase': True}, 'SAME_PROCESS_ATTENTION_DISPATCH_OUTSIDE_MODEL_PHASES'),
            ('unfamiliar_dispatch_format_is_unparsed_not_absent', {'unfamiliar_dispatch': True}, 'UNPARSED_NATIVE_ATTENTION_DISPATCH_IN_MODEL_PHASE'),
            ('decode_Q1_independently_requires_actual_metadata', {'wrong_metadata': True}, 'DECODE_ACTUAL_Q1_K2049_NOT_ESTABLISHED'),
            ('missing_debug_key_is_unobserved_not_fabricated', {'missing_key': True}, 'ATTENTION_DISPATCH_KEY_UNOBSERVED'),
            ('extra_real_launch_is_not_deduplicated', {'extra_dispatch': True}, 'PREFILL_WRAPPER_NOT_EXACTLY_ONE_ACTUAL_FA_DISPATCH'),
            ('tiling_metadata_is_not_actual_execution', {'missing_dispatch': True}, 'TILING_METADATA_WITHOUT_ACTUAL_DISPATCH_INSIDE_MODEL_PHASE'),
            ('foreign_process_dispatch_inside_phase_blocks', {'wrong_pid': True}, 'FOREIGN_PID_ATTENTION_INSIDE_MODEL_PHASE'),
            ('repeated_layer_window_blocks', {'duplicate_layer': True}, 'INCOMPLETE_OR_REPEATED_28_LAYER_SDPA_WINDOWS'),
            ('unrecognized_attention_is_not_silently_dropped', {'decode': 'UNKNOWN'}, 'UNKNOWN_ATTENTION_OPERATOR_INSIDE_MODEL_PHASE')]:
            log, case = fixture(root / str(len(checks)), **options)
            result = census(log, case)
            if issue is None:
                assert result['status'] == 'PASS_NATIVE_LAUNCH_CENSUS', result['issues']
                assert result['summary']['original_sdpa_windows'] == 56
                assert result['summary']['independently_observed_q1_calls'] == 28
                expected = 28 if options.get('decode') == 'NONE' else 56
                assert result['summary']['attention_dispatches'] == expected
                assert len(result['auxiliary_transpose_lines']) >= 56
                if options.get('decode') == 'NONE':
                    assert result['summary']['decode_route_counts'] == {'NO_NATIVE_ATTENTION_DISPATCH_OBSERVED': 28}
                for row in result['attention_dispatches']:
                    a, b = row['source']['byte_range']
                    data = log.read_bytes()[a:b]
                    assert hashlib.sha256(data).hexdigest() == row['source']['line_sha256']
            else:
                assert result['status'] == 'FAIL_NATIVE_LAUNCH_CENSUS'
                assert issue in [row['code'] for row in result['issues']], result['issues']
            checks.append({'name': name, 'passed': True})
        for name, options in [('extra_sidecar_not_in_whole_log', {'extra_sidecar': True}),
                              ('unclosed_SDPA_window', {'open_window': True})]:
            log, case = fixture(root / str(len(checks)), **options)
            try: census(log, case)
            except ValueError: pass
            else: raise AssertionError('Expected structural rejection: ' + name)
            checks.append({'name': name, 'passed': True})
    report = {'schema': 'nes.model_native_launch_census.cpu.v1', 'status': 'PASS_CPU_FIXTURES',
              'checks': checks, 'npu_calls': 0, 'containers_started': 0,
              'analyzer_sha256': sha(Path(__file__).with_name('native_launch_census.py')),
              'test_source_sha256': sha(Path(__file__))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({'status': report['status'], 'checks': len(checks), 'sha256': sha(args.output)}))


if __name__ == '__main__':
    main()
