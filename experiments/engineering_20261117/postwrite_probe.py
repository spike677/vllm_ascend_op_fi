"""Actual NPU postwrite probes, with optional pretrained HF prefill replay.

Run B/F as separate processes with the same library, inputs and collector.
The provider discovers actual operators and consumers; callers supply no layer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mechanism', choices=['39.output_proxy', '41.tail_column'], required=True)
    parser.add_argument('--arm', choices=['baseline', 'fault'], required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--statistics', action='store_true')
    parser.add_argument('--model', type=Path, help='Optional local pretrained HF model for 39 only; no layer selectors')
    parser.add_argument('--prompt', default='Explain what a checksum is in one sentence.')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu
    from fault_injection.deployments.postwrite import (NativePostwriteLibrary,
        Fault39OutputProxyProvider, Fault41TailColumnProvider)
    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False

    class Projection(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.eye(4, dtype=torch.bfloat16, device='npu'), requires_grad=False)
        def forward(self, x):
            projected = torch.mm(x, self.weight)
            return projected + 1

    class Cache(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('state', torch.zeros(3, 4, dtype=torch.bfloat16, device='npu'))
        def forward(self, x, rows):
            self.state.index_copy_(0, rows, x)
            return self.state + 1

    is39 = args.mechanism.startswith('39.')
    if args.model is not None:
        if not is39:
            parser.error('Natural model cache writer adaptation for 41 is not qualified')
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
        network = AutoModelForCausalLM.from_pretrained(args.model, local_files_only=True,
            trust_remote_code=False, torch_dtype=torch.bfloat16, attn_implementation='eager').eval().npu()
        class Prefill(torch.nn.Module):
            def __init__(self, actual):
                super().__init__()
                self.actual = actual
            def forward(self, ids, mask):
                return self.actual(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=False)[0]
        model = Prefill(network).eval()
        encoded = tokenizer(args.prompt, return_tensors='pt', truncation=True, max_length=32)
        inputs = (encoded['input_ids'].npu(), encoded['attention_mask'].npu())
    else:
        model = (Projection() if is39 else Cache()).eval()
        inputs = (torch.arange(1, 9, dtype=torch.bfloat16, device='npu').reshape(2, 4),) if is39 else (
            torch.arange(1, 5, dtype=torch.bfloat16, device='npu').reshape(1, 4),
            torch.tensor([1], dtype=torch.int64, device='npu'))
    report = dict(schema='demoncat.postwrite_probe.v1', mechanism=args.mechanism, arm=args.arm,
        scope='PRETRAINED_HF_PREFILL' if args.model else 'CONSTRUCTED_NPU_INFERENCE_MODULE',
        pretrained_model=args.model is not None, model_path=str(args.model) if args.model else None,
        worker_extension=False, continuous_generation_validated=False,
        actual_device=torch.npu.get_device_name(0), torch=torch.__version__, torch_npu=torch_npu.__version__,
        library_sha256=args.sha256, steps=[0], status='STARTED')
    provider = collector = None
    try:
        native = NativePostwriteLibrary(args.library, args.sha256)
        provider = (Fault39OutputProxyProvider(model, native) if is39 else
                    Fault41TailColumnProvider(model, native, tail_elements=2))
        report['qualification'] = provider.qualify(*inputs)
        baseline, lossless = provider.run('baseline', *inputs)
        report['lossless'] = lossless
        before = model.state.cpu().clone() if not is39 else None
        if args.statistics:
            sys.path.insert(0, str(ROOT / 'experiments/provider_adaptation_20261116'))
            from capture import capture
            collector = capture(args.out / 'capture')
        from experiments.engineering_20261117.model_capture import statistics_model_call
        window = (lambda actual: statistics_model_call(collector, actual, torch.npu.synchronize)) if collector else None
        observed, phase_report = provider.run(args.arm, *inputs, model_observation=window)
        torch.npu.synchronize()
        report['capture_scope'] = 'ACTUAL_MODEL_CALL_API_MODULE_ONLY_VALIDATION_AND_RECOVERY_EXCLUDED'
        off, off_report = provider.run('off', *inputs)
        report.update(phase=phase_report, off=off_report,
            exact_off=torch.equal(off, baseline),
            changed_elements=int((observed != baseline).sum().item()),
            observed_inf=int(torch.isinf(observed).sum().item()),
            observed_nan=int(torch.isnan(observed).sum().item()),
            live_leases=provider.query()['leases'])
        if not is39:
            report['cache_restored_exact'] = torch.equal(model.state.cpu(), before)
            report['non_target_exact'] = torch.equal(observed[[0, 2]], baseline[[0, 2]]) and torch.equal(observed[1, :2], baseline[1, :2])
        expected_count = (1 if is39 else 2) if args.arm == 'fault' else 0
        effect_ok = (report['changed_elements'] == expected_count if args.model is None else
                     (report['changed_elements'] > 0) == (args.arm == 'fault'))
        if (not effect_ok or not report['exact_off']
                or report['live_leases'] != 0 or report.get('cache_restored_exact') is False
                or report.get('non_target_exact') is False):
            raise RuntimeError('POSTWRITE_EFFECT_OR_RECOVERY_FAILED')
        report['status'] = 'NPU_PRETRAINED_PREFILL_RECOVERED' if args.model else 'NPU_CONSTRUCTED_MECHANISM_RECOVERED'
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc), traceback=traceback.format_exc())
    finally:
        if provider is not None:
            try:
                provider.recover()
            except BaseException as exc:
                report.update(status='RECOVERY_FAILED', recovery_error=repr(exc))
    sources = ['fault_injection/deployments/postwrite.py',
               'fault_injection/native_adapters/postwrite/postwrite.cpp',
               'fault_injection/native_adapters/postwrite/postwrite.h',
               'experiments/engineering_20261117/model_capture.py',
               'experiments/engineering_20261117/postwrite_probe.py']
    report['source_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    (args.out / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('mechanism', 'arm', 'status', 'error') if k in report}))
    return 0 if report['status'] in ('NPU_CONSTRUCTED_MECHANISM_RECOVERED', 'NPU_PRETRAINED_PREFILL_RECOVERED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
