"""Real NPU mechanism probes, explicitly constructed inference modules.

No pretrained-model/serving/quality claim. B/F use identical raw msprobe config.
Private numerical checks run after capture and do not select statistical first difference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/provider_adaptation_20261116'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fault', choices=['15.alias_detach', '17.spatial_padding', '18.api_half'], required=True)
    parser.add_argument('--arm', choices=['baseline', 'fault'], required=True)
    parser.add_argument('--out', type=Path, default=Path('/evidence'))
    args = parser.parse_args()
    import torch
    import torch_npu
    from fault_injection import attach
    from capture import capture
    torch.npu.set_device(0)
    torch.manual_seed(20261116)
    report = dict(fault=args.fault, arm=args.arm, status='STARTED', scope='CONSTRUCTED_NPU_INFERENCE_MODULE',
                  pretrained_model=False, worker_extension=False, actual_device=torch.npu.get_device_name(0),
                  torch=torch.__version__, torch_npu=torch_npu.__version__)
    class AliasModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Exactly representable BF16 values, identical per forward and arm.
            self.register_buffer('initial', torch.arange(256).reshape(16,16).to(torch.bfloat16).npu())
            self.register_buffer('write', torch.full((8,8), -32, dtype=torch.bfloat16, device='npu'))
            self.last_base = self.last_view = None
        def forward(self):
            base = torch_npu.npu_format_cast(self.initial.clone(),29)
            view = torch.as_strided(base,(8,8),(16,1),17)
            view.copy_(self.write)
            result = base + 1
            self.last_base, self.last_view = base, view
            return result
    class ConvModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = torch.nn.Conv3d(3,8,(2,16,16),stride=(2,16,16),bias=False)
        def forward(self,x):
            return self.patch(x).flatten(1) + 1
    class NormModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a = torch.nn.RMSNorm(32)
            self.b = torch.nn.RMSNorm(32)
        def forward(self,x):
            # Two independently observed eligible modules; shared inputs.
            return self.a(x) + self.b(x)
    model = {'15.alias_detach':AliasModule, '17.spatial_padding':ConvModule, '18.api_half':NormModule}[args.fault]()
    model = model.eval().npu()
    inputs = (() if args.fault.startswith('15.') else
              (torch.linspace(-1,1,2*3*2*16*16,device='npu').reshape(2,3,2,16,16),)
              if args.fault.startswith('17.') else (torch.linspace(-2,2,4*32,device='npu').reshape(4,32),))
    collector = None
    def cpu(value):
        torch.npu.synchronize()
        return value.detach().cpu().clone()
    def digest(value):
        return hashlib.sha256(cpu(value).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    try:
        with torch.inference_mode():
            healthy = cpu(model(*inputs)); healthy2 = cpu(model(*inputs))
            if not torch.equal(healthy,healthy2):
                raise RuntimeError('HEALTHY_REPEAT_NOT_EXACT')
            input_hashes = [digest(x) for x in inputs]
            # msprobe installs API wrappers in its constructor. Establish that
            # layer before the fault session takes ownership; reversing this
            # order correctly trips the session's foreign-wrapper cleanup gate.
            collector = capture(args.out/'capture')
            with attach(model,args.fault,arm=args.arm) as session:
                warm = cpu(model(*inputs))
                # Baseline must be lossless before the fault can be accepted.
                if args.arm == 'baseline' and not torch.equal(warm,healthy):
                    raise RuntimeError('BASELINE_NOT_LOSSLESS')
                outputs = []
                for step in range(3):
                    collector.start(model=model,rank_id=0)
                    try:
                        out = model(*inputs)
                        torch.npu.synchronize()
                    finally:
                        collector.stop()
                    outputs.append(cpu(out))
                    collector.step()
                session.assert_coverage()
                session_report = session.report()
                if args.fault.startswith('15.'):
                    initial = cpu(model.initial); after = cpu(model.last_base); view = cpu(model.last_view)
                    expected = initial.clone(); expected[1:9,1:9] = -32
                    report['alias_check'] = dict(actual_format=int(torch_npu.get_npu_format(model.last_base)),
                        view_written_exact=torch.equal(view,torch.full_like(view,-32)),
                        original_base_updated=torch.equal(after,expected),
                        original_base_retains_initial=torch.equal(after,initial),
                        consumer_exact=torch.equal(outputs[-1],after+1))
                    check=report['alias_check']
                    if not check['view_written_exact'] or not check['consumer_exact'] or not (
                        check['original_base_updated'] if args.arm=='baseline' else check['original_base_retains_initial']):
                        raise RuntimeError('ALIAS_WRITE_CONSUMER_CHECK_FAILED')
            off = cpu(model(*inputs))
            report.update(off_exact=torch.equal(off,healthy), repeat_stable=all(torch.equal(x,warm) for x in outputs),
                input_unchanged=[digest(x) for x in inputs]==input_hashes,
                changed=not torch.equal(outputs[0],healthy), finite=all(bool(torch.isfinite(x).all()) for x in outputs),
                max_abs=float((outputs[0].float()-healthy.float()).abs().max()), steps=[0,1,2],
                session=session_report)
            if args.fault.startswith('18.'):
                report['exact_half']=torch.equal(outputs[0],healthy*.5) if args.arm=='fault' else torch.equal(outputs[0],healthy)
            if not all(report[k] for k in ('off_exact','repeat_stable','input_unchanged','finite')):
                raise RuntimeError('RECOVERY_OR_STABILITY_FAILED')
            if report['changed'] != (args.arm=='fault'):
                raise RuntimeError('EXPECTED_EFFECT_NOT_OBSERVED')
            if args.fault.startswith('18.') and not report['exact_half']:
                raise RuntimeError('HALF_OUTPUT_NOT_EXACT')
            report['status']='PASS_NATIVE_MECHANISM_ONLY'
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc())
    (args.out/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('fault','arm','status','error') if k in report}))
    return 0 if report['status']=='PASS_NATIVE_MECHANISM_ONLY' else 2


if __name__ == '__main__':
    raise SystemExit(main())
