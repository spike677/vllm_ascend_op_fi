"""Minimum real NPU replays of the new 15/17/18 Provider implementation.

This constructs inference modules. It does not launch or impersonate a vLLM
Worker, and cannot grant pretrained-model or public-Worker qualification.
"""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mechanism', choices=['15.alias_detach','17.spatial_padding','18.api_half',
                                               '8.restore_offset','22.wrong_row'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    from demoncat.runtime.vllm_ascend.worker_extension.providers.tensor_modules import (
        TensorModuleProvider, install_tensor_replay, uninstall_tensor_replay)
    torch.npu.set_device(0)
    torch.manual_seed(20261117)

    class Alias(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer('initial', torch.arange(256.).reshape(16,16).to(torch.bfloat16))
        def forward(self):
            base = torch_npu.npu_format_cast(self.initial.clone(), 29)
            view = torch.as_strided(base, (8,8), (16,1), 17)
            view.copy_(torch.full((8,8), -32, dtype=base.dtype, device=base.device))
            return base + 1

    class Conv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = torch.nn.Conv3d(3,8,(2,16,16),stride=(2,16,16),bias=False)
        def forward(self, x): return self.patch(x).flatten(1) + 1

    class Norm(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.norm = torch.nn.RMSNorm(32)
        def forward(self, x): return self.norm(x) + 1

    class MoE(torch.nn.Module):
        def forward(self, hidden, indices, probs):
            return torch_npu.npu_moe_token_unpermute(hidden, indices, probs) + 1

    class RoPE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            angle=torch.arange(16.).unsqueeze(1)*torch.tensor([.1,.2,.3,.4]).unsqueeze(0)
            self.register_buffer('cos_sin_cache',torch.cat((angle.cos(),angle.sin()),1).to(torch.bfloat16))
        def forward(self, x, positions):
            lookup=self.cos_sin_cache.index_select(0,positions)
            cos,sin=lookup.chunk(2,dim=-1)
            rotated=torch.cat((-x[:,2:],x[:,:2]),dim=-1)
            return x*cos+rotated*sin

    factory = {'15.alias_detach':Alias,'17.spatial_padding':Conv,'18.api_half':Norm,
               '8.restore_offset':MoE,'22.wrong_row':RoPE}[args.mechanism]
    model = factory().eval().npu()
    inputs = (() if args.mechanism.startswith('15.') else
              (torch.linspace(-1,1,2*3*2*16*16,device='npu').reshape(2,3,2,16,16),)
              if args.mechanism.startswith('17.') else
              (torch.linspace(-2,2,4*32,device='npu').reshape(4,32),))
    if args.mechanism == '8.restore_offset':
        inputs = ((torch.arange(8*128,device='npu').reshape(8,128)/32).to(torch.bfloat16),
                  torch.arange(8,dtype=torch.int32,device='npu').flip(0),
                  torch.tensor([[.25,.75],[.5,.5],[.75,.25],[.25,.75]],dtype=torch.bfloat16,device='npu'))
    elif args.mechanism == '22.wrong_row':
        inputs = (torch.arange(16,dtype=torch.bfloat16,device='npu').reshape(4,4),
                  torch.arange(4,dtype=torch.int64,device='npu'))
    worker = SimpleNamespace(model_runner=SimpleNamespace(model=model))
    binding = install_tensor_replay(worker, args.mechanism,
        replay=lambda m, phase: m(*inputs), synchronize=torch.npu.synchronize,
        workload_kind='constructed_module')
    provider = TensorModuleProvider(worker, model, {}, args.mechanism)
    receipt = dict(scope='CONSTRUCTED_NPU_PROVIDER_REPLAY', public_vllm_worker=False,
        pretrained_model=False, mechanism=args.mechanism, torch=torch.__version__,
        torch_npu=torch_npu.__version__, device=torch.npu.get_device_name(0))
    try:
        for action in ('prepare','register','check','execute','finish'):
            state = provider.dispatch(action, {}, owner='minimal-npu-probe', generation=1)
        effect=state['qualification']['effect']
        receipt.update(status='PASS_MECHANISM_ONLY' if effect['output_changed']
                       else 'EXECUTED_WITHOUT_OUTPUT_DIFFERENCE', provider=state)
    except BaseException as exc:
        receipt.update(status='FAILED', error=repr(exc))
        raise
    finally:
        try:
            receipt['cleanup'] = provider.dispatch('unregister', {}, owner='minimal-npu-probe', generation=1)
            uninstall_tensor_replay(worker, args.mechanism, binding)
        except BaseException as exc:
            receipt.update(status='RECOVERY_FAILED', cleanup_error=repr(exc))
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(receipt, indent=2)+'\n')
        print(json.dumps({k:receipt[k] for k in ('scope','status','mechanism')}))


if __name__ == '__main__': main()
