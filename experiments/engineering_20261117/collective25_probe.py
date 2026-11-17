"""Two-rank actual collective calibration. Identical outputs are not injection.

Run with torchrun; --cpu is a Gloo integration check, not NPU acceptance.
No model weights, native-order assumption, or synthetic delta is used.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--file-store',type=Path)
    parser.add_argument('--rank',type=int)
    parser.add_argument('--world-size',type=int)
    args=parser.parse_args()
    import torch
    import torch.distributed as dist
    if not args.cpu:
        import torch_npu
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    if args.file_store:
        if not args.cpu:raise ValueError('The file-store helper is only for local CPU integration')
        dist.init_process_group('gloo',init_method=args.file_store.resolve().as_uri(),
                                rank=args.rank,world_size=args.world_size)
    else:
        dist.init_process_group('gloo' if args.cpu else 'hccl')
    rank,world=dist.get_rank(),dist.get_world_size()
    if world<2:raise RuntimeError('At least two real ranks are required')
    from fault_injection.collective25 import ReduceScatterSession
    from fault_injection.core import UnsupportedCall

    class Collective(torch.nn.Module):
        def forward(self,x):
            out=torch.empty(x.numel()//world,dtype=x.dtype,device=x.device)
            dist.reduce_scatter_tensor(out,x)
            return out+1

    model=Collective().eval()
    device='cpu' if args.cpu else f'npu:{int(os.environ["LOCAL_RANK"])}'
    source=(torch.arange(world*128,device=device,dtype=torch.float32)/32)+(rank+1)/8
    report=dict(mechanism='25.reduce_scatter',rank=rank,world_size=world,
        scope='CPU_GLOO_CONTRACT' if args.cpu else 'NPU_HCCL_COLLECTIVE_CALIBRATION',
        pretrained_model=False,public_vllm_worker=False)
    try:
        with torch.inference_mode(),ReduceScatterSession(model,simulation=args.cpu) as session:
            first=model(source);second=model(source)
            try:
                pair=session.measured_pair()
            except UnsupportedCall as exc:
                report.update(status='NO_REPLAY_SAMPLE',reason=str(exc))
                pair=None
            report['observation']=session.report()
        if pair is not None:
            with torch.inference_mode(),ReduceScatterSession(model,mode='replay_delta',pair=pair,
                                                             simulation=args.cpu) as replay:
                fault=model(source)
                report['fault']=replay.report()
            off=model(source)
            report.update(status='MEASURED_REPLAY_ATTEMPTED',off_exact=bool(torch.equal(first,off)))
        else:
            off=model(source)
            report['off_exact']=bool(torch.equal(first,off))
        report['repeat_exact']=bool(torch.equal(first,second))
        report['original_collective_calls']=len(report['observation']['events'])
    except BaseException as exc:
        report.update(status='FAILED',error=repr(exc))
        raise
    finally:
        args.out.mkdir(parents=True,exist_ok=True)
        (args.out/f'rank{rank}.json').write_text(json.dumps(report,indent=2)+'\n')
        dist.destroy_process_group()
        print(json.dumps({k:report[k] for k in ('rank','status','scope')}))


if __name__=='__main__':main()
