"""Bounded real-collective calibration and teardown diagnostics, never injection.

Independent rank receipts coordinate the experiment without inserting hidden
collectives into the adapter. Only the launcher can prove actual process exit.
"""
import argparse
import atexit
from datetime import timedelta
import faulthandler
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def save(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class RankReceipts:
    """One writer per rank, bounded shared-filesystem rendezvous."""
    def __init__(self, folder, rank, world, timeout):
        self.folder, self.rank, self.world, self.timeout = Path(folder), rank, world, timeout
        self.folder.mkdir(parents=True, exist_ok=True)

    def exchange(self, stage, payload):
        path = self.folder / f'{stage}-rank{self.rank}.json'
        if path.exists():
            raise RuntimeError('RANK_CHECKPOINT_REUSE:' + stage)
        save(path, dict(rank=self.rank, world=self.world, pid=os.getpid(),
                        stage=stage, payload=payload))
        deadline = time.monotonic() + self.timeout
        paths = [self.folder / f'{stage}-rank{rank}.json' for rank in range(self.world)]
        while not all(path.is_file() for path in paths):
            if time.monotonic() >= deadline:
                raise TimeoutError('RANK_CHECKPOINT_TIMEOUT:' + stage)
            time.sleep(0.05)
        rows = [json.loads(path.read_text()) for path in paths]
        for rank, row in enumerate(rows):
            if (row['rank'], row['world'], row['stage']) != (rank, self.world, stage):
                raise RuntimeError('RANK_CHECKPOINT_IDENTITY_MISMATCH:' + stage)
        return rows


def install_shutdown_receipt(torch_npu, mark):
    """Run the vendor's registered callback once, recording actual boundaries.

    No private native shutdown routine is called directly or replaced. Profiling
    observes the original callback's Python/C boundaries if it hangs at exit.
    """
    original = torch_npu._npu_shutdown
    source = inspect.getsource(original)
    mark('native_shutdown_hook_identified', source_sha256=hashlib.sha256(source.encode()).hexdigest(),
         source=source)
    names = {'_npu_shutdown_synchronize', '_destructor_process_group', '_npu_shutdown'}

    def observe(frame, event, argument):
        if event in ('c_call', 'c_return', 'c_exception'):
            name = getattr(argument, '__name__', '')
        elif event in ('call', 'return'):
            name = frame.f_code.co_name
        else:
            return
        if name in names:
            details = dict(function=name, event=event)
            if event == 'return' and frame.f_code is original.__code__:
                details['synchronize_success'] = frame.f_locals.get('success')
            mark('native_shutdown_boundary', **details)

    def shutdown():
        mark('native_shutdown_enter')
        previous = sys.getprofile()
        sys.setprofile(observe)
        try:
            original()
        except BaseException as error:
            mark('native_shutdown_failed', error=repr(error))
            raise
        finally:
            sys.setprofile(previous)
        mark('native_shutdown_returned')

    atexit.unregister(original)
    atexit.register(shutdown)


def case_specs(suite):
    small = [dict(id='fp32-ramp-128', dtype='float32', count=128, pattern='ramp')]
    if suite == 'smoke':
        return small
    return small + [
        dict(id='fp32-cancel-4096', dtype='float32', count=4096, pattern='cancellation'),
        dict(id='fp32-mixed-65536', dtype='float32', count=65536, pattern='mixed_scale'),
        dict(id='fp16-mixed-4096', dtype='float16', count=4096, pattern='mixed_scale'),
        dict(id='bf16-mixed-4096', dtype='bfloat16', count=4096, pattern='mixed_scale'),
        dict(id='bf16-cancel-65536', dtype='bfloat16', count=65536, pattern='cancellation'),
    ]


def make_source(torch, spec, rank, world, device):
    count = world * spec['count']
    index = torch.arange(count, dtype=torch.float32)
    if spec['pattern'] == 'ramp':
        source = index / 32 + (rank + 1) / 8
    elif spec['pattern'] == 'cancellation':
        # Legitimate, fixed finite inputs. No output or collective-order edits.
        magnitude = torch.pow(2., index.remainder(18) - 7)
        source = magnitude * (1 if rank % 2 == 0 else -1) + (rank + 1) / 32
    else:
        generator = torch.Generator().manual_seed(20261117 + rank)
        source = torch.randn(count, generator=generator) * torch.pow(2., index.remainder(20) - 10)
    return source.to(device=device, dtype=getattr(torch, spec['dtype']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--file-store', type=Path)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--world-size', type=int)
    parser.add_argument('--suite', choices=('smoke', 'bounded'), default='bounded')
    parser.add_argument('--repeats', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if not 2 <= args.repeats <= 16 or not 0 < args.timeout <= 300:
        parser.error('Use 2..16 repeats and a 0..300 second stage timeout')
    rank = args.rank if args.rank is not None else int(os.environ['RANK'])
    world = args.world_size if args.world_size is not None else int(os.environ['WORLD_SIZE'])
    if world != 2:
        raise ValueError('This bounded experiment is reviewed for exactly two ranks')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / f'rank{rank}.started.json').open('x') as stream:
        json.dump(dict(rank=rank, pid=os.getpid()), stream)
    lifecycle_path = args.out / f'rank{rank}.lifecycle.jsonl'

    def mark(stage, **details):
        row = dict(stage=stage, rank=rank, pid=os.getpid(), monotonic=time.monotonic(), **details)
        with lifecycle_path.open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        print(json.dumps(dict(rank=rank, stage=stage)), flush=True)

    # Registered before torch import, so this runs after the NPU callback.
    atexit.register(lambda: mark('python_atexit_tail'))
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.timeout, repeat=True, exit=False)
    mark('before_torch_import')
    import torch
    import torch.distributed as dist
    if not args.cpu:
        import torch_npu
        install_shutdown_receipt(torch_npu, mark)
        torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    from fault_injection.collective25 import ReduceScatterSession
    device = 'cpu' if args.cpu else f'npu:{int(os.environ["LOCAL_RANK"])}'
    receipts = RankReceipts(args.out / 'coordination', rank, world, args.timeout)
    report = dict(mechanism='25.reduce_scatter', rank=rank, world_size=world, pid=os.getpid(),
                  scope='CPU_GLOO_CONTRACT' if args.cpu else 'NPU_HCCL_COLLECTIVE_CALIBRATION',
                  pretrained_model=False, public_vllm_worker=False, injection_attempted=False,
                  hits=0, c2_pass=False, c3_pass=False, cases=[],
                  environment=dict(torch=torch.__version__,
                      torch_npu=None if args.cpu else torch_npu.__version__,
                      hccl_algo=os.environ.get('HCCL_ALGO', 'DEFAULT_UNMODIFIED'),
                      local_rank=os.environ.get('LOCAL_RANK')))
    report_path = args.out / f'rank{rank}.json'
    initialized = False
    try:
        plans = receipts.exchange('plan', dict(suite=args.suite, repeats=args.repeats,
                                               specs=case_specs(args.suite)))
        if any(row['payload'] != plans[0]['payload'] for row in plans):
            raise RuntimeError('RANK_CALIBRATION_PLAN_MISMATCH')
        mark('init_process_group_enter')
        kwargs = dict(timeout=timedelta(seconds=args.timeout))
        if args.file_store:
            if not args.cpu:
                raise ValueError('File-store mode is only for CPU contracts')
            kwargs.update(init_method=args.file_store.resolve().as_uri(), rank=rank, world_size=world)
        dist.init_process_group('gloo' if args.cpu else 'hccl', **kwargs)
        initialized = True
        mark('init_process_group_returned')

        class Collective(torch.nn.Module):
            def forward(self, source):
                output = torch.empty(source.numel() // world, dtype=source.dtype, device=source.device)
                dist.reduce_scatter_tensor(output, source)
                return output + 1

        model = Collective().eval()
        for case_index, spec in enumerate(case_specs(args.suite)):
            mark('calibration_enter', case=spec['id'])
            source = make_source(torch, spec, rank, world, device)
            outputs = []
            with torch.inference_mode(), ReduceScatterSession(model, simulation=args.cpu) as session:
                for _ in range(args.repeats):
                    outputs.append(model(source).detach().cpu().clone())
            observation = session.report()
            if len(session.observations) != args.repeats:
                raise RuntimeError('ACTUAL_NATIVE_CALIBRATION_NOT_ADMITTED')
            native_hashes = [row['event']['baseline_sha256'] for row in session.observations]
            input_hashes = [row['event']['input_sha256'] for row in session.observations]
            numeric_repeat_exact = all(torch.equal(session.observations[0]['output'], row['output'])
                                       for row in session.observations[1:])
            same_input = len(set(input_hashes)) == 1
            if not same_input:
                raise RuntimeError('CALIBRATION_INPUT_CHANGED')
            with torch.inference_mode():
                off = model(source).detach().cpu().clone()
            row = dict(**spec, repeat_exact=numeric_repeat_exact,
                       repeat_bytes_exact=len(set(native_hashes)) == 1,
                       actual_native_output_hashes=native_hashes,
                       same_input_exact=same_input, off_exact=bool(torch.equal(outputs[0], off)),
                       observation=observation, cleanup=session.runtime_state(),
                       calibration_release=session.release_calibration())
            row['status'] = ('NO_REPLAY_SAMPLE' if row['repeat_exact'] else
                             'NATIVE_DIFFERENCE_CANDIDATE_REQUIRES_GROUP_REPLAY_REVIEW')
            peers = receipts.exchange(f'case{case_index}', dict(
                case=spec['id'], input_hashes=input_hashes, native_hashes=native_hashes,
                off_exact=row['off_exact'], cleanup=row['cleanup'], hits=0))
            row['cross_rank_inputs_fixed'] = all(len(set(peer['payload']['input_hashes'])) == 1 for peer in peers)
            row['cross_rank_off_exact'] = all(peer['payload']['off_exact'] for peer in peers)
            if not row['cross_rank_off_exact']:
                raise RuntimeError('SAME_PROCESS_OFF_NOT_EXACT')
            report['cases'].append(row)
            save(report_path, report)
            del source, outputs, off, session
            mark('calibration_returned', case=spec['id'], status=row['status'])

        # An actual same-process health collective after every hook is disabled.
        health = torch.tensor([rank + 1.], device=device)
        dist.all_reduce(health)
        health_exact = bool(torch.equal(health.cpu(), torch.tensor([3.])))
        dist.barrier()
        if not args.cpu:
            torch.npu.synchronize()
        peers = receipts.exchange('drained', dict(health_exact=health_exact,
            all_sessions_closed=True, retained_calibration_observations=0,
            native_barrier_returned=True, device_synchronize_returned=True))
        report['group_drain'] = dict(status='CROSS_RANK_DRAIN_RETURNED', ranks=peers,
                                    fault_recovery_proven=False)
        if not all(peer['payload']['health_exact'] for peer in peers):
            raise RuntimeError('POST_OFF_GROUP_HEALTH_FAILED')
        del health, model
        gc.collect()
        mark('destroy_process_group_enter')
        dist.destroy_process_group()
        initialized = False
        mark('destroy_process_group_returned', distributed_initialized=dist.is_initialized())
        gc.collect()
        peers = receipts.exchange('destroyed', dict(destroy_returned=True,
            distributed_initialized=dist.is_initialized()))
        report['group_destroy'] = dict(status='ALL_RANKS_DESTROY_RETURNED', ranks=peers)
        report['status'] = ('NO_REPLAY_SAMPLE' if all(row['repeat_exact'] for row in report['cases'])
                            else 'NATIVE_DIFFERENCE_CANDIDATE_REQUIRES_GROUP_REPLAY_REVIEW')
        report['reason'] = ('NO_NATIVE_SAME_INPUT_DIFFERENCE_OBSERVED' if report['status'] == 'NO_REPLAY_SAMPLE'
                            else 'NO_REPLAY_HAS_BEEN_ARMED')
        report['normal_process_exit_proven'] = False
        report['normal_exit_evidence_owner'] = 'collective25_supervisor.py'
        save(report_path, report)
        mark('main_returned', threads=[dict(name=t.name, daemon=t.daemon) for t in threading.enumerate()])
    except BaseException as error:
        report.update(status='FAILED', error=repr(error))
        save(report_path, report)
        mark('main_failed', error=repr(error))
        if initialized:
            mark('failed_run_destroy_enter')
            dist.destroy_process_group()
            mark('failed_run_destroy_returned')
        raise


if __name__ == '__main__':
    main()
