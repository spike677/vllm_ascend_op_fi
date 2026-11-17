"""Fault25 synchronous ReduceScatter observation and measured-output replay.

The adapter never manufactures a delta. A replay is admitted only from two
actual, completed same-input collectives in the same live process group. It
does not change collective order, add collectives, or claim LCCL provenance.
"""
from dataclasses import dataclass
import hashlib
import inspect
import json
import os
from pathlib import Path
import weakref

from ._scoped_dispatch import ScopedDispatch
from .core import UnsupportedCall

_ISSUED_PAIRS = weakref.WeakValueDictionary()
_PAIR_DIGESTS = {}


def _hash(tensor):
    import torch
    return hashlib.sha256(tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def _sequence(group):
    reader=getattr(group,'_get_sequence_number_for_group',None)
    try:
        return reader() if callable(reader) else None
    except (RuntimeError,NotImplementedError):
        return None


def _pair_digest(pair):
    data=dict(signature=pair.signature,input=_hash(pair.input_cpu),baseline=_hash(pair.baseline_cpu),
              alternate=_hash(pair.alternate_cpu),group=id(pair.group),model=id(pair.model),
              sequences=list(pair.calibration_sequences),origin=id(pair.origin))
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _verify_pair(pair):
    if (_ISSUED_PAIRS.get(id(pair)) is not pair or _PAIR_DIGESTS.get(id(pair))!=_pair_digest(pair)):
        raise UnsupportedCall('Calibration was not issued by an actual observation session or its bytes changed')


@dataclass(frozen=True)
class MeasuredCollectivePair:
    """Private device-value calibration, excluded from publishable summaries."""
    group: object
    model: object
    signature: dict
    input_cpu: object
    baseline_cpu: object
    alternate_cpu: object
    calibration_sequences: tuple
    origin: object


class ReduceScatterSession(ScopedDispatch):
    mechanism = '25.reduce_scatter'

    def __init__(self, model, *, mode='observe', pair=None, simulation=False):
        if mode not in ('observe','off','replay_delta'):
            raise ValueError('Only observe, off and measured replay_delta are implemented')
        super().__init__(model, arm='fault' if mode=='replay_delta' else 'baseline',simulation=simulation)
        if mode=='replay_delta' and type(pair) is not MeasuredCollectivePair:
            raise UnsupportedCall('A measured same-input native pair is required')
        if mode=='replay_delta':
            _verify_pair(pair)
        self.mode_name = mode
        self.pair = pair
        self.original = self.wrapper = None
        self.observations = []
        self._origin = object()
        self.hits = 0
        self.loaded_libraries = None

    def dispatch(self, func, args, kwargs):
        return func(*args, **kwargs)

    def __enter__(self):
        super().__enter__()
        import torch.distributed as dist
        self.original = dist.reduce_scatter_tensor
        def wrapper(output,input,op=dist.ReduceOp.SUM,group=None,async_op=False):
            if not self.active or not self.stack or self.diagnostic:
                return self.original(output,input,op=op,group=group,async_op=async_op)
            return self._call(output,input,op,group,async_op)
        self.wrapper=wrapper
        dist.reduce_scatter_tensor=wrapper
        return self

    def runtime_state(self):
        state=super().runtime_state()
        if self.wrapper is not None:
            import torch.distributed as dist
            state['ownership_verified'] &= dist.reduce_scatter_tensor is self.wrapper
            state['hook_owned']=True
        return state

    def _call(self,output,input,op,group,async_op):
        import torch
        import torch.distributed as dist
        event=self.event('torch.distributed.reduce_scatter_tensor')
        # Unsupported signatures still execute the original collective so one
        # rank cannot accidentally skip a communication awaited by its peers.
        if async_op or op!=dist.ReduceOp.SUM or not isinstance(input,torch.Tensor) or not isinstance(output,torch.Tensor):
            result=self.original(output,input,op=op,group=group,async_op=async_op)
            event.update(status='passthrough',reason='ONLY_SYNCHRONOUS_TENSOR_SUM_ADAPTED')
            return result
        group=dist.distributed_c10d._get_default_group() if group is None else group
        before_sequence=_sequence(group)
        capture_error=None
        with self.diagnostics():
            try:
                source=self.cpu(input)
            except Exception as exc:
                source=None
                capture_error=type(exc).__name__+': '+str(exc)
        result=self.original(output,input,op=op,group=group,async_op=False)
        after_sequence=_sequence(group)
        if source is None:
            event.update(status='unsupported',reason='INPUT_OBSERVATION_FAILED',error=capture_error,
                         original_execution_verified=True,native_calls=1)
            return result
        with self.diagnostics():
            baseline=self.cpu(output)
            members=tuple(dist.get_process_group_ranks(group))
            local_rank=dist.get_rank(group)
            backend=str(dist.get_backend(group))
            if self.loaded_libraries is None:
                maps=Path('/proc/self/maps')
                paths={line.split()[-1] for line in maps.read_text().splitlines()
                       if '/' in line and any(name in line for name in ('libhccl.so','libtorch_npu.so'))} if maps.exists() else set()
                self.loaded_libraries={path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                       for path in paths if Path(path).is_file()}
            try:
                operator_source=hashlib.sha256(inspect.getsource(self.original).encode()).hexdigest()
            except (OSError,TypeError):
                operator_source=None
            signature=dict(pid=os.getpid(),group_object_id=id(group),model_object_id=id(self.model),
                members=list(members),global_rank=dist.get_rank(),local_rank=local_rank,
                backend=backend,operator='torch.distributed.reduce_scatter_tensor',reduce_op='SUM',
                dtype=str(input.dtype),send_shape=list(input.shape),receive_shape=list(output.shape),
                send_count=input.numel(),receive_count=output.numel(),path=event['path'],
                operator_source_sha256=operator_source,loaded_libraries=self.loaded_libraries)
            valid=(len(members)>=2 and 0<=local_rank<len(members)
                and members[local_rank]==signature['global_rank']
                and input.numel()==len(members)*output.numel()
                and input.dtype==output.dtype and input.device==output.device
                and input.untyped_storage()._cdata!=output.untyped_storage()._cdata
                and input.is_floating_point() and bool(torch.isfinite(source).all())
                and bool(torch.isfinite(baseline).all()) and torch.equal(self.cpu(input),source)
                and type(before_sequence) is int and type(after_sequence) is int and after_sequence>before_sequence
                and operator_source is not None
                and (self.simulation or backend=='hccl' and input.device.type=='npu'
                     and any('libhccl.so' in path for path in self.loaded_libraries)))
            event.update(original_execution_verified=True,native_calls=1,signature=signature,
                input_sha256=_hash(source),baseline_sha256=_hash(baseline),applied=False,
                source='ACTUAL_COLLECTIVE_RETURN',native_acceptance_claimed=False,
                native_sequence_before=before_sequence,native_sequence_after=after_sequence)
            if not valid:
                event.update(status='unsupported',reason='LIVE_GROUP_LAYOUT_OR_NATIVE_BACKEND_UNPROVEN')
                return result
            event['status']='observed'
            row=dict(group=group,signature=signature,input=source,output=baseline,
                     sequence=before_sequence,event=event,origin=self._origin)
            self.observations.append(row)
            if self.mode_name!='replay_delta' or self.hits:
                return result
            pair=self.pair
            _verify_pair(pair)
            if (pair.group is not group or pair.model is not self.model or pair.signature!=signature
                or not torch.equal(pair.input_cpu,source) or not torch.equal(pair.baseline_cpu,baseline)):
                event.update(status='passthrough',reason='MEASURED_GROUP_INPUT_OR_BASELINE_MISMATCH')
                return result
            # Baseline bytes are exact. Replaying the measured alternate bytes
            # avoids a second rounding error from forming/subtracting delta.
            # This is an output replay proxy, not a changed native algorithm.
            alternate=pair.alternate_cpu.to(device=output.device,dtype=output.dtype)
            self.temps.append(alternate)
            try:
                output.copy_(alternate)
                self.synchronize()
                if not torch.equal(self.cpu(output),pair.alternate_cpu):
                    raise UnsupportedCall('Measured ReduceScatter alternate readback failed')
            except BaseException:
                # The original collective already completed. Restore its local
                # result before propagating a proxy failure; never call it again.
                output.copy_(baseline.to(device=output.device,dtype=output.dtype))
                self.synchronize()
                event['failed_proxy_original_output_restored']=torch.equal(self.cpu(output),baseline)
                raise
            self.hits+=1
            event.update(status='applied',applied=True,alternate_sha256=_hash(pair.alternate_cpu),
                calibration_sequences=list(pair.calibration_sequences),exact_measured_alternate=True,
                mechanism='MEASURED_REDUCE_SCATTER_OUTPUT_REPLAY',raw_backend_nondeterminism_reproduced=False)
        return result

    def measured_pair(self, first=0, second=1):
        import torch
        if len(self.observations)<=max(first,second) or first==second:
            raise UnsupportedCall('Two actual completed observations are required')
        a,b=(self.observations[index] for index in (first,second))
        if (a['origin'] is not self._origin or b['origin'] is not self._origin
            or a['group'] is not b['group'] or a['signature']!=b['signature']
            or not torch.equal(a['input'],b['input'])):
            raise UnsupportedCall('Calibration collective scope or exact input differs')
        if torch.equal(a['output'],b['output']):
            raise UnsupportedCall('NO_NATIVE_SAME_INPUT_DIFFERENCE_OBSERVED')
        pair=MeasuredCollectivePair(a['group'],self.model,dict(a['signature']),a['input'].clone(),
            a['output'].clone(),b['output'].clone(),(a['sequence'],b['sequence']),self._origin)
        _ISSUED_PAIRS[id(pair)]=pair
        _PAIR_DIGESTS[id(pair)]=_pair_digest(pair)
        weakref.finalize(pair,_PAIR_DIGESTS.pop,id(pair),None)
        return pair

    def close(self):
        if self.wrapper is not None:
            import torch.distributed as dist
            if self.stack:
                raise RuntimeError('Collective root forward remains in flight')
            if dist.reduce_scatter_tensor is not self.wrapper:
                raise RuntimeError('Foreign ReduceScatter wrapper replaced this owner')
            dist.reduce_scatter_tensor=self.original
            self.wrapper=None
        super().close()

    def release_calibration(self):
        """Drop this closed observer's group references before group teardown.

        Closing hooks deliberately preserves observations for measured_pair().
        The owner must explicitly release them after calibration/replay is done;
        this is local reference cleanup, not proof of a cross-rank drain.
        """
        if not self.closed:
            raise RuntimeError('Close the collective session before releasing calibration')
        retained = len(self.observations)
        self.observations.clear()
        self.pair = None
        return dict(observations_released=retained, observations_retained=0,
                    group_drain_claimed=False)

    def report(self):
        return dict(super().report(),mode=self.mode_name,hits=self.hits,
            collective_calls_added=0,collective_calls_skipped=0,
            group_recovery='LOCAL_SYNCHRONOUS_DRAIN_ONLY_CROSS_RANK_RECEIPTS_REQUIRED',
            calibration='ACTUAL_SAME_INPUT_DIFFERENCE_REQUIRED',
            no_difference_is_successful_injection=False)
