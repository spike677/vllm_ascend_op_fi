"""Read-only, pinned Triton JIT-to-launch callsite evidence.

Arguments come from the actual reviewed JIT frame, not expected profiles or
injection logs. This is not a device ABI payload decoder. The report preserves
that distinction even when the actual launcher invocation and return are seen.
"""
import inspect
import sys

from .moe_census import describe

TRITON_ROPE_SOURCE = '4c579d5d22b08d7657f1406fe072c2d17f6fa0a87b0421a520b61e6e1177f7f9'
TRITON_JIT_SOURCE = '55c8b09db2780f1d5106fb1652bda7a5807c7b243b1e43a763265d4755b58dd3'


def same_span(left, right):
    return (left.untyped_storage()._cdata == right.untyped_storage()._cdata
        and left.storage_offset() == right.storage_offset() and left.numel() == right.numel()
        and left.is_contiguous() and right.is_contiguous()
        and left.dtype == right.dtype and left.device == right.device)


def same_callable(left, right):
    if left is right:
        return True
    return (getattr(left, '__self__', None) is not None
        and getattr(left, '__self__', None) is getattr(right, '__self__', None)
        and getattr(left, '__name__', None) == getattr(right, '__name__', None)
        and getattr(left, '__func__', None) is getattr(right, '__func__', None))


class TritonRoPELaunchTrace:
    def __init__(self, kernel, parent):
        self.kernel, self.parent = kernel, parent
        self.run = kernel.run.__func__
        self.run_code = self.run.__code__
        self.fn = kernel.fn
        self.fn_code = self.fn.__code__
        self.pending, self.launch_frames, self.rows = {}, {}, []

    def _reader(self, frame):
        while frame is not None:
            if id(frame) in self.parent.pending:
                return self.parent.pending[id(frame)]
            frame = frame.f_back
        raise RuntimeError('TRITON_ROPE_OUTSIDE_OBSERVED_READER')

    def _joins(self, values, actual):
        q, k = values['q_ptr'], values['k_ptr']
        return dict(positions_exact_object=values['pos_ptr'] is actual['positions'],
            cache_exact_object=values['cos_sin_ptr'] is actual['cos_sin_cache'],
            query_actual_storage=same_span(q, actual['query']),
            key_actual_storage=same_span(k, actual['key']),
            query_stride=values['q_row_stride'] == q.stride(0),
            key_stride=values['k_row_stride'] == k.stride(0),
            cache_stride=values['cos_sin_row_stride'] == actual['cos_sin_cache'].stride(0),
            full_original_rotary_dim=values['rope_dim'] == actual['rotary_dim'],
            original_head_size=values['hd'] == actual['head_size'],
            neox_style=values['IS_NEOX_STYLE'] == actual['is_neox_style'],
            uses_positions_cache=values['USE_COS_SIN'] is True,
            no_preselected_cos_sin=all(values[name] is None for name in
                ('cos_ptr','cos_row_stride','sin_ptr','sin_row_stride')),
            token_count=values['num_tokens'] == actual['positions'].numel())

    def _launch(self, jit_frame, callback):
        row, actual, entering_error = self.pending[id(jit_frame)]
        values = jit_frame.f_locals.get('bound_args')
        if not isinstance(values, dict):
            raise RuntimeError('ACTUAL_TRITON_BOUND_ARGUMENTS_MISSING')
        with self.parent.diagnostics():
            joins = self._joins(values, actual)
            nonconst = jit_frame.f_locals.get('non_constexpr_vals')
            indices = self.kernel.non_constexpr_indices
            expected = [values[self.kernel.params[index].name] for index in indices]
            packed = isinstance(nonconst, tuple) and len(nonconst) == len(expected) and all(
                a is b or type(a) in (int, bool, float, type(None)) and type(a) is type(b) and a == b
                for a, b in zip(nonconst or (), expected))
            row['launches'].append(dict(returned=False, actual_argument_join=joins,
                actual_non_constexpr_matches_bound=packed,
                arguments={name: describe(value) for name, value in values.items()},
                actual_grid=[jit_frame.f_locals.get('grid_' + str(i)) for i in range(3)],
                actual_stream=jit_frame.f_locals.get('stream'),
                callsite_line=jit_frame.f_lineno,
                argument_evidence='PINNED_JIT_CALLSITE_LIVE_LOCALS',
                raw_device_launch_payload_observed=False))
        return row['launches'][-1]

    def _launcher(self, kernel):
        # CompiledKernel can initialize native handles lazily. Inspect only
        # the already populated instance dictionary; reading a descriptor is
        # not an observation if it can cause initialization itself.
        if kernel is None:return None
        callback=vars(kernel).get('run')
        if callback is None and self.parent.simulation:
            callback=getattr(kernel,'run',None)
        return callback

    def profile(self, frame, event, result):
        if event == 'call' and frame.f_code is self.run_code and frame.f_locals.get('self') is self.kernel:
            if self.kernel.run.__func__ is not self.run or self.kernel.fn is not self.fn or self.fn.__code__ is not self.fn_code:
                raise RuntimeError('ACTUAL_TRITON_ROPE_CALLABLE_CHANGED')
            if frame.f_locals.get('warmup') is not False or getattr(self.kernel, 'pre_run_hooks', ()):
                raise RuntimeError('TRITON_WARMUP_OR_FOREIGN_PRE_RUN_HOOK')
            reader, _, actual, *_ = self._reader(frame)
            with self.parent.diagnostics():
                values = inspect.signature(self.fn).bind(*frame.f_locals['args'], **frame.f_locals['kwargs']).arguments
                row = dict(sequence=len(self.rows), reader_sequence=reader['sequence'],
                    request_id=reader['request_id'], model_path=reader['model_path'],
                    entry_argument_join=self._joins(values, actual), launches=[], returned=False,
                    simulation=self.parent.simulation,
                    kernel_source_sha256=None if self.parent.simulation else TRITON_ROPE_SOURCE,
                    jit_source_sha256=None if self.parent.simulation else TRITON_JIT_SOURCE)
            self.rows.append(row)
            self.pending[id(frame)] = (row, actual, sys.exc_info()[1])
        elif id(frame) in self.pending and event in ('c_call', 'c_return', 'c_exception'):
            kernel = frame.f_locals.get('kernel')
            callback = self._launcher(kernel)
            if callback is not None and same_callable(result, callback):
                if event == 'c_call':
                    self.launch_frames[id(frame)] = self._launch(frame, callback)
                elif id(frame) in self.launch_frames:
                    self.launch_frames.pop(id(frame))['returned'] = event == 'c_return'
        elif event == 'call' and frame.f_back is not None and id(frame.f_back) in self.pending:
            kernel = frame.f_back.f_locals.get('kernel')
            callback = self._launcher(kernel)
            method = getattr(callback, '__func__', callback)
            if not inspect.isfunction(method):
                method = getattr(getattr(callback, '__call__', None), '__func__', None)
            if method is not None and getattr(method, '__code__', None) is frame.f_code:
                self.launch_frames[id(frame)] = self._launch(frame.f_back, callback)
        elif event == 'return' and id(frame) in self.launch_frames:
            self.launch_frames.pop(id(frame))['returned'] = sys.exc_info()[1] is None
        if event == 'return' and id(frame) in self.pending:
            row, _, entering_error = self.pending.pop(id(frame))
            row['returned'] = result is frame.f_locals.get('kernel') and result is not None and sys.exc_info()[1] is entering_error
            row['kernel_name'] = getattr(result, 'name', None)
            row['kernel_hash'] = getattr(result, 'hash', None)
            row['actual_launch_call_count'] = len(row['launches'])

    @staticmethod
    def complete(row):
        return (row['returned'] and all(row['entry_argument_join'].values())
            and len(row['launches']) == 1 and row['launches'][0]['returned']
            and row['launches'][0]['actual_non_constexpr_matches_bound']
            and all(row['launches'][0]['actual_argument_join'].values()))
