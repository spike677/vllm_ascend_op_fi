"""Every eligible controlled torch.as_strided call; no write/consumer interception."""
import functools
import inspect
import os


def identity(tensor, torch_npu):
    storage = tensor.untyped_storage()
    return {
        'object_id': id(tensor), 'tensor_cdata': int(tensor._cdata),
        'storage_cdata': int(storage._cdata), 'storage_ptr': int(storage.data_ptr()),
        'storage_nbytes': int(storage.nbytes()), 'tensor_ptr': int(tensor.data_ptr()),
        'storage_offset': int(tensor.storage_offset()),
        'actual_format': int(torch_npu.get_npu_format(tensor)),
        'shape': list(tensor.shape), 'stride': list(tensor.stride()),
        'dtype': str(tensor.dtype), 'device': str(tensor.device),
    }


def safe_dispatch_identity(value, torch_npu):
    """Diagnostics must never block an unmatched CPU or other valid call."""
    basic = {'object_id': id(value), 'python_type': type(value).__name__,
             'device': str(getattr(value, 'device', None))}
    if not basic['device'].startswith('npu:'):
        return basic
    try:
        return identity(value, torch_npu)
    except Exception as exc:
        return {**basic, 'metadata_error': repr(exc)}


class AliasInjector:
    def __init__(self, torch, torch_npu, arm, bases, state, selected_callsite):
        self.torch, self.torch_npu = torch, torch_npu
        if arm not in ('B', 'F'):
            raise ValueError('Unknown arm')
        self.arm, self.state = arm, state
        self.bases = list(bases)
        if not self.bases or len({id(base) for base in self.bases}) != len(self.bases):
            raise ValueError('Distinct live base registrations required')
        self.storages = [base.untyped_storage() for base in self.bases]
        self.expected_identities = [identity(base, torch_npu) for base in self.bases]
        self.temporaries = []  # Keep all actual conversion storage alive for provenance.
        for expected in self.expected_identities:
            if (expected['actual_format'] != 29 or expected['dtype'] != 'torch.bfloat16'
                    or expected['shape'] != [16, 16] or expected['stride'] != [16, 1]
                    or expected['storage_offset'] != 0):
                raise RuntimeError('Actual registered base does not satisfy NZ29/BF16 geometry')
        self.callsite = selected_callsite.__code__
        self.pid = os.getpid()
        self.record = {
            'arm': arm, 'hits': 0, 'selected_original_calls': 0,
            'unmatched_calls': 0, 'unmatched_original_calls': 0, 'errors': [], 'events': [],
            'policy': 'every_eligible_call',
            'selector': {'pid': self.pid, 'rank': 0, 'active_forward': True,
                         'forward_filter': None, 'hit_limit': None,
                         'callsite_file': self.callsite.co_filename,
                         'callsite_function': self.callsite.co_name,
                         'callsite_line': self.callsite.co_firstlineno + 1,
                         'base_identities': self.expected_identities,
                         'view_size': [8, 8], 'view_stride': [16, 1], 'view_offset': 17},
            'coverage': 'Only this controlled Python torch.as_strided call; no Tensor/ATen/native internal coverage claim',
        }
        self.original = None
        self.wrapper = None

    def install(self):
        if self.original is not None:
            raise RuntimeError('Injector already installed')
        # Install after msprobe.start: the original may include its ordinary API
        # observer. Calling it exactly once preserves the real native operation.
        self.original = self.torch.as_strided
        self.record['native_callable'] = {
            'module': getattr(self.original, '__module__', None),
            'name': getattr(self.original, '__qualname__', repr(self.original)),
            'called_once_per_dispatch': True,
        }

        @functools.wraps(self.original)
        def dispatch(input, size, stride, storage_offset=None):
            frame = inspect.currentframe().f_back
            location = {'file': frame.f_code.co_filename, 'function': frame.f_code.co_name,
                        'line': frame.f_lineno}
            same_callsite = (frame.f_code is self.callsite and
                             frame.f_lineno == self.callsite.co_firstlineno + 1)
            del frame
            actual = safe_dispatch_identity(input, self.torch_npu)
            slot = next((i for i, base in enumerate(self.bases) if input is base), None)
            matched = (os.getpid() == self.pid and self.state.rank == 0 and
                       self.state.forward_active and same_callsite and slot is not None and
                       actual == self.expected_identities[slot] and
                       list(size) == [8, 8] and list(stride) == [16, 1] and storage_offset == 17)
            event = {'forward': self.state.forward_index, 'call': self.state.call_index,
                     'case_id': self.state.case_id, 'registered_base_slot': slot,
                     'callsite': location, 'input_identity': actual,
                     'requested_size': list(size), 'requested_stride': list(stride),
                     'requested_offset': storage_offset, 'matched': matched}
            self.record['events'].append(event)
            try:
                argument = input
                if matched:
                    if self.arm == 'F':
                        # The only fault factor. Never clone, rearrange, or repair.
                        argument = self.torch_npu.npu_format_cast(input, 2)
                        self.state.etensor = argument
                        self.temporaries.append(argument)
                        self.record['hits'] += 1
                    self.record['selected_original_calls'] += 1
                else:
                    self.record['unmatched_calls'] += 1
                    self.record['unmatched_original_calls'] += 1
                event['original_argument_identity'] = safe_dispatch_identity(argument, self.torch_npu)
                result = self.original(argument, size=size, stride=stride, storage_offset=storage_offset)
                event['native_return_identity'] = safe_dispatch_identity(result, self.torch_npu)
                event['completed'] = True
                return result
            except BaseException as exc:
                event['completed'] = False
                event['error'] = repr(exc)
                self.record['errors'].append(repr(exc))
                raise

        self.wrapper = dispatch
        self.torch.as_strided = dispatch

    def restore(self):
        if self.wrapper is not None:
            if self.torch.as_strided is not self.wrapper:
                self.record['errors'].append('Controlled torch.as_strided binding changed unexpectedly')
                raise RuntimeError(self.record['errors'][-1])
            self.torch.as_strided = self.original
