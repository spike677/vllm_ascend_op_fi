"""Constructed all-eligible NPU alias-write-consume probe, two real forwards.

The common observer reads physical objects and values, never the selected arm or
the injector's success counters. Control truth is saved separately. No model
stage, historical-defect, or standalone-statistics root-cause claim is made.
"""
import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import struct
import sys
import time
import traceback
import warnings

from alias_injector import AliasInjector, identity


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def restore_registered_custom_apis(restorer, registered, report):
    """Restore only successful registrations, using the official (module, api) API."""
    errors = []
    for entry in list(reversed(registered)):
        row = {'api': entry['label'], 'restored': False}
        try:
            restorer(entry['module'], entry['api'])
            if getattr(entry['module'], entry['api']) is not entry['original']:
                raise RuntimeError('Official restoration did not restore the original callable')
            row['restored'] = True
            registered.remove(entry)
        except Exception as exc:
            row['error'] = repr(exc)
            errors.append(entry['label'] + ': ' + repr(exc))
        report.setdefault('restoration_results', []).append(row)
    report['pending_restorations'] = [entry['label'] for entry in registered]
    report['restored'] = not registered
    if errors:
        raise RuntimeError('Custom API restoration failed: ' + '; '.join(errors))


def selected_as_strided(base):
    return torch.as_strided(base, size=(8, 8), stride=(16, 1), storage_offset=17)


def unmatched_as_strided(base):
    return torch.as_strided(base, size=(8, 8), stride=(16, 1), storage_offset=17)


class State:
    def __init__(self, bases):
        self.bases = list(bases)
        self.base_storages = [base.untyped_storage() for base in bases]
        self.base = None
        self.etensor = None
        self.view = None
        self.keepalive = []
        self.rank = 0
        self.forward_index = -1
        self.forward_active = False
        self.call_index = None
        self.case_id = None
        self.current_case = None


class CommonObserver:
    """Shared readback path; its API nodes are diagnostics, not fault boundaries."""
    def __init__(self, runtime, state):
        self.runtime, self.state = runtime, state
        self.readback_sequence = 0

    @staticmethod
    def cpu_payload(cpu):
        cpu = cpu.contiguous()
        bits = [int(value) & 65535 for value in cpu.reshape(-1).view(torch.int16).tolist()]
        raw = struct.pack('<' + 'H' * len(bits), *bits)
        return {'values_flat': cpu.float().reshape(-1).tolist(), 'bf16_bits': bits, 'sha256': sha(raw)}

    def read_tensor(self, tensor):
        if tensor is None:
            return None
        torch_npu.npu.synchronize()
        before = identity(tensor, torch_npu)
        self.readback_sequence += 1
        readback_sequence = self.readback_sequence

        def marker(phase):
            if before['storage_offset'] != 0:
                torch_npu.npu.synchronize()
                print(json.dumps({'event': 'offset_readback_phase', 'pid': os.getpid(),
                                  'epoch_ns': time.time_ns(), 'sequence': readback_sequence,
                                  'forward': self.state.forward_index, 'case_id': self.state.case_id,
                                  'phase': phase, 'object_id': before['object_id'],
                                  'storage_offset': before['storage_offset'],
                                  'actual_format': before['actual_format']}), file=sys.stderr, flush=True)

        # Direct CPU reads before/after independently check that the explicit ND
        # observation did not change the source's logical values or identity.
        marker('before_direct_cpu')
        direct_before = self.cpu_payload(tensor.detach().to('cpu'))
        marker('after_direct_cpu')
        marker('before_nd_cast')
        nd = torch_npu.npu_format_cast(tensor, 2)
        torch_npu.npu.synchronize()
        marker('after_nd_cast')
        marker('before_nd_cpu')
        payload = self.cpu_payload(nd.detach().to('cpu'))
        marker('after_nd_cpu')
        marker('before_direct_cpu_recheck')
        direct_after = self.cpu_payload(tensor.detach().to('cpu'))
        marker('after_direct_cpu_recheck')
        torch_npu.npu.synchronize()
        after = identity(tensor, torch_npu)
        return {'identity': before, **payload,
                'readback': {'method': 'explicit_npu_format_cast_to_ND_then_cpu; direct_cpu_before_after_check',
                             'source_identity_before': before, 'source_identity_after': after,
                             'identity_unchanged': before == after,
                             'nd_observation_identity': identity(nd, torch_npu),
                             'direct_before_sha256': direct_before['sha256'],
                             'direct_after_sha256': direct_after['sha256'],
                             'direct_nd_values_equal': direct_before == payload == direct_after,
                             'source_bf16_bits_before': direct_before['bf16_bits'],
                             'source_bf16_bits_after': direct_after['bf16_bits']}}

    def stage(self, name, **extras):
        torch_npu.npu.synchronize()
        # The neutral temporary slot is observed when an actual object exists;
        # no arm, hit counter, or injector conclusion is read here.
        row = {'stage': name, 'synchronized': True,
               'base': self.read_tensor(self.state.base),
               'view': self.read_tensor(self.state.view),
               'etensor': self.read_tensor(self.state.etensor), **extras}
        self.state.current_case['stages'].append(row)
        return row


def create_model(observer, state, writes, guard_bases, guard_values):
    class ReadBaseBefore(torch.nn.Module):
        def forward(self, base):
            return torch_npu.npu_format_cast(base, 2)

    class CreateView(torch.nn.Module):
        def forward(self, base):
            return selected_as_strided(base)

    class WriteView(torch.nn.Module):
        def forward(self, view, values):
            return view.copy_(values)

    class ReadOriginalBase(torch.nn.Module):
        def forward(self, base):
            return torch_npu.npu_format_cast(base, 2)

    class ConsumeOriginalBase(torch.nn.Module):
        def forward(self, base):
            return base + 1

    class InvocationBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.read_base_before = ReadBaseBefore()
            self.create_view = CreateView()
            self.write_view = WriteView()
            self.read_original_base = ReadOriginalBase()
            self.consume_original_base = ConsumeOriginalBase()

        def forward(self, base, values):
            # Clear observation slots, not original data. Same base is reused for
            # all three calls, so previous write effects remain real inputs.
            state.view, state.etensor = None, None
            before_read = self.read_base_before(base)
            observer.stage('pre_view', base_before_read_nd=observer.read_tensor(before_read))
            state.view = self.create_view(base)
            state.keepalive.append(state.view)
            observer.stage('view_created')
            copy_return = self.write_view(state.view, values)
            observer.stage('view_written', copy_return_identity=identity(copy_return, torch_npu),
                           write_values=observer.read_tensor(values))
            original_read = self.read_original_base(base)
            observer.stage('base_read', original_base_read_nd=observer.read_tensor(original_read))
            consumer_identity = identity(base, torch_npu)
            output = self.consume_original_base(base)
            observer.stage('consumed', consumer_input_identity=consumer_identity,
                           consumer=observer.read_tensor(output))
            return output

    class GuardOtherCallsite(torch.nn.Module):
        def forward(self, base):
            row = {'forward': state.forward_index, 'kind': 'other_callsite',
                   'base_before': observer.read_tensor(base)}
            view = unmatched_as_strided(base)
            state.keepalive.append(view)
            row.update(view_created=observer.read_tensor(view), base_after=observer.read_tensor(base),
                       view_written=None, consumer_input_identity=None, consumer=None, write_values=None)
            observer.runtime['guards'].append(row)
            return view

    class GuardUnregisteredBase(torch.nn.Module):
        def forward(self, base, values):
            row = {'forward': state.forward_index, 'kind': 'unregistered_base',
                   'base_before': observer.read_tensor(base)}
            view = selected_as_strided(base)
            state.keepalive.append(view)
            row['view_created'] = observer.read_tensor(view)
            result = view.copy_(values)
            row.update(view_written=observer.read_tensor(result), base_after=observer.read_tensor(base),
                       consumer_input_identity=identity(base, torch_npu),
                       consumer=observer.read_tensor(base + 1), write_values=observer.read_tensor(values))
            observer.runtime['guards'].append(row)
            return result

    class AliasMechanism(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.guards = torch.nn.ModuleDict({'other_callsite': GuardOtherCallsite(),
                                             'unregistered_base': GuardUnregisteredBase()})
            self.invocations = torch.nn.ModuleList([InvocationBlock() for _ in range(3)])

        def forward(self):
            state.forward_index += 1
            if state.forward_index >= len(state.bases):
                raise RuntimeError('No fresh registered base for this experiment forward')
            base = state.bases[state.forward_index]
            state.base = base
            state.forward_active = True
            try:
                state.call_index = None
                state.case_id = 'f' + str(state.forward_index) + '_guard_other_callsite'
                self.guards['other_callsite'](base)
                state.case_id = 'f' + str(state.forward_index) + '_guard_unregistered_base'
                self.guards['unregistered_base'](guard_bases[state.forward_index], guard_values)
                outputs = []
                for call, block in enumerate(self.invocations):
                    state.call_index = call
                    state.case_id = 'f' + str(state.forward_index) + '_c' + str(call)
                    state.current_case = {'case_id': state.case_id, 'forward': state.forward_index,
                                          'call': call, 'base_slot': state.forward_index, 'stages': []}
                    observer.runtime['cases'].append(state.current_case)
                    outputs.append(block(base, writes[call]))
                return tuple(outputs)
            finally:
                state.forward_active = False
                state.current_case = None
                state.case_id = None
                state.call_index = None

    return AliasMechanism().eval()


def internal_option():
    try:
        value = torch_npu._C._npu_getOption('ALLOW_INTERNAL_FORMAT')
        return value.decode(errors='replace') if isinstance(value, bytes) else str(value)
    except BaseException as exc:
        return {'getter_error': repr(exc)}


def main():
    global torch, torch_npu
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('B', 'F'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True, help='Independent CPU logical reference JSON')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--device', type=int, default=0, help='Container-visible device index; logical rank remains zero')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parent
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    if (reference['base_shape'], reference['view_size'], reference['view_stride'], reference['offset']) != ([16, 16], [8, 8], [16, 1], 17):
        raise ValueError('Reference geometry is not this fixed mechanism')
    runtime = {'schema': 'fault15.alias_detach.allhit.runtime.v1', 'status': 'started',
               'pid': os.getpid(), 'rank': 0, 'forwards': [0, 1], 'steps': [0, 1],
               'reference_sha256': sha(reference_bytes),
               'source_sha256': {name: sha((source_root / name).read_bytes())
                                 for name in ('alias_probe.py', 'alias_injector.py')},
               'scope': 'Constructed small-module mechanism; not a model run or historical defect reproduction',
               'observer_stack_marker': 'CommonObserver.read_tensor', 'cases': [], 'guards': [], 'warnings': []}
    receipt = {'schema': 'fault15.alias_detach.allhit.capture.v1', 'status': 'started',
               'capture_requested': args.capture, 'capture_completed': False,
               'steps': [0, 1], 'rank': 0, 'forwards': [0, 1], 'native_files': [], 'dump_nodes': {},
               'full_tensor_files': []}
    injector, debugger = None, None
    custom_api_restorer = None
    registered_custom_apis = []
    control = {'arm': args.arm, 'hits': 0, 'selected_original_calls': 0,
               'unmatched_calls': 0, 'unmatched_original_calls': 0, 'errors': []}
    code = 2
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            import torch
            import torch_npu
            runtime['versions'] = {p: importlib.metadata.version(p) for p in ('torch', 'torch-npu')}
            runtime['module_paths'] = {'torch': torch.__file__, 'torch_npu': torch_npu.__file__}
            runtime['environment'] = {'device_visible_index': args.device,
                                      'rank_environment': os.environ.get('RANK'),
                                      'world_size_environment': os.environ.get('WORLD_SIZE')}
            if os.environ.get('RANK', '0') != '0' or os.environ.get('WORLD_SIZE', '1') != '1':
                raise RuntimeError('Only single-process logical rank0 is supported')
            torch_npu.npu.config.allow_internal_format = True
            runtime['allow_internal_format_requested'] = True
            runtime['allow_internal_format_actual'] = internal_option()
            torch_npu.npu.set_device(args.device)
            device = torch.device('npu', args.device)
            # Construct values independently of the reference; the analyzer owns
            # the oracle comparison. No observation is reused as program input.
            def fresh_private_base():
                data = torch.arange(256, dtype=torch.bfloat16).reshape(16, 16).to(device)
                return torch_npu.npu_format_cast(data, 29)
            bases = [fresh_private_base() for _ in range(2)]
            guard_bases = [fresh_private_base() for _ in range(2)]
            writes = [(-64 * (call + 1) - torch.arange(64, dtype=torch.bfloat16)).reshape(8, 8).to(device)
                      for call in range(3)]
            guard_values = (-16 - torch.arange(64, dtype=torch.bfloat16)).reshape(8, 8).to(device)
            torch_npu.npu.synchronize()
            state = State(bases)
            observer = CommonObserver(runtime, state)
            model = create_model(observer, state, writes, guard_bases, guard_values)
            injector = AliasInjector(torch, torch_npu, args.arm, bases, state, selected_as_strided)
            config = {'task': 'statistics', 'level': 'mix', 'rank': [], 'step': [0, 1],
                      'dump_path': str(out / 'observation'), 'async_dump': False,
                      'statistics': {'scope': [], 'list': [], 'data_mode': ['all'],
                                     'summary_mode': 'statistics', 'tensor_list': []}}
            write(out / 'config.json', config)
            if args.capture:
                from msprobe.pytorch import PrecisionDebugger
                runtime['versions']['mindstudio-probe'] = importlib.metadata.version('mindstudio-probe')
                debugger_source = Path(inspect.getsourcefile(PrecisionDebugger))
                runtime['debugger_source'] = {'path': str(debugger_source),
                                              'sha256': sha(debugger_source.read_bytes())}
                if runtime['debugger_source']['sha256'] != '43bb9d834c736ebeb5c7c89e4770c23e1c6fc813000d12aeee8c2b72753a02fb':
                    raise RuntimeError('msprobe PrecisionDebugger source differs from the verified installation')
                debugger = PrecisionDebugger(config_path=str(out / 'config.json'))
                debugger.start(model=model, rank_id=0)
                # Installed mix hooks omit these two Python entries by default.
                # The official extension wraps their unchanged native functions;
                # it does not rewrite the dump or alter installed package files.
                custom_api_restorer = getattr(debugger, 'restore_custom_api', None)
                restore_owner = 'PrecisionDebugger'
                if not callable(custom_api_restorer):
                    custom_api_restorer = getattr(debugger.service, 'restore_custom_api', None)
                    restore_owner = 'debugger.service'
                if not callable(custom_api_restorer):
                    raise RuntimeError('Official custom API restoration interface is unavailable')
                # Confirm the bound installation's real restoration signature.
                # The reviewed core API requires a module and an API name.
                restore_signature = inspect.signature(custom_api_restorer)
                restore_signature.bind(torch, 'as_strided')
                registration_source = Path(inspect.getsourcefile(debugger.register_custom_api))
                restoration_source = Path(inspect.getsourcefile(custom_api_restorer))
                runtime['capture_api_registration'] = {
                    'interface': 'PrecisionDebugger.register_custom_api',
                    'reason': 'Installed mix Python API support omits torch.as_strided and Tensor.copy_',
                    'registrations': [], 'installed_files_modified': False,
                    'restoration_interface': restore_owner + '.restore_custom_api', 'restored': False,
                    'restoration_signature': str(restore_signature),
                    'restoration_results': [], 'pending_restorations': [],
                    'registration_source_sha256': sha(registration_source.read_bytes()),
                    'restoration_source_sha256': sha(restoration_source.read_bytes())}
                for module, api, prefix in ((torch, 'as_strided', 'Torch'), (torch.Tensor, 'copy_', 'Tensor')):
                    original_api = getattr(module, api)
                    debugger.register_custom_api(module, api, prefix)
                    label = prefix + '.' + api
                    registered_custom_apis.append({'module': module, 'api': api,
                                                   'label': label, 'original': original_api})
                    runtime['capture_api_registration']['registrations'].append(label)
                    runtime['capture_api_registration']['pending_restorations'].append(label)
            injector.install()
            observed_copy = torch.Tensor.copy_
            runtime['completed_forwards'] = []
            for forward in range(2):
                if debugger is not None and forward:
                    debugger.start(model=model, rank_id=0)
                if torch.as_strided is not injector.wrapper or torch.Tensor.copy_ is not observed_copy:
                    raise RuntimeError('Capture lifecycle changed installed native API bindings')
                with torch.no_grad():
                    model()
                torch_npu.npu.synchronize()
                runtime['completed_forwards'].append(forward)
                if debugger is not None:
                    debugger.stop()
                    debugger.step()
            injector.restore()
            if debugger is not None:
                if registered_custom_apis:
                    restore_registered_custom_apis(custom_api_restorer, registered_custom_apis,
                                                   runtime['capture_api_registration'])
                for step in range(2):
                    native_root = out / ('observation/step' + str(step) + '/rank0')
                    for name in ('dump.json', 'stack.json', 'construct.json'):
                        path = native_root / name
                        data = path.read_bytes()
                        json.loads(data)
                        receipt['native_files'].append({'path': str(path.relative_to(out)),
                                                        'sha256': sha(data), 'bytes': len(data)})
                    dump = json.loads((native_root / 'dump.json').read_bytes())
                    if dump['task'] != 'statistics' or dump['level'] != 'mix':
                        raise RuntimeError('Native dump does not match statistics/mix')
                    receipt['dump_nodes'][str(step)] = len(dump['data'])
                receipt['full_tensor_files'] = [str(p.relative_to(out)) for p in out.rglob('*.pt')]
                if receipt['full_tensor_files']:
                    raise RuntimeError('Unexpected tensor binary output')
                receipt['capture_completed'] = True
            runtime['status'] = 'completed'
            receipt['status'] = 'completed'
            code = 0
        except BaseException as exc:
            runtime.update(status='error', exception_type=type(exc).__name__, exception=str(exc),
                           traceback=traceback.format_exc())
            receipt.update(status='error', exception=str(exc))
            if injector is not None and injector.torch.as_strided is injector.wrapper:
                try:
                    injector.restore()
                except BaseException:
                    pass
            if debugger is not None:
                try:
                    debugger.stop()
                except BaseException:
                    pass
            if registered_custom_apis and callable(custom_api_restorer):
                try:
                    restore_registered_custom_apis(custom_api_restorer, registered_custom_apis,
                                                   runtime['capture_api_registration'])
                except BaseException as cleanup_exc:
                    runtime['custom_api_restore_error'] = repr(cleanup_exc)
            traceback.print_exc()
        finally:
            if injector is not None:
                control = injector.record
            runtime['warnings'] = [{'category': item.category.__name__, 'message': str(item.message),
                                    'filename': item.filename, 'lineno': item.lineno} for item in caught]
            runtime['native_warning_capture'] = 'Controller must retain raw stdout and stderr; Python warnings do not cover native C++ warnings.'
            write(out / 'runtime.json', runtime)
            write(out / 'control.json', control)
            receipt['runtime_sha256'] = sha((out / 'runtime.json').read_bytes())
            receipt['control_sha256'] = sha((out / 'control.json').read_bytes())
            receipt['config_sha256'] = sha((out / 'config.json').read_bytes()) if (out / 'config.json').exists() else None
            receipt['exit_code'] = code
            write(out / 'receipt.json', receipt)
    print(json.dumps({'status': receipt['status'], 'capture_completed': receipt['capture_completed'],
                      'output': str(out), 'exit_code': code}), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
