"""Natural whole-request MatMul observation and bounded exact-cache dispatch.

The root and actual LogitsProcessor keep their implementations. Owned module
hooks provide object ownership; TorchDispatchMode observes every actual API
boundary; an always-on native observer supplies the independent denominator.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from .serving import ServingReplaySession, complete_tensor
from .serving_context import require, tensor_metadata


def public_pair(operator, args, kwargs):
    """Only reviewed two-dimensional native MatMul input lowering is eligible."""
    import torch
    import torch_npu
    if operator in ('aten.mm.default', 'aten.matmul.default'):
        a = args[0] if args else kwargs.get('self')
        b = args[1] if len(args) > 1 else kwargs.get('mat2', kwargs.get('other'))
    elif operator == 'aten.linear.default':
        a = args[0] if args else kwargs.get('input')
        weight = args[1] if len(args) > 1 else kwargs.get('weight')
        bias = args[2] if len(args) > 2 else kwargs.get('bias')
        if bias is not None or not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            return None
        b = weight.t()
    else:
        return None
    if (not isinstance(a, torch.Tensor) or not isinstance(b, torch.Tensor)
            or a.ndim != 2 or b.ndim != 2 or a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16
            or a.device != b.device or a.shape[1] != b.shape[0] or a.shape[1] % 2
            or min(*a.shape, *b.shape) <= 0
            or torch_npu.get_npu_format(a) != 2 or torch_npu.get_npu_format(b) != 2):
        return None
    values = [a.untyped_storage().data_ptr(), a.untyped_storage().nbytes(),
              b.untyped_storage().data_ptr(), b.untyped_storage().nbytes(),
              int(a.shape[0]), int(a.shape[1]), int(b.shape[1]),
              a.storage_offset(), b.storage_offset(), *a.stride(), *b.stride()]
    return a, b, values


class Serving39CoverageSession(ServingReplaySession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from . import collector_dispatch
        self.packet_attributes = collector_dispatch.CollectorPacketAttributes()
        self.backend.observer_sha256 = hashlib.sha256(
            Path(__file__).read_bytes() + Path(collector_dispatch.__file__).read_bytes()).hexdigest()
        self.mode = None
        self.module_stack = []
        self.ordinals = Counter()
        self.modules = {id(module): (path or '<root>', module) for path, module in self.model.named_modules()}
        self.dispatches = 0

    def expect(self, request_id):
        super().expect(request_id)
        try:
            self.packet_attributes.install()
        except BaseException as error:
            self._failure(error)
            raise

    def collect(self, *, force=False):
        if self.needs_collect or force:
            self.packet_attributes.restore()
        super().collect(force=force)

    def install(self):
        super().install()
        try:
            for _, module in self.modules.values():
                self.handles.append(module.register_forward_pre_hook(self._module_pre, with_kwargs=True))
                self.handles.append(module.register_forward_hook(self._module_post, with_kwargs=True, always_call=True))
            self.backend.owned_hooks = len(self.handles)
        except BaseException:
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            self.installed = False
            raise

    def _module_pre(self, module, args, kwargs):
        if self.current is not None:
            require(id(module) in self.modules, 'UNREGISTERED_ACTUAL_MODULE')
            self.module_stack.append(module)

    def _module_post(self, module, args, kwargs, output):
        if self.module_stack:
            require(self.module_stack[-1] is module, 'ACTUAL_MODULE_STACK_CHANGED')
            self.module_stack.pop()

    def _start_mode(self):
        from torch.utils._python_dispatch import TorchDispatchMode
        session = self
        require(self.mode is None, 'OVERLAPPING_DISPATCH_OBSERVER')

        class WholeRequestObserver(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return session._dispatch(func, args, kwargs or {})

        self.mode = WholeRequestObserver()
        self.mode.__enter__()

    def _stop_mode(self):
        if self.mode is not None:
            mode, self.mode = self.mode, None
            mode.__exit__(None, None, None)

    def _dispatch(self, function, args, kwargs):
        from demoncat_fault39.coverage_adapter import digest
        self._same_worker()
        require(self.current is not None and self.module_stack, 'DISPATCH_OUTSIDE_ACTUAL_MODULE')
        module = self.module_stack[-1]
        path = self.modules[id(module)][0]
        operator = str(function)
        ordinal = self.ordinals[(id(module), operator)]
        self.ordinals[(id(module), operator)] += 1
        site = (id(module), operator, ordinal)
        pair = public_pair(operator, args, kwargs)
        row = dict(module_object_identity=id(module), module_path=path, operator=operator,
                   ordinal=ordinal, native_non_target_reason='NO_ELIGIBLE_NATIVE_MATMUL_RUN_OBSERVED')
        values = None
        if pair is not None:
            a, b, values = pair
            geometry = [(str(t.dtype), list(t.shape), list(t.stride()), t.storage_offset(),
                         t.untyped_storage().nbytes()) for t in (a, b)]
            row.update(tensor_signature=digest(geometry), tensor_geometry=geometry,
                       input_storage_bytes=[values[1], values[3]], output_bytes=int(a.shape[0] * b.shape[1] * 2))
        self.dispatches += 1
        return self.backend.dispatch(site, row, values, lambda: function(*args, **kwargs))

    def _root_pre(self, module, args, kwargs):
        super()._root_pre(module, args, kwargs)
        try:
            self.ordinals.clear()
            require(not self.module_stack, 'MODULE_STACK_NOT_EMPTY_BETWEEN_REQUESTS')
            self.backend.begin(self.current['actual_internal_request_id'], self.current['input_sha256'])
            self._start_mode()
        except BaseException as error:
            self._failure(error)
            raise

    def _root_post(self, module, args, kwargs, output):
        self._stop_mode()
        if self.current is None:
            return
        if output is None and sys.exc_info()[1] is not None:
            self._failure(sys.exc_info()[1])
            return  # Preserve the actual original exception, not a cleanup error.
        try:
            require(output is not None and output.ndim == 2 and output.shape[0] == len(self.prompt),
                    'COMPLETE_PREFILL_HIDDEN_STATES_REQUIRED')
            _, value, finite = complete_tensor(output[-1:].contiguous())
            require(finite or self.backend.phase in ('fault', 'fault_repeat'), 'HEALTHY_HIDDEN_STATES_NOT_FINITE')
            self.hidden_digest = value
            self.current['root_returned'] = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            if self.boundary_guard is not None:
                self.boundary_guard()
            require(self.current is not None and self.current['root_returned'] and not self.logits_active,
                    'LOGITS_NOT_BOUND_TO_ONE_COMPLETE_FORWARD')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(actual.get('lm_head') is self.head and actual.get('embedding_bias') is None,
                    'LOGITS_HEAD_OR_BIAS_CHANGED')
            _, value, finite = complete_tensor(actual['hidden_states'])
            require(value == self.hidden_digest and (finite or self.backend.phase in ('fault', 'fault_repeat')),
                    'LOGITS_ROWS_NOT_LAST_FRESH_QUERY')
            self.logits_active = True
            self._start_mode()
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        self._stop_mode()
        if not self.logits_active:
            return
        if output is None and sys.exc_info()[1] is not None:
            self._failure(sys.exc_info()[1])
            return
        try:
            require(output is not None and list(output.shape) == self.boundary_report['output_contract'],
                    'COMPLETE_GLOBAL_VOCAB_OUTPUT_REQUIRED')
            cpu, value, finite = complete_tensor(output)
            self.current.update(model_output_sha256=value, model_output_finite=finite,
                                output_shape=list(cpu.shape), top1=int(cpu.float().argmax(dim=-1).item()))
            self.backend.end(value, model_output_finite=finite)
            self.rows.append(self.current)
            self.current = None
            self.needs_collect = True
            self._save()
        except BaseException as error:
            self._failure(error)
            raise
        finally:
            self.logits_active = False

    def _failure(self, error):
        self._stop_mode()
        super()._failure(error)

    def close(self):
        self._same_worker()
        self.collect(force=True)
        self._stop_mode()
        require(not self.backend.active, 'ACTIVE_GROUP_FORWARD_CANNOT_DETACH')
        require(not self.backend.fault_started or self.backend.phase == 'complete',
                'GROUP_OFF_RECOVERY_REQUIRED_BEFORE_DETACH')
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.installed = False
        self.expected = self.current = None
        self.module_stack.clear()
        self.backend.owned_hooks = 0
        require(not self.packet_attributes.owned, 'COLLECTOR_PACKET_ATTRIBUTES_STILL_OWNED')
        if self.backend.phase == 'complete' and not self.backend.recovered:
            self.backend.finalize()
            if self.backend.trigger == 'once':
                self.worker._demoncat39_live_c2_proof = self.backend.proof
        self._save()

    def _save(self):
        super()._save()
        (self.folder / 'collector_dispatch.json').write_text(
            json.dumps(self.packet_attributes.report(), indent=2) + '\n')

    def report(self):
        self.collect()
        return dict(installed=self.installed, completed_requests=len(self.rows),
                    actual_request_ids=[row['actual_internal_request_id'] for row in self.rows],
                    independent_public_dispatches=self.dispatches, owned_hooks=len(self.handles),
                    native_denominator='ALWAYS_ON_NATIVE_MATMUL_OBSERVER',
                    collector_dispatch=self.packet_attributes.report(),
                    expected_request_id=self.expected, receipt=str(self.folder / 'serving_receipt.json'))


def install_fault39_coverage_binding(worker, *, prompt_token_ids, evidence_dir,
                                     collect_statistics=False, trigger='once'):
    from demoncat_fault39.provider import CtypesProvider
    from fault_injection.providers import register_runtime
    from .providers.fault39 import Fault39CoverageReplay
    require(trigger in ('once', 'every_match') and not hasattr(worker, '_demoncat_serving_factory'),
            'ONE_BOUNDED_COVERAGE_DEPLOYMENT_REQUIRED')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'),
            'READY_UNQUARANTINED_WORKER_REQUIRED')
    native = CtypesProvider.attach_preloaded()
    state = native.snapshot()
    require(state.get('runtime_identity') == 'MATCHED_BUILD' and state.get('exact_cache_api_version') == 2
            and state.get('independent_observer_complete') is True, 'MATCHED_V2_BUILD_WITH_INDEPENDENT_OBSERVER_REQUIRED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')),
            'PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
    c2_proof = getattr(worker, '_demoncat39_live_c2_proof', None) if trigger == 'every_match' else None
    require(trigger == 'once' or c2_proof is not None, 'SAME_WORKER_REAL_C2_PROOF_REQUIRED')
    collector = None
    if collect_statistics:
        import msprobe
        from msprobe.pytorch import PrecisionDebugger
        capture = folder / 'statistics'
        capture.mkdir(parents=True, exist_ok=False)
        config = dict(task='statistics', level='mix', precision='high', rank=[], step=[2, 6],
            dump_path=str(capture / 'observation'), async_dump=False,
            statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
        config_path = capture / 'config.json'
        config_path.write_text(json.dumps(config, indent=2) + '\n')
        package = Path(msprobe.__file__).resolve().parent
        files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}
        (capture / 'collector_identity.json').write_text(json.dumps(dict(package=str(package), files=files,
            edited_by_experiment=False, steps={'2': 'reference', '6': 'fault'},
            independent_comparison_inputs='RAW_STATISTICS_CONFIG_STACK_CONSTRUCT_ONLY'), indent=2) + '\n')
        collector = PrecisionDebugger(config_path=str(config_path))
    model = worker.model_runner.model
    def make_session(w, m, backend, n, owner):
        return Serving39CoverageSession(w, m, backend, n, owner, prompt_token_ids=prompt_token_ids,
                                        evidence_dir=folder / 'runtime', collector=collector)
    def factory(actual_model):
        require(actual_model is model and worker.model_runner.model is model, 'MODEL_LOAD_CHANGED')
        return Fault39CoverageReplay(state['build_profile_sha256'], make_session, folder / 'native', trigger, c2_proof)
    register_runtime('39.matmul_range', factory)
    worker._demoncat_serving_factory = factory
    return dict(status='COVERAGE_DEPLOYMENT_BOUND_NOT_QUALIFIED', trigger=trigger,
                build_profile_sha256=state['build_profile_sha256'], core_methods_replaced=False, workers_created=0)
