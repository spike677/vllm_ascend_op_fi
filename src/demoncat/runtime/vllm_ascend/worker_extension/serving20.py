"""Fault20 at the actual post-forward LogitsProcessor serving boundary.

The processor owns the native invocation. Its lm_head is an observed argument,
never a fictitious forward invocation or a module selected by its name.
"""
import hashlib
import json
from pathlib import Path

from .call_binding import same_callable
from .serving import ServingReplaySession, complete_tensor
from .serving_context import require, tensor_metadata


def live_tensor(value):
    import torch_npu
    return dict(tensor_metadata(value), storage_addr=value.untyped_storage().data_ptr(),
                storage_bytes=value.untyped_storage().nbytes(), element_size=value.element_size(),
                actual_format=int(torch_npu.get_npu_format(value)))


class Serving20Session(ServingReplaySession):
    """Share the inspected fresh-prefill validation; use Fault20's own native gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quant_method = self.head.quant_method
        self.quant_apply = self.quant_method.apply
        self.weight = self.head.weight

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            if self.boundary_guard is not None:
                self.boundary_guard()
            require(self.current is not None and self.current['root_returned'] and not self.logits_active,
                    'LOGITS_NOT_BOUND_TO_ONE_COMPLETE_FORWARD')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(module is self.processor and actual.get('lm_head') is self.head
                    and actual.get('embedding_bias') is None
                    and self.head.quant_method is self.quant_method
                    and same_callable(self.head.quant_method.apply, self.quant_apply)
                    and self.head.weight is self.weight, 'ACTUAL_HEAD_OR_QUANT_METHOD_CHANGED')
            hidden, weight = actual['hidden_states'], self.head.weight
            _, hidden_hash, hidden_finite = complete_tensor(hidden)
            _, weight_hash, weight_finite = complete_tensor(weight)
            require(hidden_finite and weight_finite and hidden_hash == self.hidden_digest,
                    'LOGITS_ROWS_NOT_LAST_FRESH_QUERY')
            require(hidden.ndim == weight.ndim == 2 and hidden.shape[0] == 1
                    and hidden.shape[1] == weight.shape[1]
                    and weight.shape[0] >= self.boundary_report['vocab_size'],
                    'ACTUAL_OUTPUT_PROJECTION_GEOMETRY_UNPROVEN')
            transaction = dict(root_forward_returned=True, processor_object_id=id(module),
                               head_argument_object_id=id(self.head), hidden_sha256=hidden_hash,
                               weight_sha256=weight_hash, hidden=live_tensor(hidden), weight=live_tensor(weight),
                               scope='REAL_ROOT_FORWARD_RETURNED_THEN_REAL_PROCESSOR_SCOPE',
                               native_x1_equals_hidden_storage=(self.native.operator == 'aclnnMatmulWeightNz'))
            self.current['live_inputs'] = transaction
            self.context = self.backend.processor_scope(self.processor)
            self.context.__enter__()
            self.backend.begin(self.current['actual_internal_request_id'], self.current['input_sha256'],
                               transaction=transaction)
            self.logits_active = True
        except BaseException as error:
            self._failure(error)
            raise


def _statistics_collector(folder):
    import msprobe
    from msprobe.pytorch import PrecisionDebugger
    capture = folder / 'statistics'
    capture.mkdir(parents=True, exist_ok=False)
    config = dict(task='statistics', level='mix', precision='high', rank=[], step=[2, 5],
                  dump_path=str(capture / 'observation'), async_dump=False,
                  statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
    config_path = capture / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    package = Path(msprobe.__file__).resolve().parent
    files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in package.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}
    (capture / 'collector_identity.json').write_text(json.dumps(dict(
        package=str(package), files=files, edited_by_experiment=False,
        steps={'2': 'reference', '5': 'fault'},
        native_replacement_below_python_input_observation=True,
        independent_comparison_inputs='RAW_STATISTICS_CONFIG_STACK_CONSTRUCT_ONLY'), indent=2) + '\n')
    return PrecisionDebugger(config_path=str(config_path))


def install_fault20_serving_binding(worker, *, prompt_token_ids, evidence_dir,
                                   reviewed_manifest_path, first_live_validation=False,
                                   collect_statistics=False, baseline_control=False):
    """Trusted standard Worker RPC installation; no user layer/slot/address setting."""
    from fault_injection.providers import register_runtime
    from fault_injection.serving_qualification import ServingQualifiedNativeSession, load_reviewed_manifest
    from fault_injection.deployments.matmul import CtypesMatmulRuntime
    from .providers.fault20 import Fault20ServingReplay
    from .serving_context import discover_logits_boundary
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    require(not hasattr(worker, '_demoncat_serving20_factory'), 'SERVING_DEPLOYMENT_ALREADY_BOUND')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'),
            'READY_UNQUARANTINED_WORKER_REQUIRED')
    require(first_live_validation is True, 'EXPLICIT_FIRST_LIVE_VALIDATION_REQUIRED')
    require(type(baseline_control) is bool, 'EXPLICIT_BASELINE_CONTROL_REQUIRED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')),
            'PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
    # Manifest inspection is inert; the actual callback remains owned by the
    # provider lease and is installed atomically paused only at execute.
    manifest_path = Path(reviewed_manifest_path)
    require(manifest_path.is_absolute() and manifest_path.is_file() and not manifest_path.is_symlink()
            and manifest_path.stat().st_size <= 1024 * 1024, 'REVIEWED_MANIFEST_FILE_REQUIRED')
    manifest = json.loads(manifest_path.read_text())
    reviewed = load_reviewed_manifest(manifest_path, build_profile=manifest.get('build_profile_sha256'),
                                     first_live_validation=first_live_validation)
    model = worker.model_runner.model
    boundary = discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor,
        actual_vocab_size=worker.model_runner.vllm_config.model_config.hf_config.vocab_size)
    collector = _statistics_collector(folder) if collect_statistics else None

    def make_session(actual_worker, actual_model, runtime, owner, generation):
        require(actual_worker is worker and actual_model is model and worker.model_runner.model is model,
                'MODEL_LOAD_CHANGED')
        require(type(runtime) is CtypesMatmulRuntime
                and runtime.build_profile == reviewed['build_profile_sha256'], 'REVIEWED_BUILD_PROFILE_MISMATCH')
        runtime.evidence_dir = folder / 'native_proofs'
        backend = ServingQualifiedNativeSession(model, boundary['processor'], boundary['lm_head'],
            boundary['report'], runtime, owner=owner, generation=generation,
            reviewed_manifest_path=manifest_path, first_live_validation=first_live_validation,
            evidence_dir=folder / 'qualification', baseline_control=baseline_control)
        session = Serving20Session(worker, model, backend, runtime, owner,
            prompt_token_ids=prompt_token_ids, evidence_dir=folder / 'runtime', collector=collector)
        require(session.processor is backend.processor and session.head is backend.head,
                'DISCOVERED_PROCESSOR_CHANGED')
        return session

    def factory(actual_model):
        require(actual_model is model and worker.model_runner.model is model, 'MODEL_LOAD_CHANGED')
        return Fault20ServingReplay(reviewed['build_profile_sha256'], make_session)

    register_runtime('20.buffer_substitute', factory)
    worker._demoncat_serving20_factory = factory
    return dict(status='SERVING_DEPLOYMENT_BOUND_NOT_QUALIFIED',
                build_profile_sha256=reviewed['build_profile_sha256'], manifest_sha256=reviewed['manifest_sha256'],
                statistics=bool(collector), first_live_validation=True,
                baseline_control=baseline_control,
                core_methods_replaced=False, workers_created=0,
                coverage_scope='ONE_NATURAL_OUTPUT_PROJECTION_NOT_ALL_ELIGIBLE')


def remove_fault20_serving_binding(worker):
    from fault_injection.providers import unregister_runtime
    import ctypes
    factory = getattr(worker, '_demoncat_serving20_factory', None)
    require(factory is not None, 'SERVING_DEPLOYMENT_NOT_BOUND')
    library = ctypes.CDLL(None)
    for prefix in ('dcat20_weight_nz_', 'dcat20_qmm_'):
        state = getattr(library, prefix + 'state_v1')
        state.argtypes, state.restype = [ctypes.c_void_p], ctypes.c_uint32
        # Owned/installed/inflight/recovery bits must all be clear; build-match
        # and the optional paused-state bit do not represent a live lease.
        require(state(None) & (1 | 2 | 4 | 8) == 0, 'OWNER_CLEAN_REQUIRED')
    unregister_runtime('20.buffer_substitute', factory)
    del worker._demoncat_serving20_factory
    return dict(status='SERVING_DEPLOYMENT_REMOVED')
