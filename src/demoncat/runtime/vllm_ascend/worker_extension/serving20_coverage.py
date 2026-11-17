"""Trusted standard Worker binding for actual whole-request QMM target groups."""
import hashlib
import inspect
import json
from pathlib import Path
import sys

from .quant_group import QuantGroupBackend
from .quant_inventory import PublicQuantInventory
from .serving import ServingReplaySession, complete_tensor
from .serving_context import require
from .serving_quant_inventory import ServingQuantInventorySession


def load_group_review(path):
    from fault_injection.deployments.matmul import CtypesMatmulGroupRuntime, _REQUIRED_BUNDLE
    path = Path(path)
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and path.stat().st_size < 1024 * 1024, 'BOUNDED_NATIVE_GROUP_REVIEW_REQUIRED')
    raw = path.read_bytes()
    review = json.loads(raw)
    require(review.get('schema') == 'demoncat.fault20.group_review.v3'
            and review.get('validation_level') == 'CPU_VERIFIED'
            and review.get('reviewed_contract') == 'fault20.exact_target_set.native_group_v3'
            and _REQUIRED_BUNDLE | {'libopapi_math.so', 'libopapi_math_real.so'} <= set(review.get('required_libraries', [])),
            'EXACT_NATIVE_GROUP_BUILD_REVIEW_REQUIRED')
    for descriptor in [review['build_evidence'], review['profile_evidence'], *review['cpu_evidence']]:
        evidence = Path(descriptor['path'])
        require(evidence.is_absolute() and evidence.is_file() and not evidence.is_symlink()
                and evidence.stat().st_size <= 16 * 1024 * 1024
                and hashlib.sha256(evidence.read_bytes()).hexdigest() == descriptor['sha256'],
                'NATIVE_GROUP_BUILD_REVIEW_BYTES_CHANGED')
    profile = json.loads(Path(review['profile_evidence']['path']).read_text())
    actual = Path(inspect.getfile(CtypesMatmulGroupRuntime)).read_bytes()
    require(profile['profile_sha256'] == review['build_profile_sha256']
            and profile['sources']['fault_injection/deployments/matmul.py'] == hashlib.sha256(actual).hexdigest(),
            'ACTUAL_CTYPES_GROUP_BINDING_NOT_FROM_REVIEWED_BUILD')
    return dict(review, manifest_sha256=hashlib.sha256(raw).hexdigest())


class Serving20CoverageSession(ServingQuantInventorySession):
    def install(self):
        super().install()
        self.backend.owned_hooks = len(self.handles)

    def _root_post(self, module, args, kwargs, output):
        if self.current is None:
            return
        if output is None and sys.exc_info()[1] is not None:
            self._failure(sys.exc_info()[1])
            return
        try:
            require(output is not None and output.ndim == 2 and output.shape[0] == len(self.prompt),
                    'COMPLETE_ACTUAL_PREFILL_HIDDEN_STATES_REQUIRED')
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
                    'ACTUAL_COMPLETE_ROOT_BEFORE_LOGITS_REQUIRED')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(actual.get('lm_head') is self.head and actual.get('embedding_bias') is None, 'ACTUAL_LOGITS_HEAD_CHANGED')
            _, value, finite = complete_tensor(actual['hidden_states'])
            require(value == self.hidden_digest and (finite or self.backend.phase in ('fault', 'fault_repeat')),
                    'ACTUAL_LOGITS_QUERY_ROWS_CHANGED')
            self.logits_active = True
        except BaseException as error:
            self._failure(error)
            raise

    def _logits_post(self, module, args, kwargs, output):
        if self.logits_active and output is None and sys.exc_info()[1] is not None:
            self._failure(sys.exc_info()[1])
            return
        super()._logits_post(module, args, kwargs, output)

    def close(self):
        self._same_worker()
        self.collect(force=True)
        require(not self.backend.active and not self.logits_active, 'ACTIVE_GROUP_FORWARD_CANNOT_DETACH')
        require(not self.backend.fault_started or self.backend.phase == 'complete', 'GROUP_OFF_REQUIRED_BEFORE_DETACH')
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.installed = False
        self.current = self.expected = None
        PublicQuantInventory.close(self.backend)
        self.backend.owned_hooks = 0
        if self.backend.phase == 'complete' and not self.backend.recovered:
            self.backend.finalize()
            if self.backend.trigger == 'once':
                self.worker._demoncat20_live_c2_proof = self.backend.proof
        self.backend.close()
        self._save()

    def _save(self):
        ServingReplaySession._save(self)
        (self.folder / 'public_quant_inventory.json').write_text(json.dumps(
            PublicQuantInventory.report(self.backend), indent=2, allow_nan=False) + '\n')

    def report(self):
        self.collect()
        return dict(installed=self.installed, completed_requests=len(self.rows),
            owned_hooks=len(self.handles), owned_method_observers=len(self.backend.patches),
            owned_api_observers=len(self.backend.api_patches), public_calls=self.backend.public_total,
            native_scope='ALL_ACTUAL_ELIGIBLE_QMM_IN_FIXED_FRESH_PREFILL',
            receipt=str(self.folder / 'serving_receipt.json'))


def statistics_collector(folder):
    import msprobe
    from msprobe.pytorch import PrecisionDebugger
    capture = folder / 'statistics'
    capture.mkdir(parents=True, exist_ok=False)
    config = dict(task='statistics', level='mix', precision='high', rank=[], step=[5, 6],
        dump_path=str(capture / 'observation'), async_dump=False,
        statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
    config_path = capture / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    package = Path(msprobe.__file__).resolve().parent
    files = {path.relative_to(package).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in package.rglob('*') if path.is_file() and path.suffix in ('.py', '.json', '.yaml')}
    (capture / 'collector_identity.json').write_text(json.dumps(dict(package=str(package), files=files,
        edited_by_experiment=False, steps={'5': 'baseline', '6': 'fault'},
        native_replacement_below_python_input_observation=True,
        independent_comparison_inputs='RAW_STATISTICS_CONFIG_STACK_CONSTRUCT_ONLY'), indent=2) + '\n')
    return PrecisionDebugger(config_path=str(config_path))


def install_fault20_coverage_binding(worker, *, prompt_token_ids, evidence_dir,
                                     reviewed_manifest_path='/native/group_review.json', trigger='once', collect_statistics=False):
    from fault_injection.deployments.matmul import CtypesMatmulRuntime, CtypesMatmulGroupRuntime
    from fault_injection.providers import register_runtime
    from fault_injection.quantization import _discover
    from .providers.fault20 import Fault20CoverageReplay
    require(trigger in ('once', 'every_match') and not hasattr(worker, '_demoncat_serving20_factory')
            and not hasattr(worker, '_demoncat_quant_inventory') and not hasattr(worker, '_demoncat_quant_factory'),
            'ONE_EXCLUSIVE_QMM_GROUP_DEPLOYMENT_REQUIRED')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'), 'READY_ACTUAL_WORKER_REQUIRED')
    folder = Path(evidence_dir)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')), 'PRIVATE_GROUP_EVIDENCE_REQUIRED')
    review = load_group_review(reviewed_manifest_path)
    model = worker.model_runner.model
    candidates, rejected = _discover(model, (1,))
    require(candidates, 'ACTUAL_DYNAMIC_QUANT_CANDIDATES_REQUIRED')
    c2_proof = getattr(worker, '_demoncat20_live_c2_proof', None) if trigger == 'every_match' else None
    require(trigger != 'every_match' or c2_proof is not None, 'SAME_WORKER_REAL_C2_PROOF_REQUIRED')
    collector = statistics_collector(folder) if collect_statistics else None
    def make_session(actual_worker, actual_model, inspection_runtime, owner, generation):
        require(actual_worker is worker and actual_model is model and worker.model_runner.model is model
                and type(inspection_runtime) is CtypesMatmulRuntime
                and inspection_runtime.operator == 'aclnnQuantMatmulWeightNz'
                and inspection_runtime.build_profile == review['build_profile_sha256'], 'TRUSTED_QMM_GROUP_WORKER_BUILD_REQUIRED')
        runtime = CtypesMatmulGroupRuntime('20.buffer_substitute', inspection_runtime.worker, folder / 'native_proofs',
                                          operator='aclnnQuantMatmulWeightNz')
        backend = QuantGroupBackend(model, candidates, runtime=runtime, worker_identity=identity['worker'],
            owner=owner, generation=generation, folder=folder / 'group', required_libraries=review['required_libraries'],
            trigger=trigger, c2_proof=c2_proof)
        backend.observer_sha = hashlib.sha256(Path(__file__).read_bytes() +
            Path(inspect.getfile(PublicQuantInventory)).read_bytes()).hexdigest()
        return Serving20CoverageSession(worker, model, backend, runtime, owner,
            prompt_token_ids=prompt_token_ids, evidence_dir=folder / 'runtime', collector=collector)
    def factory(actual_model):
        require(actual_model is model and worker.model_runner.model is model, 'MODEL_LOAD_CHANGED')
        return Fault20CoverageReplay(review['build_profile_sha256'], make_session)
    register_runtime('20.buffer_substitute', factory)
    worker._demoncat_serving20_factory = factory
    return dict(status='GROUP_DEPLOYMENT_BOUND_NOT_QUALIFIED', build_profile_sha256=review['build_profile_sha256'],
        manifest_sha256=review['manifest_sha256'], structural_candidates=len(candidates), structurally_rejected=rejected,
        trigger=trigger, core_methods_replaced=False, workers_created=0)
