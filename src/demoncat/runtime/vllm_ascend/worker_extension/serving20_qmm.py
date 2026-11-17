"""Natural intermediate QMM qualification through standard serving requests.

Only owned hooks and pass-through quantization API observers are installed.
The first actual dynamic projection on the embedding/norm lineage is frozen;
two healthy requests must identify the same real native call before mutation.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path

from .serving import ServingReplaySession, complete_tensor
from .serving20 import live_tensor, _statistics_collector
from .serving_context import require
from .serving_quant import same_tensor


def bind_qmm_arguments(args, kwargs):
    """Reviewed public torch_npu signature; unknown arguments fail closed."""
    names = ('x1', 'x2', 'scale')
    require(len(args) <= 3 and not set(kwargs) - set(names + (
        'offset', 'pertoken_scale', 'bias', 'output_dtype')), 'UNREVIEWED_PUBLIC_QMM_SIGNATURE')
    bound = dict(zip(names, args))
    require(not set(bound).intersection(kwargs), 'DUPLICATE_PUBLIC_QMM_ARGUMENT')
    bound.update(kwargs)
    require(set(names) <= set(bound) and bound.get('offset') is None
            and bound.get('pertoken_scale') is not None, 'DYNAMIC_QMM_SCALE_CONTRACT_REQUIRED')
    return bound


def same_storage(a, b):
    return (a is not None and b is not None and a.device == b.device
            and a.untyped_storage()._cdata == b.untyped_storage()._cdata
            and a.untyped_storage().nbytes() == b.untyped_storage().nbytes())


class Serving20QmmSession(ServingReplaySession):
    """Input mapping comes from the real root; output completion from real logits."""

    def __init__(self, *args, candidates, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidates = tuple(candidates)
        self.by_id = {id(module): (path, module, method, chain) for path, module, method, chain in candidates}
        self.patches, self.api_patches = [], []
        self.selected = self.embedding = self.normalized = self.actual_input_ids = None
        self.quant_call = self.quant_result = None
        self.quant_outputs = self.qmm_calls = 0
        self.selection_report = None

    def install(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
        from vllm.model_executor.layers.layernorm import RMSNorm
        super().install()
        try:
            for _, module in self.model.named_modules():
                if isinstance(module, VocabParallelEmbedding) and not isinstance(module, ParallelLMHead):
                    self.handles.append(module.register_forward_hook(self._embedding, with_kwargs=True))
                if isinstance(module, RMSNorm):
                    self.handles.append(module.register_forward_hook(self._normalizer, with_kwargs=True))
            for method in {id(row[2]): row[2] for row in self.candidates}.values():
                original = method.apply
                local, saved = 'apply' in vars(method), vars(method).get('apply')
                source = Path(inspect.getsourcefile(original))
                source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
                ownership = {}

                @functools.wraps(original)
                def apply(module, *args, _original=original, _method=method, _source=source_hash,
                          _ownership=ownership, **kwargs):
                    x = args[0] if args else kwargs.get('x')
                    match = (self.current is not None and not self.current['root_returned']
                             and id(module) in self.by_id and same_tensor(x, self.normalized))
                    if match and self.selected is None:
                        require(self.backend.phase == 'discovery_1', 'HEALTHY_FIRST_PROJECTION_REQUIRED')
                        self.selected = module
                        self.selection_report = dict(path=self.by_id[id(module)][0],
                            object_identity=id(module), method_class=type(_method).__module__ + '.' + type(_method).__qualname__,
                            weight_object_identity=id(module.weight),
                            quant_apply_source_sha256=_source,
                            selection='FIRST_ACTUAL_EMBED_NORM_DYNAMIC_PROJECTION',
                            candidate_role='DYNAMIC_W8A8_LINEAR_FROM_EMBEDDING_NORMALIZER',
                            structural_candidates=len(self.candidates), all_eligible_covered=False)
                    selected = match and module is self.selected
                    if not selected:
                        return _original(module, *args, **kwargs)
                    require(self.quant_call is None and self.qmm_calls == 0
                            and module.quant_method is _method and _method.apply is _ownership['observer']
                            and id(module.weight) == self.selection_report['weight_object_identity'],
                            'SELECTED_PROJECTION_REENTERED_OR_CHANGED')
                    self.quant_call = dict(module=module, input=x, method=_method, apply=_original, source=_source)
                    self.quant_result = None
                    self.quant_outputs = 0
                    try:
                        result = _original(module, *args, **kwargs)
                        require(self.quant_outputs == 1 and self.qmm_calls == 1,
                                'ONE_REAL_DYNAMIC_QUANT_AND_QMM_REQUIRED')
                        return result
                    finally:
                        self.quant_call = self.quant_result = None

                ownership['observer'] = apply
                method.apply = apply
                self.patches.append((method, original, apply, local, saved))
        except BaseException:
            self._restore_methods()
            for handle in reversed(self.handles):
                handle.remove()
            self.handles.clear()
            self.installed = False
            raise

    def expect(self, request_id):
        super().expect(request_id)
        # Install after msprobe, so the observer does not replace/bypass its
        # real Python argument capture. Restore before collector.stop().
        import torch_npu
        require(not self.api_patches, 'QUANT_API_OBSERVER_ALREADY_ACTIVE')
        for name, callback in (('npu_dynamic_quant', self._dynamic_quant), ('npu_quant_matmul', self._qmm)):
            original = getattr(torch_npu, name)

            @functools.wraps(original)
            def observe(*args, _original=original, _callback=callback, **kwargs):
                return _callback(_original, args, kwargs)

            setattr(torch_npu, name, observe)
            self.api_patches.append((torch_npu, name, original, observe))

    def _dynamic_quant(self, original, args, kwargs):
        output = original(*args, **kwargs)
        if self.quant_call is not None:
            import torch
            x = args[0] if args else kwargs.get('input')
            require(same_tensor(x, self.quant_call['input']) and self.quant_outputs == 0
                    and isinstance(output, tuple) and len(output) == 2, 'REAL_DYNAMIC_QUANT_LINEAGE_REQUIRED')
            q, scale = output
            require(q.dtype == torch.int8 and q.shape == x.shape
                    and scale.dtype == torch.float32 and scale.numel() == x.shape[0],
                    'ACTUAL_DYNAMIC_QUANT_GEOMETRY_REQUIRED')
            self.quant_result = output  # Strong ownership until real QMM returns.
            self.quant_outputs += 1
        return output

    def _qmm(self, original, args, kwargs):
        if self.quant_call is None:
            return original(*args, **kwargs)
        try:
            import torch
            self._same_worker()
            call, actual = self.quant_call, bind_qmm_arguments(args, kwargs)
            module, x = call['module'], call['input']
            require(self.current is not None and not self.current['root_returned']
                    and self.quant_outputs == 1 and self.qmm_calls == 0, 'QMM_OUTSIDE_ACTUAL_PROJECTION')
            q, per_token_scale = self.quant_result
            require(same_tensor(actual['x1'], q) and same_tensor(actual['pertoken_scale'], per_token_scale)
                    and not same_storage(q, x), 'QMM_NOT_REAL_DYNAMIC_QUANT_OUTPUT')
            require(same_storage(actual['x2'], module.weight) and module.weight.dtype == torch.int8,
                    'QMM_NOT_ACTUAL_PROJECTION_WEIGHT')
            owners = [value for _, value in module.named_parameters(recurse=False)]
            owners += [value for _, value in module.named_buffers(recurse=False)]
            require(any(same_storage(actual['scale'], value) for value in owners),
                    'QMM_WEIGHT_SCALE_NOT_OWNED_BY_PROJECTION')
            if actual.get('bias') is not None:
                require(any(same_storage(actual['bias'], value) for value in owners),
                        'QMM_BIAS_NOT_OWNED_BY_PROJECTION')
            require(actual.get('output_dtype') in (torch.float16, torch.bfloat16)
                    and actual['output_dtype'] == x.dtype,
                    'QMM_FLOAT_OUTPUT_DTYPE_REQUIRED')
            values = dict(x1=actual['x1'], x2=actual['x2'], x1_scale=per_token_scale, x2_scale=actual['scale'])
            if actual.get('bias') is not None:
                values['bias'] = actual['bias']
            hashes = {}
            for name, value in dict(values, float_input=x).items():
                _, digest, finite = complete_tensor(value)
                require(finite, 'FINITE_PREINJECTION_PROJECTION_INPUT_REQUIRED:' + name)
                hashes[name] = digest
            transaction = dict(root_forward_active=True, projection_object_id=id(module),
                quant_method_object_id=id(call['method']), quant_apply_source_sha256=call['source'],
                quant_apply_live_identity=id(call['method'].apply),
                source='ACTUAL_QUANT_APPLY_AND_PUBLIC_QMM', input_hashes=hashes,
                float_input=live_tensor(x), formals={name: live_tensor(value) for name, value in values.items()},
                output_dtype=str(actual['output_dtype']),
                quantization_output_storage_verified=True, native_x1_equals_float_input_storage=False,
                selection=self.selection_report, actual_request_input_sha256=self.current['input_sha256'])
            with self.backend.projection_scope(module):
                self.backend.begin(self.current['actual_internal_request_id'], self.current['input_sha256'],
                                   transaction=transaction)
                try:
                    result = original(*args, **kwargs)
                finally:
                    self.backend.pause_projection()
            require(result is not None and list(result.shape) == [q.shape[0], actual['x2'].shape[1]],
                    'COMPLETE_PROJECTION_OUTPUT_GEOMETRY_REQUIRED')
            self.qmm_calls += 1
            self.current['projection_transaction'] = transaction
            return result
        except BaseException as error:
            self._failure(error)
            raise

    def _root_pre(self, module, args, kwargs):
        super()._root_pre(module, args, kwargs)
        self.actual_input_ids = self.root_binding.bind(module, args, kwargs)['input_ids']
        self.embedding = self.normalized = None
        self.qmm_calls = self.quant_outputs = 0

    def _embedding(self, module, args, kwargs, output):
        if self.current is not None and not self.current['root_returned']:
            x = args[0] if args else kwargs.get('input_')
            if same_tensor(x, self.actual_input_ids):
                require(self.embedding is None, 'MULTIPLE_EMBEDDING_LINEAGES')
                self.embedding = output

    def _normalizer(self, module, args, kwargs, output):
        if self.current is not None and not self.current['root_returned'] and self.embedding is not None:
            x = args[0] if args else kwargs.get('x')
            if same_tensor(x, self.embedding) and self.normalized is None:
                self.normalized = output[0] if isinstance(output, tuple) else output
                require(self.normalized.ndim == 2 and self.normalized.shape == self.embedding.shape,
                        'NORM_ROW_GEOMETRY_CHANGED')

    def _root_post(self, module, args, kwargs, output):
        super()._root_post(module, args, kwargs, output)
        if self.current is not None:
            try:
                require(self.qmm_calls == 1 and self.backend.active, 'NATURAL_PROJECTION_QMM_NOT_OBSERVED')
            except BaseException as error:
                self._failure(error)
                raise

    def _logits_pre(self, module, args, kwargs):
        try:
            self._same_worker()
            if self.boundary_guard is not None:
                self.boundary_guard()
            require(self.current is not None and self.current['root_returned'] and not self.logits_active
                    and self.backend.active and self.qmm_calls == 1, 'COMPLETE_ROOT_AFTER_QMM_REQUIRED')
            actual = self.logits_binding.bind(module, args, kwargs)
            require(module is self.processor and actual.get('lm_head') is self.head
                    and actual.get('embedding_bias') is None, 'ACTUAL_LOGITS_BOUNDARY_CHANGED')
            _, digest, finite = complete_tensor(actual['hidden_states'])
            require(finite and digest == self.hidden_digest, 'LOGITS_ROWS_NOT_LAST_FRESH_QUERY')
            self.logits_active = True
        except BaseException as error:
            self._failure(error)
            raise

    def _restore_apis(self):
        for module, name, original, observer in reversed(self.api_patches):
            require(getattr(module, name) is observer, 'QUANT_API_OBSERVER_OWNER_CHANGED')
            setattr(module, name, original)
        self.api_patches.clear()

    def _restore_methods(self):
        for method, original, observer, local, saved in reversed(self.patches):
            require(method.apply is observer, 'QUANT_METHOD_OBSERVER_OWNER_CHANGED')
            if local:
                method.apply = saved
            else:
                del method.apply
        self.patches.clear()

    def collect(self, *, force=False):
        if self.needs_collect or force:
            self._restore_apis()
        super().collect(force=force)

    def close(self):
        super().close()
        self._restore_methods()

    def report(self):
        value = super().report()
        value.update(projection_scope=self.selection_report,
                     coverage_scope='ONE_FIRST_EMBED_NORM_QUANT_PROJECTION_NOT_ALL_ELIGIBLE',
                     core_methods_replaced=False)
        return value


def install_fault20_qmm_serving_binding(worker, *, prompt_token_ids, evidence_dir,
                                       reviewed_manifest_path, first_live_validation=False,
                                       collect_statistics=False, baseline_control=False):
    from fault_injection.providers import register_runtime
    from fault_injection.quantization import _discover
    from fault_injection.serving_qualification import ProjectionQualifiedNativeSession, load_reviewed_manifest
    from fault_injection.deployments.matmul import CtypesMatmulRuntime
    from .providers.fault20 import Fault20ServingReplay
    require(not hasattr(worker, '_demoncat_serving20_factory'), 'SERVING_DEPLOYMENT_ALREADY_BOUND')
    identity = worker.demoncat_worker_identity()
    require(identity['status'] == 'WORKER_IDENTIFIED' and not identity.get('quarantined'),
            'READY_UNQUARANTINED_WORKER_REQUIRED')
    require(type(baseline_control) is bool, 'EXPLICIT_BASELINE_CONTROL_REQUIRED')
    folder, manifest_path = Path(evidence_dir), Path(reviewed_manifest_path)
    require(folder.is_absolute() and folder.resolve().is_relative_to(Path('/evidence')),
            'PRIVATE_EVIDENCE_DIRECTORY_REQUIRED')
    require(manifest_path.is_absolute() and manifest_path.is_file() and not manifest_path.is_symlink()
            and manifest_path.stat().st_size <= 1024 * 1024, 'REVIEWED_MANIFEST_FILE_REQUIRED')
    manifest = json.loads(manifest_path.read_text())
    reviewed = load_reviewed_manifest(manifest_path, build_profile=manifest.get('build_profile_sha256'),
                                     first_live_validation=first_live_validation)
    model = worker.model_runner.model
    candidates, rejected = _discover(model, (1,))
    require(bool(candidates), 'NO_ACTUAL_DYNAMIC_QUANT_PROJECTION')
    collector = _statistics_collector(folder) if collect_statistics else None

    def make_session(actual_worker, actual_model, runtime, owner, generation):
        require(actual_worker is worker and actual_model is model and worker.model_runner.model is model,
                'MODEL_LOAD_CHANGED')
        require(type(runtime) is CtypesMatmulRuntime and runtime.operator == 'aclnnQuantMatmulWeightNz'
                and runtime.build_profile == reviewed['build_profile_sha256'], 'REVIEWED_QMM_BUILD_PROFILE_MISMATCH')
        runtime.evidence_dir = folder / 'native_proofs'
        backend = ProjectionQualifiedNativeSession(model, candidates, runtime, owner=owner, generation=generation,
            reviewed_manifest_path=manifest_path, first_live_validation=first_live_validation,
            evidence_dir=folder / 'qualification', baseline_control=baseline_control)
        return Serving20QmmSession(worker, model, backend, runtime, owner, candidates=candidates,
            prompt_token_ids=prompt_token_ids, evidence_dir=folder / 'runtime', collector=collector)

    def factory(actual_model):
        require(actual_model is model and worker.model_runner.model is model, 'MODEL_LOAD_CHANGED')
        return Fault20ServingReplay(reviewed['build_profile_sha256'], make_session)

    register_runtime('20.buffer_substitute', factory)
    worker._demoncat_serving20_factory = factory
    return dict(status='SERVING_QMM_DEPLOYMENT_BOUND_NOT_QUALIFIED',
                build_profile_sha256=reviewed['build_profile_sha256'], manifest_sha256=reviewed['manifest_sha256'],
                structural_candidates=[dict(path=p, method_chain=c) for p, _, _, c in candidates],
                rejected_candidates=rejected, statistics=bool(collector), first_live_validation=True,
                baseline_control=baseline_control, core_methods_replaced=False, workers_created=0,
                coverage_scope='ONE_FIRST_EMBED_NORM_QUANT_PROJECTION_NOT_ALL_ELIGIBLE')
