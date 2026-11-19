"""One ordinary W8A8 projection through the existing Worker/QuantProvider.

This is a runtime adapter, not the all-projection qualification experiment.
The unchanged Native ABI requires three single-site calibration modes. Every
step consumes a separately enqueued real batch; no control method runs a model.
No census, routed-token adapter, collector, tensor dump or C2/C3 is installed.
"""
from dataclasses import dataclass
import functools
import hashlib
import inspect
import os
from pathlib import Path
import threading

from fault_injection.effect_summary import numeric_change, current_completed_row
from fault_injection.quant_group_native import Config, Lease, QuantGroupNative, key, snapshot as native_snapshot
from fault_injection.deployments.quant_group_c1 import public_views_match, verify_completion
from fault_injection.deployments.quant_group_inventory import verify_public_group_call
from . import serving_quant as serving
from .call_binding import ForwardBinding
from .quant_row_provenance import RequestRowProvenance, digest, metadata, snapshot
from .serving import complete_tensor
from .serving_context import require
from .serving20_qmm import bind_qmm_arguments, same_storage


PHASES = ('normal', 'reference', 'lossless', 'calibration_off', 'fault', 'off')
MODES = dict(reference=1, lossless=2, calibration_off=4, fault=3)
SCOPE = 'ONE_ORDINARY_W8A8_PROJECTION_TP1_B8_FIRST_DECODE'
MAX_READBACK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class QuantProjectionServingBinding:
    make_session: object


def geometry(view):
    return {name: view[name] for name in ('bytes', 'rows', 'columns', 'stride0', 'stride1',
        'offset', 'dtype', 'format', 'dimensions', 'reserved')}


def function_owner(function):
    actual = getattr(function, '__func__', function)
    require(inspect.isfunction(actual), 'QUANT_PYTHON_METHOD_SOURCE_REQUIRED')
    source = Path(inspect.getsourcefile(actual)).resolve()
    return dict(function=actual, code=actual.__code__, globals=actual.__globals__, source=source,
                sha256=hashlib.sha256(source.read_bytes()).hexdigest())


def verify_function(owner):
    require(owner['function'].__code__ is owner['code']
            and owner['function'].__globals__ is owner['globals']
            and hashlib.sha256(owner['source'].read_bytes()).hexdigest() == owner['sha256'],
            'QUANT_CURRENT_METHOD_SOURCE_CHANGED')


def module_forward_owner(module):
    bound = module.forward
    require(getattr(bound, '__self__', None) is module
            and getattr(bound, '__func__', None) is getattr(type(module), 'forward', None)
            and 'forward' not in vars(module), 'UNREVIEWED_INSTANCE_FORWARD_REPLACEMENT')
    return function_owner(bound)


class QuantProjectionServingSession(serving.QuantServingSession):
    """Reuse scheduler/embedding/norm hooks; own only one actual projection.

The six-step plan is explicit Native calibration, not a hidden replay. A fresh
provider/task creates a fresh session; neither attempted nor Native flags reset.
"""
    def __init__(self, worker, model, *, root, build_sha256, owner, generation, worker_identity):
        from fault_injection.quantization import _discover
        from vllm.model_executor.layers.logits_processor import LogitsProcessor
        require(isinstance(owner, str) and owner and type(generation) is int and generation > 0,
                'ACTUAL_WORKER_OWNER_REQUIRED')
        self.worker, self.runner, self.model = worker, worker.model_runner, model
        serving.require_serial_eager(self.runner, expected_max_seqs=8)
        require(self.runner.model is model and not model.training, 'ACTUAL_EVAL_MODEL_REQUIRED')
        require(getattr(self.runner.vllm_config.model_config.hf_config, 'model_type', None) in ('qwen2', 'qwen3'),
                'ORDINARY_QWEN2_OR_QWEN3_ONLY')
        self.thread, self.owner, self.generation = threading.get_ident(), owner, generation
        self.rpc_identity = snapshot(worker_identity)
        self.native = QuantGroupNative.load(root, approved_build_sha256=build_sha256)
        self.candidates, self.rejected = _discover(model, (1,))
        require(self.candidates, 'NO_ORDINARY_TP1_W8A8_CANDIDATE')
        boundary = serving.discover_logits_boundary(model, actual_logits_processor_type=LogitsProcessor,
            actual_vocab_size=self.runner.vllm_config.model_config.hf_config.vocab_size)
        self.processor, self.guard = boundary['processor'], boundary.get('branch_guard')
        self.binding = ForwardBinding(model)
        self.handles, self.patches, self.api_patches = [], [], []
        self.rows, self.seen, self.receipts, self.recovery = [], [], [], []
        self.expected = self.active = self.error = self.embedding = self.normalized = None
        self.target = self.frame = self.current_call = self.current_lease = None
        self.healthy = self.current_observation = None
        self.prefill_result = self.decode_result = None
        self.phase_attempts = set()
        self.pending_callbacks = {}
        self.callback_counts = dict(observed=0, completed=0, admitted=0, passthrough=0)
        self.installed = self.recovered = self.aborted = self.capture_started = False
        self.root_running = False
        self.recovery_only = False
        self.failed_batches = []
        self.capture_scope = 'first_decode'
        self.quant_session = None  # No legacy execution session is installed.
        self.forward_count = 0
        self.row_observer = RequestRowProvenance(max_calls=256)
        self.prefix = None
        self.norm_owner = None
        self.embedding_owner = None
        self._normal_values = None
        self._identity()

    def _identity(self):
        from fault_injection.runtime_discovery import _process_start_identity
        import torch_npu
        actual, configured = self.rpc_identity, self.worker.device
        require(actual['pid'] == os.getpid() and actual['process_start_identity'] == _process_start_identity()
                and self.worker.rank == actual['rank'] == 0 and self.worker.model_runner is self.runner
                and self.runner.model is self.model and threading.get_ident() == self.thread
                and not self.model.training and configured.type == 'npu'
                and configured.index == actual['configured_logical_device'] == actual['logical_device']
                == actual['observed_logical_device'] == int(torch_npu.npu.current_device())
                and actual['worker_start_identity'] and actual['model_instance_id'],
                'SAME_ACTUAL_TP1_INFERENCE_WORKER_REQUIRED')
        return snapshot(actual)

    def state(self):
        self._identity()
        value = self.native.query()
        core, census = value['core'], value['census']
        return dict(owned=int(self.native.config is not None or core['installed'] or value['quarantined']
                        or value['abi_exceptions'] or value['orphan_descriptors']),
            armed=int(bool(core['pending_parent'])),
            pending=core['pending_parent'] + census['preparing'] + census['pending'],
            executing=census['executing'] + int(bool(value['native_stack_active']))
                + value['callback_active'] + value['completion_active'] + int(self.root_running), raw=value)

    def provider_call_leases(self, state):
        return state['pending'] + state['raw']['public_active'] + int(self.current_call is not None)

    def _read(self, tensor):
        require(tensor.numel() * tensor.element_size() <= MAX_READBACK_BYTES,
                'QUANT_COMPLETE_READBACK_BUDGET_EXCEEDED')
        return complete_tensor(tensor)

    def _failure(self, error):
        if self.error is None:
            self.error = repr(error)

    def save(self):
        pass  # Common Host receipts own persistence. No experimental files.

    def _hook(self, function):
        @functools.wraps(function)
        def observed(*args, **kwargs):
            try:
                require(threading.get_ident() == self.thread, 'FOREIGN_QUANT_HOOK_THREAD')
                return function(*args, **kwargs)
            except BaseException as error:
                self._failure(error)
                raise
        return observed

    def phase_name(self):
        if self.recovered:
            return 'complete'
        if self.recovery_only:
            return 'off'
        return PHASES[len(self.rows)] if len(self.rows) < len(PHASES) else 'complete'

    def install(self):
        from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding, ParallelLMHead
        from vllm.model_executor.layers.layernorm import RMSNorm
        require(not self.installed and not any(self.state()[name] for name in ('owned', 'pending', 'executing')),
                'IDLE_UNSPENT_QUANT_SCOPE_REQUIRED')
        require(not self.rows and not self.phase_attempts, 'QUANT_SESSION_CANNOT_BE_REUSED')
        by_id = {id(module): (path, module, method) for path, module, method, _ in self.candidates}
        require(len(by_id) == len(self.candidates), 'ALIASED_PROJECTION_CANDIDATES_REFUSED')
        try:
            self.handles.append(self.model.register_forward_pre_hook(self._hook(self.pre), with_kwargs=True))
            self.handles.append(self.model.register_forward_hook(self._hook(self.post), with_kwargs=True, always_call=True))
            self.handles.append(self.processor.register_forward_hook(self._hook(self.logits), with_kwargs=True, always_call=True))
            for _, module in self.model.named_modules():
                if isinstance(module, VocabParallelEmbedding) and not isinstance(module, ParallelLMHead):
                    self.handles.append(module.register_forward_hook(self._hook(self.embed), with_kwargs=True))
                if isinstance(module, RMSNorm):
                    self.handles.append(module.register_forward_hook(self._hook(self.norm), with_kwargs=True))
            for method in {id(row[2]): row[2] for row in self.candidates}.values():
                original = method.apply
                local, saved = 'apply' in vars(method), vars(method).get('apply')
                source, ownership = function_owner(original), {}
                @functools.wraps(original)
                def observed(module, *args, _method=method, _original=original, _source=source,
                             _ownership=ownership, **kwargs):
                    x = args[0] if args else kwargs.get('x')
                    if not (self.active and self.active.get('decode') and id(module) in by_id
                            and x is self.normalized and (self.target is None or module is self.target['module'])):
                        return _original(module, *args, **kwargs)
                    try:
                        require(self.frame is None and _method.apply is _ownership['observer']
                                and module.quant_method is _method, 'CURRENT_NONREENTRANT_QUANT_METHOD_REQUIRED')
                        verify_function(_source)
                        return self._projection(by_id[id(module)], _source, x,
                                                lambda: _original(module, *args, **kwargs))
                    except BaseException as error:
                        self._failure(error)
                        raise
                ownership['observer'] = observed
                method.apply = observed
                self.patches.append((method, original, observed, local, saved))
            self.installed = True
        except BaseException:
            self.close(abort=True)
            raise

    def expect(self, request_ids):
        require(self.installed and not self.aborted and (self.error is None or self.recovery_only) and self.expected is None
                and self.phase_name() != 'complete', 'IDLE_CURRENT_QUANT_BATCH_REQUIRED')
        require(isinstance(request_ids, list) and len(request_ids) == len(set(request_ids)) == 8
                and all(isinstance(value, str) and value and value not in self.seen for value in request_ids),
                'EIGHT_FRESH_ACTUAL_ENQUEUED_REQUEST_IDS_REQUIRED')
        # Prompt tokens are obtained from the actual runner when this batch is
        # delivered, not caller assertions or an experimental prompt fixture.
        self.expected = {value: dict(logical_id=index, prompt=None) for index, value in enumerate(request_ids)}
        self.prefill_result = self.decode_result = None
        self.forward_count, self.capture_started = 0, False
        self.current_observation = self.prefix = None

    def pre(self, module, args, kwargs):
        from vllm.forward_context import get_forward_context
        try:
            self.state()
            require(self.expected is not None and self.active is None and (self.error is None or self.recovery_only),
                    'UNOWNED_OR_REENTRANT_QUANT_FORWARD')
            self.root_running = True
            if self.expected is not None and self.runner.with_prefill is True:
                for request_id, expected in self.expected.items():
                    request = self.runner.requests.get(request_id)
                    require(request is not None, 'ACTUAL_RUNNER_REQUEST_NOT_DELIVERED')
                    tokens = list(request.prompt_token_ids)
                    require(0 < len(tokens) <= 1024 and all(type(value) is int and value >= 0 for value in tokens),
                            'BOUNDED_ACTUAL_PROMPT_TOKENS_REQUIRED')
                    require(expected['prompt'] is None or expected['prompt'] == tokens,
                            'ACTUAL_PROMPT_CHANGED_WITHIN_BATCH')
                    expected['prompt'] = tokens
            actual = self.binding.bind(module, args, kwargs)
            context = get_forward_context()
            self.active = dict(phase=self.phase_name(), decode=None, returned=False, logits_returned=False)
            self.embedding = self.normalized = None
            self.actual = actual
            self.forward_count += 1
            if self.runner.with_prefill is False:
                require(self.capture_started, 'DECODE_WITHOUT_HOST_CAPTURE_ADVANCE')
                rows = serving.snapshot_first_decode(self.runner, self.model, actual, context, self.expected)
                ids, _, _ = self._read(actual['input_ids'])
                positions, _, _ = self._read(actual['positions'])
                indices, _, _ = self._read(self.runner.logits_indices)
                require(ids.tolist() == rows['input_ids'] and positions.tolist() == rows['row_token_positions']
                        and indices.tolist() == list(range(8)), 'DEVICE_ROW_MAPPING_MISMATCH')
                self.active['decode'] = rows
            if self.guard:
                self.guard()
        except BaseException as error:
            self._failure(error)
            raise

    def begin_decode_capture(self):
        require(self.installed and self.expected is not None and not self.capture_started and self.active is None
                and self.prefill_result is not None and self.decode_result is None and self.forward_count == 1
                and (self.error is None or self.recovery_only), 'COMPLETED_PREFILL_BEFORE_CAPTURE_REQUIRED')
        require(set(self.runner.input_batch.req_ids) == set(self.expected), 'PENDING_DECODE_REQUESTS_CHANGED')
        for request_id, expected in self.expected.items():
            request = self.runner.requests[request_id]
            require(request.prompt_token_ids == expected['prompt'] and len(request.output_token_ids) == 1,
                    'REAL_PREFILL_TOKEN_REQUIRED_BEFORE_CAPTURE')
        self.begin_observation()

    def begin_observation(self):
        require(not self.capture_started and not self.api_patches and self.current_call is None,
                'QUANT_CAPTURE_ALREADY_ACTIVE')
        require(not any(self.state()[name] for name in ('pending', 'executing')), 'NATIVE_QUANT_NOT_IDLE')
        phase = self.phase_name()
        if self.rows:
            require(self.prefill_result['logits_sha256'] == self.rows[0]['prefill']['logits_sha256'],
                    'NORMAL_PREFILL_CHANGED_BEFORE_SELECTED_DECODE')
        if phase == 'reference':
            require(self.target is not None and self.native.config is None, 'CURRENT_NORMAL_BEFORE_NATIVE_CALIBRATION')
            value = bytes.fromhex(digest(dict(owner=self.owner, worker=self._identity())))
            self.native.install(Config(owner_hi=int.from_bytes(value[:8], 'little'),
                owner_lo=int.from_bytes(value[8:16], 'little'), generation=self.generation,
                thread=threading.get_native_id(), device=self.worker.device.index, max_targets=1, max_calls=4,
                max_device_bytes=64 * 1024 * 1024, max_host_bytes=128 * 1024 * 1024), self._observed, self._completed)
        if phase == 'fault':
            require([row['phase'] for row in self.receipts] == ['reference', 'lossless', 'calibration_off']
                    and 'fault' not in self.phase_attempts, 'SINGLE_SITE_NATIVE_CALIBRATION_REQUIRED')
            self.native.seal([self.target['sha256']], digest(dict(scope=SCOPE, owner=self.owner,
                target=self.target['sha256'], normal=self.healthy, receipts=self.receipts)), unresolved_candidates=0)
        import torch_npu
        for name in ('npu_dynamic_quant', 'npu_quant_matmul'):
            original = getattr(torch_npu, name)
            @functools.wraps(original)
            def observed(*args, _name=name, _original=original, **kwargs):
                if self.frame is None:
                    return _original(*args, **kwargs)
                try:
                    return (self._dynamic(_original, args, kwargs) if _name == 'npu_dynamic_quant'
                            else self._qmm(_original, args, kwargs))
                except BaseException as error:
                    self._failure(error)
                    raise
            setattr(torch_npu, name, observed)
            self.api_patches.append((torch_npu, name, original, observed))
        self.capture_started = True

    def embed(self, module, args, kwargs, output):
        if self.active and self.active.get('decode'):
            x = args[0] if args else kwargs.get('input_')
            if x is self.actual['input_ids']:
                require(self.embedding is None, 'ONE_ACTUAL_EMBEDDING_LINEAGE_REQUIRED')
                source = module_forward_owner(module)
                if self.embedding_owner is None:
                    self.embedding_owner = (module, source)
                require(self.embedding_owner[0] is module
                        and self.embedding_owner[1]['function'] is source['function'], 'EMBEDDING_OWNER_CHANGED')
                verify_function(self.embedding_owner[1])
                self.embedding = output

    def norm(self, module, args, kwargs, output):
        if self.active and self.active.get('decode') and self.embedding is not None:
            x = args[0] if args else kwargs.get('x')
            if x is self.embedding and self.normalized is None:
                import torch
                require(not torch.is_grad_enabled() and not module.training, 'INFERENCE_ONLY_QUANT_PREFIX_REQUIRED')
                self.normalized = output[0] if isinstance(output, tuple) else output
                require(self.normalized.ndim == 2 and self.normalized.shape == self.embedding.shape,
                        'ACTUAL_NORMALIZATION_ROW_GEOMETRY_CHANGED')
                owner = module_forward_owner(module)
                if self.norm_owner is None:
                    self.norm_owner = (module, owner)
                require(self.norm_owner[0] is module and self.norm_owner[1]['function'] is owner['function'],
                        'NORMALIZATION_OWNER_CHANGED')
                verify_function(self.norm_owner[1])
                self.row_observer.begin(self.active['decode']['request_ids'][0])
                self.row_observer.seed(self.normalized, dict(verified=True, actual_rows=self.active['decode'],
                    source='ACTUAL_SCHEDULER_EMBEDDING_RMSNORM_SAME_TENSOR_PROJECTION',
                    embedding_source_sha256=self.embedding_owner[1]['sha256'],
                    embedding_module=id(self.embedding_owner[0]),
                    norm_source_sha256=owner['sha256'], norm_module=id(module),
                    embedding=metadata(self.embedding), normalized=metadata(self.normalized)))

    def _projection(self, candidate, source, x, original):
        path, module, method = candidate
        require(self.current_observation is None, 'ONE_SELECTED_PROJECTION_PER_BATCH')
        proof = self.row_observer.query(x)
        require(proof['verified'] and proof['actual_rows'] == self.active['decode'], 'CURRENT_ORDINARY_REQUEST_ROWS_REQUIRED')
        _, input_sha, finite = self._read(x)
        require(finite, 'FINITE_ACTUAL_QUANT_INPUT_REQUIRED')
        if self.target is None:
            require(self.phase_name() == 'normal', 'ACTUAL_NORMAL_AUTOMATIC_SELECTION_REQUIRED')
            identity = dict(scope=SCOPE, worker=self._identity(), path=path, module=id(module), method=id(method),
                weight=id(module.weight), weight_metadata=metadata(module.weight), source_sha256=source['sha256'],
                rows=self.row_signature(self.active['decode']), input_sha256=input_sha,
                input_geometry=geometry(native_snapshot(QuantGroupNative.tensor_view(x, 'input', self.worker.device.index))))
            self.target = dict(identity=identity, sha256=digest(identity), module=module, method=method,
                               weight=module.weight, source=source)
        target = self.target
        require(module is target['module'] and method is target['method'] and module.weight is target['weight']
                and metadata(module.weight) == target['identity']['weight_metadata']
                and source['function'] is target['source']['function'], 'FROZEN_ORDINARY_PROJECTION_OWNER_CHANGED')
        require(input_sha == target['identity']['input_sha256']
                and self.row_signature(self.active['decode']) == target['identity']['rows'],
                'MATCHED_NORMAL_INPUT_PHASE_AND_REQUEST_ROWS_REQUIRED')
        self.frame = dict(module=module, input=x, proof=proof, quant=None, qmm_output=None, dynamic_calls=0, qmm_calls=0)
        try:
            result = original()  # The actual scheduler-owned model call only.
            require(self.frame['dynamic_calls'] == self.frame['qmm_calls'] == 1
                    and serving.same_tensor(result, self.frame['qmm_output']), 'ACTUAL_Q_SCALE_QMM_CONSUMER_REQUIRED')
            require(self.row_observer.query(x)['verified'], 'QUANT_INPUT_ROW_PREFIX_INVALIDATED_BEFORE_CONSUMER')
            self.prefix = self.row_observer.end()
            require(self.prefix['complete'] and not self.prefix['active'], 'ORDINARY_PREFIX_OBSERVER_NOT_COMPLETE')
            self.current_observation['consumer'] = dict(original_calls=1, module=id(module),
                qmm_output=metadata(result), method_source_sha256=source['sha256'])
            return result
        finally:
            self.frame = None

    def _dynamic(self, original, args, kwargs):
        frame, phase = self.frame, self.phase_name()
        x = args[0] if args else kwargs.get('input')
        require(x is frame['input'] and frame['dynamic_calls'] == 0
                and len(args) <= 1 and not (set(kwargs) - {'input', 'smooth_scales', 'group_index'})
                and kwargs.get('smooth_scales') is None and kwargs.get('group_index') is None,
                'EXACT_UNSMOOTHED_ACTUAL_DYNAMIC_QUANT_REQUIRED')
        frame['dynamic_calls'] += 1
        proof = self.row_observer.query(x)
        require(proof['verified'], 'CURRENT_QUANT_PUBLIC_ROW_PROOF_REQUIRED')
        before = self.state()['raw']
        require(not before['census']['overflow'] and not before['census']['errors'], 'COMPLETE_NATIVE_JOURNAL_REQUIRED')
        done = None
        if phase in MODES:
            import torch
            try:
                operator = torch.ops.npu.npu_dynamic_quant.default
            except AttributeError:
                operator = None
            context = dict(input=x, proof=proof, row_context=self.row_observer.capture_native_input_context(),
                operator=operator, schema=None if operator is None else str(operator._schema), completions=[])
            self.current_call, self.current_lease = context, None
            try:
                output, done = self.native.invoke(x, lambda: original(*args, **kwargs))
                require(len(context['completions']) == 1 and context['completions'][0]['call'] == done['call'],
                        'ONE_ACTUAL_NATIVE_COMPLETION_REQUIRED')
            finally:
                self.current_call = None
        else:
            output = original(*args, **kwargs)
        require(isinstance(output, (tuple, list)) and len(output) == 2, 'ACTUAL_PUBLIC_Q_SCALE_PAIR_REQUIRED')
        q, scale = output
        after = self.state()['raw']
        require(not after['census']['overflow'] and not after['census']['errors']
                and 0 < after['last_sequence'] - before['last_sequence'] <= 9, 'BOUNDED_COMPLETE_NATIVE_CALL_REQUIRED')
        events = self.native.raw_range(before['last_sequence'] + 1, after['last_sequence'])
        if done is None:
            execution = verify_public_group_call(events, x, q, scale, threading.get_native_id(),
                                                 expected_device=self.worker.device.index)
            if phase == 'normal':
                self.target['identity']['native_geometry'] = {
                    name: geometry(events[0]['parent'][name]) for name in ('input', 'quant', 'scale')}
                self.target['sha256'] = digest(self.target['identity'])
        else:
            public_views_match(done['call'], x, q, scale, self.worker.device.index)
            execution = verify_completion(done, events, self.current_lease, released=True)
            packet = dict(phase=phase, completion=done, lease=native_snapshot(self.current_lease), raw_events=events,
                          execution=execution)
            packet['sha256'] = digest(packet)
            self.receipts.append(packet)
        values = {}
        for name, tensor in (('input', x), ('quant', q), ('scale', scale)):
            cpu, sha, finite = self._read(tensor)
            require(finite, 'FINITE_COMPLETE_PUBLIC_QUANT_READBACK_REQUIRED:' + name)
            values[name] = (cpu, sha)
        require(values['input'][1] == self.target['identity']['input_sha256'], 'ACTUAL_QUANT_INPUT_CHANGED')
        if phase == 'normal':
            self._normal_values = (values['quant'][0], values['scale'][0])
            self.healthy = {name: value[1] for name, value in values.items()}
        effect = None
        if phase == 'fault':
            effect = dict(comparison='MATCHED_CURRENT_INPUT_NORMAL_VS_FAULT_WITH_NATIVE_CURRENT_REFERENCE',
                request_ids=list(self.active['decode']['request_ids']), forward=self.row_observer.forward_id,
                phase='decode', semantic_position=self.target['identity']['path'], input_sha256=values['input'][1],
                quant=numeric_change(self._normal_values[0], values['quant'][0]),
                scale=numeric_change(self._normal_values[1], values['scale'][0]),
                native_changed_quant_bytes=done['core']['changed_quant_bytes'],
                native_changed_scale_bytes=done['core']['changed_scale_bytes'])
            require(effect['quant']['changed_elements'] > 0 and effect['scale']['changed_elements'] > 0
                    and effect['native_changed_quant_bytes'] > 0 and effect['native_changed_scale_bytes'] > 0,
                    'BOTH_ACTUAL_Q_AND_SCALE_MUST_CHANGE')
        elif phase != 'normal':
            require(all(value[1] == self.healthy[name] for name, value in values.items()),
                    'ACTUAL_NORMAL_Q_SCALE_INPUT_NOT_RESTORED')
        frame['quant'] = output
        self.current_observation = dict(phase=phase, execution=execution, native_parent=events[0]['parent'],
            hashes={name: value[1] for name, value in values.items()}, effect=effect, consumer=None)
        return output

    def _qmm(self, original, args, kwargs):
        import torch
        frame = self.frame
        require(frame['quant'] is not None and frame['qmm_calls'] == 0, 'ONE_ACTUAL_QUANT_BEFORE_QMM_REQUIRED')
        actual = bind_qmm_arguments(args, kwargs)
        q, scale = frame['quant']
        module = frame['module']
        require(serving.same_tensor(actual['x1'], q) and serving.same_tensor(actual['pertoken_scale'], scale),
                'QMM_MUST_CONSUME_CURRENT_Q_AND_SCALE')
        require(self._read(q)[1] == self.current_observation['hashes']['quant']
                and self._read(scale)[1] == self.current_observation['hashes']['scale'],
                'QMM_INPUT_Q_OR_SCALE_CHANGED_AFTER_NATIVE_PUBLICATION')
        require(same_storage(actual['x2'], module.weight) and module.weight.dtype == torch.int8,
                'QMM_ACTUAL_PROJECTION_WEIGHT_REQUIRED')
        owners = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
        require(any(same_storage(actual['scale'], value) for value in owners)
                and (actual.get('bias') is None or any(same_storage(actual['bias'], value) for value in owners)),
                'QMM_ACTUAL_PROJECTION_SCALE_AND_BIAS_REQUIRED')
        require(actual['output_dtype'] == frame['input'].dtype, 'QMM_CURRENT_OUTPUT_DTYPE_REQUIRED')
        frame['qmm_calls'] += 1
        output = original(*args, **kwargs)
        frame['qmm_output'] = output
        return output

    def _observed(self, call):
        raw = self.native.raw(call.token)
        require(raw['kind'] == 1 and raw['parent'] == native_snapshot(call), 'INDEPENDENT_NATIVE_PARENT_REQUIRED')
        require(call.token not in self.pending_callbacks and self.callback_counts['observed'] < 8192,
                'BOUNDED_UNIQUE_NATIVE_CALLBACK_REQUIRED')
        selected = self.current_call is not None
        self.pending_callbacks[call.token] = selected
        self.callback_counts['observed'] += 1
        if not selected:
            self.callback_counts['passthrough'] += 1
            return self.native.pass_call(call)
        self._identity()
        context, phase = self.current_call, self.phase_name()
        require(phase in MODES and phase not in self.phase_attempts and call.thread == threading.get_native_id()
                and call.device == self.worker.device.index and call.lifetime_token and not call.reserved,
                'ACTUAL_UNSPENT_NATIVE_LIFETIME_REQUIRED')
        fresh = self.row_observer.query_native_input(context['input'], public_proof=context['proof'],
            context=context['row_context'], operator=context['operator'], schema=context['schema'])
        require(fresh['actual_rows'] == self.active['decode'], 'ACTUAL_NATIVE_REQUEST_ROWS_CHANGED')
        view = native_snapshot(QuantGroupNative.tensor_view(context['input'], 'input', call.device))
        require(all(native_snapshot(call.input)[name] == value for name, value in view.items() if name != 'descriptor')
                and geometry(view) == self.target['identity']['input_geometry']
                and all(geometry(native_snapshot(getattr(call, name))) == self.target['identity']['native_geometry'][name]
                        for name in ('input', 'quant', 'scale')), 'ACTUAL_NATIVE_INPUT_FORMALS_CHANGED')
        config = self.native.config
        lease = Lease(mode=MODES[phase], owner_hi=config.owner_hi, owner_lo=config.owner_lo, generation=self.generation,
            thread=call.thread, target=key(self.target['sha256']), actual_row_proof=key(digest(dict(proof=fresh,
                native_call=native_snapshot(call), owner=self.owner, generation=self.generation))), expected=call)
        self.phase_attempts.add(phase)  # Admission, including a failed one, consumes the attempt.
        self.current_lease = lease
        self.callback_counts['admitted'] += 1
        return self.native.admit(lease)

    def _completed(self, completion):
        token = completion.call.token
        require(token in self.pending_callbacks, 'CURRENT_NATIVE_COMPLETION_TICKET_REQUIRED')
        selected = self.pending_callbacks[token]
        done = native_snapshot(completion)
        require(done['status'] == 0 and bool(done['admitted']) == selected, 'NATIVE_CALL_FAILED_OR_FOREIGN_INTERVENTION')
        if selected:
            require(self.current_call is not None and self.current_lease is not None, 'LIVE_PUBLIC_NATIVE_OWNER_REQUIRED')
            self.current_call['completions'].append(done)
            end = max(done['core']['reference']['execute_exit'], done['core']['effect']['execute_exit'])
            verify_completion(done, self.native.raw_range(token, end), self.current_lease, released=False)
        self.pending_callbacks.pop(token)
        self.callback_counts['completed'] += 1
        return 0

    def post(self, module, args, kwargs, output):
        first_error = self.error
        try:
            if self.row_observer.active:
                # Failed/unknown target paths keep a retryable owned mode. Do
                # not leave it installed after a normal model return.
                self.prefix = self.row_observer.end()
            if self.active is not None:
                require(output is not None and output.numel() * output.element_size() <= MAX_READBACK_BYTES,
                        'BOUNDED_COMPLETE_HIDDEN_READBACK_REQUIRED')
            super().post(module, args, kwargs, output)
        finally:
            self.root_running = False
            if first_error is not None:
                self.error = first_error

    def logits(self, module, args, kwargs, output):
        if self.active is not None:
            require(output is not None and output.numel() * output.element_size() <= MAX_READBACK_BYTES,
                    'BOUNDED_COMPLETE_LOGITS_READBACK_REQUIRED')
        first_error = self.error
        try:
            super().logits(module, args, kwargs, output)
        finally:
            if first_error is not None:
                self.error = first_error

    def _restore_apis(self):
        while self.api_patches:
            module, name, original, observer = self.api_patches[-1]
            require(getattr(module, name) is observer, 'QUANT_API_OBSERVER_OWNER_CHANGED')
            setattr(module, name, original)
            require(getattr(module, name) is original, 'QUANT_API_RESTORE_UNVERIFIED')
            self.api_patches.pop()

    def collect(self):
        if self.expected is None or self.decode_result is None:
            return
        try:
            require((self.error is None or self.recovery_only) and self.active is None and self.prefill_result is not None
                    and self.forward_count == 2, 'COMPLETE_NORMAL_PREFILL_AND_ONE_B8_DECODE_REQUIRED')
            require(self.current_observation is not None and self.current_observation['consumer'] is not None
                    and self.prefix is not None and self.prefix['complete'], 'COMPLETE_ACTUAL_SINGLE_PROJECTION_REQUIRED')
            state = self.state()
            require(not state['pending'] and not state['executing'] and not self.pending_callbacks,
                    'ACTUAL_NATIVE_CALLS_NOT_COMPLETE')
            phase = self.phase_name()
            if self.rows:
                require(self.prefill_result['logits_sha256'] == self.rows[0]['prefill']['logits_sha256'],
                        'NORMAL_PREFILL_CHANGED_BEFORE_SELECTED_DECODE')
                if phase != 'fault':
                    require(all(self.decode_result[name] == self.rows[0]['decode'][name]
                                for name in ('hidden_sha256', 'logits_sha256')), 'SAME_WORKER_NORMAL_OUTPUT_NOT_RESTORED')
            self._restore_apis()
            if phase == 'fault':
                self._recover_native()
            if phase == 'off':
                require(not any(self.state()[name] for name in ('owned', 'armed', 'pending', 'executing')),
                        'OFF_NATIVE_OWNER_OR_RESOURCES_REMAIN')
            self.rows.append(dict(phase=phase, prefill=snapshot(self.prefill_result), decode=snapshot(self.decode_result),
                                  projection=snapshot(self.current_observation)))
            self.seen.extend(self.expected)
            self.expected = None
            self.recovered = phase == 'off'
        except BaseException as error:
            self._failure(error)
            raise

    def _recover_native(self):
        if self.native.config is None:
            return
        require(self.frame is None and self.current_call is None, 'ACTUAL_PUBLIC_PROJECTION_STILL_ACTIVE')
        state = self.native.query()
        require(not any(state[name] for name in ('public_active', 'native_stack_active', 'callback_active', 'completion_active')),
                'ACTUAL_NATIVE_CALL_STACK_STILL_ACTIVE')
        if state['core']['installed']:
            if state['core']['targets']:
                require(state['core']['targets'] == 1 and self.target is not None, 'SINGLE_NATIVE_TARGET_REQUIRED')
                self.native.stop(self.target['sha256'])
                import torch_npu
                torch_npu.npu.synchronize()
                require(self.native.drain(self.target['sha256']) is True, 'ACTUAL_NATIVE_DRAIN_PENDING')
                self.native.recover()
            else:
                require(self.native.cancel_empty() is True, 'EMPTY_NATIVE_OWNER_DRAIN_PENDING')
        state = self.native.query()
        require(not any(state['core'][name] for name in ('installed', 'pending_parent', 'device_bytes', 'host_bytes',
                    'owned_descriptors', 'owned_workspace_bytes', 'borrowed_workspace_bytes', 'quarantined_bytes'))
                and not any(state['census'][name] for name in ('preparing', 'pending', 'executing'))
                and not any(state[name] for name in ('quarantined', 'abi_exceptions', 'orphan_descriptors')),
                'ACTUAL_QUANT_RESOURCES_NOT_ZERO')
        self.native.remove()
        self.recovery.append(dict(native=state, worker=self._identity(), late_public_restore_requested=False))

    def close(self, *, abort=False):
        if not abort:
            self.collect()
        require(not self.root_running and self.frame is None and self.current_call is None, 'QUANT_PUBLIC_CALL_STILL_ACTIVE')
        if self.row_observer.active:
            self.prefix = self.row_observer.end()
        self._recover_native()  # Any refusal retains hooks, input and provider owner.
        require('fault' not in self.phase_attempts or self.recovered,
                'REAL_SAME_WORKER_OFF_REQUIRED_AFTER_FAULT_ATTEMPT')
        require(not any(self.state()[name] for name in ('owned', 'armed', 'pending', 'executing')),
                'QUANT_RECOVERY_RESOURCES_STILL_LIVE')
        self._restore_apis()
        while self.patches:
            method, original, observer, local, saved = self.patches[-1]
            require(method.apply is observer, 'QUANT_METHOD_OBSERVER_OWNER_CHANGED')
            if local:
                method.apply = saved
            else:
                del method.apply
            self.patches.pop()
        while self.handles:
            self.handles[-1].remove()
            self.handles.pop()
        self.installed = False
        self._normal_values = self.embedding = self.normalized = None
        self.actual = None
        self.row_observer.proofs.clear()
        self.row_observer.events.clear()
        self.row_observer.nodes.clear()
        self.row_observer.parameters.clear()
        self.row_observer.epochs.clear()
        if abort:
            self.aborted = True
            self.active = self.expected = None

    def prepare_recovery(self):
        """Keep the existing owner for a real OFF batch after a failed F.

        Only the common recover action enters this pass-only mode. First error,
        failed request IDs and the spent Native attempt remain immutable.
        """
        require(not self.root_running and self.frame is None and self.current_call is None, 'QUANT_PUBLIC_CALL_STILL_ACTIVE')
        if self.row_observer.active:
            self.prefix = self.row_observer.end()
        self._recover_native()
        if 'fault' not in self.phase_attempts or self.recovered:
            self.close(abort=not self.recovered)
            return
        self._restore_apis()
        require(not any(self.state()[name] for name in ('owned', 'armed', 'pending', 'executing')),
                'RECOVERY_OFF_REQUIRES_CONFIRMED_NATIVE_RELEASE')
        if self.expected is not None:
            self.failed_batches.append(dict(phase=self.phase_name(), request_ids=list(self.expected), error=self.error,
                projection=snapshot(self.current_observation), prefill=snapshot(self.prefill_result),
                decode=snapshot(self.decode_result)))
            self.seen.extend(self.expected)
        self.expected = self.active = None
        self.embedding = self.normalized = self.actual = None
        self.current_observation = None
        self.prefill_result = self.decode_result = None
        self.recovery_only = True

    def report(self):
        effect = next((row['projection']['effect'] for row in self.rows if row['phase'] == 'fault'), None)
        current = current_completed_row(self.rows[-1] if self.rows else None,
                                        pending=self.expected is not None)
        clean = not self.installed and self.native.config is None and not self.row_observer.active
        return snapshot(dict(schema='demoncat.quant.ordinary_projection.v1', scope=SCOPE,
            status='FUNCTIONAL_EFFECT_AND_OFF_OBSERVED' if self.recovered and clean and not self.error else 'PENDING_OR_REFUSED',
            error=self.error, installed=self.installed, recovered=self.recovered, aborted=self.aborted,
            owner=self.owner, generation=self.generation,
            recovery_only=self.recovery_only, failed_batches=self.failed_batches,
            stage=('RECOVERED' if self.recovered else 'RECOVERY_OFF' if self.recovery_only or self.phase_name() == 'off'
                   else 'SINGLE_FAULT_READY' if self.phase_name() == 'fault' else 'CURRENT_SCOPE_CALIBRATION'),
            next_phase=self.phase_name(), next_action='COLLECT' if self.decode_result is not None and self.expected is not None else
                ('FINISH' if self.recovered else 'ENQUEUE_MATCHED_B8_BATCH'),
            calibration=dict(required_real_batches=6, ordinary_execution_after_current_scope_calibration=True,
                native_prevalidation_modes=['reference', 'lossless', 'calibration_off'], hidden_model_calls=0,
                limitation='CURRENT_NATIVE_ABI_REQUIRES_CALIBRATION_PER_NEW_OWNER_NOT_A_THREE_BATCH_API'),
            rows=self.rows, selected_site=None if self.target is None else self.target['identity'],
            structural_candidate_count=len(self.candidates), all_target_inventory_run=False,
            native_build=self.native.identity, native_receipts=self.receipts, callback_counts=self.callback_counts,
            recovery=self.recovery, fault_effect=effect,
            current_request_effect=current['projection']['effect'] if current is not None else None,
            functional_effect_observed=effect is not None, recovery_proven=self.recovered and clean,
            owned_hooks=len(self.handles), owned_method_patches=len(self.patches),
            owned_api_patches=len(self.api_patches), owned_row_modes=int(self.row_observer.active),
            owned_normal_readback_bytes=(sum(value.numel() * value.element_size() for value in self._normal_values)
                                         if self._normal_values is not None else 0),
            native_registered=self.native.config is not None, fault_attempted='fault' in self.phase_attempts,
            actual_fault_hits=sum(row['phase'] == 'fault' for row in self.receipts), c2=False, c3=False,
            e5='NOT_ASSESSED_NO_STATISTICS_RUN',
            main_project_wiring='NOT_AVAILABLE_NOT_ACCEPTED', npu_acceptance='PENDING_NEW_ORDINARY_ENTRY_RUN'))


def install_quant_projection_binding(worker, *, root, build_sha256):
    """Trusted deployment metadata only; the existing provider owns execution."""
    from fault_injection.providers import register_runtime
    require(not hasattr(worker, '_dcat_quant_binding') and not hasattr(worker, '_demoncat_shared_scale_inventory'),
            'EXCLUSIVE_QUANT_DEPLOYMENT_BINDING_REQUIRED')
    require(os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING') == '0' and os.environ.get('TASK_QUEUE_ENABLE') == '0',
            'HOST_DRIVEN_SYNCHRONOUS_QUANT_REQUIRED')
    serving.require_serial_eager(worker.model_runner, expected_max_seqs=8)
    for value in (worker, worker.model_runner):
        require(not any(name in vars(value) for name in ('execute_model', '_prepare_inputs')), 'CORE_OVERRIDE_REFUSED')
    model = worker.model_runner.model
    def make_session(actual_worker, actual_model, *, owner, generation, worker_identity):
        require(actual_worker is worker and actual_model is model and worker.model_runner.model is model, 'FOREIGN_MODEL')
        return QuantProjectionServingSession(worker, model, root=root, build_sha256=build_sha256,
            owner=owner, generation=generation, worker_identity=worker_identity)
    def factory(actual_model):
        require(actual_model is model, 'FOREIGN_MODEL')
        return QuantProjectionServingBinding(make_session)
    register_runtime(serving.MECHANISM, factory)
    worker._dcat_quant_binding = factory
    return dict(status='DEPLOYMENT_BOUND_NOT_ACCEPTED', scope=SCOPE, native_owner_installed=False,
                actual_main_project_wiring=False, c2=False, c3=False, required_real_batches=6)
