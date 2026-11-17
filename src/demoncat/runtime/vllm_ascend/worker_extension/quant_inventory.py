"""Read-only all-candidate dynamic-quant/QMM lineage, independent of injection.

Structural candidates, actual public chains and native-qualified targets are
different sets. This observer issues no native qualification or injection lease.
It records every public dynamic_quant/QMM call, including unowned/rejected calls,
and never chooses the first projection as a coverage denominator.
"""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import threading

from .serving20 import live_tensor
from .serving20_qmm import bind_qmm_arguments, same_storage
from .serving_quant import same_tensor
from .serving_context import require


class PublicQuantInventory:
    def __init__(self, model, candidates, *, worker_identity, max_calls=4096):
        self.model = model
        self.candidates = tuple(candidates)
        self.by_id = {id(m): (path, m, method, chain) for path, m, method, chain in self.candidates}
        require(len(self.by_id) == len(self.candidates) and bool(self.by_id), 'UNALIASED_ACTUAL_DYNAMIC_CANDIDATES_REQUIRED')
        self.worker_identity = dict(worker_identity)
        require(type(max_calls) is int and 0 < max_calls <= 4096, 'BOUNDED_PUBLIC_CALL_CAPACITY_REQUIRED')
        self.max_calls = max_calls
        self.thread = threading.get_ident()
        self.patches = []
        self.api_patches = []
        self.rows = []
        self.public_calls = []
        self.public_total = self.overflow_calls = 0
        self.current = self.frame = None
        self.installed = False
        self.requests = set()
        self.failure = None
        self.fault_started = self.recovering = self.recovered = False

    @property
    def active(self):
        return self.current is not None

    @property
    def phase(self):
        return ('inventory_1', 'inventory_2')[len(self.rows)] if len(self.rows) < 2 else 'complete'

    def _thread(self):
        require(threading.get_ident() == self.thread, 'FOREIGN_QUANT_OBSERVER_THREAD')

    def install(self):
        self._thread()
        require(not self.installed and not self.patches, 'QUANT_INVENTORY_ALREADY_INSTALLED')
        try:
            for method in {id(row[2]): row[2] for row in self.candidates}.values():
                original = method.apply
                local, saved = 'apply' in vars(method), vars(method).get('apply')
                source = Path(inspect.getsourcefile(original))
                source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
                ownership = {}

                @functools.wraps(original)
                def apply(module, *args, _method=method, _original=original,
                          _source=source_sha, _ownership=ownership, **kwargs):
                    if self.current is None or id(module) not in self.by_id:
                        return _original(module, *args, **kwargs)
                    self._thread()
                    require(self.frame is None and _method.apply is _ownership['observer'],
                            'REENTRANT_OR_REPLACED_QUANT_OBSERVER')
                    x = args[0] if args else kwargs.get('x')
                    frame = dict(module=module, input=x, quant=None, qmm_output=None, dynamic_calls=0, qmm_calls=0,
                        row=dict(path=self.by_id[id(module)][0], module_object_identity=id(module),
                            quant_method_object_identity=id(_method), method_source_sha256=_source,
                            weight_object_identity=id(getattr(module, 'weight', None)),
                            request_ids=list(self.current['request_ids']), forward_id=self.current['forward_id'],
                            rank=self.worker_identity['rank'], public_chain_verified=False, native_qualified=False,
                            errors=[], api_call_indices=[]))
                    if getattr(module, 'quant_method', None) is not _method:
                        frame['row']['errors'].append('QUANT_METHOD_NOT_CURRENT_MODULE_OWNER')
                    self.frame = frame
                    try:
                        result = _original(module, *args, **kwargs)
                        if frame['dynamic_calls'] != 1 or frame['qmm_calls'] != 1:
                            frame['row']['errors'].append('ONE_ACTUAL_DYNAMIC_QUANT_AND_QMM_REQUIRED')
                        if not same_tensor(result, frame['qmm_output']):
                            frame['row']['errors'].append('PROJECTION_RESULT_NOT_ACTUAL_QMM_OUTPUT')
                        frame['row']['public_chain_verified'] = not frame['row']['errors']
                        self._projection_complete(frame, result)
                        return result
                    except BaseException as error:
                        frame['row']['errors'].append('ORIGINAL_PROJECTION_FAILED:' + type(error).__name__)
                        raise
                    finally:
                        frame['row'].update(dynamic_calls=frame['dynamic_calls'], qmm_calls=frame['qmm_calls'])
                        self.current['projections'].append(frame['row'])
                        self.frame = None
                ownership['observer'] = apply
                method.apply = apply
                self.patches.append((method, original, apply, local, saved))
            self.installed = True
        except BaseException:
            self.close()
            raise

    def _projection_complete(self, frame, result):
        pass

    def begin(self, forward_id, input_sha256, *, request_ids, row_mapping=None):
        self._thread()
        require(self.installed and self.current is None and self.phase != 'complete' and not self.failure,
                'QUANT_INVENTORY_FORWARD_NOT_AVAILABLE')
        require(isinstance(forward_id, str) and forward_id and forward_id not in self.requests
                and isinstance(request_ids, tuple) and request_ids and len(set(request_ids)) == len(request_ids)
                and all(isinstance(request, str) and request for request in request_ids),
                'ACTUAL_UNREUSED_FORWARD_AND_REQUEST_IDENTITIES_REQUIRED')
        require(isinstance(input_sha256, str) and len(input_sha256) == 64, 'COMPLETE_ACTUAL_INPUT_HASH_REQUIRED')
        self.current = dict(forward_id=forward_id, phase=self.phase, request_ids=list(request_ids),
            input_sha256=input_sha256, projections=[], public_start=len(self.public_calls), row_mapping=row_mapping)
        self.requests.add(forward_id)
        import torch_npu
        try:
            for name in ('npu_dynamic_quant', 'npu_quant_matmul'):
                original = getattr(torch_npu, name)
                @functools.wraps(original)
                def observe(*args, _name=name, _original=original, **kwargs):
                    return self._api(_name, _original, args, kwargs)
                setattr(torch_npu, name, observe)
                self.api_patches.append((torch_npu, name, original, observe))
        except BaseException:
            self._restore_apis()
            self.current = None
            raise

    def _api(self, name, original, args, kwargs):
        self._thread()
        require(self.current is not None, 'PUBLIC_QUANT_CONTEXT_MISSING')
        self.public_total += 1
        if len(self.public_calls) >= self.max_calls:
            self.overflow_calls += 1
            self.failure = 'PUBLIC_QUANT_DENOMINATOR_CAPACITY_EXCEEDED'
            # Keep the read-only observer transparent; explicitly retain the
            # missing-denominator count and refuse a complete inventory.
            return original(*args, **kwargs)
        frame = self.frame
        row = dict(operator=name, forward_id=self.current['forward_id'], request_ids=self.current['request_ids'],
            module_object_identity=id(frame['module']) if frame else None,
            original_calls=0, status='PENDING', errors=[])
        index = len(self.public_calls)
        self.public_calls.append(row)  # Always before any verification/admission.
        if frame:
            frame['row']['api_call_indices'].append(index)
        else:
            row['errors'].append('PUBLIC_CALL_OUTSIDE_DISCOVERED_DYNAMIC_PROJECTION')
        try:
            if name == 'npu_quant_matmul' and frame:
                frame['qmm_calls'] += 1
                try:
                    self._qmm_lineage(frame, row, args, kwargs)
                except Exception as error:
                    row['errors'].append(str(error))
                    frame['row']['errors'].append(str(error))
            row['original_calls'] += 1
            output = original(*args, **kwargs)
            if name == 'npu_quant_matmul' and frame:
                frame['qmm_output'] = output
                row['actual_output'] = live_tensor(output)
            if name == 'npu_dynamic_quant' and frame:
                frame['dynamic_calls'] += 1
                try:
                    self._dynamic_lineage(frame, row, args, kwargs, output)
                except Exception as error:
                    row['errors'].append(str(error))
                    frame['row']['errors'].append(str(error))
            row['status'] = 'OBSERVED_UNQUALIFIED' if not row['errors'] else 'PUBLIC_LINEAGE_REJECTED'
            return output
        except BaseException as error:
            row.update(status='ORIGINAL_API_FAILED', exception=repr(error))
            raise

    def _dynamic_lineage(self, frame, row, args, kwargs, output):
        import torch
        x = args[0] if args else kwargs.get('input')
        require(same_tensor(x, frame['input']) and frame['dynamic_calls'] == 1
                and isinstance(output, tuple) and len(output) == 2,
                'ACTUAL_DYNAMIC_QUANT_PROJECTION_LINEAGE_MISSING')
        q, scale = output
        require(q.dtype == torch.int8 and q.shape == x.shape and x.ndim == 2
                and scale.dtype == torch.float32 and scale.numel() == x.shape[0]
                and q.device == scale.device == x.device
                and not same_storage(x, q), 'ACTUAL_DYNAMIC_QUANT_GEOMETRY_UNSUPPORTED')
        frame['quant'] = output
        row.update(float_input=live_tensor(x), quantized_output=live_tensor(q), per_token_scale=live_tensor(scale))

    def _qmm_lineage(self, frame, row, args, kwargs):
        import torch
        actual = bind_qmm_arguments(args, kwargs)
        require(frame['quant'] is not None and frame['dynamic_calls'] == frame['qmm_calls'] == 1,
                'QMM_WITHOUT_ONE_ACTUAL_DYNAMIC_QUANT')
        module = frame['module']
        q, scale = frame['quant']
        require(same_tensor(actual['x1'], q) and same_tensor(actual['pertoken_scale'], scale),
                'QMM_INPUT_NOT_ACTUAL_DYNAMIC_QUANT_OUTPUT_STORAGE')
        require(same_storage(actual['x2'], module.weight) and module.weight.dtype == torch.int8,
                'QMM_WEIGHT_NOT_OWNED_BY_ACTUAL_PROJECTION')
        owners = [value for _, value in module.named_parameters(recurse=False)]
        owners += [value for _, value in module.named_buffers(recurse=False)]
        require(any(same_storage(actual['scale'], value) for value in owners),
                'QMM_SCALE_NOT_OWNED_BY_ACTUAL_PROJECTION')
        require(actual.get('bias') is None or any(same_storage(actual['bias'], value) for value in owners),
                'QMM_BIAS_NOT_OWNED_BY_ACTUAL_PROJECTION')
        require(actual.get('output_dtype') == frame['input'].dtype and actual['output_dtype'] in (torch.float16, torch.bfloat16),
                'QMM_OUTPUT_DTYPE_UNSUPPORTED')
        row.update(formals={name: live_tensor(actual[name]) for name in ('x1', 'x2', 'scale', 'pertoken_scale')},
                   output_dtype=str(actual['output_dtype']), native_qualified=False,
                   native_gate='CURRENT_TYPED_GWS_RUN_METADATA_AND_LEASE_NOT_OBSERVED')

    def _restore_apis(self):
        for module, name, original, observer in reversed(self.api_patches):
            require(getattr(module, name) is observer, 'PUBLIC_QUANT_OBSERVER_OWNER_CHANGED')
            setattr(module, name, original)
        self.api_patches.clear()

    def end(self, model_output_sha256, *, model_output_finite):
        self._thread()
        require(self.current is not None and self.frame is None, 'COMPLETE_QUANT_FORWARD_REQUIRED')
        self._restore_apis()
        row = self.current
        row.update(model_output_sha256=model_output_sha256, model_output_finite=model_output_finite,
                   public_end=len(self.public_calls))
        self.rows.append(row)
        self.current = None
        if len(self.rows) == 2:
            first, second = self.rows
            require(model_output_finite and first['model_output_finite']
                    and first['input_sha256'] == second['input_sha256']
                    and first['model_output_sha256'] == second['model_output_sha256'],
                    'INVENTORY_HEALTHY_WORKLOAD_NOT_REPEATABLE')
            self.recovered = not self.failure  # Observer-only control, never Native C2.

    def fail(self, reason):
        self.failure = str(reason)
        self._restore_apis()
        self.current = None

    def close(self):
        self._thread()
        self._restore_apis()
        require(self.frame is None, 'ACTIVE_QUANT_PROJECTION_CANNOT_REMOVE_OBSERVER')
        for method, original, observer, local, saved in reversed(self.patches):
            require(method.apply is observer, 'QUANT_METHOD_OBSERVER_OWNER_CHANGED')
            if local:
                method.apply = saved
            else:
                del method.apply
        self.patches.clear()
        self.installed = False

    def report(self):
        counts = {identity: [] for identity in self.by_id}
        for forward in self.rows:
            for row in forward['projections']:
                counts[row['module_object_identity']].append(row)
        candidates = []
        for identity, (path, module, method, chain) in self.by_id.items():
            rows = counts[identity]
            candidates.append(dict(path=path, module_object_identity=identity, method_chain=chain,
                actual_calls=len(rows), public_verified_calls=sum(row['public_chain_verified'] for row in rows),
                native_qualified=False, status='NOT_OBSERVED' if not rows else
                    'PUBLIC_CHAIN_ONLY_NATIVE_UNVERIFIED' if all(row['public_chain_verified'] for row in rows) else 'PUBLIC_CHAIN_REJECTED'))
        signatures = [tuple((r['module_object_identity'], r['quant_method_object_identity'],
                            r['weight_object_identity'], r['method_source_sha256'], r['dynamic_calls'],
                            r['qmm_calls'], r['public_chain_verified'], tuple(r['errors']))
                           for r in forward['projections']) for forward in self.rows]
        return dict(schema='demoncat.public_quant_inventory.v1', structural_candidates=candidates,
            structural_candidate_count=len(candidates), actual_projection_calls=sum(len(r['projections']) for r in self.rows),
            actual_public_api_calls=self.public_total, retained_public_api_calls=len(self.public_calls),
            missing_public_call_details=self.overflow_calls, public_observer_complete=not self.overflow_calls,
            public_calls=self.public_calls, forwards=self.rows,
            unowned_public_calls=sum(row['module_object_identity'] is None for row in self.public_calls),
            complete_healthy_replays=self.recovered and not self.failure, failure=self.failure,
            repeated_public_inventory_stable=len(signatures) == 2 and signatures[0] == signatures[1],
            installed=self.installed, active=self.active, owned_method_patches=len(self.patches), owned_api_patches=len(self.api_patches),
            qualified_native_targets=0, c1=False, c2=False, c3=False,
            remaining_gates=['INDEPENDENT_NATIVE_ELIGIBLE_CALL_OBSERVER', 'CURRENT_TYPED_FORMALS_ALL_PROJECTIONS',
                'PER_TARGET_NATIVE_LEASE', 'LOSSLESS_EFFECT_AND_GROUP_OFF'],
            worker_identity=self.worker_identity)

    def query(self):
        report = self.report()
        return {'qualification': dict(status='PUBLIC_QUANT_INVENTORY_ONLY_NATIVE_UNVERIFIED',
            next_phase=self.phase, failure=self.failure, recovery_verified=False,
            actual_projection_calls=report['actual_projection_calls'], qualified_native_targets=0)}
