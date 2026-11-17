"""Whole-model public/native quant inventory with explicit request-row uncertainty."""
import hashlib
import json
from pathlib import Path
import threading

from fault_injection.deployments.dynamic_quant_inventory import verify_public_native_call
from .quant_inventory import PublicQuantInventory
from .serving import complete_tensor
from .serving_context import require


class SharedScaleInventory(PublicQuantInventory):
    def __init__(self, model, candidates, *, worker_identity, native, folder, row_lineage):
        super().__init__(model, candidates, worker_identity=worker_identity)
        self.native = native
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.native_thread = threading.get_native_id()
        from fault_injection.runtime_discovery import _process_start_identity
        self.lifetime = dict(worker=worker_identity, process_start_identity=_process_start_identity(),
            model_object_identity=id(model), native_thread=self.native_thread, device=native.device)
        self.row_lineage = row_lineage
        self.native_call = None
        self.native_forwards = []
        self._initial = None

    def begin(self, *args, **kwargs):
        self._initial = self.native.state()
        require(not self._initial['pending'] and not self._initial['executing'], 'NATIVE_QUANT_FORWARD_NOT_IDLE')
        super().begin(*args, **kwargs)

    def _api(self, name, original, args, kwargs):
        if name != 'npu_dynamic_quant':
            return super()._api(name, original, args, kwargs)
        require(self.native_call is None, 'REENTRANT_PUBLIC_DYNAMIC_QUANT')
        record = dict(errors=[], native_events=[])
        public_index = len(self.public_calls)
        self.native_call = record
        def observed(*values, **options):
            before = self.native.state()
            result = original(*values, **options)
            try:
                after, events = self.native.delta(before)
                record.update(native_before=before, native_after=after, native_events=events)
            except BaseException as error:
                record['errors'].append(str(error))
            return result
        try:
            return super()._api(name, observed, args, kwargs)
        finally:
            if len(self.public_calls) > public_index:
                self.public_calls[public_index]['independent_native'] = record
            self.native_call = None

    def _dynamic_lineage(self, frame, row, args, kwargs, output):
        super()._dynamic_lineage(frame, row, args, kwargs, output)
        row['independent_native'] = self.native_call
        x = args[0] if args else kwargs.get('input')
        q, scale = output
        self.native_call['binding'] = verify_public_native_call(self.native_call['native_events'], x, q, scale,
                                                                 self.native_thread, expected_device=self.native.device)
        row['request_row_lineage'] = self.row_lineage(x)
        row['native_execution_observed'] = True
        # Auxiliary exact healthy arrays establish no fault outcome or E5 result.
        hashes = {}
        import torch
        for name, tensor in (('input', x), ('quant', q), ('scale', scale)):
            cpu, sha, finite = complete_tensor(tensor)
            hashes[name] = dict(sha256=sha, finite=finite, shape=list(cpu.shape))
            if name == 'input':
                hashes[name]['row_sha256'] = [hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                              for v in cpu]
        row['healthy_formals'] = hashes
        row['native_qualified'] = False

    def end(self, *args, **kwargs):
        after, native = self.native.delta(self._initial)
        require(not after['pending'] and not after['executing'], 'NATIVE_QUANT_EXECUTION_STILL_ACTIVE')
        public = self.public_calls[self.current['public_start']:]
        public_indices = {id(row): index for index, row in enumerate(self.public_calls)}
        by_call = {}
        for row in public:
            for event in row.get('independent_native', {}).get('native_events', []):
                if event['kind'] == 1:
                    require(event['call'] not in by_call, 'DUPLICATE_NATIVE_QUANT_PUBLIC_OWNER')
                    by_call[event['call']] = row
        candidates = []
        for event in native:
            if event['kind'] != 1:
                continue
            row = by_call.get(event['call'])
            reasons = [] if row is not None else ['NATIVE_QUANT_HAS_NO_OWNED_PUBLIC_CHAIN']
            if row is not None:
                reasons.extend(row['errors'])
                if not row.get('request_row_lineage', {}).get('verified'):
                    reasons.append('ACTUAL_PER_REQUEST_PROJECTION_ROW_LINEAGE_UNPROVEN')
            candidates.append(dict(native=event, public_api_index=public_indices[id(row)] if row is not None else None,
                module_object_identity=row['module_object_identity'] if row else None, reasons=reasons,
                status='NATIVE_CALL_AND_REQUEST_ROWS_OBSERVED_NOT_LOSSLESS_QUALIFIED' if not reasons else 'UNVERIFIED',
                qualified=False))
        self.native_forwards.append(dict(forward_id=self.current['forward_id'], before=self._initial, after=after,
            native_events=native, candidates=candidates, complete_native_denominator=True))
        super().end(*args, **kwargs)
        self.save()

    def report(self):
        report = super().report()
        dynamic = [row for row in self.public_calls if row['operator'] == 'npu_dynamic_quant']
        native_calls = sum(len(row['candidates']) for row in self.native_forwards)
        complete_binding = (bool(dynamic) and native_calls == len(dynamic)
            and all(row.get('independent_native', {}).get('binding', {}).get('complete_formals')
                    and not row['errors'] for row in dynamic)
            and all(c['public_api_index'] is not None for forward in self.native_forwards for c in forward['candidates'])
            and all(row['public_chain_verified'] for forward in self.rows for row in forward['projections']))
        report.update(schema='demoncat.shared_scale.native_inventory.v1', native_forwards=self.native_forwards,
            lifetime=self.lifetime,
            native_build=self.native.build_identity,
            actual_native_calls=native_calls, native_public_binding_complete=bool(complete_binding),
            request_row_proven_calls=sum(not c['reasons'] for row in self.native_forwards for c in row['candidates']),
            remaining_gates=['ACTUAL_PER_REQUEST_ROW_LINEAGE_FOR_EVERY_UNRESOLVED_PROJECTION',
                'PER_CALL_OWNER_GENERATION_NATIVE_LEASE', 'PER_TARGET_LOSSLESS_AND_EFFECT', 'ALL_TARGET_OFF_DRAIN_RELEASE'])
        return report

    def save(self):
        (self.folder / 'shared_scale_inventory.json').write_text(json.dumps(self.report(), indent=2, allow_nan=False) + '\n')

    def fail(self, reason):
        if self.current is not None:
            failed = dict(self.current, failure=str(reason), public_calls=self.public_calls[self.current['public_start']:])
            try:
                after, events = self.native.delta(self._initial)
                failed.update(native_after=after, native_events=events)
            except BaseException as error:
                failed['independent_inventory_error'] = str(error)
            (self.folder / 'failed_forward.json').write_text(json.dumps(failed, indent=2, allow_nan=False) + '\n')
        super().fail(reason)
        self.save()

    def close(self):
        super().close()
        self.save()
