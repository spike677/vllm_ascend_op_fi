"""Natural all-projection census using only the independent group Native DSO."""
import hashlib

from fault_injection.deployments.quant_group_inventory import verify_public_group_call
from .quant_inventory import PublicQuantInventory
from .shared_scale_inventory import SharedScaleInventory
from .serving import complete_tensor
from .serving_context import require


class SharedScaleGroupInventory(SharedScaleInventory):
    def _api(self, name, original, args, kwargs):
        if name != 'npu_dynamic_quant':
            return PublicQuantInventory._api(self, name, original, args, kwargs)
        require(self.native_call is None, 'REENTRANT_GROUP_PUBLIC_DYNAMIC_QUANT')
        record = dict(errors=[], native_events=[])
        public_index = len(self.public_calls)
        self.native_call = record
        def observed(*values, **options):
            before = self.native.state()
            try:
                return original(*values, **options)
            finally:
                # Failed public calls also keep their actual prepares/executes.
                try:
                    after, events = self.native.delta(before)
                    record.update(native_before=before, native_after=after, native_events=events)
                except BaseException as error:
                    record['errors'].append(str(error))
        try:
            return PublicQuantInventory._api(self, name, observed, args, kwargs)
        finally:
            if len(self.public_calls) > public_index:
                self.public_calls[public_index]['independent_native'] = record
            self.native_call = None

    def _dynamic_lineage(self, frame, row, args, kwargs, output):
        PublicQuantInventory._dynamic_lineage(self, frame, row, args, kwargs, output)
        row['independent_native'] = self.native_call
        x = args[0] if args else kwargs.get('input')
        q, scale = output
        self.native_call['binding'] = verify_public_group_call(self.native_call['native_events'], x, q, scale,
            self.native_thread, expected_device=self.native.device)
        row['request_row_lineage'] = self.row_lineage(x)
        row['native_execution_observed'] = True
        import torch
        hashes = {}
        for name, tensor in (('input', x), ('quant', q), ('scale', scale)):
            cpu, sha, finite = complete_tensor(tensor)
            hashes[name] = dict(sha256=sha, finite=finite, shape=list(cpu.shape))
            if name == 'input':
                hashes[name]['row_sha256'] = [hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                             for v in cpu]
        row['healthy_formals'], row['native_qualified'] = hashes, False

    def end(self, *args, **kwargs):
        after, native = self.native.delta(self._initial)
        public = self.public_calls[self.current['public_start']:]
        public_indices = {id(row): index for index, row in enumerate(self.public_calls)}
        by_parent = {}
        owned_sequences = set()
        for row in public:
            for event in row.get('independent_native', {}).get('native_events', []):
                sequence = event['parent']['token'] if event['kind'] == 1 else event['native']['sequence']
                require(sequence not in owned_sequences, 'DUPLICATE_GROUP_RAW_PUBLIC_OWNER')
                owned_sequences.add(sequence)
                if event['kind'] == 1:
                    require(sequence not in by_parent, 'DUPLICATE_GROUP_MODEL_PARENT')
                    by_parent[sequence] = row
        candidates = []
        for event in native:
            if event['kind'] != 1:
                continue
            row = by_parent.get(event['parent']['token'])
            reasons = [] if row else ['GROUP_NATIVE_MODEL_PARENT_HAS_NO_PUBLIC_OWNER']
            if row:
                reasons.extend(row['errors'])
                reasons.extend(row.get('independent_native', {}).get('errors', []))
                if not row.get('independent_native', {}).get('binding', {}).get('complete_formals'):
                    reasons.append('GROUP_CURRENT_NATIVE_FORMALS_UNPROVEN')
                if not row.get('request_row_lineage', {}).get('verified'):
                    reasons.append('ACTUAL_PER_REQUEST_PROJECTION_ROW_LINEAGE_UNPROVEN')
            if self.row_trace is None or not self.row_trace['complete']:
                reasons.append('COMPLETE_REQUEST_ROW_OPERATOR_OBSERVATION_UNPROVEN')
            if after['pending'] or after['executing']:
                reasons.append('GROUP_NATIVE_EXECUTION_STILL_ACTIVE')
            candidates.append(dict(native=event, public_api_index=public_indices[id(row)] if row else None,
                module_object_identity=row['module_object_identity'] if row else None, reasons=reasons,
                status='NATIVE_CALL_AND_REQUEST_ROWS_OBSERVED_NOT_LOSSLESS_QUALIFIED' if not reasons else 'UNVERIFIED',
                qualified=False))
        unowned = [event for event in native if (event['parent']['token'] if event['kind'] == 1
                   else event['native']['sequence']) not in owned_sequences]
        self.native_forwards.append(dict(forward_id=self.current['forward_id'], before=self._initial, after=after,
            native_events=native, candidates=candidates, unowned_native_events=unowned,
            complete_native_denominator=True, row_provenance=self.row_trace))
        self.save()
        require(not after['pending'] and not after['executing'], 'GROUP_NATIVE_EXECUTION_STILL_ACTIVE')
        PublicQuantInventory.end(self, *args, **kwargs)
        self.save()

    def report(self):
        value = super().report()
        value.update(schema='demoncat.shared_scale.group_native_inventory.v1',
            native_public_binding_complete=value['native_public_binding_complete'] and all(
                not row.get('unowned_native_events') for row in self.native_forwards),
            actual_native_prepare_calls=sum(sum(event['kind'] == 2 and event['native']['kind'] == 1
                for event in row['native_events']) for row in self.native_forwards),
            actual_native_execute_calls=sum(sum(event['kind'] == 2 and event['native']['kind'] == 3
                for event in row['native_events']) for row in self.native_forwards),
            legacy_library_loaded=False, group_owner_installed=False)
        return value
