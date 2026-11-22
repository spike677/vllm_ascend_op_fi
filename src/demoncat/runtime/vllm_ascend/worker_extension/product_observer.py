"""Automatic request discovery on the pinned Worker scheduler boundary.

No host request IDs, conditional collectives or inference initiation. The existing
Runtime still owns enable/disable and all region leases.
"""
from copy import deepcopy
from .serving_context import require
from .product_request import ProductRequest


class AutomaticRequests:
    def __init__(self, binding, *, canary=False, factory=ProductRequest):
        self.binding, self.runner = binding, binding.worker.model_runner
        self.canary, self.factory = canary, factory
        self.session = None
        self.records = []
        self.seen = set()
        self.patches = []
        self.inside = 0
        self.error = None
        self.completed_canary = False

    def install(self):
        self.binding.check()
        for name, handler in (('execute_model', self.execute), ('sample_tokens', self.sample)):
            require(name not in vars(self.runner), 'PRODUCT_AUTOMATIC_BOUNDARY_ALREADY_OWNED:' + name)
            original = getattr(self.runner, name)
            def wrapper(*args, _original=original, _handler=handler, **kwargs):
                self.inside += 1
                try:
                    return _handler(_original, *args, **kwargs)
                except BaseException as error:
                    self.error = repr(error)
                    raise
                finally:
                    self.inside -= 1
            setattr(self.runner, name, wrapper)
            self.patches.append((name, wrapper))

    def finish_request(self, reason):
        s = self.session
        require(s is not None and s.inside == 0, 'PRODUCT_REQUEST_STILL_EXECUTING')
        s.finished = True
        row = s.collect()
        row['completion_source'] = reason
        self.records.append(row)
        self.session = None
        if self.canary:
            require(not row['reasons'] and len(row['output']) == 4, 'PRODUCT_CANARY_MAPPING_OR_LENGTH')
            self.binding.warmups.append(row)
            self.completed_canary = True

    def execute(self, original, scheduler, intermediate_tensors=None):
        self.binding.check()
        require(self.error is None, 'PRODUCT_AUTOMATIC_OBSERVER_FAILED')
        finished = getattr(scheduler, 'finished_req_ids', set())
        if self.session is not None and self.session.ids[0] in finished:
            self.finish_request('SCHEDULER_FINISHED_REQUEST')
        ids = list(scheduler.num_scheduled_tokens)
        if not ids or self.completed_canary:
            return original(scheduler, intermediate_tensors)
        require(len(ids) == 1, 'PRODUCT_SERIAL_SCHEDULER_CONTRACT_CHANGED')
        rid = ids[0]
        if self.session is None:
            require(rid not in self.seen, 'PRODUCT_FINISHED_ID_RESCHEDULED')
            new = [r for r in scheduler.scheduled_new_reqs if r.req_id == rid]
            if self.canary:
                if len(new) != 1 or tuple(getattr(new[0], 'prompt_token_ids', ()) or ()) != self.binding.prompts[0]:
                    return original(scheduler, intermediate_tensors)
            self.seen.add(rid)
            self.session = self.factory(self.binding, [rid], inject=not self.canary,
                forced_skip=None if len(new) == 1 else 'REQUEST_STARTED_BEFORE_ENABLE')
            self.session.install(managed=True)
        require(self.session.ids == ids, 'PRODUCT_ACTIVE_REQUEST_REPLACED_WITHOUT_COMPLETION')
        s = self.session
        s.inside += 1
        try:
            return s.managed_execute(original, scheduler, intermediate_tensors)
        finally:
            s.inside -= 1

    def sample(self, original, *args, **kwargs):
        s = self.session
        if s is None:
            return original(*args, **kwargs)
        s.inside += 1
        try:
            result = s.managed_sample(original, *args, **kwargs)
        finally:
            s.inside -= 1
        limit = getattr(s, 'range_facts', {}).get('max_tokens')
        # A real sample per step plus the actual configured cap is sufficient;
        # early EOS/abort is finalized only by scheduler.finished_req_ids.
        if type(limit) is int and s.samples == limit:
            self.finish_request('ACTUAL_SAMPLES_REACHED_REQUEST_MAX_TOKENS')
        return result

    def close(self):
        require(self.inside == 0, 'PRODUCT_AUTOMATIC_OBSERVER_EXECUTING')
        if self.session is not None:
            s = self.session
            require(s.inside == 0 and not getattr(s.site, 'pending', 0), 'PRODUCT_DEVICE_CONSUMPTION_PENDING')
            # No new samples are manufactured. A control-boundary stop can end
            # observation without claiming that the user's inference completed.
            s.finished = True
            row = s.collect()
            row.update(completed=False, completion_source='OBSERVER_DISABLED_AT_CONTROL_BOUNDARY')
            self.records.append(row)
            self.session = None
        require(all(vars(self.runner).get(n) is w for n,w in self.patches), 'PRODUCT_AUTOMATIC_WRAPPER_OWNER_CHANGED')
        for n,w in reversed(self.patches):
            delattr(self.runner, n)
        self.patches.clear()

    def snapshot(self):
        s = self.session
        pending = getattr(getattr(s, 'site', None), 'pending', 0)
        return dict(requests=deepcopy(self.records), request_count=len(self.seen),
            active_request_id=s.ids[0] if s else None,
            leases=len(pending) if isinstance(pending, dict) else pending,
            owned_hooks=len(self.patches) + (len(s.patches) + len(s.hooks) if s else 0),
            inflight=self.inside + (s.inside if s else 0), failure=self.error)
