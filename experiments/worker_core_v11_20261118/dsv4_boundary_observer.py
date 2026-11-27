"""Source-pinned Python boundary observation core; no inferred V4 site mapping.

Trusted integration supplies reviewed objects, a bounded tensor snapshotter, and
the current runner step/layer context. This core grants no model qualification.
"""
import inspect
import json
import threading
from experiments.worker_core_v11_20261118.dsv4_probe_support import (
    MethodLease, ProbeRefused, restore_all, source,
)


class BoundaryObserver:
    def __init__(self, targets, snapshot, context, *, max_calls=256, max_bytes=1048576):
        if type(max_calls) is not int or not 1 <= max_calls <= 4096:
            raise ProbeRefused("BOUNDARY_CALL_BUDGET_INVALID")
        if type(max_bytes) is not int or not 1 <= max_bytes <= 4194304:
            raise ProbeRefused("BOUNDARY_BYTE_BUDGET_INVALID")
        if not 1 <= len(targets) <= 16:
            raise ProbeRefused("BOUNDARY_TARGET_COUNT_INVALID")
        self.snapshot, self.context = snapshot, context
        self.max_calls, self.max_bytes = max_calls, max_bytes
        self.records, self.errors, self.leases = [], [], []
        self.active, self.next_call = 0, 0
        self.local = threading.local()
        self.used_bytes, self.truncated, self.installed = 0, False, False
        self.targets = []
        seen = set()
        # Validate every target before changing any object. SHA never bypasses
        # MethodLease's strict class/bound-method guard.
        for target in targets:
            label, obj, method, sha, fields = target
            if not isinstance(label, str) or not label or label in seen:
                raise ProbeRefused("BOUNDARY_LABEL_INVALID")
            seen.add(label)
            if (id(obj), method) in seen:
                raise ProbeRefused("BOUNDARY_DUPLICATE_TARGET")
            seen.add((id(obj), method))
            lease = MethodLease(obj, method)
            identity = source(lease.original)
            if not isinstance(sha, str) or len(sha) != 64 or sha != identity["sha256"]:
                raise ProbeRefused("BOUNDARY_SOURCE_SHA_MISMATCH:" + label)
            signature = inspect.signature(lease.original)
            if not isinstance(fields, (list, tuple)) or any(f not in signature.parameters for f in fields):
                raise ProbeRefused("BOUNDARY_ARGUMENT_UNPROVEN:" + label)
            self.targets.append((label, lease, signature, tuple(fields), identity))
            self.leases.append(lease)

    def _observe(self, function):
        try:
            return function()
        except Exception as error:
            if len(self.errors) < 32:
                self.errors.append(dict(type=type(error).__name__, reason=str(error)[:240]))
            else:
                self.truncated = True
            return None

    def _wrapper(self, label, lease, signature, fields, identity):
        def call(*args, **kwargs):
            self.active += 1
            self.next_call += 1
            call_id = self.next_call
            stack = getattr(self.local, "stack", None)
            if stack is None:
                stack = self.local.stack = []
            parent = stack[-1] if stack else None
            stack.append(call_id)
            eligible = call_id <= self.max_calls and not self.truncated
            row = None
            if eligible:
                def before():
                    bound = signature.bind(*args, **kwargs)
                    # Missing/default arguments remain absent; don't fabricate values.
                    return dict(call_id=call_id, parent_call_id=parent, boundary=label,
                                thread_id=threading.get_ident(),
                                source=identity, context=self.context(),
                                inputs={key: self.snapshot(bound.arguments[key]) for key in fields if key in bound.arguments},
                                qualification_granted=False)
                row = self._observe(before)
            else:
                self.truncated = True
            try:
                # Exactly one original call, original return object and exception.
                result = lease.original(*args, **kwargs)
                if row is not None:
                    row["output"] = self._observe(lambda: self.snapshot(result))
                    row["completion"] = "SNAPSHOTTER_MUST_SUPPLY_DEVICE_EVENT_EVIDENCE"
                    def append():
                        raw = json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
                        if self.used_bytes + len(raw) > self.max_bytes:
                            self.truncated = True
                        else:
                            self.used_bytes += len(raw)
                            self.records.append(row)
                    self._observe(append)
                return result
            finally:
                stack.pop()
                self.active -= 1
        return call

    def install(self):
        if self.installed:
            raise ProbeRefused("BOUNDARY_OBSERVER_ALREADY_INSTALLED")
        try:
            for args in self.targets:
                args[1].install(self._wrapper(*args))
        except BaseException:
            restore_all(self.leases)
            raise
        self.installed = True

    def stop(self):
        if self.active:
            raise ProbeRefused("BOUNDARY_REQUEST_INFLIGHT")
        restore_all(self.leases)
        self.installed = False
        return dict(schema="dcat.v4.python_boundaries.v1", records=self.records,
                    errors=self.errors, truncated=self.truncated, observed_calls=self.next_call,
                    bytes=self.used_bytes, observer_removed=True, qualification_granted=False)
