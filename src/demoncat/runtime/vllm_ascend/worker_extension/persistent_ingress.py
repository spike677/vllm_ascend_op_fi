"""Dedicated ingress exclusion for v2, independent of the frozen v1 path.

Application completion is not device completion. pause_and_drain requires the
deployment's real backend drain callback; Workers still fence their own stream.
The gate is owned by the trusted host, never controlled by HTTP request fields.
"""
from contextlib import contextmanager
import threading
import time
import uuid


class IngressRefused(RuntimeError):
    exit_code=4


class PersistentIngress:
    def __init__(self,*,validate_request,drain_backend):
        if not callable(validate_request) or not callable(drain_backend):
            raise TypeError('V2_TRUSTED_INGRESS_CALLBACKS_REQUIRED')
        self.validate_request=validate_request;self.drain_backend=drain_backend
        self._condition=threading.Condition(threading.RLock())
        self._owner=None;self._active={};self._paused=False;self._drained=False
        self._admitted=0;self._rejected=0

    @contextmanager
    def request(self,request):
        """Wrap actual backend submission AND response/cancellation completion."""
        with self._condition:
            if self._paused:
                self._rejected+=1;raise IngressRefused('V2_INGRESS_PAUSED')
            reason=self.validate_request(request)
            if reason is not None:
                self._rejected+=1;raise IngressRefused(str(reason))
            token=uuid.uuid4().hex
            self._active[token]=True;self._admitted+=1;self._drained=False
        try:yield token
        finally:
            with self._condition:
                del self._active[token];self._condition.notify_all()

    @contextmanager
    def canary_request(self,owner,request):
        """Trusted host only; never expose this path to request body parameters.

        Keep public ingress closed during warmup/recovery. At most one canary
        may run, and its completion still requires backend drain afterward.
        """
        with self._condition:
            if self._owner!=owner or not self._paused or not self._drained or self._active:
                raise IngressRefused('V2_EXCLUSIVE_CANARY_BOUNDARY_REQUIRED')
            reason=self.validate_request(request)
            if reason is not None:raise IngressRefused(str(reason))
            token=uuid.uuid4().hex;self._active[token]=True;self._drained=False
        try:yield token
        finally:
            with self._condition:
                del self._active[token];self._condition.notify_all()

    def pause_and_drain(self,owner,*,timeout):
        if not isinstance(owner,str) or not owner or not 0<timeout<float('inf'):
            raise ValueError('V2_BOUNDED_INGRESS_OWNER_REQUIRED')
        deadline=time.monotonic()+timeout
        with self._condition:
            if self._owner not in (None,owner):raise IngressRefused('V2_INGRESS_OTHER_OWNER')
            self._owner=owner;self._paused=True;self._drained=False
            while self._active:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise RuntimeError('V2_INGRESS_DRAIN_TIMEOUT_OWNERSHIP_RETAINED')
                self._condition.wait(remaining)
        # No new requests can enter during this call. A cancelled HTTP request
        # may still be running in the engine; only backend observation closes it.
        remaining=deadline-time.monotonic()
        if remaining<=0:raise RuntimeError('V2_BACKEND_DRAIN_TIMEOUT_OWNERSHIP_RETAINED')
        proof=self.drain_backend(timeout=remaining)
        with self._condition:
            if not (isinstance(proof,dict) and proof.get('inflight')==0
                    and proof.get('pending_submissions')==0 and proof.get('aborts_completed') is True):
                raise RuntimeError('V2_BACKEND_DRAIN_UNPROVEN_OWNERSHIP_RETAINED')
            self._drained=True
            return self.snapshot()

    def resume(self,owner):
        with self._condition:
            if self._owner!=owner or not self._paused or not self._drained:
                raise IngressRefused('V2_INGRESS_DRAINED_OWNER_REQUIRED')
            self._paused=False;self._drained=False

    def release(self,owner):
        with self._condition:
            if self._owner!=owner or not self._paused or not self._drained:
                raise IngressRefused('V2_INGRESS_DRAINED_OWNER_REQUIRED')
            self._owner=None;self._paused=False;self._drained=False

    def snapshot(self):
        with self._condition:
            return dict(owner=self._owner,accepting=not self._paused,inflight=len(self._active),
                        backend_drained=self._drained,admitted=self._admitted,rejected=self._rejected,
                        device_readers_complete=None)
