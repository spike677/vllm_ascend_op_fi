"""Common lifetime/report interface; native acceptance is never inferred here."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import threading
import uuid

_LEASE = threading.Lock()


class ModelFaultSession:
    def __init__(self, adapter, capability, arm, evidence_dir=None):
        self.adapter, self.capability, self.arm = adapter, capability, arm
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None
        self.session_id = uuid.uuid4().hex
        self._active = self._closed = self._locked = False
        self._owner = None
        self.cleanup_completed = None
        self._saved = False
        self.error = None
        self.errors = []
        self._cleanup_pending = False
        self._quarantined = False
        self._report_revision = 0

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError('Session cannot be reused')
        if not _LEASE.acquire(blocking=False):
            raise RuntimeError('Another unified fault session is active')
        self._locked = True
        self._owner = threading.get_ident()
        try:
            self.adapter.__enter__()
            self._active = True
            return self
        except BaseException as exc:
            self._record_error('installation', exc)
            # Installation may have partially patched the model/native bridge.
            # A raised __enter__ is not evidence that its rollback succeeded.
            try:
                self.close()
            except BaseException as cleanup:
                raise RuntimeError('Installation failed: '+repr(exc)+'; restore failed: '+repr(cleanup)) from exc
            raise

    def report(self):
        result = dict(self.adapter.report())
        result.update(adapter_schema=result.get('schema'), schema='demoncat.model_fault.session.v1',
            session_id=self.session_id, fault=self.capability.name, arm=self.arm,
            capability=asdict(self.capability), active=self._active, closed=self._closed,
            session_error=self.error, cleanup_completed=self.cleanup_completed, npu_validation_claimed=False,
            error_history=[dict(row) for row in self.errors],
            cleanup_pending=self._cleanup_pending, quarantined=self._quarantined,
            report_scope='Automatic routing/observation only; not msprobe or native/model acceptance')
        return result

    def assert_coverage(self):
        self.adapter.assert_coverage()
        if self.error is not None:
            raise RuntimeError('Session failed: ' + self.error)
        return self.report()

    def _release(self):
        if self._locked:
            _LEASE.release()
            self._locked = False

    def _record_error(self, stage, error):
        detail = repr(error)
        self.errors.append(dict(stage=stage, error=detail))
        self.error = detail if self.error is None else self.error+'; '+stage+': '+detail

    def close(self):
        if self._closed:
            self._save_report()
            return
        if self._owner is not None and threading.get_ident() != self._owner:
            # A pre-cleanup refusal must leave the owner able to restore.
            raise RuntimeError('Unified session must be closed on its owning thread')
        if self._quarantined:
            raise RuntimeError('Adapter discarded failed cleanup state; session remains quarantined until process restart')
        cleanup = None
        try:
            self.adapter.close()
            self.cleanup_completed = True
            self._cleanup_pending = False
            self._active, self._closed = False, True
            self._release()
        except BaseException as exc:
            self._record_error('cleanup', exc)
            cleanup = exc
            self.cleanup_completed = False
            self._cleanup_pending = True
            # Old/unknown adapters that mark themselves closed after losing
            # removal handles cannot prove recovery on a subsequent no-op.
            terminal = getattr(self.adapter, '_closed', False) or getattr(self.adapter, 'closed', False)
            self._quarantined = bool(terminal and not getattr(self.adapter, '_cleanup_retryable', False))
            # Retain the global lease and owned adapter reference. New sessions
            # must not enter while this one's hooks or native leases are unknown.
        if self._saved:
            self._report_revision += 1
            self._saved = False
        try:
            self._save_report()
        except BaseException as writing:
            self._record_error('report_persistence', writing)
            if cleanup is not None:
                raise RuntimeError('Restore failed: '+repr(cleanup)+'; report failed: '+repr(writing)) from cleanup
            raise
        if cleanup is not None:
            raise cleanup

    def _save_report(self):
        if self.evidence_dir is None or self._saved:
            return
        # Serialize before creating a file. Never leave partial JSON as evidence.
        data = (json.dumps(self.report(), ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        suffix = '' if self._report_revision == 0 else '-revision'+str(self._report_revision)
        target = self.evidence_dir / ('session-' + self.session_id + suffix + '.json')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.evidence_dir, prefix='.session-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(data)
            if target.exists():
                raise FileExistsError('Refusing to overwrite an existing session report: '+str(target))
            os.replace(temporary, target)
            self._saved = True
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is not None:
            self._record_error('model', exc_value)
        try:
            self.close()
        except BaseException as cleanup:
            if exc_value is not None:
                raise RuntimeError('Model failed: '+repr(exc_value)+'; restore failed: '+repr(cleanup)) from exc_value
            raise
        return False
