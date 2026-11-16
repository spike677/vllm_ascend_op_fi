"""Owner-scoped orchestration with fail-closed partial enable rollback."""
from dataclasses import dataclass
import threading
import uuid
from .provider import NativeProvider
from .schema import Fault39Config, INTEGRATION_STATUS, REQUEST_MAPPING
from .target import select_stable_target


@dataclass(frozen=True)
class WorkerEndpoint:
    rank: int
    device: int
    provider: NativeProvider


class EnableError(RuntimeError):
    def __init__(self, cause, rollback):
        self.cause, self.rollback = cause, rollback
        super().__init__(f"ENABLE_FAILED: {cause}; rollback={rollback['status']}")


class NativeBackend:
    api = "demoncat_fault39.NativeBackend"
    schema_version = 1

    def __init__(self, workers, owner=None):
        self.workers = dict(workers)
        if any(key != value.rank for key, value in self.workers.items()):
            raise ValueError("Worker map must be keyed by verified rank")
        self.owner = owner or "fault39-" + uuid.uuid4().hex
        if not self.owner or len(self.owner.encode()) >= 64 or "\0" in self.owner:
            raise ValueError("Invalid owner")
        self.config = None
        self.generations = {}
        self._uncertain_ranks = set()
        self.targets = {}
        self._inventory = {}
        self._prepared = False
        self._state = "UNLOADED"
        self._lock = threading.RLock()

    def _selected(self):
        if self.config is None:
            raise RuntimeError("NOT_REGISTERED")
        return {rank: worker for rank, worker in self.workers.items()
                if rank in self.config.ranks and (not self.config.devices or worker.device in self.config.devices)}

    def _validated_ack(self, report, generation, clean=False):
        if not isinstance(report, dict):
            return {"status": "DISABLE_PENDING", "error": "INVALID_NATIVE_ACK"}
        if report.get("status") != "DISABLED_CONFIRMED":
            return report if report.get("status") in ("DISABLE_PENDING", "RECOVERY_FAILED") else dict(
                report, status="DISABLE_PENDING", error="INVALID_NATIVE_ACK_STATUS")
        state = report.get("snapshot", {})
        valid = (report.get("native_code") == 0 and report.get("owner") == self.owner
            and report.get("generation") == generation and state.get("enabled") is False
            and state.get("inflight") == 0 and state.get("leases") == 0)
        if not clean:
            valid = valid and state.get("owner") == self.owner and state.get("generation") == generation
        if valid:
            return report
        return dict(report, status="DISABLE_PENDING", error="DRAIN_ACK_EVIDENCE_INCOMPLETE")

    def prepare(self):
        with self._lock:
            reports = {rank: worker.provider.prepare() for rank, worker in self.workers.items()}
            self._prepared = True
            if self.config is None:
                self._state = "LOADED"
            return reports

    def register(self, config):
        with self._lock:
            if not self._prepared:
                raise RuntimeError("PREPARE_REQUIRED")
            if self.config is not None or self.generations or self._uncertain_ranks:
                raise RuntimeError("ALREADY_REGISTERED")
            if not isinstance(config, Fault39Config):
                raise TypeError("Fault39Config required")
            if not set(config.ranks).issubset(self.workers):
                raise ValueError("SELECTED_RANK_UNAVAILABLE")
            if config.devices and any(self.workers[r].device not in config.devices for r in config.ranks):
                raise ValueError("SELECTED_RANK_DEVICE_MISMATCH")
            self.config = config
            self._state = "REGISTERED"
            return self.query()

    def _arm(self, mode):
        if self._uncertain_ranks or self.generations and self._state != "DISABLED_CONFIRMED":
            raise RuntimeError("DISABLE_REQUIRED_BEFORE_REENABLE")
        try:
            for rank, worker in self._selected().items():
                self._uncertain_ranks.add(rank)
                generation = worker.provider.configure(self.owner, self.config, rank, worker.device,
                    mode=mode, target_signature=self.targets.get(rank, ""))
                self.generations[rank] = generation
                self._uncertain_ranks.discard(rank)
            self._state = "DISCOVERING" if mode == "discover" else "ARMED"
            return self.query()
        except Exception as exc:
            # A transport may lose a configure reply after the worker armed.
            # Recover only this owner's real generation, never another owner.
            for rank, worker in self._selected().items():
                try:
                    state = worker.provider.snapshot()
                    generation = state.get("generation")
                    if state.get("owner") == self.owner and type(generation) is int and generation > 0:
                        self.generations[rank] = generation
                    elif state.get("owner") == self.owner:
                        continue
                    self._uncertain_ranks.discard(rank)
                except Exception:
                    pass
            rollback = self.disable()
            raise EnableError(exc, rollback) from exc

    def _recover_uncertain(self):
        for rank in tuple(self._uncertain_ranks):
            try:
                state = self.workers[rank].provider.snapshot()
                generation = state.get("generation")
                if state.get("owner") == self.owner:
                    if type(generation) is not int or generation < 1:
                        continue
                    self.generations[rank] = generation
                self._uncertain_ranks.discard(rank)
            except Exception:
                pass

    def discover(self):
        with self._lock:
            self.targets.clear()
            self._inventory.clear()
            return self._arm("discover")

    def observe_inventory(self):
        """Call after one complete representative workload; then repeat it.

        This method never runs a workload itself. Two adjacent control reads
        cannot establish cross-workload stability and callers must not claim it.
        """
        with self._lock:
            if self._state != "DISCOVERING":
                raise RuntimeError("DISCOVERY_REQUIRED")
            current = {rank: worker.provider.snapshot().get("candidates", [])
                       for rank, worker in self._selected().items()}
            previous = self._inventory
            self._inventory = current
            self.targets.clear()
            if not previous:
                return {"status": "STABILITY_WORKLOAD_REQUIRED", "inventories": current}
            targets = {rank: select_stable_target(items, previous.get(rank), self.config.target_policy) for rank, items in current.items()}
            self.targets = targets
            return {"status": "STABLE_UNIQUE_TARGET", "targets": dict(targets)}

    def end_discovery(self, timeout_ms=10000):
        return self.disable(timeout_ms)

    def enable(self):
        with self._lock:
            if set(self.targets) != set(self._selected()):
                raise RuntimeError("STABLE_UNIQUE_DISCOVERY_REQUIRED")
            return self._arm("inject")

    def disable(self, timeout_ms=10000):
        with self._lock:
            self._recover_uncertain()
            reports = {rank: {"status": "DISABLE_PENDING", "error": "CONFIGURE_ACK_AND_STATE_UNAVAILABLE"}
                       for rank in self._uncertain_ranks}
            for rank, generation in list(self.generations.items()):
                if rank in self._uncertain_ranks:
                    continue
                try:
                    report = self._validated_ack(self.workers[rank].provider.disable(self.owner, generation, timeout_ms), generation)
                except Exception as exc:
                    report = {"status": "DISABLE_PENDING", "error": str(exc)}
                reports[rank] = report
            failed = any(r["status"] == "RECOVERY_FAILED" for r in reports.values())
            pending = any(r["status"] != "DISABLED_CONFIRMED" for r in reports.values())
            status = "RECOVERY_FAILED" if failed else "DISABLE_PENDING" if pending else "DISABLED_CONFIRMED"
            self._state = status
            # Retain generation ownership even after disable, for scoped clean.
            return {"status": status, "workers": reports}

    def _release_disabled_generations(self, timeout_ms):
        self._recover_uncertain()
        reports = {rank: {"status": "DISABLE_PENDING", "error": "CONFIGURE_ACK_AND_STATE_UNAVAILABLE"}
                   for rank in self._uncertain_ranks}
        for rank, generation in list(self.generations.items()):
            if rank in self._uncertain_ranks:
                continue
            try:
                report = self._validated_ack(self.workers[rank].provider.clean(self.owner, generation, timeout_ms), generation, clean=True)
            except Exception as exc:
                report = {"status": "DISABLE_PENDING", "error": str(exc)}
            reports[rank] = report
            if report["status"] == "DISABLED_CONFIRMED":
                del self.generations[rank]
        if any(r["status"] == "RECOVERY_FAILED" for r in reports.values()):
            status = "RECOVERY_FAILED"
        elif self.generations or self._uncertain_ranks:
            status = "DISABLE_PENDING"
        else:
            status = "DISABLED_CONFIRMED"
        self._state = status
        return {"status": status, "workers": reports}

    def clean(self, timeout_ms=10000):
        with self._lock:
            # Native clean atomically disarms, drains and releases only owner.
            result = self._release_disabled_generations(timeout_ms)
            if result["status"] == "DISABLED_CONFIRMED":
                self.targets.clear()
                self._inventory.clear()
            return result

    def unregister(self, timeout_ms=10000):
        with self._lock:
            report = self.clean(timeout_ms)
            if report["status"] == "DISABLED_CONFIRMED":
                self.config = None
                self.targets.clear()
                self._inventory.clear()
                self._state = "LOADED"
            return report

    def query(self):
        with self._lock:
            return {"api": self.api, "schema_version": self.schema_version,
                    "owner": self.owner, "state": self._state,
                    "config": self.config.as_dict() if self.config else None,
                    "generations": dict(self.generations), "targets": dict(self.targets),
                    "uncertain_workers": sorted(self._uncertain_ranks),
                    "workers": {r: w.provider.snapshot() for r, w in self.workers.items()},
                    "inner_framework_integration": INTEGRATION_STATUS,
                    "request_mapping": REQUEST_MAPPING}
