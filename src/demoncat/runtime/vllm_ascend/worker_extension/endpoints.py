"""Explicit existing Worker routes; each rank retains its own Core lease.

This is host-side routing, not distributed atomic enable/recovery. It creates no
Worker and issues no model work. A failed rank never erases another rank's lease.
"""

from dataclasses import dataclass
import json
import threading

from .config import FaultConfig
from .host import HostAdapter, HostContractError, WorkerFaultClient


def _key(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


@dataclass(frozen=True)
class WorkerEndpoint:
    rank: int
    _identity_json: str

    @property
    def identity(self):
        return json.loads(self._identity_json)


class WorkerEndpointSet:
    """Pin an explicit rank set from the existing executor's live census.

    ``expected_ranks`` is deployment policy, not a user fault/target selector.
    ``client_for`` returns the same client on repeat access, preserving uncertain
    leases. Actions and cleanup are explicit on each client. Re-discover a new
    set after recovery/restart; no stale lease is transferred to the new set.
    """

    def __init__(self, executor, *, expected_ranks, prepare_binding=None, timeout=30):
        if (not isinstance(expected_ranks, (list, tuple, set, frozenset))
                or not expected_ranks
                or any(type(rank) is not int or rank < 0 for rank in expected_ranks)
                or len(set(expected_ranks)) != len(expected_ranks)):
            raise ValueError('An explicit nonempty unique rank set is required')
        self._host = HostAdapter(executor, prepare_binding=prepare_binding, timeout=timeout)
        rows = self._host.discover_workers()
        identities = {row['worker']['rank']: row['worker'] for row in rows}
        if set(identities) != set(expected_ranks):
            raise HostContractError('WORKER_ENDPOINT_SET_MISMATCH')
        if any(row.get('quarantined') is not False for row in rows):
            raise HostContractError('WORKER_RECOVERY_REQUIRED')
        self._endpoints = {rank: WorkerEndpoint(rank, _key(identities[rank]))
                           for rank in sorted(identities)}
        self._clients = {}
        self._lock = threading.RLock()

    @property
    def endpoints(self):
        return tuple(self._endpoints.values())

    def _select(self, rank, identities):
        live = {item['rank']: _key(item) for item in identities}
        pinned = {item.rank: item._identity_json for item in self.endpoints}
        if live != pinned:
            raise HostContractError('WORKER_ENDPOINT_SET_CHANGED')
        return self._endpoints[rank].identity

    def client_for(self, rank, config):
        """Create one inert client for this rank; its attach rechecks the census."""
        if type(rank) is not int or rank not in self._endpoints:
            raise HostContractError('WORKER_ENDPOINT_NOT_SELECTED')
        if isinstance(config, dict):
            config = FaultConfig.from_dict(config)
        if not isinstance(config, FaultConfig):
            raise TypeError('Worker FaultConfig is required')
        with self._lock:
            if rank in self._clients:
                client = self._clients[rank]
                if client.config != config:
                    raise HostContractError('WORKER_ENDPOINT_ALREADY_BOUND')
                return client
            host = HostAdapter(self._host.executor,
                               select_worker=lambda cfg, rows: self._select(rank, rows),
                               prepare_binding=self._host.prepare_binding,
                               timeout=self._host.timeout)
            client = WorkerFaultClient(host, config)
            self._clients[rank] = client
            return client

    def snapshot(self):
        """Report each lease independently; never grant a group qualification."""
        with self._lock:
            rows = [{'rank': endpoint.rank, 'worker': endpoint.identity,
                     'client': self._clients[endpoint.rank].snapshot()
                     if endpoint.rank in self._clients else None}
                    for endpoint in self.endpoints]
            return {'schema': 'demoncat.worker_endpoints.v1', 'endpoints': rows,
                    'atomic_group_operation': False,
                    'ownership_retained': any(row['client'] is not None
                                              and row['client']['ownership_retained']
                                              for row in rows)}
