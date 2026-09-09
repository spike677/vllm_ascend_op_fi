"""Small developer client reusing the existing OpenAI HTTP transport."""
from ascend_fi.evaluation.live import OpenAICompletionClient
from .extension import ACTIONS


class InspectClient:
    def __init__(self, base_url, transport=None):
        self.transport = transport or OpenAICompletionClient(base_url, timeout_seconds=30)

    def call(self, action, **kwargs):
        if action not in ACTIONS:
            raise ValueError('ACTION_NOT_ALLOWED')
        status, payload, error = self.transport._post('/collective_rpc', {
            'method': 'demoncat_inspect', 'kwargs': {'action': action, **kwargs}})
        if status != 200 or error or not isinstance(payload, dict):
            raise RuntimeError('INSPECT_RPC_FAILED:' + str(error or status))
        rows = payload.get('results')
        if not isinstance(rows, list) or any(not isinstance(r, dict) or r.get('ok') is not True for r in rows):
            raise RuntimeError('INSPECT_WORKER_FAILED:' + str(rows)[:1000])
        if not rows or all(r.get('skipped') for r in rows):
            raise RuntimeError('NO_CURRENT_WORKER_SELECTED; stale epoch?')
        # Preserve worker-provided ranks; array index is never rank.
        return rows
