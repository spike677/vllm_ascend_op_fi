"""Small developer client reusing the existing OpenAI HTTP transport."""
from ascend_fi.evaluation.live import OpenAICompletionClient
from .extension import ACTIONS
import json
from pathlib import Path
import urllib.request
import urllib.error


class AuthenticatedInspectTransport:
    def __init__(self, base_url, api_key_file):
        self.base_url = base_url.rstrip('/')
        self.token = Path(api_key_file).read_text().strip()
        if not self.token:
            raise ValueError('EMPTY_API_KEY')

    def _post(self, path, payload):
        req = urllib.request.Request(self.base_url + path, json.dumps(payload).encode(),
            {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = json.load(r)
                if isinstance(raw, dict) and isinstance(raw.get('results'), list):
                    raw['results'] = [json.loads(v) if isinstance(v, str) else v for v in raw['results']]
                return r.status, raw, None
        except urllib.error.HTTPError as e:
            return e.code, None, e.read().decode(errors='replace')


class InspectClient:
    def __init__(self, base_url, transport=None, api_key_file=None):
        if transport is not None and api_key_file is not None:
            raise ValueError('CHOOSE_TRANSPORT_OR_API_KEY_FILE')
        self.transport = transport or (AuthenticatedInspectTransport(base_url, api_key_file)
            if api_key_file else OpenAICompletionClient(base_url, timeout_seconds=30))

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
