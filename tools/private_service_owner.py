"""Explicit private-service deployment of the existing VerifiedOwner lifecycle.

No replacement fault math, no implicit R1 paths, and no inferred lock ownership.
"""
import copy
import hashlib
import json
import shutil
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

from run_normal_diagnostic_live import VerifiedOwner
from ascend_fi.evaluation.distortion.contracts import content_hash
from ascend_fi.evaluation.distortion.mechanism import cleanup_gate

REFERENCE_SHA = '56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7'


class PrivateTransport:
    def __init__(self, token):
        self.token = token

    def _post(self, url, payload, timeout):
        req = urllib.request.Request(url, json.dumps(payload, ensure_ascii=False).encode(),
            {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.token})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.load(r), None
        except urllib.error.HTTPError as e:
            return e.code, None, e.read().decode(errors='replace')
        except (OSError, ValueError) as e:
            return None, None, repr(e)

    def _rpc(self, base, method, kwargs, timeout):
        status, raw, error = self._post(base + '/collective_rpc', {'method': method, 'kwargs': kwargs}, timeout)
        rows = raw.get('results') if isinstance(raw, dict) else None
        # Some vLLM builds JSON-encode each worker result; retain the raw envelope.
        if isinstance(rows, list):
            rows = [json.loads(r) if isinstance(r, str) else r for r in rows]
        if status != 200 or not isinstance(rows, list) or not rows:
            raise RuntimeError('RPC_FAILED:' + str(error or raw)[:1000])
        return {'http_status': status, 'results': rows, 'raw_response': raw}

    def _health(self, base):
        try:
            with urllib.request.urlopen(base + '/health', timeout=10) as r:
                return r.status == 200
        except OSError:
            return False

    @staticmethod
    def _choice_row(sample, request, status, raw, error):
        choice = (raw or {}).get('choices', [{}])[0]
        pieces = (choice.get('logprobs') or {}).get('tokens')
        text = choice.get('text')
        ids = choice.get('token_ids')
        if not isinstance(ids, list) or not all(type(n) is int for n in ids):
            ids = None
        return {'sample_id': sample['id'], 'category': sample['category'], 'request': request,
            'http_status': status, 'error': error, 'response_text': text,
            'response_sha256': hashlib.sha256(text.encode()).hexdigest() if isinstance(text, str) else None,
            'token_ids': ids, 'token_pieces': pieces,
            'comparison_basis': 'token_ids' if ids is not None else 'token_pieces',
            'finish_reason': choice.get('finish_reason'),
            'generation_length': (raw or {}).get('usage', {}).get('completion_tokens'),
            'usage': (raw or {}).get('usage')}


class PrivateServiceOwner(VerifiedOwner):
    def __init__(self, session, suite, deployment):
        self.session, self.suite = Path(session).resolve(), suite
        self.deployment = copy.deepcopy(deployment)
        self.base = deployment['base_url'].rstrip('/')
        parsed = urllib.parse.urlparse(self.base)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
            raise ValueError('LOOPBACK_PRIVATE_SERVICE_REQUIRED')
        self.worker_count = deployment['tp_size']
        self.device_ids = deployment['device_ids']
        if self.worker_count != len(self.device_ids) or self.worker_count < 1:
            raise ValueError('DEPLOYMENT_TOPOLOGY_MISMATCH')
        self.reference_path = Path(deployment['reference_path'])
        self.observer_path = Path(deployment['source_root']) / 'src/ascend_fi/inspect/experiment_extension.py'
        self.access_token = (self.session / 'access.token').read_text().strip()
        if len(self.access_token) < 32:
            raise ValueError('PRIVATE_GENERATION_CREDENTIAL_REQUIRED')
        self.client = PrivateTransport(self.access_token)
        self.model = deployment['served_model_name']
        self.arm = copy.deepcopy(deployment['reference_arm'])
        self.manifest = {'model': self.model, 'r1_arm_config': self.arm,
            'r1_arm_config_sha256': 'sha256:' + content_hash(self.arm),
            'configuration_origin': 'EXPLICIT_NEW_MODEL_DEMO_NOT_FROZEN_R1'}
        self.generation, self.events = 10000, 0
        self.rawroot = self.session / 'raw'
        self.attempt = 'private_model_demo'
        self.experiment = self.target_plan = None
        self.plan_discovery_window = True
        self.request_timeout = 600

    def reserve_single_instance(self):
        """Durable one-driver reservation; never reset after failure or hot remove."""
        self.guard()
        reservation = uuid.uuid4().hex
        with (self.session / 'single_lossy_instance_reserved.json').open('x', encoding='utf-8') as f:
            json.dump({'reservation_id': reservation, 'lease_id': self.deployment['lease_id'],
                       'policy': 'SINGLE_LOSSY_THEN_STOP_INSTANCE'}, f)
        self._single_instance_reservation = reservation

    def guard(self):
        marker = self.session / 'single_lossy_instance_reserved.json'
        if marker.exists() and json.loads(marker.read_text()).get('reservation_id') != getattr(self, '_single_instance_reservation', None):
            raise RuntimeError('INSTANCE_RESERVED_OR_CONSUMED; restart complete model in new session')
        if (self.session / 'request_completion_unproven.json').exists():
            raise RuntimeError('PRIOR_REQUEST_COMPLETION_UNPROVEN')
        lock = json.loads(Path(self.deployment['lock_owner_path']).read_text())
        if (lock.get('owner') != self.deployment['owner'] or lock.get('result_dir') != str(self.session)
                or lock.get('lease_id') != self.deployment['lease_id']):
            raise RuntimeError('EXCLUSIVE_LEASE_MISMATCH')
        if hashlib.sha256(self.reference_path.read_bytes()).hexdigest() != REFERENCE_SHA:
            raise RuntimeError('PINNED_REFERENCE_CHANGED')
        if shutil.disk_usage(self.session).free < 12 * 1024**3:
            raise RuntimeError('DISK_GUARD_12_GIB')
        if self.events >= 20000 or sum(p.stat().st_size for p in self.session.rglob('*') if p.is_file()) > 4 * 1024**3:
            raise RuntimeError('BOUNDED_SESSION_QUOTA')

    def preflight(self):
        self.guard()
        req = urllib.request.Request(self.base + '/v1/models', headers={'Authorization': 'Bearer ' + self.access_token})
        with urllib.request.urlopen(req, timeout=15) as r:
            models = json.load(r)
        if self.model not in [m['id'] for m in models['data']]:
            raise RuntimeError('SERVED_MODEL_MISMATCH')
        for endpoint in ('/v1/models', '/collective_rpc'):
            req = urllib.request.Request(self.base + endpoint,
                data=b'{"method":"fake_requant_query"}' if endpoint == '/collective_rpc' else None,
                headers={'Content-Type': 'application/json'})
            try:
                urllib.request.urlopen(req, timeout=15)
            except urllib.error.HTTPError as e:
                if e.code != 401:
                    raise
            else:
                raise RuntimeError('PRIVATE_AUTH_NOT_PROVEN')
        clean = cleanup_gate(self.clean_query(), self.service_healthy(), self.worker_count)
        if clean['status'] != 'PASS':
            raise RuntimeError('OWNER_NOT_CLEAN:' + str(clean))
        counters = self.metrics()
        if counters['running'] or counters['waiting']:
            raise RuntimeError('ENGINE_NOT_IDLE')
        return {'status': 'PASS', 'exclusive_lock_owned': True, 'models': models, 'clean': clean,
            'tp_size': self.worker_count, 'device_ids': self.device_ids,
            'reference_sha256': REFERENCE_SHA, 'private_auth': 'PASS', 'engine_counters': counters}


class InspectTransport:
    def __init__(self, owner):
        self.owner = owner

    def _post(self, path, payload):
        status, raw, error = self.owner.client._post(self.owner.base + path, payload, 180)
        if isinstance(raw, dict) and isinstance(raw.get('results'), list):
            raw['results'] = [json.loads(r) if isinstance(r, str) else r for r in raw['results']]
        return status, raw, error
