"""Strict, runtime-free primitives for saved Worker Core group evidence."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('DUPLICATE_JSON_KEY:' + key)
            value[key] = item
        return value
    def bad(value):
        raise ValueError('NONFINITE_JSON:' + value)
    def floating(value):
        result = float(value)
        return result if math.isfinite(result) else bad(value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs,
                      parse_constant=bad, parse_float=floating)


def sha(value):
    return isinstance(value, str) and re.fullmatch('[a-f0-9]{64}', value) is not None


def positive(value):
    return type(value) is int and value > 0


def under(root, name):
    root = Path(root).resolve(strict=True)
    if not isinstance(name, str) or '\\' in name or ':' in name or PurePosixPath(name).is_absolute() or any(
            part in ('', '.', '..') for part in name.split('/')):
        raise ValueError('UNSAFE_ORIGINAL_PATH')
    value = (root / name).resolve(strict=True)
    if not value.is_relative_to(root) or not value.is_file():
        raise ValueError('ORIGINAL_OUTSIDE_EVIDENCE_ROOT')
    return value


def pure_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Audit:
    def __init__(self):
        self.checks = 0
        self.issues = []

    def check(self, condition, code, where=None):
        self.checks += 1
        if condition is not True:
            self.issues.append(dict(code=code, where=where))

    def section(self, label, function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (KeyError, ValueError, TypeError, IndexError, AttributeError, OSError, AssertionError) as error:
            self.check(False, 'MALFORMED_OR_MISSING_ORIGINAL', label + ':' + repr(error))
            return None

    def report(self):
        return dict(consistent=not self.issues, checks=self.checks, issues=self.issues,
                    public_qualification_granted=False, c2=False, c3=False, e5=False)


class Originals:
    def __init__(self, root, *, natural=True):
        self.root = Path(root).resolve(strict=True)
        self.natural = natural
        self.checked = {}

    def load(self, reference, *, prefix='', mounted=False):
        name = reference['path']
        if mounted:
            if name.startswith('/evidence/'):
                name = name[len('/evidence/'):]
            elif not self.natural:
                name = Path(name).resolve(strict=True).relative_to(self.root).as_posix()
            else:
                raise ValueError('ORIGINAL_EVIDENCE_MOUNT_REQUIRED')
        elif prefix:
            name = prefix + '/' + name
        path = under(self.root, name)
        if not sha(reference['sha256']) or file_sha(path) != reference['sha256']:
            raise ValueError('ORIGINAL_FILE_SHA_MISMATCH:' + name)
        if 'bytes' in reference and (type(reference['bytes']) is not int or path.stat().st_size != reference['bytes']):
            raise ValueError('ORIGINAL_FILE_SIZE_MISMATCH:' + name)
        if name in self.checked and self.checked[name] != reference['sha256']:
            raise ValueError('ORIGINAL_FILENAME_REUSED')
        self.checked[name] = reference['sha256']
        return read(path)


def load_rpc(originals, document, *, prefix='probe'):
    host = originals.load(document['host_receipt'], prefix=prefix)
    records = []
    for index, ref in enumerate(document['rpc_receipts']):
        if ref['path'] != f'{index:03d}.json':
            raise ValueError('ORDERED_ORIGINAL_RPC_SET_REQUIRED')
        raw = originals.load(ref, prefix=(prefix + '/' if prefix else '') + 'worker_core_rpc')
        if raw['method'] != ref['method'] or ('error' in raw) == ('responses' in raw):
            raise ValueError('ONE_RPC_RESULT_OR_EXCEPTION_REQUIRED')
        records.append(dict(raw, _reference=ref))
    return host, records


def rpc_action(record):
    names = {'demoncat_attach': 'attach', 'demoncat_detach': 'detach', 'demoncat_fault_query': 'query'}
    if record['method'] == 'demoncat_fault_rpc':
        return record['args'][0], record['args'][1]
    if record['method'] in names:
        return names[record['method']], record['args'][0]
    return None, None


def protocol(a, document, host, records, mechanism, *, natural):
    calls = [raw for raw in records if rpc_action(raw)[0]]
    history = host['history']; worker = host['worker']
    a.check(len(calls) == len(history), 'COMPLETE_HOST_RPC_ACTION_DENOMINATOR')
    ticket = attachment = internal = None; states = []
    for index, (saved, raw) in enumerate(zip(history, calls)):
        action, payload = rpc_action(raw); reply = saved['response']; loc = f'{index}:{action}'
        a.check(saved['action'] == action, 'ORIGINAL_HOST_RPC_ORDER', loc)
        if 'error' in raw:
            a.check(reply['status'] == 'RPC_UNCERTAIN' and bool(raw['error']), 'ORIGINAL_RPC_EXCEPTION_RETAINED', loc)
            continue
        a.check(len(raw['responses']) == 1 and raw['responses'][0] == reply, 'ORIGINAL_RPC_REPLY_EXACT', loc)
        if action == 'attach':
            attachment = reply.get('ticket')
            a.check(reply['status'] == 'ATTACHED' and attachment and attachment['worker'] == worker and
                    attachment['tenant_id'] == host['tenant_id'], 'ACTUAL_WORKER_ATTACH', loc)
        else:
            a.check(attachment and all(payload.get(k) == v for k, v in attachment.items()), 'RPC_ATTACHMENT_SCOPE', loc)
        supplied = reply.get('ticket')
        if supplied and 'generation' in supplied:
            ticket = ticket or supplied
            a.check(supplied == ticket and ticket['worker'] == worker and ticket['owner'] == host['owner'] and
                    ticket['mechanism'] == mechanism and positive(ticket['generation']), 'RPC_OWNER_GENERATION', loc)
        if ticket and action not in ('attach', 'prepare', 'detach'):
            a.check(all(payload.get(k) == v for k, v in ticket.items()), 'RPC_TICKET_PAYLOAD', loc)
        settings = payload.get('settings', {})
        allowed = (settings == {'collect': True} or set(settings) == {'request_id'} and
                   isinstance(settings['request_id'], str) and bool(settings['request_id'])) if action == 'advance' else settings == {}
        a.check(allowed, 'NO_TARGET_SELECTION_IN_PUBLIC_RPC', loc)
        state = reply.get('provider_state')
        if state is not None:
            internal = internal or state['owner']
            a.check(ticket and state['generation'] == ticket['generation'] and reply.get('native') == state,
                    'CURRENT_PROVIDER_READBACK_JOIN', loc)
            a.check(state['owner'] == internal, 'PRIVATE_PROVIDER_OWNER_STABLE', loc)
            states.append(dict(action=action, state=state, reply=reply, raw=raw, index=index))
    if natural:
        a.check(worker == document['worker'] and worker['rank'] == 0 and positive(worker['pid']) and
                worker['logical_device'] == worker['configured_logical_device'] == worker['observed_logical_device'] == 0 and
                bool(worker['process_start_identity']) and bool(worker['worker_start_identity']) and
                bool(worker['model_instance_id']), 'ONE_ACTUAL_STANDARD_WORKER_LIFETIME')
    for raw in records:
        if rpc_action(raw)[0]:
            continue
        method = raw['method']
        a.check(method in ('demoncat_worker_identity', 'demoncat_capabilities') or method.endswith(('.setup', '.remove', '.remove_binding')),
                'UNREVIEWED_RPC_METHOD', method)
        if method == 'demoncat_worker_identity' and 'responses' in raw:
            a.check(len(raw['responses']) == 1 and raw['responses'][0]['worker'] == worker, 'RPC_ACTUAL_WORKER_ROUTE')
        if method.endswith('.setup') and 'responses' in raw:
            a.check(raw['responses'] == [document.get('setup', document.get('binding'))], 'INERT_SETUP_ORIGINAL_JOIN')
    return dict(states=states, ticket=ticket, provider_owner=internal,
                actions=[x['action'] for x in history], worker=worker)


def final_cleanup(a, document, host, context):
    released = [row for row in context['states'] if row['action'] == 'unregister' and row['reply']['status'] == 'ACTION_COMPLETED']
    a.check(len(released) == 1, 'EXPLICIT_SUCCESSFUL_UNREGISTER')
    if released:
        state = released[0]['state']
        a.check(state['leases'] == state['inflight'] == 0 and state['enabled'] is state['registered'] is False and
                state.get('binding_reserved', False) is False and state.get('native_leases', []) == [], 'FINAL_OWNED_LEASE_ZERO')
        serving = state.get('serving')
        if serving:
            a.check(serving['installed'] is False and serving['owned_hooks'] == 0 and
                    serving.get('expected') is serving.get('current') is None and
                    not any(serving.get(k) for k in ('pending', 'needs_collect', 'inflight', 'root_running', 'logits_running')),
                    'FINAL_OWNED_SERVING_ZERO')
    a.check(host['status'] == 'DETACHED' and host['ownership_retained'] is document['ownership_retained'] is False and
            host['attachment'] is host['ticket'] is host['census_ticket'] is None and
            host['history'][-1] == dict(action='detach', response=dict(status='DETACHED', recoveries=[])),
            'PUBLIC_OWNER_DETACHED')
    return len(released) == 1


def deny_runtime(event, args):
    if event in ('ctypes.dlopen', 'subprocess.Popen', 'socket.connect', 'os.system') or (
            event == 'import' and args[0].split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend',
                                                         'fault_injection', 'demoncat')):
        raise RuntimeError('OFFLINE_AUDIT_FORBIDS_RUNTIME:' + event)
