"""Prepare an in-memory controller patch; only analysis artifacts are written."""
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
CONTROLLER=HERE.parent/'controller.py'
RELATIVE='experiments/fault33_paged_tiling_20261105/controller.py'

RECOVERY=r'''def recover_owned_id(token, name, recovery=None,
                     timeout_seconds=UNCERTAIN_CREATE_RECOVERY_SECONDS,
                     poll_seconds=UNCERTAIN_CREATE_RECOVERY_POLL_SECONDS):
    """Find an exact owner within a deadline; no observation is proof of absence."""
    recovery = {} if recovery is None else recovery
    recovery.clear()
    started = time.monotonic()
    deadline = started + timeout_seconds
    recovery.update({'status': 'SEARCHING', 'container_id': None,
                     'controller_token': token, 'task': TASK, 'name': name,
                     'expected_image': IMAGE, 'ownership_verified': False,
                     'cleanup_claimed': False, 'timeout_seconds': timeout_seconds,
                     'poll_seconds': poll_seconds, 'attempts': []})
    try:
        require(timeout_seconds > 0 and poll_seconds > 0,
                'Uncertain-create recovery requires a positive bounded window')
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            event = {'attempt': len(recovery['attempts']) + 1,
                     'elapsed_seconds': time.monotonic() - started,
                     'status': 'QUERYING'}
            recovery['attempts'].append(event)
            try:
                ids = run(['docker', 'ps', '-a', '-q', '--no-trunc',
                           '--filter', 'label=demoncat.controller_token=' + token,
                           '--filter', 'label=demoncat.task=' + TASK],
                          min(UNCERTAIN_CREATE_RECOVERY_QUERY_SECONDS, remaining)).split()
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                event.update(status='QUERY_ERROR', error=str(error)[-2000:])
            else:
                event['candidate_ids'] = ids
                if len(ids) > 1 or not all(re.fullmatch('[0-9a-f]{64}', value) for value in ids):
                    recovery['status'] = 'AMBIGUOUS_OR_INVALID_IDENTITY'
                    raise RuntimeError('Ambiguous token-owned container recovery')
                if ids:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        event['status'] = 'INSPECT_DEADLINE_EXHAUSTED'
                        break
                    try:
                        objects = json.loads(run(['docker', 'inspect', ids[0]],
                            min(UNCERTAIN_CREATE_RECOVERY_QUERY_SECONDS, remaining)))
                    except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
                        event.update(status='INSPECT_ERROR', error=str(error)[-2000:])
                    else:
                        if not isinstance(objects, list) or len(objects) != 1:
                            recovery['status'] = 'AMBIGUOUS_INSPECT_RESULT'
                            raise RuntimeError('Ambiguous recovered inspect result')
                        try:
                            verify_owned(objects[0], token, name, ids[0])
                        except BaseException:
                            recovery['status'] = 'OWNERSHIP_REJECTED'
                            event['status'] = 'OWNERSHIP_REJECTED'
                            raise
                        if time.monotonic() >= deadline:
                            event['status'] = 'VERIFIED_AFTER_DEADLINE'
                            break
                        obj = objects[0]
                        event['status'] = 'EXACT_OWNER_VERIFIED'
                        recovery.update(status='RECOVERED_EXACT_OWNER', container_id=ids[0],
                            ownership_verified=True, elapsed_seconds=time.monotonic() - started,
                            verified_identity={'id': obj['Id'], 'image': obj['Image'],
                                'name': obj['Name'].lstrip('/'),
                                'task': obj['Config']['Labels']['demoncat.task'],
                                'controller_token': obj['Config']['Labels']['demoncat.controller_token']})
                        return ids[0]
                else:
                    event['status'] = 'NO_MATCH_OBSERVED'
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_seconds, remaining))
        recovery.update(status='UNRESOLVED_CREATE_OUTCOME', elapsed_seconds=time.monotonic() - started)
        raise RuntimeError('Uncertain container-create outcome remains unresolved after bounded recovery; '
                           'no absence or cleanup is proven')
    except BaseException:
        recovery['elapsed_seconds'] = time.monotonic() - started
        if recovery['status'] == 'SEARCHING':
            recovery['status'] = 'RECOVERY_ERROR'
        raise
'''


def candidate(before):
    anchor='OWNED_MISSING_PID_RETRY_SECONDS = 0.5\n'
    if before.count(anchor)!=1:
        raise ValueError('Recovery constant anchor differs')
    after=before.replace(anchor,anchor+
        'UNCERTAIN_CREATE_RECOVERY_SECONDS = 60.0\n'
        'UNCERTAIN_CREATE_RECOVERY_POLL_SECONDS = 0.5\n'
        'UNCERTAIN_CREATE_RECOVERY_QUERY_SECONDS = 5.0\n',1)
    start=after.index('def recover_owned_id(token, name):\n')
    end=after.index('\n\ndef device_capability(',start)
    after=after[:start]+RECOVERY.rstrip()+after[end:]
    old="""            if owned is None and create_attempted:
                owned = recover_owned_id(token, name)
                save(evidence / 'uncertain_create_recovery.json',
                     {'container_id': owned, 'controller_token': token,
                      'task': TASK, 'name': name, 'token_query_completed': True})
"""
    new="""            if owned is None and create_attempted:
                recovery = {}
                result['owned_container_removed'] = False
                try:
                    owned = recover_owned_id(token, name, recovery)
                finally:
                    save(evidence / 'uncertain_create_recovery.json', recovery)
                    result['uncertain_create_recovery'] = {
                        'status': recovery.get('status', 'RECOVERY_ERROR'),
                        'ownership_verified': recovery.get('ownership_verified', False),
                        'receipt_file': 'uncertain_create_recovery.json'}
"""
    if after.count(old)!=1:
        raise ValueError('Uncertain-create cleanup anchor differs')
    return after.replace(old,new,1)


def diff(before,after):
    return ''.join(difflib.unified_diff(before.splitlines(keepends=True),after.splitlines(keepends=True),
        fromfile='a/'+RELATIVE,tofile='b/'+RELATIVE))


def main():
    original=CONTROLLER.read_bytes()
    before=original.decode('utf-8')
    after=candidate(before)
    patch=diff(before,after).encode('utf-8')
    (HERE/'controller_recovery.patch').write_bytes(patch)
    receipt={'schema':'fault33.controller_recovery.proposal.v1','status':'PROPOSED_NOT_APPLIED',
        'controller':RELATIVE,'base_sha256':hashlib.sha256(original).hexdigest(),
        'candidate_sha256':hashlib.sha256(after.encode('utf-8')).hexdigest(),
        'patch_sha256':hashlib.sha256(patch).hexdigest(),
        'recovery_window_seconds':60.0,'per_query_timeout_cap_seconds':5.0,'poll_seconds':0.5,
        'normal_successful_create_path_changed':False,'controller_written':False,'npu_executed':False}
    (HERE/'controller_recovery_proposal.json').write_bytes((json.dumps(receipt,indent=2)+'\n').encode('utf-8'))
    if CONTROLLER.read_bytes()!=original:
        raise RuntimeError('Read-only controller invariant failed')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
