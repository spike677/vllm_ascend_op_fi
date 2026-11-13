#!/usr/bin/python3
"""Task-private Docker client: retry a vanished foreign inventory object only.

This is infrastructure added after preregistration, not a scientific source
change. The real host Docker entry and all frozen controller files stay intact.
No mutating Docker command is retried or changed. No inspect payload is logged.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

REAL = '/usr/bin/docker'
ROOT = Path('/home/fault28-reduce-replay-20261113')
STATE = ROOT / 'build' / 'inventory_client_a1'
TASK = 'fault28-reduce-replay-20261113'
FULL_ID = re.compile(r'[0-9a-f]{64}')
MISSING = re.compile(r'Error(?: response from daemon)?: No such (?:object|container):\s*([0-9a-f]{64})')


def retry_inventory(requested, initial, owned, query, event):
    """Return a stable fresh inventory, or the original failure (fail closed)."""
    messages = [x.strip() for x in initial.stderr.decode().splitlines() if x.strip()]
    matches = [MISSING.fullmatch(x) for x in messages]
    if not matches or not all(matches):
        return initial
    missing = {x.group(1) for x in matches}
    if not missing <= set(requested) or missing & owned:
        return initial
    began = time.monotonic()
    for number in range(3):
        first = query(['ps', '-q', '--no-trunc'])
        if first.returncode:
            return first
        ids = first.stdout.decode().split()
        if len(ids) != len(set(ids)) or not all(FULL_ID.fullmatch(x) for x in ids):
            return initial
        # A requested owned object disappearing is never normalized to success.
        if (set(requested) & owned) - set(ids):
            return initial
        snapshot = query(['inspect', *ids]) if ids else subprocess.CompletedProcess([], 0, b'[]\n', b'')
        if snapshot.returncode:
            lines = [x.strip() for x in snapshot.stderr.decode().splitlines() if x.strip()]
            mm = [MISSING.fullmatch(x) for x in lines]
            if not mm or not all(mm) or {x.group(1) for x in mm} & owned:
                return snapshot
            continue
        try:
            objects = json.loads(snapshot.stdout)
            actual = [x['Id'] for x in objects]
            assert len(actual) == len(ids) and set(actual) == set(ids)
        except (ValueError, TypeError, KeyError, AssertionError):
            return initial
        second = query(['ps', '-q', '--no-trunc'])
        if second.returncode:
            return second
        second_ids = second.stdout.decode().split()
        if len(second_ids) != len(set(second_ids)) or set(second_ids) != set(ids):
            continue
        event({'event': 'foreign_inventory_disappearance_recovered', 'epoch': time.time(),
               'original_requested_ids': requested, 'disappeared_foreign_ids': sorted(missing),
               'fresh_running_ids': ids, 'retry_number': number + 1,
               'elapsed_seconds': time.monotonic() - began,
               'scope': 'Fresh complete inspect returned to unchanged resource gate; no device approval here.'})
        return snapshot
    return initial


def main():
    argv = sys.argv[1:]
    inspect = len(argv) > 1 and argv[0] == 'inspect' and all(FULL_ID.fullmatch(x) for x in argv[1:])
    own_run = argv[:2] == ['run', '-d'] and '--label' in argv and ('demoncat.task=' + TASK) in argv
    if not inspect and not own_run:
        os.execv(REAL, [REAL, *argv])
    STATE.mkdir(parents=True, exist_ok=True)
    if own_run:
        # Exact original arguments; do not record launch/environment contents.
        result = subprocess.run([REAL, *argv], stdout=subprocess.PIPE)
        if result.returncode == 0:
            cid = result.stdout.decode().strip()
            if FULL_ID.fullmatch(cid):
                with (STATE / 'owned_ids.txt').open('a') as stream:
                    stream.write(cid + '\n')
        sys.stdout.buffer.write(result.stdout)
        return result.returncode
    owned_file = STATE / 'owned_ids.txt'
    owned = set(owned_file.read_text().split()) if owned_file.exists() else set()
    # The controller already bounds each client call to 25 seconds. Keep all
    # read-only retries inside a stricter 18-second aggregate deadline.
    deadline = time.monotonic() + 18
    def query(args):
        return subprocess.run([REAL, *args], capture_output=True, timeout=max(.01, deadline - time.monotonic()))
    def event(value):
        with (STATE / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n')
    initial = query(argv)
    if initial.returncode:
        initial = retry_inventory(argv[1:], initial, owned, query, event)
    sys.stdout.buffer.write(initial.stdout)
    sys.stderr.buffer.write(initial.stderr)
    return initial.returncode


if __name__ == '__main__':
    raise SystemExit(main())
