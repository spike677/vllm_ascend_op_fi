"""Protocol regressions for inventory races; no Docker or NPU calls."""
import json
import subprocess
import unittest
from inventory_client import retry_inventory

A, B, C = 'a' * 64, 'b' * 64, 'c' * 64


def result(value=b'', rc=0, error=b''):
    return subprocess.CompletedProcess([], rc, value, error)


def missing(cid):
    return result(rc=1, error=('Error: No such object: ' + cid + '\n').encode())


class RetryTests(unittest.TestCase):
    def call(self, replies, owned=(), failure=None):
        answers = iter(replies)
        events = []
        calls = []
        def query(argv):
            calls.append(argv)
            return next(answers)
        initial = failure or missing(A)
        return retry_inventory([A], initial, set(owned), query, events.append), events, calls

    def test_deleted_foreign_empty_verified_twice(self):
        r, events, calls = self.call([result(), result()])
        self.assertEqual(r.returncode, 0)
        self.assertEqual(json.loads(r.stdout), [])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(events), 1)

    def test_new_blocker_kept_in_fresh_full_inventory(self):
        raw = json.dumps([{'Id': B, 'HostConfig': {'Privileged': True}}]).encode()
        r, _, _ = self.call([result((B + '\n').encode()), result(raw), result((B + '\n').encode())])
        self.assertEqual(json.loads(r.stdout)[0]['Id'], B)
        self.assertTrue(json.loads(r.stdout)[0]['HostConfig']['Privileged'])

    def test_missing_owned_never_retried(self):
        r, events, calls = self.call([], owned=[A])
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual((events, calls), ([], []))

    def test_daemon_error_never_hidden(self):
        r, events, calls = self.call([], failure=result(rc=1, error=b'permission denied\n'))
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual((events, calls), ([], []))

    def test_inventory_instability_bounded_and_rejected(self):
        replies = []
        for _ in range(3):
            replies += [result((B + '\n').encode()), result(json.dumps([{'Id': B}]).encode()), result((C + '\n').encode())]
        r, events, calls = self.call(replies)
        self.assertNotEqual(r.returncode, 0)
        self.assertEqual(len(calls), 9)
        self.assertFalse(events)

    def test_forged_or_incomplete_inspect_rejected(self):
        r, _, _ = self.call([result((B + '\n').encode()), result(b'[]')])
        self.assertNotEqual(r.returncode, 0)


if __name__ == '__main__':
    unittest.main()
