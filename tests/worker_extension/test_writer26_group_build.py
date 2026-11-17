"""Explicit CPU mocks for immutable image identity and owned build cleanup."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[2] / 'experiments/c2c3_closure_20261117/providers26_build_group.py'
SPEC = importlib.util.spec_from_file_location('writer26_group_build', PATH)
build = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build)


class GroupBuildTests(unittest.TestCase):
    def test_actual_image_id_is_required(self):
        actual = dict(Id='sha256:' + 'a' * 64, Architecture='arm64', Os='linux')
        with patch.object(build.subprocess, 'run', return_value=NS(stdout=json.dumps([actual]))) as run:
            self.assertEqual(build.inspect_image(), actual)
            self.assertEqual(run.call_args.args[0], ['docker', 'image', 'inspect', build.IMAGE])
        actual['Id'] = build.IMAGE
        with patch.object(build.subprocess, 'run', return_value=NS(stdout=json.dumps([actual]))):
            with self.assertRaisesRegex(RuntimeError, 'ACTUAL_LOCAL_IMAGE_ID_REQUIRED'):
                build.inspect_image()

    def test_cleanup_checks_both_owner_and_image(self):
        actual = dict(Name='/owned', Image='sha256:' + 'a' * 64,
                      Config=dict(Labels={'demoncat.writer26.build-owner': 'nonce'}))
        inspection = NS(returncode=0, stdout=json.dumps([actual]), stderr='')
        removed = NS(returncode=0, stdout='owned', stderr='')
        with patch.object(build.subprocess, 'run', side_effect=[inspection, removed]) as run:
            self.assertTrue(build.cleanup_owned('owned', 'nonce', actual['Image'])['confirmed'])
            self.assertEqual(run.call_args.args[0], ['docker', 'container', 'rm', '-f', 'owned'])
        for nonce, image in [('foreign', actual['Image']), ('nonce', 'sha256:' + 'b' * 64)]:
            with patch.object(build.subprocess, 'run', return_value=inspection) as run:
                self.assertFalse(build.cleanup_owned('owned', nonce, image)['confirmed'])
                self.assertEqual(run.call_count, 1)

    def test_absent_is_distinct_from_daemon_failure(self):
        for error, confirmed in [('Error: No such container: owned', True), ('daemon unavailable', False)]:
            with patch.object(build.subprocess, 'run', return_value=NS(returncode=1, stderr=error)):
                self.assertIs(build.cleanup_owned('owned', 'nonce', 'sha256:' + 'a' * 64)['confirmed'], confirmed)


if __name__ == '__main__':
    unittest.main()
