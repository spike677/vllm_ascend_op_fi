"""Fresh-build recipe safety contracts; no compiler/SDK or NPU qualification."""
import importlib.util
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


FILE = Path(__file__).resolve().parents[1] / 'fault_injection/native_adapters/attention28/fia/build_fresh.py'
spec = importlib.util.spec_from_file_location('dcat_fia_fresh_build', FILE)
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class FreshBuildTests(unittest.TestCase):
    def fixtures(self, root):
        deps = Path(root) / 'dependencies';repo = deps / 'opbase'
        folder = repo / 'tests/mock';folder.mkdir(parents=True)
        data = b'\x7fELFofficial test fixture'
        (folder / 'test.o').write_bytes(data)
        (repo / 'source.h').write_text('source')
        commit, blob = 'a' * 40, 'b' * 40
        row = dict(path='tests/mock/test.o', sha256=hashlib.sha256(data).hexdigest(), git_blob=blob, bytes=len(data))
        def git(repo_arg, *args):
            if args == ('rev-parse', '--show-toplevel'):return str(repo)
            if args == ('rev-parse', 'HEAD'):return commit
            if args[:2] == ('ls-tree', '--full-tree'):return f'100644 blob {blob}\t{args[-1]}'
            if args[0] == 'status':return ''
            raise AssertionError(args)
        return deps, repo, commit, row, data, git

    def test_source_inventory_refuses_compiled_objects(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'x.h').write_text('source')
            self.assertEqual(build.source_files(root)[0]['path'], 'x.h')
            (root / 'previous.o').write_bytes(b'old binary')
            with self.assertRaisesRegex(RuntimeError, 'BUILT_OBJECT_IN_SOURCE'):
                build.source_files(root)

    def test_policy_verifies_exact_bytes_and_confines_paths(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            (root / 'policy.h').write_text('actual patched content')
            receipt = dict(source_commit=build.COMMIT, status='PATCHED_NOT_BUILT_NOT_NPU_VALIDATED',
                           files=[dict(path='policy.h', after_sha256=build.sha(root / 'policy.h'))])
            with patch.object(build, 'git', return_value=build.COMMIT):
                build.verify_policy(root, receipt)
                (root / 'policy.h').write_text('changed')
                with self.assertRaisesRegex(RuntimeError, 'PATCHED_SOURCE_CHANGED'):
                    build.verify_policy(root, receipt)
                receipt['files'][0]['path'] = '../policy.h'
                with self.assertRaisesRegex(RuntimeError, 'INVALID_POLICY_PATH'):
                    build.verify_policy(root, receipt)

    def test_output_must_be_a_fresh_attempt(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, 'FRESH_BUILD_DIRECTORY_REQUIRED'):
                build.Builder(SimpleNamespace(work=Path(root)))

    def test_only_exact_committed_fixture_removed_unknown_object_still_refused(self):
        with tempfile.TemporaryDirectory() as root:
            deps, repo, commit, row, data, git = self.fixtures(root)
            unknown = repo / 'tests/mock/unlisted.o';unknown.write_bytes(data)
            with patch.object(build, 'git', git), patch.object(build.subprocess, 'check_output', return_value=data):
                receipt = build.remove_disabled_test_fixtures(deps, 'opbase', commit, [row])
            self.assertFalse((repo / row['path']).exists())
            self.assertTrue(unknown.exists())
            self.assertEqual(receipt[0]['disposition'], 'REMOVED_FROM_NEW_CLONE_BEFORE_CONFIGURE')
            with self.assertRaisesRegex(RuntimeError, 'BUILT_OBJECT_IN_SOURCE'):build.source_files(repo)

    def test_all_fixture_hashes_validated_before_any_removal(self):
        with tempfile.TemporaryDirectory() as root:
            deps, repo, commit, row, data, git = self.fixtures(root)
            other = repo / 'tests/mock/second.o';other.write_bytes(data)
            bad = dict(row, path='tests/mock/second.o', sha256='0' * 64)
            with patch.object(build, 'git', git), patch.object(build.subprocess, 'check_output', return_value=data):
                with self.assertRaisesRegex(RuntimeError, 'FIXTURE_CONTENT_HASH_MISMATCH'):
                    build.remove_disabled_test_fixtures(deps, 'opbase', commit, [row, bad])
            self.assertTrue((repo / row['path']).exists())
            self.assertTrue(other.exists())

    def test_untracked_or_non_test_binary_cannot_be_excluded(self):
        with tempfile.TemporaryDirectory() as root:
            deps, repo, commit, row, data, git = self.fixtures(root)
            def untracked(repo_arg, *args):
                return '' if args[0] == 'ls-tree' else git(repo_arg, *args)
            with patch.object(build, 'git', untracked):
                with self.assertRaisesRegex(RuntimeError, 'FIXTURE_NOT_EXACT_COMMITTED_BLOB'):
                    build.remove_disabled_test_fixtures(deps, 'opbase', commit, [row])
            with patch.object(build, 'git', git):
                with self.assertRaisesRegex(RuntimeError, 'FIXTURE_EXCLUSION_PATH_INVALID'):
                    build.remove_disabled_test_fixtures(deps, 'opbase', commit, [dict(row, path='source.o')])
            self.assertTrue((repo / row['path']).exists())


if __name__ == '__main__':unittest.main()
