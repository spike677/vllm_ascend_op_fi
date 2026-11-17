"""Root-queue mount regression; no Docker, framework or NPU imports."""
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('fa_case_preparer_tests',
    ROOT / 'experiments/c2c3_closure_20261117/prepare_attention28_fa_cases.py')
prepare = importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def inventory(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


class FaCaseMountTests(unittest.TestCase):
    def test_generated_case_preserves_absolute_paths_without_readonly_parent_mount(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);source = root / 'source';observer = root / 'observer/cann910'
            build = root / 'fresh-build/build/run';out = root / 'new-cases.json'
            observer_source = source / 'fault_injection/native_adapters/attention28/launch_observer.cpp'
            write(observer_source, b'CPU fixture observer source')
            write(source / 'native_execution_strategy/native/fa_pod_decoder.cpp', b'CPU fixture decoder source')
            write(build / 'source/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_tiling.h',
                  b'CPU fixture public POD header')
            write(build / 'evidence/completion.json', dict(schema='demoncat.fa.fresh_build.v1',
                status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION', compiled_keys=[22619644976],
                historical_certificate_inherited=False))
            write(build.parents[1] / 'controller.json', dict(status='BUILD_COMMAND_COMPLETED',
                actual_npu_calls=0, image_id='sha256:' + 'a' * 64))
            receipt = dict(observer_compiled=True, actual_npu_calls=0,
                observer_source=dict(path='/repo/fault_injection/native_adapters/attention28/launch_observer.cpp',
                    sha256=prepare.sha(observer_source)),
                observer_library=dict(path='/build/attention-preflight/libdcat_attention_observer.so'),
                sdk_interface_manifest=dict(path='/build/attention-preflight/sdk_interfaces.json'))
            write(observer / 'attention-preflight/result.json', receipt)
            write(observer / 'attention-preflight/libdcat_attention_observer.so', b'CPU fixture observer ELF')
            write(observer / 'attention-preflight/sdk_interfaces.json', {})
            before = inventory(root)
            self.assertFalse((observer / 'run').exists())
            argv = ['prepare', '--source', str(source), '--build-root', str(build),
                    '--observer', str(observer), '--out', str(out)]
            with patch('sys.argv', argv), redirect_stdout(io.StringIO()):prepare.main()
            case, = json.loads(out.read_bytes())
            self.assertEqual(case['timeout'], 90)
            self.assertEqual(case['review']['resource_budget']['timeout_seconds'], 90)
            self.assertEqual(case['review']['physical_cards'], [5])
            self.assertIn('--mode sequence', case['command'])
            self.assertNotIn('--statistics', case['command'])
            mounts = [(Path(src), PurePosixPath(dst), mode) for src, dst, mode in case['mounts']]
            for _, dst, mode in mounts:
                self.assertEqual(mode, 'ro')
                self.assertFalse(any(other != dst and other.is_relative_to(dst) for _, other, _ in mounts),
                    'A read-only parent mount would require immutable source mountpoint creation')
            def mapped(path):
                requested = PurePosixPath(path)
                matches = [src / str(requested.relative_to(dst)) for src, dst, _ in mounts
                           if requested.is_relative_to(dst)]
                self.assertEqual(len(matches), 1)
                self.assertTrue(matches[0].is_file())
                return matches[0]
            self.assertEqual(mapped('/build/attention-preflight/result.json'), observer / 'attention-preflight/result.json')
            mapped(receipt['observer_library']['path']);mapped(receipt['sdk_interface_manifest']['path'])
            self.assertEqual(mapped('/build/run/evidence/completion.json'), build / 'evidence/completion.json')
            mapped('/build/run/source/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_tiling.h')
            mapped('/driver/native_execution_strategy/native/fa_pod_decoder.cpp')
            self.assertEqual({name: digest for name, digest in inventory(root).items() if name != 'new-cases.json'}, before)
            self.assertFalse((observer / 'run').exists())


if __name__ == '__main__':unittest.main()
