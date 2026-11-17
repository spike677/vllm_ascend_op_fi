"""Host-only ABI wrapper tests with explicit CPU memory; not ACL/NPU evidence."""
import ctypes
from types import SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import torch

from fault_injection.deployments.postwrite import NativePostwriteLibrary, PostwriteRefused


class CpuMemoryAsNpuMetadata:
    # A test-only descriptor; the production provider cannot construct this.
    device = NS(type='npu', index=0)
    def __init__(self, tensor):self.tensor = tensor
    def __getattr__(self, name):return getattr(self.tensor, name)


class NativeSnapshotWrapperTests(unittest.TestCase):
    def library(self):
        result = NativePostwriteLibrary.__new__(NativePostwriteLibrary)
        result.calls = []
        def read(pointer, size, device, host):
            result.calls.append((pointer, size, device))
            ctypes.memmove(host, pointer, size)
            return 0
        result._snapshot = read
        return result

    def test_exact_bounded_read_of_live_nonzero_offset_view(self):
        source = torch.arange(24, dtype=torch.bfloat16)
        tensor = CpuMemoryAsNpuMetadata(source[8:20])
        before = source.clone()
        library = self.library()
        with patch.dict(sys.modules, {'torch_npu':NS(get_npu_format=lambda unused:2)}):
            actual = library.snapshot(tensor, 2, 4)
        self.assertEqual(actual, source[10:14].view(torch.uint8).numpy().tobytes())
        self.assertEqual(library.calls, [(source.data_ptr() + 20, 8, 0)])
        self.assertTrue(torch.equal(source, before))

    def test_missing_fresh_export_refuses(self):
        library = self.library(); library._snapshot = None
        with self.assertRaisesRegex(PostwriteRefused, 'FRESH_READ_SPAN'):
            library.snapshot(CpuMemoryAsNpuMetadata(torch.ones(4, dtype=torch.bfloat16)))

    def test_bounds_format_and_device_errors_never_copy(self):
        library = self.library()
        tensor = CpuMemoryAsNpuMetadata(torch.ones(8, dtype=torch.bfloat16))
        with patch.dict(sys.modules, {'torch_npu':NS(get_npu_format=lambda unused:2)}):
            for offset, count in ((-1, 1), (7, 2), (0, -1)):
                with self.assertRaisesRegex(PostwriteRefused, 'SPAN_OUT_OF_BOUNDS'):
                    library.snapshot(tensor, offset, count)
        with patch.dict(sys.modules, {'torch_npu':NS(get_npu_format=lambda unused:29)}):
            with self.assertRaisesRegex(PostwriteRefused, 'BASE_FORMAT'):
                library.snapshot(tensor)
        with self.assertRaisesRegex(PostwriteRefused, 'CONTIGUOUS_NPU'):
            library.snapshot(torch.ones(8, dtype=torch.bfloat16))
        self.assertEqual(library.calls, [])

    def test_layout_change_during_native_read_is_rejected(self):
        library = self.library()
        tensor = CpuMemoryAsNpuMetadata(torch.ones(8, dtype=torch.bfloat16))
        read = library._snapshot
        def changed(*args):
            result = read(*args)
            tensor.tensor = tensor.tensor.view(2, 4)
            return result
        library._snapshot = changed
        with patch.dict(sys.modules, {'torch_npu':NS(get_npu_format=lambda unused:2)}):
            with self.assertRaisesRegex(PostwriteRefused, 'SNAPSHOT_FAILED'):
                library.snapshot(tensor)


if __name__ == '__main__':unittest.main()
