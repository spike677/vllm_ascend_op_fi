"""Explicit test identity cannot become a real Native receipt."""
import unittest
from unittest.mock import patch
import torch
from fault_injection.cpu_build_identity import cpu_test_build_identity, current_cpu_identity, require_cpu_values
from fault_injection.deployments.postwrite import NativePostwriteLibrary
from fault_injection.deployments import postwrite_npu_consumers as consumers
from demoncat.runtime.vllm_ascend.worker_extension import gemm_lineage

class CpuBuildIdentityTests(unittest.TestCase):
    def profile(self):
        return cpu_test_build_identity(name='CPU_FIXTURE_IDENTITY_TEST',torch_version=torch.__version__,
            torch_npu_version='CPU_DOUBLE',source_hashes={'fixture':'a'*64},registrations={})

    def test_does_not_mutate_production_pins_and_restores_on_failure(self):
        pins=(consumers.TORCH_VERSION,consumers.TORCH_NPU_VERSION,consumers.META_SHA256,
              gemm_lineage.DISPATCH_TORCH_VERSION,gemm_lineage.PATCH_SHA256)
        with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
            with self.profile() as profile:
                self.assertTrue(profile.report()['test_only'])
                self.assertFalse(profile.report()['deployment_qualification'])
                require_cpu_values(torch.ones(1))
                with self.assertRaises(TypeError): profile.source_hashes['fixture']='b'*64
                raise RuntimeError('fixture failure')
        self.assertIsNone(current_cpu_identity())
        self.assertEqual(pins,(consumers.TORCH_VERSION,consumers.TORCH_NPU_VERSION,consumers.META_SHA256,
              gemm_lineage.DISPATCH_TORCH_VERSION,gemm_lineage.PATCH_SHA256))

    def test_device_tensor_and_actual_native_load_refused_before_io(self):
        with self.profile():
            with self.assertRaisesRegex(RuntimeError,'CPU_TEST_BUILD_CANNOT_ATTEST_DEVICE_TENSOR'):
                require_cpu_values(torch.empty(1,device='meta'))
            with patch('ctypes.CDLL',side_effect=AssertionError('native load')):
                with self.assertRaisesRegex(RuntimeError,'CPU_TEST_BUILD_CANNOT_LOAD_NATIVE_TRANSPORT'):
                    NativePostwriteLibrary('does-not-exist','a'*64)
            value=NativePostwriteLibrary.__new__(NativePostwriteLibrary)
            with self.assertRaisesRegex(RuntimeError,'CPU_TEST_BUILD_CANNOT_OPEN_NATIVE_LEASE'):
                value.open(None,1,1)
            with self.assertRaisesRegex(RuntimeError,'CPU_TEST_BUILD_CANNOT_OPEN_NATIVE_LEASE'):
                value.open_span(None,0,1,1,1)

    def test_nested_scope_cannot_replace_identity(self):
        with self.profile() as value:
            with self.assertRaisesRegex(RuntimeError,'EXPLICIT_UNNESTED_CPU_BUILD_IDENTITY_REQUIRED'):
                with self.profile(): pass
            self.assertIs(current_cpu_identity(),value)

if __name__=='__main__':unittest.main()
