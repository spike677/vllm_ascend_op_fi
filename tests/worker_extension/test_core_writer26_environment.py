"""Source71 status-shaped CPU fixtures; no environment or Native qualification."""
from copy import deepcopy
import json
import os
import unittest
from unittest.mock import patch

import torch

from fault_injection import environment as env
from fault_injection.coverage_session import CoverageScope
import test_core_writer26_coverage as complete


def environment_fixture():
    names = ('libdcat26_group_v2.so', 'libfault26_readonly_trace_abi0.so', 'libnnopbase.so', 'libascendcl.so')
    return dict(pid=os.getpid(), machine='CPU_ENVIRONMENT_FIXTURE', python='fixture', packages={},
        torch_cxx11_abi=False, driver=dict(installation=None, running_module_versions={}), maps_status='OBSERVED',
        loaded_libraries=[dict(name=name, path='/fixture/'+name, status='MAPPED_FILE_VERIFIED',
            sha256=str(index)*64, elf_build_id=None) for index, name in enumerate(names, 1)])


class CoreWriterEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.case = complete.CoreWriter26CoverageTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        self.case.make(fia=True, count=3)
        self.case.start()
        # Only test the actual-scope branch; all tensor/Native effects remain CPU doubles.
        self.backend = self.case.provider.backend
        self.backend.evidence_kind = 'natural_worker'

    def test_source71_ambiguous_and_changed_mappings_refuse_before_f_and_report_cause(self):
        case = self.case
        for index in range(5):
            case.enqueue(index)
        actual = environment_fixture()
        for row in actual['loaded_libraries'][2:]:
            row.update(status='MAPPED_FILE_IDENTITY_CHANGED', sha256=None)
        actual['loaded_libraries'].extend(dict(name='libcust_opapi.so', path=path,
            status='AMBIGUOUS_MAPPING', sha256=None, elf_build_id=None) for path in
            ('/native/libcust_opapi.so', '/vllm-workspace/_cann_ops_custom/libcust_opapi.so'))
        case.actual_request_id = 'actual-cpu-environment-refusal'
        case.ok(case.client.advance(case.actual_request_id))
        with patch.object(env, 'inspect_environment', return_value=actual), torch.inference_mode():
            with self.assertRaisesRegex(RuntimeError, 'LOADED_WORKER_ENVIRONMENT_UNPROVEN'):
                case.forward(case.provider.session)
        self.assertIsNone(self.backend.coverage)
        self.assertIsNone(self.backend.group)
        self.assertIsNone(self.backend.proof)
        self.assertFalse(case.native.sealed)
        self.assertFalse(any(mode == 'fault' for _, mode, _ in case.native.executes))
        self.assertEqual(case.model.root_calls, 5)
        state = case.client.query()['response']['provider_state']
        refusal = state['evidence']['environment_refusal']
        self.assertEqual(refusal['missing_required_libraries'], [])
        self.assertEqual(len(refusal['unverified_libraries']), 4)
        self.assertFalse(refusal['scope_constructed'] or refusal['qualification_granted'])
        self.assertEqual(json.loads((self.backend.folder/'actual-environment.json').read_text()), actual)
        self.assertTrue(case.client.snapshot()['ownership_retained'])
        case.ok(case.client.unregister())
        self.assertEqual(case.client.detach()['status'], 'DETACHED')

    def test_exact_verified_environment_passes_unchanged_scope_validator(self):
        actual = environment_fixture()
        with patch.object(env, 'inspect_environment', return_value=actual):
            self.backend._environment()
        expected = env.environment_fingerprint(actual, tuple(row['name'] for row in actual['loaded_libraries']))
        self.assertEqual(self.backend.actual_environment_sha, expected)
        scope = CoverageScope(self.case.mechanism, 'e'*64, expected, self.backend.implementation_sha)
        self.assertEqual(scope.environment_sha256, expected)
        self.assertIsNone(self.backend.environment_refusal)
        self.assertFalse(self.case.native.executes or self.case.model.root_calls)

    def test_foreign_pid_is_not_current_worker_environment(self):
        actual = deepcopy(environment_fixture())
        actual['pid'] += 1
        with patch.object(env, 'inspect_environment', return_value=actual):
            with self.assertRaisesRegex(RuntimeError, 'LOADED_WORKER_ENVIRONMENT_UNPROVEN'):
                self.backend._environment()
        self.assertEqual(self.backend.environment_refusal['actual_pid'], actual['pid'])
        self.assertFalse(self.case.native.executes or self.case.model.root_calls)


if __name__ == '__main__':
    unittest.main()
