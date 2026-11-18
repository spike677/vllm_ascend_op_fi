"""Offline reviewers consume the real CPU public-protocol fixture's originals.

CPU doubles never become Native qualification. Mutations change in-memory audit
inputs only; no original saved receipt is overwritten.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.worker_core_v11_20261118 import review39_group_originals as r


class ProcessStartRepresentationTests(unittest.TestCase):
    """Fixed saved examples exercise both OS branches on either test platform."""
    def test_current_cpu_complete_process_local_requires_exact_pid_and_nonce(self):
        marker='process_local:123:0123456789abcdef0123456789abcdef'
        worker=dict(pid=123,process_start_identity=marker)
        self.assertTrue(r.process_start_matches(marker,worker,natural=False))
        self.assertFalse(r.process_start_matches(marker,dict(worker,pid=124),natural=False))
        self.assertFalse(r.process_start_matches(marker[:-1]+'0',worker,natural=False))
        for other in ('process_local:123:short',marker+':extra','process_local:123:'+'g'*32):
            self.assertFalse(r.process_start_matches(other,dict(pid=123,process_start_identity=other),natural=False))

    def test_natural_rejects_matching_process_local_and_malformed_proc(self):
        for marker in ('process_local:123:0123456789abcdef0123456789abcdef',
                       'cpu-contract:123:45678','45678','proc::123:45678','proc:boot:124:45678'):
            self.assertFalse(r.process_start_matches(marker,dict(pid=123,process_start_identity=marker),natural=True))

    def test_current_cpu_complete_proc_identity_requires_exact_pid_and_marker(self):
        marker='proc:e15e1842-4541-4ec5-a763-0666d2cf3ed1:1:64770165'
        worker=dict(pid=1,process_start_identity=marker)
        self.assertTrue(r.process_start_matches(marker,worker,natural=False))
        self.assertFalse(r.process_start_matches(marker,dict(worker,pid=2),natural=False))
        for other in ('proc:foreign-boot:1:64770165',marker[:-1]+'6','proc::1:64770165'):
            self.assertFalse(r.process_start_matches(other,worker,natural=False))
        invalid='proc::1:64770165'
        self.assertFalse(r.process_start_matches(invalid,dict(pid=1,process_start_identity=invalid),natural=False))

    def test_linux_cpu_tick_joins_the_exact_recorded_worker_tick_and_pid(self):
        worker=dict(pid=123,process_start_identity='proc:fixture-boot:123:45678')
        self.assertTrue(r.process_start_matches('45678',worker,natural=False))
        self.assertFalse(r.process_start_matches('45679',worker,natural=False))
        self.assertFalse(r.process_start_matches('045678',worker,natural=False))
        self.assertFalse(r.process_start_matches('45678',dict(worker,pid=124),natural=False))

    def test_linux_tick_needs_a_complete_worker_proc_identity(self):
        for marker in ('45678','proc::123:45678','proc:boot:124:45678','process_local:123:fixture'):
            self.assertFalse(r.process_start_matches('45678',dict(pid=123,process_start_identity=marker),natural=False))

    def test_windows_cpu_fallback_has_exact_pid_and_numeric_nonce(self):
        worker=dict(pid=123,process_start_identity='process_local:123:fixture')
        self.assertTrue(r.process_start_matches('cpu-contract:123:45678',worker,natural=False))
        for marker in ('cpu-contract:124:45678','cpu-contract:123:unknown','cpu-contract:123:', 'cpu-contract:123:45678:extra'):
            self.assertFalse(r.process_start_matches(marker,worker,natural=False))

    def test_natural_worker_still_requires_the_complete_exact_identity(self):
        worker=dict(pid=123,process_start_identity='proc:fixture-boot:123:45678')
        self.assertTrue(r.process_start_matches(worker['process_start_identity'],worker,natural=True))
        for marker in ('45678','cpu-contract:123:45678','proc:foreign-boot:123:45678','proc:fixture-boot:123:45679'):
            self.assertFalse(r.process_start_matches(marker,worker,natural=True))


class CurrentProducerPlatformReviewTests(unittest.TestCase):
    """Run the current producer and real CPU RPC/collection chain for both OS forms.

    Only the platform identity reader is a deterministic CPU test input. No
    provider, collector, auditor or saved original is replaced or rewritten.
    """
    def check_current_producer(self,pattern):
        from test_output39_group_worker_core_probe import Output39GroupProbeTests
        from demoncat.runtime.vllm_ascend.worker_extension import identity
        with patch.object(identity,'_process_start',side_effect=lambda pid:pattern.format(pid=pid)):
            case=Output39GroupProbeTests();case.setUp()
            try:
                document=case.drive()
                host,rpc=r.C.load_rpc(r.C.Originals(case.out,natural=False),document,prefix='')
                root=os.path.commonpath([str(case.out),document['requests'][0]['original_observations']['path']])
                originals=r.C.Originals(root,natural=False)
                final=originals.load(document['requests'][-1]['original_observations'],mounted=True)
                marker=pattern.format(pid=host['worker']['pid'])
                self.assertEqual(host['worker']['process_start_identity'],marker)
                for group in final['finished_groups'].values():
                    self.assertEqual(group['scope']['evidence_kind'],'cpu_contract')
                    for row in group['coverage']['recovery']['targets']:
                        self.assertEqual(row['lease']['target']['process_start_identity'],marker)
                result=r.review(document,host,rpc,originals,natural=False)
                self.assertTrue(result['consistent'],result['issues'])
                self.assertEqual(result['classification'],'CPU_CONTRACT_ONLY')
                self.assertFalse(result['c2'] or result['c3'] or result['e5'] or result['public_qualification_granted'])
            finally:
                case.doCleanups()

    def test_current_producer_linux_proc_full_public_protocol(self):
        self.check_current_producer('proc:fixed-cpu-test-boot:{pid}:64770165')

    def test_current_producer_windows_process_local_full_public_protocol(self):
        self.check_current_producer('process_local:{pid}:0123456789abcdef0123456789abcdef')


class Output39OriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_output39_group_worker_core_probe import Output39GroupProbeTests
        cls.case = Output39GroupProbeTests(); cls.case.setUp()
        cls.addClassCleanup(cls.case.doCleanups)
        cls.document = cls.case.drive()
        cls.host, cls.rpc = r.C.load_rpc(r.C.Originals(cls.case.out,natural=False),cls.document,prefix='')
        cls.root = os.path.commonpath([str(cls.case.out),cls.document['requests'][0]['original_observations']['path']])
        cls.final = r.C.read(cls.document['requests'][-1]['original_observations']['path'])
        cls.c, cls.p = r.OLD.helpers(r.OLD.CUSTOM_PROFILE)

    def group(self, group, name='c2', natural=False):
        audit = r.C.Audit()
        audit.section(name,r.check_group,audit,self.p,group,name,self.final['actual_requests'],self.host['worker'],natural=natural)
        return audit.report()

    def test_actual_complete_protocol_cpu_only(self):
        result=r.review(self.document,self.host,self.rpc,r.C.Originals(self.root,natural=False),natural=False)
        self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual((result['details']['c2']['recomputed_hits'],result['details']['c3']['recomputed_hits']),(2,4))
        self.assertFalse(result['c2'] or result['c3'] or result['e5'] or result['public_qualification_granted'])

    def test_probe_success_boolean_is_not_acceptance_oracle(self):
        d=deepcopy(self.document); d.update(status='ARBITRARY_SUCCESS',c2=True,c3=True)
        result=r.review(d,self.host,self.rpc,r.C.Originals(self.root,natural=False),natural=False)
        self.assertTrue(result['consistent'],result['issues'])
        self.assertFalse(result['c2'] or result['c3'])

    def test_saved_contract_boolean_is_not_acceptance_oracle(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['coverage']['contract_satisfied']=False
        self.assertTrue(self.group(g)['consistent'])

    def test_cpu_original_cannot_be_natural_worker_evidence(self):
        self.assertFalse(self.group(self.final['finished_groups']['c2'],natural=True)['consistent'])

    def test_missing_target_recovery_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['coverage']['recovery']['targets'].pop()
        self.assertFalse(self.group(g)['consistent'])

    def test_missing_effect_rejected_even_if_hit_total_unchanged(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['execution_artifacts'].pop()
        self.assertFalse(self.group(g)['consistent'])

    def test_changed_consumer_indices_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['execution_artifacts'][0]['consumer_observations'][0]['reads_selected_indices']=False
        # The digest reference must reject any alteration, even an unknown field.
        self.assertFalse(self.group(g)['consistent'])

    def test_no_actual_value_change_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);effect=g['execution_artifacts'][0]
        effect['mutated_output_sha256']=effect['original_output_sha256'];effect['changed_elements']=0
        self.assertFalse(self.group(g)['consistent'])

    def test_off_bundle_changed_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['forward_artifacts'][-1]['output_sha256']='0'*64
        self.assertFalse(self.group(g)['consistent'])

    def test_resource_leak_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['coverage']['recovery']['workers'][0]['host_bytes']=8
        self.assertFalse(self.group(g)['consistent'])

    def test_c3_missing_second_forward_rejected(self):
        g=deepcopy(self.final['finished_groups']['c3']);g['forward_artifacts'].pop(2)
        self.assertFalse(self.group(g,name='c3')['consistent'])

    def test_ghost_scope_worker_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['worker']['worker_start_identity']='foreign'
        self.assertFalse(self.group(g)['consistent'])

    def test_inventory_projection_omission_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['independent_discovery'][0]['calls'].pop()
        self.assertFalse(self.group(g)['consistent'])

    def test_false_admission_or_native_readback_rejected(self):
        g=deepcopy(self.final['finished_groups']['c2']);g['coverage']['calls'][0]['receipt']['native_applied']=False
        self.assertFalse(self.group(g)['consistent'])

    def test_corrupt_hash_and_path_escape_refused(self):
        o=r.C.Originals(self.root,natural=False);ref=deepcopy(self.document['requests'][0]['original_observations'])
        ref['sha256']='0'*64
        with self.assertRaises(ValueError):o.load(ref,mounted=True)
        with self.assertRaises(ValueError):r.C.under(self.root,'../foreign.json')

    def test_duplicate_keys_and_nonfinite_json_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'input.json'
            for value in ('{"a":1,"a":2}','{"x":NaN}','{"x":1e999}'):
                p.write_text(value)
                with self.assertRaises(ValueError):r.C.read(p)


if __name__=='__main__':unittest.main()
