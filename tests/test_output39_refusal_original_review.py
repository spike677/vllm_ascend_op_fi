"""Actual public CPU driver originals, plus offline negative evidence cases."""
from copy import deepcopy
import json
import os
from pathlib import Path
import unittest

import test_output39_refusal_probe as fixture_module
from experiments.worker_core_v11_20261118 import review39_refusal_originals as R


class RefusalOriginalReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture=fixture_module.DiagnosticProbeTests();fixture.setUp();cls.addClassCleanup(fixture.doCleanups)
        # These are actual entry metadata supplied before the CPU experiment,
        # not corrections to a purported NPU receipt.
        fixture.report.update(schema='demoncat.output39.fixed_c3_refusal_probe.v1',mechanism='39.output_proxy',
            c2=False,c3=False,e5=False,qualification_granted=False,sources=fixture.pins,plan=fixture.plan.__dict__)
        result=fixture.run_driver();path=fixture.out/'result.json'
        path.write_text(json.dumps(result,allow_nan=False),encoding='utf-8')
        cls.document=R.C.read(path);cls.fixture=fixture
        cls.host=R.C.read(fixture.out/result['host_receipt']['path'])
        cls.records=[dict(R.C.read(fixture.out/'rpc'/ref['path']),_reference=ref) for ref in result['rpc_receipts']]
        cls.diagnostic=R.C.read(result['diagnostic_original']['path'])
        cls.root=os.path.commonpath([str(fixture.out),result['requests'][0]['original_observations']['path']])
        cls.final=R.C.read(result['recovery_requests'][-1]['original_observations']['path'])

    def run_review(self,**changes):
        values=dict(d=deepcopy(self.document),host=deepcopy(self.host),records=deepcopy(self.records),
                    originals=R.C.Originals(self.root,natural=False),diagnostic=deepcopy(self.diagnostic),natural=False)
        values.update(changes)
        return R.review(**values)

    def test_real_public_driver_originals_are_consistent_but_never_qualify(self):
        report=self.run_review()
        self.assertEqual(report['issues'],[],report)
        self.assertTrue(report['consistent'])
        self.assertTrue(report['details']['core_originals']['consistent'])
        self.assertEqual(report['details']['native_universe']['unique_leases'],10)
        self.assertEqual(report['details']['counterfactual_full_joins']['single_predecessor_fault']['actual_return_codes'],[0])
        self.assertEqual(report['details']['counterfactual_full_joins']['cumulative_fault']['actual_return_codes'],[0,4])
        for key in ('c2','c3','e5','public_qualification_granted','causal_qualification_granted'):
            self.assertIs(report[key],False)
        self.assertTrue(report['native_build_provenance_required_separately'])

    def test_missing_helper_preserves_completed_core_facts_but_refuses_full_join(self):
        records=[r for r in deepcopy(self.records) if r['method']!=R.HELPER+'finish_diagnostic']
        result=self.run_review(records=records)
        self.assertFalse(result['consistent'])
        self.assertTrue(result['details']['core_originals']['consistent'])

    def test_unknown_rpc_cannot_be_filtered_with_helpers(self):
        records=deepcopy(self.records);records.append(dict(method='unknown.intervention',args=[],responses=[{}]))
        result=self.run_review(records=records)
        self.assertFalse(result['consistent'])
        self.assertIn('UNREVIEWED_RPC_METHOD',[r['code'] for r in result['details']['core_originals']['issues']])

    def test_helper_source_ticket_thread_and_scope_tampering_are_rejected(self):
        for kind in ('source','ticket','thread','scope'):
            with self.subTest(kind=kind):
                diag=deepcopy(self.diagnostic)
                if kind=='source':diag['helper']['source_sha256']='0'*64
                elif kind=='ticket':diag['helper']['actual_provider_ticket'][1]+=1
                elif kind=='thread':diag['helper']['thread']+=1
                else:diag['explicit_cpu_contract']=False
                self.assertFalse(self.run_review(diagnostic=diag)['consistent'])

    def test_original_file_byte_change_is_rejected_without_new_reference(self):
        path=Path(self.document['requests'][6]['original_observations']['path']);old=path.read_bytes()
        try:
            path.write_bytes(old+b' ')
            result=self.run_review()
            self.assertFalse(result['consistent'])
            self.assertTrue(any('ORIGINAL_FILE_SHA_MISMATCH' in str(r) for r in result['issues']))
        finally:path.write_bytes(old)

    def test_same_event_number_changed_in_a_later_snapshot_is_rejected(self):
        first=deepcopy(self.final);last=deepcopy(self.final)
        last['finished_groups']['c3']['native_lease_receipts'][0]['receipt']['events'][1]['changed_elements']=2
        a=R.C.Audit();R.native_universe(a,[first,last],last,natural=False)
        self.assertIn('NATIVE_DUPLICATE_OR_PREFIX_EXACT',[r['code'] for r in a.issues])

    def audit_causal(self,final=None,diag=None):
        a=R.C.Audit();final=deepcopy(self.final) if final is None else final
        unique,_=R.native_universe(a,[final],final,natural=False)
        R.causal_joins(a,self.document,diag or self.diagnostic,final,unique)
        return a

    def test_extra_prior_write_is_not_hidden_by_five_record_pattern(self):
        final=deepcopy(self.final);g=final['finished_groups']['c2'];extra=deepcopy(g['native_lease_receipts'][1])
        extra['actual_request_id']=g['native_lease_receipts'][0]['actual_request_id'];g['native_lease_receipts'].append(extra)
        a=self.audit_causal(final)
        self.assertIn('ALL_ACTUAL_APPLICATIONS_IN_ARM',[r['code'] for r in a.issues])

    def test_wrong_native_target_or_missing_call_is_not_accepted(self):
        for kind in ('target','call'):
            with self.subTest(kind=kind):
                final=deepcopy(self.final);g=final['finished_groups']['c3']
                if kind=='target':g['native_lease_receipts'][0]['target_key']=g['native_lease_receipts'][1]['target_key']
                else:g['forward_artifacts'][1]['inventory']['calls'].pop(0)
                self.assertTrue(self.audit_causal(final).issues)

    def test_matching_fake_span_hashes_still_must_match_original_tensor_digest(self):
        diag=deepcopy(self.diagnostic)
        for row in diag['records']:
            if row['target']=='refused' and row['arm']!='healthy':row['span']['sha256']='a'*64
        self.assertEqual(R.NARROW.review(diag)['issues'],[])
        self.assertIn('COMPLETE_RAW_SPAN_SHA_REBUILDS_ORIGINAL_TENSOR_DIGEST',
            [r['code'] for r in self.audit_causal(diag=diag).issues])

    def test_recovery_bundle_or_owned_resources_changed_is_rejected(self):
        for kind in ('off','resource','proof'):
            with self.subTest(kind=kind):
                final=deepcopy(self.final);g=final['finished_groups']['c3']
                if kind=='off':g['forward_artifacts'][-1]['output_sha256']='f'*64
                elif kind=='resource':g['coverage']['recovery']['workers'][0]['active_leases']=1
                else:g['coverage']['recovery']['targets'][0]['proof_sha256']='f'*64
                a=R.C.Audit();R.recovery(a,self.document,self.host,self.diagnostic,final,natural=False)
                self.assertTrue(a.issues)

    def test_npu_mode_never_accepts_cpu_abi_device_events(self):
        a=R.C.Audit();R.native_universe(a,[self.final],self.final,natural=True)
        self.assertIn('NATIVE_DEVICE_SCOPE',[r['code'] for r in a.issues])

    def test_completed_refusal_is_not_saved_success_boolean_oracle(self):
        d=deepcopy(self.document);d['status']='SUCCESS';d['c3']=True
        result=self.run_review(d=d)
        self.assertFalse(result['consistent']);self.assertFalse(result['c3'])


if __name__=='__main__':unittest.main()
