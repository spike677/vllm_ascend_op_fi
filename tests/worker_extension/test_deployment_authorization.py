from types import SimpleNamespace as NS
from unittest.mock import patch
from dataclasses import replace
import unittest
from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as a
from demoncat.runtime.vllm_ascend.worker_extension import deployment_authorization as d
from demoncat.runtime.vllm_ascend.worker_extension import deployment_summary as s

class DeploymentAuthorizationTests(unittest.TestCase):
    def setUp(self):
        self.model=object(); self.worker=NS(model_runner=NS(model=self.model))
        self.authority=d.configure_deployment_acceptance(self.worker,source='YELLOW_TEST_DEPLOYMENT',
            authorize_operator=lambda who,op:who=='operator')
        record=a._CONFIGS['18.api_half'];self.actual=dict(a._COMMON,model_config_sha256=record[0],model_class=record[1],live_config_matches_file=True)
    def permit(self,**kw):
        args=dict(authority=self.authority,operator='operator',mechanisms=['18.api_half'],attempt='try-1',source_manifest_sha256='a'*64);args.update(kw)
        return d.install_deployment_acceptance_permit(self.worker,**args)
    def report(self):
        with patch.object(a,'observe_deployment',return_value=self.actual):return a.deployment_acceptance_report('18.api_half',self.worker,self.model)
    def test_yellow_cannot_borrow_identical_blue_identity(self):
        r=self.report();self.assertFalse(r['can_prepare']);self.assertEqual(r['registration_reason'],'NO_REVIEWED_COMBINATION_FOR_SOURCE')
    def test_operator_authentication_and_scope_validation(self):
        for kw,err in [({'operator':'user'},PermissionError),({'authority':replace(self.authority)},RuntimeError),({'attempt':''},ValueError),({'source_manifest_sha256':'x'*64},ValueError)]:
            with self.subTest(kw=kw),self.assertRaises(err):self.permit(**kw)
        self.assertFalse(hasattr(self.worker,a._BLUE))
    def test_yellow_permit_is_not_a_registration(self):
        self.permit();r=self.report();self.assertTrue(r['can_prepare']);self.assertFalse(r['accepted']);self.assertEqual(r['mode'],'DEPLOYMENT_ACCEPTANCE_ONLY')
        self.assertEqual(r['acceptance_permit']['source_manifest_sha256'],'a'*64)
        with patch.object(a,'observe_deployment',return_value=self.actual):self.assertFalse(a.deployment_acceptance_report('39.output_proxy',self.worker,self.model)['can_prepare'])
        d.remove_deployment_acceptance_permit(self.worker,authority=self.authority,operator='operator');self.assertFalse(self.report()['can_prepare'])
    def test_blue_compatibility_cannot_change_yellow_context(self):
        with self.assertRaisesRegex(RuntimeError,'REFUSES_YELLOW'):a.install_blue_acceptance_permit(self.worker,mechanisms=['18.api_half'],attempt='x',source_manifest_sha256='a'*64)
    def test_model_process_thread_bindings(self):
        self.permit()
        for target in ('os.getpid','threading.get_ident'):
            with self.subTest(target=target),patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance.'+target,return_value=-1):self.assertFalse(self.report()['can_prepare'])
        self.worker.model_runner.model=object();self.assertFalse(self.report()['can_prepare'])
    def test_per_action_record_and_revocation_allows_recovery(self):
        self.permit();p=NS(_deployment_admission=(self.worker,self.model,self.report()))
        x=d.acceptance_use(p,'advance',enforce=True);self.assertEqual(x['permit']['attempt'],'try-1');self.assertFalse(x['accepted_record_created'])
        d.remove_deployment_acceptance_permit(self.worker,authority=self.authority,operator='operator')
        with self.assertRaisesRegex(RuntimeError,'SCOPE_CHANGED'):d.acceptance_use(p,'advance',enforce=True)
        self.assertFalse(d.acceptance_use(p,'clean',enforce=False)['scope_valid'])
    def test_revoke_refuses_owned_lease(self):
        self.permit();self.worker._demoncat_worker_fault_runtime=NS(_leases={'18.api_half':object()})
        with self.assertRaisesRegex(RuntimeError,'OWNER_CLEAN'):d.remove_deployment_acceptance_permit(self.worker,authority=self.authority,operator='operator')
    def test_hardware_is_part_of_blue_identity(self):
        worker=NS(model_runner=NS(model=self.model));d.configure_deployment_acceptance(worker,source='BLUE_COMBINATION',authorize_operator=lambda *_:False)
        for soc,expected in [('A2',True),('A3',False),(None,False)]:
            with patch.object(a,'observe_deployment',return_value=dict(self.actual,hardware_family=soc)):
                self.assertEqual(a.deployment_acceptance_report('18.api_half',worker,self.model)['accepted'],expected)
    def test_reviewed_yellow_record_requires_same_source_and_actual(self):
        row=dict(source='YELLOW_TEST_DEPLOYMENT',mechanism='18.api_half',expected=self.actual,review_reference='reviewed-in-test')
        with patch('demoncat.runtime.vllm_ascend.worker_extension.reviewed_deployments.YELLOW_RECORDS',(row,)):
            self.assertTrue(self.report()['accepted']);self.actual=dict(self.actual,tp=2);self.assertFalse(self.report()['accepted'])

class SummaryTests(unittest.TestCase):
    def receipt(self,owner):
        fields=dict(a._COMMON,model_config_sha256='a'*64,model_class='ActualModel',live_config_matches_file=True)
        state=dict(inflight=0,leases=0,registered=False,enabled=False,native_registered=False,evidence_complete=True,
            deployment_acceptance_use=dict(actual=fields,scope_valid=True,registration_source='YELLOW_TEST_DEPLOYMENT',permit={'attempt':owner}),
            evidence=dict(actual_requests=[dict(phase=p,request_id=str(i),output={'logits_changed':p=='fault'}) for i,p in enumerate(('normal','fault','off','health'))],
                matched_off_and_health_confirmed=True,last_fault_effect={'changed_elements':5,'max_abs_diff_finite':.5}))
        return dict(mechanism='18.api_half',owner=owner,worker={'pid':1},history=[dict(response={'provider_state':state})],status='DETACHED',ownership_retained=False)
    def test_two_scopes_bounded_text_without_registration(self):
        r=s.summarize_deployment([self.receipt('a'),self.receipt('b')]);self.assertTrue(r['review_ready']);self.assertFalse(r['acceptance_granted']);self.assertFalse(r['registration_written']);self.assertEqual(len(r['scopes'][0]['requests']),4)
    def test_duplicate_scope_refused(self):
        r=s.summarize_deployment([self.receipt('a'),self.receipt('a')]);self.assertIn('INDEPENDENT_OWNER_REQUIRED',r['missing_or_failed'])
    def test_missing_or_nonzero_resource_is_incomplete(self):
        for value in (None,1):
            r=self.receipt('b');st=r['history'][0]['response']['provider_state'];st['leases']=value
            self.assertFalse(s.summarize_deployment([self.receipt('a'),r])['review_ready'])
    def test_missing_hardware_and_old_receipts_are_not_upgraded(self):
        r=self.receipt('b');del r['history'][0]['response']['provider_state']['deployment_acceptance_use']
        self.assertFalse(s.summarize_deployment([self.receipt('a'),r])['review_ready'])
    def test_incomplete_wire_data_cannot_become_success(self):
        r=self.receipt('b');r['history'][0]['response']['provider_state']['evidence_complete']=False
        self.assertFalse(s.summarize_deployment([self.receipt('a'),r])['review_ready'])
    def test_oversize_never_silently_truncates_success(self):
        with patch.object(s,'MAX_SUMMARY_BYTES',100):
            r=s.summarize_deployment([self.receipt('a'),self.receipt('b')]);self.assertFalse(r['review_ready']);self.assertTrue(r['summary_payload_omitted'])
    def test_batch_off_proof_does_not_certify_external_health(self):
        r=self.receipt('b');r['mechanism']='quant.batch_shared_scale'
        ev=r['history'][0]['response']['provider_state']['evidence'];ev['actual_requests']=[{'phase':p} for p in ('normal','reference','lossless','calibration_off','fault','off')]
        self.assertIn('BATCH_EXTERNAL_HEALTH_RECEIPT_REQUIRES_SEPARATE_REVIEW',s.summarize_scope(r)['missing_or_failed'])
    def test_cross_deployment_scopes_cannot_merge(self):
        r=self.receipt('b');r['history'][0]['response']['provider_state']['deployment_acceptance_use']['actual']['hardware_family']='A3'
        self.assertIn('SCOPE_MISMATCH:actual',s.summarize_deployment([self.receipt('a'),r])['missing_or_failed'])

if __name__=='__main__':unittest.main()
