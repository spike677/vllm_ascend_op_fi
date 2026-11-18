"""Independent saved-evidence contracts; explicit CPU Worker/Native doubles."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.c2c3_postwrite_20261117 import review39_worker_core_result as review


def dictionaries(value):
    if isinstance(value,dict):
        yield value
        for child in value.values(): yield from dictionaries(child)
    elif isinstance(value,(list,tuple)):
        for child in value: yield from dictionaries(child)


class WorkerOutput39ReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_output39_worker_core_probe import Output39ProbeTests
        from fault_injection.deployments.postwrite import PostwriteRefused
        cls.c,cls.p = review.helpers(); cls.cases = {}; cls.saved = {}
        for mode in ('full','no_eligible','precondition','failed_fault','uncertain_arm','cleanup_failure'):
            case=Output39ProbeTests(); case.setUp(); cls.cases[mode]=case
            if mode=='no_eligible':
                case.enterContext(patch.object(case.case.transport,'validate',side_effect=PostwriteRefused('CPU_UNSUPPORTED_PROJECTION')))
            if mode=='precondition':
                def wrong_positions(module,args,kwargs): return args,dict(kwargs,positions=kwargs['positions']+1)
                handle=case.case.model.register_forward_pre_hook(wrong_positions,with_kwargs=True)
                case.addCleanup(handle.remove)
            if mode=='failed_fault': case.driver.fail_index=3
            if mode=='uncertain_arm':
                def lose(method,args,replies,case=case):
                    if method=='demoncat_fault_rpc' and args[0]=='advance' and len(case.driver.enqueued)==4:
                        case.case.executor.after=None
                        raise TimeoutError('CPU_LOST_FAULT_ARM_RESPONSE')
                    return replies
                case.case.executor.after=lose
            if mode=='cleanup_failure':
                original=case.driver.wait_for_completion
                def fail_restore(use_tqdm,case=case,original=original):
                    if len(case.driver.enqueued)==4: case.case.transport.restore_failure=True
                    return original(use_tqdm)
                case.driver.wait_for_completion=fail_restore
            try:
                document=case.drive(); host,raw,artifacts=review.load_artifacts(cls.c,case.out,document)
                serving=json.loads((case.case.provider.session.folder/'serving_receipt.json').read_text())
                cls.saved[mode]=(document,host,raw,serving)
                if not artifacts['consistent']: raise AssertionError(artifacts)
            except BaseException:
                for held in reversed(list(cls.cases.values())): held.doCleanups()
                raise

    @classmethod
    def tearDownClass(cls):
        for case in reversed(list(cls.cases.values())): case.doCleanups()

    def evaluate(self,mode='full',saved=None):
        return review.review_evidence(*(self.saved[mode] if saved is None else saved),natural=False)

    def mutated(self,change,code):
        saved=deepcopy(self.saved['full']);change(saved)
        result=self.evaluate(saved=saved)
        self.assertFalse(result['consistent'],result)
        self.assertIn(code,[v['code'] for v in result['issues']],result['issues'])
        self.assertFalse(result['source_bound_c1_receipts_reconciled'])

    def test_six_actual_cpu_requests_and_seventh_ordinary_request_reconcile(self):
        result=self.evaluate()
        self.assertTrue(result['consistent'],result['issues'])
        self.assertTrue(result['source_bound_c1_receipts_reconciled'] and result['cleanup_reconciled'])
        self.assertEqual(result['details']['c1']['observed_projection_calls'],2)
        self.assertEqual(result['details']['c1']['eligible_pairs'],1)
        self.assertFalse(result['public_qualification_granted'] or result['c2'] or result['c3'] or result['e5'])

    def test_run_status_and_c1_success_flag_are_not_authority(self):
        saved=deepcopy(self.saved['full']);saved[0]['status']='UNTRUSTED_LABEL'
        for row in dictionaries(saved):
            if 'c1_evidence_complete' in row: row['c1_evidence_complete']=False
        result=self.evaluate(saved=saved)
        self.assertTrue(result['consistent'],result['issues'])

    def test_no_eligible_keeps_partial_second_census_and_never_manufactures_c1(self):
        result=self.evaluate('no_eligible')
        self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual(result['classification'],'NO_ELIGIBLE_OBSERVED_PAIR')
        self.assertEqual(result['details']['observations']['completed_requests'],1)
        self.assertEqual(result['details']['observations']['partial_failures'][0]['eligible'],0)
        self.assertTrue(result['cleanup_reconciled'])
        self.assertFalse(result['source_bound_c1_receipts_reconciled'])
        self.assertFalse(self.cases['no_eligible'].case.transport.opened)

    def test_precondition_refusal_has_no_natural_completed_request_or_native_open(self):
        result=self.evaluate('precondition')
        self.assertTrue(result['consistent'],result['issues'])
        self.assertEqual(result['classification'],'PRE_C1_REFUSAL')
        self.assertIn('ACTUAL_DEVICE_INPUT_BYTES_NOT_FROZEN_PROMPT',result['original_driver_error'])
        self.assertEqual(self.cases['precondition'].case.model.forward_calls,0)
        self.assertFalse(result['source_bound_c1_receipts_reconciled'])

    def test_failed_fault_recovery_does_not_erase_original_failure_or_grant_c1(self):
        result=self.evaluate('failed_fault')
        self.assertTrue(result['consistent'],result['issues'])
        self.assertTrue(result['cleanup_reconciled'])
        self.assertIn('CPU_ROOT_FORWARD_FAILED',result['original_driver_error'])
        self.assertFalse(result['source_bound_c1_receipts_reconciled'])
        self.assertEqual(self.cases['failed_fault'].driver.aborted,[self.cases['failed_fault'].driver.enqueued[3]])

    def test_uncertain_fault_arm_retains_original_rpc_exception_and_is_not_fault_success(self):
        result=self.evaluate('uncertain_arm')
        self.assertTrue(result['consistent'],result['issues'])
        self.assertTrue(result['cleanup_reconciled'])
        self.assertFalse(result['source_bound_c1_receipts_reconciled'])
        case=self.cases['uncertain_arm']
        self.assertNotIn(case.driver.enqueued[3],case.driver.executed)

    def test_native_restore_failure_keeps_cleanup_unconfirmed_and_owner_retained(self):
        result=self.evaluate('cleanup_failure')
        self.assertFalse(result['consistent'] or result['cleanup_reconciled'] or result['source_bound_c1_receipts_reconciled'])
        self.assertTrue(self.saved['cleanup_failure'][1]['ownership_retained'])
        self.assertIn('EXPLICIT_PUBLIC_UNREGISTER_WITH_NATIVE_READBACK',[v['code'] for v in result['issues']])

    def test_wrong_rpc_owner_and_generation_are_not_reconciled(self):
        def change(saved):
            for row in dictionaries(saved):
                if 'rpc_generation' in row: row['rpc_generation']+=1
        self.mutated(change,'HOST_RUNTIME_PROVIDER_OWNER_JOIN')

    def test_advance_is_bound_to_actual_request_and_cannot_report_another_request(self):
        def change(saved):
            raw=next(r for r in saved[2] if review.action(r)[0]=='advance')
            raw['args'][1]['settings']['request_id']='foreign-enqueue'
        self.mutated(change,'ADVANCE_DOES_NOT_EXECUTE_REQUEST')

    def test_false_native_effect_rejected_even_when_all_hit_totals_still_one(self):
        def change(saved):
            for row in dictionaries(saved):
                if row.get('boundary')=='post_operator_tensor_output' and row.get('new_fault_hits')==1:
                    row['changed_elements']=0
        self.mutated(change,'ACTUAL_ORIGINAL_NATIVE_EFFECT_READBACK')

    def test_same_value_clone_is_not_the_observed_original_return(self):
        def change(saved):
            for row in dictionaries(saved):
                if 'producer_proof' in row and 'generation' in row and row['generation']==3:
                    row['producer_proof']['returned_tensor']['storage_identity']+=1
        self.mutated(change,'LITERAL_ORIGINAL_PRODUCER_STORAGE_AND_OPERANDS')

    def test_consumer_must_read_selected_real_storage_index(self):
        def change(saved):
            for row in dictionaries(saved):
                if 'selected_elements_read' in row: row['selected_elements_read']=[1]
        self.mutated(change,'ACTUAL_VALUE_CONSUMER')

    def test_zero_hit_off_still_requires_full_logits_restore(self):
        def change(saved):
            for row in dictionaries(saved):
                if row.get('phase')=='off' and 'output_sha256' in row: row['output_sha256']='0'*64
        self.mutated(change,'LOSSLESS_OFF_HEALTH_COMPLETE_LOGITS_HASH')

    def test_partial_denominator_and_hidden_extra_call_are_rejected(self):
        def change(saved):
            for row in dictionaries(saved):
                if 'admission_independent' in row: row['operator_counts']['aten.mm.default']+=1
        self.mutated(change,'PROJECTION_OPERATOR_DENOMINATOR')

    def test_unregistered_reply_with_owned_hook_or_pending_mode_is_not_clean(self):
        def change(saved):
            for row in dictionaries(saved):
                if row.get('registered') is False and row.get('next_phase')=='complete': row['owned_hooks']=1
        self.mutated(change,'FINAL_OWNED_NATIVE_HOOK_AND_RESERVATION_ZERO')

    def test_retained_host_owner_prevents_release_despite_clean_native_totals(self):
        self.mutated(lambda saved:saved[1].update(ownership_retained=True),'ACTUAL_PUBLIC_OWNER_RELEASE_AND_DETACH')

    def test_original_rpc_file_is_rehashed_before_reading(self):
        case=self.cases['full'];item=case.report['rpc_receipts'][0];path=case.recorder.folder/item['path'];original=path.read_bytes()
        try:
            path.write_bytes(original+b'\n')
            _,_,result=review.load_artifacts(self.c,case.out,case.report)
            self.assertFalse(result['consistent'])
            self.assertIn('ORIGINAL_RPC_FILE_SHA',[v['code'] for v in result['issues']])
        finally:path.write_bytes(original)

    def test_natural_scheduler_value_hash_is_recomputed_without_device_access(self):
        row=deepcopy(self.saved['full'][0]['c1']['actual_requests'][0])
        saved=deepcopy(self.saved['full'][3]['requests'][0]);rid=row['actual_internal_request_id']
        # Explicit saved-metadata double; the digest is from the real CPU input
        # tensors above, not fabricated NPU execution or model qualification.
        metadata={name:dict(shape=[4],stride=[1],storage_offset=0,dtype=dtype,device='npu:0',object_id=i,storage_identity=10+i)
            for i,(name,dtype) in enumerate((('input_ids','torch.int32'),('positions','torch.int64')),1)}
        snap=dict(status='READ_ONLY_SCHEDULER_SNAPSHOT_CORROBORATED_NOT_NPU_ACCEPTANCE',internal_request_id=rid,
            device_input_bytes_equality='COMPLETE_DEVICE_READBACK_VERIFIED',token_ids=[1,2,3,4],positions=[0,1,2,3],
            num_requests=1,num_computed_tokens=0,num_prompt_tokens=4,num_scheduled_tokens=4,request_object_id=1,
            token_ids_sha256=hashlib.sha256(b'[1,2,3,4]').hexdigest(),input_tensor_metadata=metadata,
            block_ids=[7],block_size=16,slot_mapping=[112,113,114,115],attention_metadata_keys=['actual'],
            attention_metadata_evidence={'actual':dict(adapter='DENSE_EXPLICIT_LENGTHS_AND_CAUSAL',seq_lens_list=[4],actual_seq_lengths_q=[4],causal=True)})
        row['snapshot']=saved['snapshot']=snap
        a=self.c.Audit();review.scheduler(self.c,a,row,rid,saved,True)
        self.assertTrue(a.report()['consistent'],a.report())
        saved['input_sha256']='0'*64
        a=self.c.Audit();review.scheduler(self.c,a,row,rid,saved,True)
        self.assertIn('COMPLETE_INPUT_HASH_RECOMPUTED',[i['code'] for i in a.issues])

    def test_no_missing_serving_file_is_treated_as_complete_input_proof(self):
        saved=deepcopy(self.saved['full']);saved=saved[:3]+(None,)
        result=self.evaluate(saved=saved)
        self.assertFalse(result['source_bound_c1_receipts_reconciled'])
        self.assertIn('ORIGINAL_SERVING_RECEIPT_COMPLETE',[v['code'] for v in result['issues']])

    def test_explicit_custom_profile_cannot_inherit_legacy_worker_receipts(self):
        result=review.review_evidence(*self.saved['full'],natural=False,profile=review.CUSTOM_PROFILE)
        self.assertFalse(result['consistent'] or result['source_bound_c1_receipts_reconciled'])
        self.assertFalse(result['public_qualification_granted'] or result['c2'] or result['c3'] or result['e5'])


class OfflineOnlyTests(unittest.TestCase):
    def test_import_has_no_framework_or_native_loading(self):
        code=('import sys;from experiments.c2c3_postwrite_20261117 import review39_worker_core_result as r;'
              'sys.addaudithook(r.deny_runtime);r.helpers();'
              'assert not any(x.split(".")[0] in ("torch","torch_npu","vllm","ctypes") for x in sys.modules)')
        result=subprocess.run([sys.executable,'-B','-c',code],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_cli_rejects_original_hash_before_attempting_other_artifacts(self):
        with tempfile.TemporaryDirectory() as root:
            result=Path(root)/'result.json';result.write_text('{}')
            run=subprocess.run([sys.executable,'-B',review.__file__,'--result',str(result),'--result-sha256','0'*64,
                '--source-root',root,'--source-manifest-sha256','0'*64,'--source-commit','a'*40,'--native-library',root],capture_output=True,text=True)
            self.assertEqual(run.returncode,2)
            report=json.loads(run.stdout)
            self.assertIn('ORIGINAL_RESULT_SHA_MISMATCH',report['error'])
            self.assertFalse(report['source_bound_c1_receipts_reconciled'])


if __name__=='__main__':unittest.main()
