import hashlib,json,tempfile,unittest
from pathlib import Path
from demoncat.runtime.vllm_ascend.worker_extension.deployment_identity import model_artifact_identity,registration_identity_issues
from fault_injection.engines.matmul_evidence import compact_matmul_calls

class IdentityEvidenceTests(unittest.TestCase):
    def test_identity_exact_file_bytes_and_all_index_names(self):
        with tempfile.TemporaryDirectory() as root:
            p=Path(root)
            for name,data in [('quant_model_description.json',b'{}'),('model.safetensors.index.json',b'{ }'),('quant_model_weights.safetensors.index.json',b'[]')]:
                (p/name).write_bytes(data)
            value=model_artifact_identity(root)
            self.assertEqual(value['quant_model_description_sha256'],hashlib.sha256(b'{}').hexdigest())
            self.assertEqual(len(value['safetensors_index_sha256']),2)
            value.update(max_model_len=1024,max_num_batched_tokens=1024,max_num_seqs=1)
            self.assertEqual(registration_identity_issues(value),[])
            for key in tuple(value):
                missing=dict(value);missing.pop(key)
                self.assertTrue(registration_identity_issues(missing),key)
    def test_dedicated_registration_requires_identity_even_if_record_matches(self):
        from types import SimpleNamespace as NS
        from unittest.mock import patch
        from demoncat.runtime.vllm_ascend.worker_extension import deployment_acceptance as a
        worker=NS(model_runner=NS(model=object()));model=worker.model_runner.model
        actual=dict(model_config_sha256='c'*64,model_class='Candidate')
        record=dict(source='YELLOW_DEDICATED_INSTANCE',mechanism='18.api_half',expected=actual,review_reference='fixture')
        with patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization.authority_for',return_value=NS(source='YELLOW_DEDICATED_INSTANCE')), patch('demoncat.runtime.vllm_ascend.worker_extension.reviewed_deployments.YELLOW_RECORDS',[record]), patch.object(a,'observe_deployment',return_value=actual):
            report=a.deployment_acceptance_report('18.api_half',worker,model)
            self.assertFalse(report['accepted']);self.assertEqual(len(report['registration_identity_issues']),5)
            actual.update(quant_model_description_sha256='d'*64,safetensors_index_sha256={'model.safetensors.index.json':'e'*64},max_model_len=1024,max_num_batched_tokens=1024,max_num_seqs=1)
            report=a.deployment_acceptance_report('18.api_half',worker,model)
            self.assertTrue(report['accepted']);self.assertFalse(report['registration_identity_issues'])

    def test_missing_not_defaulted(self):
        self.assertEqual(model_artifact_identity(None),dict(quant_model_description_sha256=None,safetensors_index_sha256={}))
        self.assertEqual(len(registration_identity_issues({})),5)
    def test_positive_int_not_boolean(self):
        self.assertIn('max_num_seqs: MISSING_OR_INVALID',registration_identity_issues(dict(max_num_seqs=True)))
    def data(self):
        sites=['model.layers.'+str(i)+'.projection' for i in range(304)]
        e=[dict(site=s,eligible=True,applied=True,consumed=True,restored=True,changed=1,
            before_sha256='a'*64,restored_sha256='a'*64,full_range_control_passed=True) for _ in range(4) for s in sites]
        return e,[dict.fromkeys(sites,1) for _ in range(4)]
    def test_full_order_hash_compaction_and_budget(self):
        e,f=self.data();p=compact_matmul_calls(e,f)
        self.assertEqual(p['call_count'],1216)
        self.assertEqual(p['ordered_calls_sha256'],hashlib.sha256(json.dumps(e,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        self.assertLess(len(json.dumps(p).encode()),100000)
        e[2]['restored']=False
        with self.assertRaisesRegex(ValueError,'CALL_PROOF'):compact_matmul_calls(e,f)
    def test_reordered_sequence_and_missing_call_not_hidden(self):
        e,f=self.data();one=compact_matmul_calls(e,f)
        e[0],e[1]=e[1],e[0]
        self.assertNotEqual(one['ordered_calls_sha256'],compact_matmul_calls(e,f)['ordered_calls_sha256'])
        e.pop()
        with self.assertRaisesRegex(ValueError,'JOIN'):compact_matmul_calls(e,f)
    def test_independent_review_checks_proofs_but_exports_bounded_summary(self):
        from copy import deepcopy
        from test_continuous_public_review import fixture
        from tools.review_continuous_public import review
        report=fixture();report['mechanism']='39.matmul_range'
        e,f=self.data()
        for row in e:
            if row['site'].endswith('0.projection'):
                row.update(eligible=False,applied=False,consumed=False,restored=False,changed=0,reason='ONLY_LOCAL_COLUMN_SHARDS_NO_COLLECTIVE')
        proof=compact_matmul_calls(e,f)
        count=sum(site['eligible'] for site in proof['sites'])
        for scope in report['scopes']:
            for host in scope['requests']:
                for state in host['result']['response']['provider_state']['rank_states'].values():
                    for row in state['evidence']['actual_requests']:
                        row['coverage']={k:(0 if k=='failed' else 1216 if k=='observed' else count) for k in row['coverage']}
                        row.pop('inapplicable_reasons',None)
                        row.update(site_count=304,events_sha256=proof['ordered_calls_sha256'])
                        row['matmul_effect']=dict(reference_controls=count,rules=['omit_final_quarter'],call_proofs=deepcopy(proof))
        result=review(report)
        self.assertLess(len(json.dumps(result).encode()),60000)
        self.assertEqual(result['scopes'][0]['rank_effects'][0]['matmul_effect']['call_proofs_summary']['site_count'],304)
        bad=report['scopes'][1]['requests'][-1]['result']['response']['provider_state']['rank_states']['0']['evidence']['actual_requests'][1]
        bad['matmul_effect']['call_proofs']['sites'][2]['restored']=0
        with self.assertRaisesRegex(ValueError,'MATMUL_SITE_PROOFS'):review(report)

    def test_normal_and_ineligible_failures_preserved(self):
        e,f=self.data()
        for row in e:
            for k in ('applied','restored','changed','full_range_control_passed'):row[k]=False
        result=compact_matmul_calls(e,f,fault=False)
        self.assertEqual(sum(r['observed'] for r in result['sites']),1216)
        self.assertEqual(sum(r['applied'] for r in result['sites']),0)
        e[0].update(eligible=False,failed=True)
        with self.assertRaisesRegex(ValueError,'CALL_PROOF'):compact_matmul_calls(e,f,fault=False)
if __name__=='__main__':unittest.main()

