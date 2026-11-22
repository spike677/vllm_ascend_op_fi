import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock
from fault_injection.product_contract import RequestScope, request_range, summarize_request
from demoncat.runtime.vllm_ascend.worker_extension.product_binding import product_binding_rpc
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig


class ProductContractTests(unittest.TestCase):
    def facts(self, **changes):
        data = dict(request_count=1, single_chunk=True, fresh_cache=True, speculative=False,
                    lora=False, prompt_embeds=False, prompt_length=30, max_tokens=4,
                    image_count=0, video_count=0)
        return dict(data, **changes)

    def test_scope_count_includes_ineligible_arrivals(self):
        scope = RequestScope('next', 2)
        self.assertEqual([scope.includes(n) for n in range(3)], [True, True, False])
        for value in (0, -1, True, 1.5):
            with self.assertRaises(ValueError): RequestScope('next', value)
        with self.assertRaises(ValueError): RequestScope('all', 1)

    def test_arbitrary_prompt_content_is_not_a_policy_input(self):
        for length in (2, 37, 1020):
            self.assertTrue(request_range(self.facts(prompt_length=length))['eligible'])
        self.assertFalse(request_range(self.facts(prompt_length=1021))['eligible'])
        for change in (dict(single_chunk=False), dict(image_count=1), dict(max_tokens=5), dict(fresh_cache=False)):
            report = request_range(self.facts(**change))
            self.assertFalse(report['eligible']); self.assertTrue(report['reasons'])

    def test_missing_and_wrong_typed_facts_never_admit(self):
        for key in self.facts():
            facts = self.facts(); del facts[key]
            self.assertFalse(request_range(facts)['eligible'], key)
        self.assertFalse(request_range(self.facts(request_count=True))['eligible'])

    def test_visual_range_does_not_require_file_or_expected_sha(self):
        facts = self.facts(image_count=1, image_size=[64, 64])
        self.assertTrue(request_range(facts, visual=True)['eligible'])
        self.assertFalse(request_range(dict(facts, image_size=[128, 64]), visual=True)['eligible'])

    def test_effect_zero_change_is_distinct_from_failed_injection(self):
        event = dict(site='live', eligible=True, applied=True, consumed=True, restored=True,
                     changed=0, before_sha256='same', restored_sha256='same', consumer_operator='actual')
        result = summarize_request('user-1', 0, [event], completed=True)
        self.assertEqual(result['coverage']['applied'], 1)
        self.assertEqual(result['coverage']['changed'], 0)
        self.assertTrue(result['bytes_restored'])
        self.assertIsNone(result['sites'][0]['nonfinite_after'])
        self.assertEqual(result['model_baseline_comparison'], 'NOT_PERFORMED_FOR_USER_REQUEST')
        event['restored_sha256'] = 'wrong'
        self.assertFalse(summarize_request('user-1', 0, [event], completed=True)['bytes_restored'])

    def test_skip_keeps_reason_and_zero_counters(self):
        report = summarize_request('request', 1, [], skipped=['MULTI_CHUNK'], completed=True)
        self.assertEqual(report['status'], 'NOT_INJECTED')
        self.assertEqual(report['reasons'], ['MULTI_CHUNK'])
        self.assertEqual(sum(report['coverage'].values()), 0)

    def test_binding_rpc_without_trusted_bootstrap_refuses(self):
        with self.assertRaisesRegex(RuntimeError, 'PRODUCT_RPC_OPERATOR_NOT_AUTHORIZED'):
            product_binding_rpc(NS(model_runner=NS(model=object())), 'attach', {'operator':'user', 'profile':'x'})

    def test_release_requires_counters_not_just_action_success(self):
        from copy import deepcopy
        from test_product_review import receipt
        from demoncat.runtime.vllm_ascend.worker_extension.product_host import released_resources
        cleanup=receipt()['scopes'][0]['disable']['cleanup']
        self.assertTrue(released_resources(cleanup))
        for field,value in (('leases',1),('owned_hooks',None),('inflight',True),('registered',True)):
            bad=deepcopy(cleanup);bad['unregister'][1]['response']['provider_state'][field]=value
            self.assertFalse(released_resources(bad))
        cleanup['detach'][1]['ownership_retained']=True
        self.assertFalse(released_resources(cleanup))

    def test_world_size_one_uses_same_host_group_path(self):
        worker = dict(rank=0)
        executor = Mock()
        executor.collective_rpc.return_value = [dict(status='WORKER_IDENTIFIED', worker=worker, quarantined=False)]
        client = ProductClient(executor, FaultConfig(18,'api_half'), profile='norm', operator='deployment',
                               run_canary=lambda before: None, drain=lambda: None)
        result = client.attach()
        self.assertEqual(result['world_size'], 1)
        self.assertEqual(len(client.clients), 1)
        self.assertTrue(client.clients[0].owner.startswith('worker-group-'))

    def test_visual_product_control_accepts_different_images_without_registered_input(self):
        import torch
        from test_vision17_continuous import Visual
        from fault_injection.sites.vision17 import Vision17Site
        model = Visual().eval()
        for value in (1., 2.):
            site = Vision17Site(model, fault=True, references={}, product_mode=True,
                request_ids=lambda:['user'], stream=lambda:0, complete=lambda stream:None)
            site.install()
            model(torch.full((1,3,2,16,16), value))
            self.assertTrue(all(e['changed'] > 0 and e['consumed'] and e['restored'] for e in site.events))
            self.assertTrue(all(e['native_calls'] == 2 and e['effect_reference'] == 'SAME_INPUT_LOCAL_NATIVE_CONTROL' for e in site.events))
            site.close()

    def test_deployment_file_is_not_an_unchecked_permission_or_native_path(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from demoncat.runtime.vllm_ascend.worker_extension.product_config import load_product_deployment
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'source-review.json';manifest.write_bytes(b'{}')
            config=dict(schema='demoncat.product.deployment.v1', profile='norm',
                fault={'fault_id':18,'variant':'api_half','implementation_version':'1'}, model='/model',tp=2,
                mode='PRODUCT_ACCEPTANCE_ONLY', source='BLUE_COMBINATION',source_manifest=str(manifest),
                source_manifest_sha256=hashlib.sha256(b'{}').hexdigest(),native_library=None,native_sha256=None,
                canary={'prompt':'a canary','image':None,'max_tokens':4},evidence_directory='/receipts')
            path=root/'deployment.json'
            path.write_text(json.dumps(config))
            self.assertEqual(load_product_deployment(path)['fault']['fault_id'],18)
            from demoncat.runtime.vllm_ascend.worker_extension.product_config import configure_product_deployments
            from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import configure_deployment_acceptance
            worker=NS(model_runner=NS(model=object(),vllm_config=NS(model_config=NS(model='/wrong'),
                parallel_config=NS(tensor_parallel_size=2))))
            authority=configure_deployment_acceptance(worker,source='BLUE_COMBINATION',authorize_operator=lambda p,a:p=='trusted')
            with self.assertRaisesRegex(ValueError,'PRODUCT_LIVE_DEPLOYMENT_IDENTITY_MISMATCH'):
                configure_product_deployments(worker,[path],authority=authority,operator='trusted',canary_tokens={'norm':[1,2]})
            self.assertFalse(hasattr(worker,'_demoncat_product_profiles'))
            worker.model_runner.vllm_config.model_config.model='/model'
            result=configure_product_deployments(worker,[path],authority=authority,operator='trusted',canary_tokens={'norm':[1,2]})
            self.assertFalse(result['qualification_granted'])
            self.assertEqual(worker._demoncat_product_profiles['norm']['canary_prompts'],[[1,2]])
            executor=Mock();executor.collective_rpc.return_value=[dict(status='WORKER_IDENTIFIED',worker={'rank':0},quarantined=False)]
            canary=Mock()
            client=ProductClient.from_deployment(executor,path,operator='trusted',run_canary=canary,drain=lambda:None)
            callback=Mock();client.run_canary(callback)
            canary.assert_called_once_with(config['canary'],callback)
            with self.assertRaisesRegex(RuntimeError,'PRODUCT_DEPLOYMENT_WORLD_SIZE_MISMATCH'):client.attach()
            for edit in ({'mode':'PRODUCT_QUALIFIED'}, {'native_library':'/arbitrary.so'}, {'tp':True},
                         {'source_manifest_sha256':'0'*64}, {'rank':0}):
                path.write_text(json.dumps(dict(config,**edit)))
                with self.assertRaises(ValueError):load_product_deployment(path)


if __name__ == '__main__': unittest.main()
