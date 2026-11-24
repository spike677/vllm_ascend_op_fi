import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch, Mock
import torch
from fault_injection.engines.region_lease import RegionLeases
from fault_injection.product_contract import request_range
from demoncat.runtime.vllm_ascend.worker_extension.external_provider import ExternalSite
from demoncat.runtime.vllm_ascend.worker_extension.product_request import ProductRequest
import test_external_provider as fixture
from experiments.worker_core_v11_20261118.external_vision_example import ExternalVisionGainSite


class VisionRegistrationTests(unittest.TestCase):
    setUp=fixture.ExternalProviderTests.setUp
    install=fixture.ExternalProviderTests.install
    def test_trusted_vision_stage_is_explicit_not_grant(self):
        result=self.install(ExternalSite(self.spec.mechanism,self.spec.factory,self.spec.source_sha256,
                                        self.actual,stage='vision'))
        self.assertFalse(result['qualification_granted'])
        from demoncat.runtime.vllm_ascend.worker_extension.external_provider import external_site
        self.assertEqual(external_site(self.worker,self.spec.mechanism).stage,'vision')


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__();self.patch_embed=torch.nn.Module()
        self.patch_embed.proj=torch.nn.Conv3d(1,2,1,bias=False).to(torch.bfloat16)
        self.patch_embed.proj.weight.data.fill_(1)
    def forward(self,x):return self.patch_embed.proj(x).flatten(1)+1


class VisionTests(unittest.TestCase):
    def test_actual_conv_output_consumer_and_readonly_weights(self):
        model=torch.nn.Module();model.visual=Visual()
        runner=NS(input_batch=NS(req_ids=['image-request']))
        leases=RegionLeases('owner',1,stream=lambda:0,complete=lambda s:None)
        site=ExternalVisionGainSite(model,runner,fault=True,region_leases=leases)
        x=torch.ones(1,1,2,2,2,dtype=torch.bfloat16);before=x.clone()
        site.install()
        for i in range(2):
            y=model.visual(x);site.finish_forward()
            self.assertTrue(torch.equal(y,torch.full_like(y,1.75)))
        self.assertTrue(torch.equal(x,before));self.assertTrue(torch.all(model.visual.patch_embed.proj.weight==1))
        self.assertEqual(len(site.events),2)
        self.assertTrue(all(e['changed']==1 and e['changed_elements']==16 and e['consumed'] and
                            e['restored'] and e['readonly_inputs_weights_unchanged'] for e in site.events))
        site.close();self.assertEqual(leases.count(),0)
        self.assertTrue(torch.all(model.visual(x)==2))

    def test_only_registered_vision_routes_images(self):
        worker=NS(model_runner=NS(model=object()))
        b=NS(worker=worker,model=worker.model_runner.model,mechanism='external.conv',prompts=((1,2),))
        for stage in ('model','sampler','vision'):
            with patch('demoncat.runtime.vllm_ascend.worker_extension.external_provider.external_site',return_value=NS(stage=stage)):
                request=ProductRequest(b,['r'])
                self.assertEqual(request.visual,stage=='vision')
                facts=dict(request_count=1,single_chunk=True,fresh_cache=True,speculative=False,lora=False,
                    prompt_embeds=False,prompt_length=20,max_tokens=4,image_count=1,video_count=0,image_size=[64,64])
                self.assertEqual(request_range(facts,visual=request.visual)['eligible'],stage=='vision')
        b.mechanism='17.spatial_padding';self.assertTrue(ProductRequest(b,['r']).visual)
        b.mechanism='18.api_half';self.assertFalse(ProductRequest(b,['r']).visual)

    def test_external_image_reports_byte_restore_not_argument_only(self):
        from test_product_automatic import ProductCoverageTests
        s=ProductCoverageTests().make(True);s.binding.mechanism='external.conv';s.binding.worker=object()
        e=s.site.events[0];e.update(request_ids=['r'],failed=False)
        with patch('demoncat.runtime.vllm_ascend.worker_extension.external_provider.external_site',return_value=NS(mechanism='external.conv',source_sha256='a'*64,stage='vision')):
            report=s.collect()
        self.assertEqual(report['restore_kind'],'LIVE_REGION_BYTES')
        self.assertFalse(report['external_provider']['qualification_granted'])
