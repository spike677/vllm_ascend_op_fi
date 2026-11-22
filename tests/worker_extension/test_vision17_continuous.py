import unittest
import torch
from types import SimpleNamespace as NS
from unittest.mock import patch
from fault_injection.sites.vision17 import Vision17Site


class Visual(torch.nn.Module):
    def __init__(self):
        super().__init__();self.patch=torch.nn.Conv3d(3,4,(2,16,16),stride=(2,16,16)).eval()
        with torch.no_grad():self.patch.weight.fill_(0.001);self.patch.bias.zero_()
    def forward(self,x):
        return self.patch(x).reshape(x.shape[0],4)+1


class VisionContinuousTests(unittest.TestCase):
    def test_four_token_request_observer_consumes_live_encoder_and_recovers(self):
        from demoncat.runtime.vllm_ascend.worker_extension.continuous_vision17 import ContinuousVisionRequest
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.visual=Visual().eval()
            def embed_input_ids(self,ids,*,multimodal_embeddings):return multimodal_embeddings[0]
            def forward(self,input_ids,inputs_embeds,positions,intermediate_tensors):return inputs_embeds+2
        model=Model().eval();runner=NS(model=model,encoder_cache={},execute_model_state=None)
        x=torch.arange(1536,dtype=torch.float32).reshape(1,3,2,16,16)/1536
        # Use class methods as the live adapter refuses pre-existing instance wrappers.
        class Runner:
            def _batch_mm_inputs_from_scheduler(self,s):
                return ([self.feature.identifier],[('image',self.feature.data)],[(self.rid,self.feature.mm_position)]) if s.scheduled_encoder_inputs else ([],[],[])
            def _execute_mm_encoder(self,s):
                if not s.scheduled_encoder_inputs:return []
                value=self.model.visual(x);self.encoder_cache[self.feature.identifier]=value;return [value]
            def _preprocess(self,s,total,intermediate_tensors=None):
                self._execute_mm_encoder(s)
                embeds=self.model.embed_input_ids(torch.tensor([1,2]),multimodal_embeddings=[self.encoder_cache[self.feature.identifier]])
                return torch.tensor([1,2]),embeds,torch.tensor([0,1]),None,{},None
            def execute_model(self,s,intermediate_tensors=None):
                args=self._preprocess(s,s.total_num_scheduled_tokens,intermediate_tensors)
                value=self.model(**dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),args[:4])))
                self.execute_model_state=NS(scheduler_output=s,hidden_states=value,logits=value)
            def sample_tokens(self):self.execute_model_state=None;return NS(req_ids=[self.rid])
        runner=Runner();runner.model=model;runner.encoder_cache={};runner.execute_model_state=None
        binding=NS(worker=NS(model_runner=runner),model=model,prompts=((1,2),),visual_references={},check=lambda:None)
        class Event:
            def record(self,s):pass
            def synchronize(self):pass
        rows=[]
        with patch.object(torch,'npu',NS(current_stream=lambda:0,Event=Event),create=True):
            for phase in ('normal','fault','off','health'):
                rid=phase;runner.rid=rid
                runner.feature=NS(identifier=rid,modality='image',data=object(),mm_position=NS(offset=0,length=1))
                runner.requests={rid:NS(req_id=rid,prompt_token_ids=[1,2],prompt_embeds=None,lora_request=None,mm_features=[runner.feature])}
                runner.input_batch=NS(req_ids=[rid],req_id_to_index={rid:0},num_computed_tokens_cpu=[0])
                request=ContinuousVisionRequest(binding,[rid],phase);request.install()
                for step in range(4):
                    runner.step=step;runner.input_batch.num_computed_tokens_cpu=[0 if step==0 else 1+step]
                    total=2 if step==0 else 1
                    schedule=NS(total_num_scheduled_tokens=total,num_scheduled_tokens={rid:total},
                        scheduled_spec_decode_tokens={},scheduled_encoder_inputs={rid:[0]} if step==0 else {})
                    runner.execute_model(schedule);runner.sample_tokens()
                rows.append(request.collect());self.assertFalse(request.hooks);self.assertFalse(request.patches)
        self.assertEqual(rows[1]['coverage']['eligible'],1)
        self.assertEqual(rows[1]['coverage']['consumed'],1)
        self.assertNotEqual(rows[0]['output'],rows[1]['output'])
        self.assertEqual(rows[0]['output'],rows[2]['output']);self.assertEqual(rows[0]['output'],rows[3]['output'])

    def test_real_padding_all_calls_and_off_exact_with_value_consumer(self):
        model=Visual().eval();x=torch.arange(1536,dtype=torch.float32).reshape(1,3,2,16,16)/1536
        references={};outputs=[];normal_weight=model.patch.weight.detach().clone();completions=[]
        for fault in (False,True,False,False):
            site=Vision17Site(model,fault=fault,references=references,request_ids=lambda:['actual'],
                stream=lambda:'same',complete=lambda s:completions.append(s));site.install()
            outputs.append([model(x) for _ in range(2)])
            self.assertEqual(len(site.events),2)
            self.assertTrue(all(e['consumed'] and e['restored'] and not e['failed'] for e in site.events))
            self.assertTrue(all(e['actual_padding']==([0,7,7] if fault else [0,0,0]) for e in site.events))
            self.assertTrue(all(e['changed']==int(fault) and e['native_calls']==1 for e in site.events))
            self.assertTrue(all(e['before_sha256']==e['restored_sha256'] for e in site.events))
            site.close();self.assertFalse(site.hooks)
        self.assertFalse(torch.equal(outputs[0][0],outputs[1][0]))
        self.assertTrue(all(torch.equal(outputs[0][0],row[0]) for row in outputs[2:]))
        self.assertTrue(torch.equal(model.patch.weight,normal_weight));self.assertEqual(len(completions),8)

    def test_missing_normal_reference_refuses_before_conv(self):
        model=Visual().eval();site=Vision17Site(model,fault=True,references={},request_ids=lambda:['actual'],
            stream=lambda:0,complete=lambda s:None);site.install()
        try:
            with self.assertRaisesRegex(RuntimeError,'NORMAL_REFERENCE_REQUIRED'):
                model(torch.ones(1,3,2,16,16))
        finally:
            if site.active:site.mode.__exit__(None,None,None);site.active=False
            site.parents.clear();site.close()

    def test_cross_stream_consumer_is_refused(self):
        model=Visual().eval();counter=iter((0,1))
        site=Vision17Site(model,fault=False,references={},request_ids=lambda:['actual'],
            stream=lambda:next(counter),complete=lambda s:None);site.install()
        try:
            with self.assertRaisesRegex(RuntimeError,'CONSUMER_STREAM_CHANGED'):
                model(torch.ones(1,3,2,16,16))
        finally:
            if site.active:site.mode.__exit__(None,None,None);site.active=False
            site.parents.clear();site.mode.pending.clear();site.close()
