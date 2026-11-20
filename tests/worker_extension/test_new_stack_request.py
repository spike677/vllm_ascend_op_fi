import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch
from contextlib import nullcontext
import torch
from demoncat.runtime.vllm_ascend.worker_extension.new_stack_request import NewStackTaskRequest

class RequestBridgeTests(unittest.TestCase):
    def fixture(self):
        class Model(torch.nn.Module):
            def __init__(self):super().__init__();self.visual=torch.nn.Identity()
            def embed_input_ids(self,input_ids,**kwargs):raise AssertionError('unexpected embedding branch')
            def forward(self,input_ids=None,positions=None,intermediate_tensors=None,inputs_embeds=None):
                return input_ids.float().reshape(-1,1).expand(-1,4).contiguous()
        Model.__name__='Qwen3_5ForConditionalGeneration';Model.__module__='vllm.model_executor.models.qwen3_5'
        class Runner:
            def _preprocess(self,scheduler,total,intermediate=None):return self.input_ids.gpu,None,self.mrope_positions.gpu,None,{},None
            def _model_forward(self,**kwargs):return self.model(**kwargs)
            def execute_model(self,scheduler,intermediate=None):
                ids,embeds,pos,it,kw,ec=self._preprocess(scheduler,3,intermediate)
                h=self._model_forward(input_ids=ids,inputs_embeds=embeds,positions=pos,intermediate_tensors=it)
                self.execute_model_state=NS(scheduler_output=scheduler,hidden_states=h,sample_hidden_states=h[-1:].contiguous(),logits=h[-1:].contiguous())
            def sample_tokens(self):self.execute_model_state=None;return NS(req_ids=['r'])
        Runner.__name__='NPUModelRunner';Runner.__module__='vllm_ascend.worker.model_runner_v1'
        t=torch.tensor([3,4,5],dtype=torch.int32);pos=torch.tensor([[0,1,2]]*3)
        r=Runner();r.model=Model().eval();r.execute_model_state=None;r.device='npu:0'
        r.supports_mm_inputs=False;r.enable_prompt_embeds=False;r.uses_mrope=True;r.use_async_scheduling=False;r.speculative_config=None
        r.input_ids=NS(cpu=t,gpu=t);r.mrope_positions=NS(cpu=pos,gpu=pos)
        r.input_batch=NS(req_ids=['r'],req_id_to_index={'r':0},num_computed_tokens_cpu=[0])
        r.requests={'r':NS(req_id='r',prompt_token_ids=[3,4,5],mm_features=[],prompt_embeds=None,lora_request=None,output_token_ids=[])}
        r.vllm_config=NS(scheduler_config=NS(enable_chunked_prefill=True,max_num_batched_tokens=3),model_config=NS(enforce_eager=True,multimodal_config=NS(get_limit_per_prompt=lambda _:0)))
        s=NS(num_scheduled_tokens={'r':3},total_num_scheduled_tokens=3,scheduled_encoder_inputs={},scheduled_spec_decode_tokens={})
        rows=[]
        p=NS(worker=NS(model_runner=r),model=r.model,binding=NS(prompt=[3,4,5],execution_profile='v026'),phase='off',requests=set(),_context=lambda:None,fail=lambda e:None)
        def begin(row):p.requests.add(row['request_id'])
        p.begin_actual=begin;p.end_actual=lambda row:rows.append(row)
        session=NewStackTaskRequest(p,lambda _:self.fail('OFF must not create a fault window'))
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.v026_text_observer._verify_function',return_value={'cpu_fixture':True}))
        self.enterContext(patch('demoncat.runtime.vllm_ascend.worker_extension.v026_text_observer._parallel_report',return_value={'cpu_fixture':True}))
        session.install();session.expect('r');return r,s,p,session,rows
    def test_actual_execute_sample_completes_and_all_owned_resources_removed(self):
        r,s,p,q,rows=self.fixture();r.execute_model(s)
        self.assertEqual(rows,[]);self.assertTrue(q.readback()['pending'])
        r.sample_tokens();self.assertEqual(len(rows),1);self.assertFalse(q.readback()['pending'])
        self.assertEqual(rows[0]['phase'],'off');self.assertEqual(rows[0]['request_trace']['status'],'OBSERVED')
        self.assertEqual(rows[0]['device_input_bytes_equality'],'COMPLETE_DEVICE_READBACK_VERIFIED')
        q.close();self.assertEqual(q.readback()['owned_hooks'],0);self.assertEqual(q.readback()['owned_wrappers'],0)
        self.assertNotIn('execute_model',vars(r));self.assertFalse(q.readback()['qualification_granted'])
    def test_wrong_model_output_state_cannot_complete(self):
        r,s,p,q,rows=self.fixture();original=q.state_return
        def changed(state):state.hidden_states=state.hidden_states.clone();return original(state)
        q.state_return=changed
        with self.assertRaisesRegex(RuntimeError,'STATE_NOT_MODEL_OUTPUT'):r.execute_model(s)
        self.assertEqual(rows,[]);q.close()
        self.assertEqual(q.readback()['owned_hooks'],0);self.assertEqual(q.readback()['owned_wrappers'],0)
    def test_cannot_complete_before_sample_or_accept_reused_id(self):
        r,s,p,q,rows=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'EFFECT_BEFORE_ACTUAL_SAMPLE'):q.complete_request()
        r.execute_model(s);r.sample_tokens()
        with self.assertRaisesRegex(RuntimeError,'FRESH_ENQUEUED_ID'):q.expect('r')
        q.close()
    def test_changed_device_ids_rejected_even_when_cpu_mirror_looks_correct(self):
        r,s,p,q,rows=self.fixture();r.input_ids.gpu=r.input_ids.cpu.clone();r.input_ids.gpu[0]=99
        with self.assertRaisesRegex(RuntimeError,'DEVICE_INPUT_BYTES_DIFFER'):r.execute_model(s)
        self.assertEqual(rows,[]);q.close()
        self.assertFalse(q.readback()['pending']);self.assertEqual(q.readback()['owned_wrappers'],0)

if __name__=='__main__':unittest.main()
