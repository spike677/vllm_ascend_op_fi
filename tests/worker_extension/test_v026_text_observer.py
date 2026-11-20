import copy
import unittest
from types import SimpleNamespace as NS
import torch
from demoncat.runtime.vllm_ascend.worker_extension.v026_text_observer import TextPrefillTrace, SourceBoundTextObserver

class TextTraceTests(unittest.TestCase):
    def fixture(self):
        tokens=torch.tensor([3,4,5,1,2],dtype=torch.int32)
        positions=torch.tensor([[0,1,2,0,1]]*3)
        expected={'a':[1,2],'b':[3,4,5]}
        runner=NS(model=object(),execute_model_state=None,
            vllm_config=NS(scheduler_config=NS(enable_chunked_prefill=True,max_num_batched_tokens=5)),
            input_batch=NS(req_ids=['b','a'],req_id_to_index={'b':0,'a':1},num_computed_tokens_cpu=[0,0]),
            requests={k:NS(req_id=k,prompt_token_ids=v,mm_features=[],prompt_embeds=None,lora_request=None,output_token_ids=[]) for k,v in expected.items()},
            input_ids=NS(cpu=tokens,gpu=tokens),mrope_positions=NS(cpu=positions,gpu=positions),
            inputs_embeds=NS(gpu=torch.zeros(5,8)))
        scheduler=NS(num_scheduled_tokens={'a':2,'b':3},total_num_scheduled_tokens=5,
                     scheduled_encoder_inputs={},scheduled_spec_decode_tokens={})
        trace=TextPrefillTrace(runner,expected)
        return runner,scheduler,trace
    def prepared(self):
        r,s,t=self.fixture();t.enter(s);t.before_preprocess(s,5,None)
        t.embedding(r.input_ids.gpu,{'multimodal_embeddings':[],'is_multimodal':torch.zeros(5,dtype=torch.bool)})
        result=(None,r.inputs_embeds.gpu,r.mrope_positions.gpu,None,{},None)
        t.after_preprocess(result)
        kwargs=dict(input_ids=None,inputs_embeds=result[1],positions=result[2],intermediate_tensors=None)
        return r,s,t,kwargs
    def executed(self):
        r,s,t,k=self.prepared();t.model_enter(r.model,(),k);t.model_return(r.model,(),k,torch.zeros(5,8))
        r.execute_model_state=NS(scheduler_output=s,logits=torch.zeros(2,20),hidden_states=torch.zeros(5,8),sample_hidden_states=torch.zeros(2,8))
        t.execute_return(None);return r,s,t
    def test_reordered_ids_one_chunk_embedding_and_sampling(self):
        r,s,t=self.executed();t.before_sample();r.execute_model_state=None;t.after_sample(NS(req_ids=['b','a']))
        report=t.snapshot();self.assertEqual(report['status'],'OBSERVED')
        self.assertTrue(report['visual_execution_excluded']);self.assertTrue(report['model_input_association_verified'])
        self.assertFalse(report['execution_authorized']);self.assertFalse(report['device_input_values_compared'])
    def test_short_chunk_refused_even_sufficient_budget(self):
        r,s,t=self.fixture();s.num_scheduled_tokens['a']=1;t.enter(s)
        with self.assertRaisesRegex(RuntimeError,'SINGLE_CHUNK'):t.before_preprocess(s,5,None)
    def test_second_execute_refused(self):
        r,s,t=self.fixture();t.enter(s)
        with self.assertRaisesRegex(RuntimeError,'EXECUTE_ORDER'):t.enter(s)
    def test_encoder_inputs_refused_before_model(self):
        r,s,t=self.fixture();s.scheduled_encoder_inputs={'a':[0]};t.enter(s)
        with self.assertRaisesRegex(RuntimeError,'SINGLE_CHUNK'):t.before_preprocess(s,5,None)
    def test_visual_is_blocked_before_computation(self):
        r,s,t=self.fixture()
        with self.assertRaisesRegex(RuntimeError,'VISUAL_EXECUTION'):t.visual()
        self.assertEqual(t.visual_calls,1);self.assertFalse(t.snapshot()['visual_execution_excluded'])
    def test_embedding_not_runner_ids_refused(self):
        r,s,t=self.fixture();t.enter(s);t.before_preprocess(s,5,None)
        with self.assertRaisesRegex(RuntimeError,'BUFFER_PREFIX'):t.embedding(r.input_ids.gpu.clone(),{'multimodal_embeddings':[]})
    def test_mm_embeddings_refused(self):
        r,s,t=self.fixture();t.enter(s);t.before_preprocess(s,5,None)
        with self.assertRaisesRegex(RuntimeError,'MULTIMODAL_EMBEDDINGS'):t.embedding(r.input_ids.gpu,{'multimodal_embeddings':[object()]})
    def test_cpu_token_mirror_mismatch_refused(self):
        r,s,t=self.fixture();t.enter(s);t.before_preprocess(s,5,None);r.input_ids.cpu[0]=7
        with self.assertRaisesRegex(RuntimeError,'CPU_MIRROR'):t.embedding(r.input_ids.gpu,{'multimodal_embeddings':[]})
    def test_position_mirror_not_plain_text_refused(self):
        r,s,t=self.fixture();t.enter(s);t.before_preprocess(s,5,None)
        t.embedding(r.input_ids.gpu,{'multimodal_embeddings':[]});r.mrope_positions.cpu[1,0]=7
        with self.assertRaisesRegex(RuntimeError,'MROPE_CPU_MIRROR'):t.after_preprocess((None,r.inputs_embeds.gpu,r.mrope_positions.gpu,None,{},None))
    def test_model_input_replacement_refused(self):
        r,s,t,k=self.prepared();k['inputs_embeds']=k['inputs_embeds'].clone()
        with self.assertRaisesRegex(RuntimeError,'MODEL_INPUT'):t.model_enter(r.model,(),k)
    def test_state_wrong_scheduler_refused(self):
        r,s,t,k=self.prepared();t.model_enter(r.model,(),k);t.model_return(r.model,(),k,torch.zeros(5,8))
        r.execute_model_state=NS(scheduler_output=copy.deepcopy(s),logits=torch.zeros(2,20),hidden_states=torch.zeros(5,8),sample_hidden_states=torch.zeros(2,8))
        with self.assertRaisesRegex(RuntimeError,'EXECUTE_STATE'):t.execute_return(None)
    def test_sample_requires_exact_pending_state(self):
        r,s,t=self.executed();r.execute_model_state=copy.copy(r.execute_model_state)
        with self.assertRaisesRegex(RuntimeError,'STATE_REPLACED'):t.before_sample()
    def test_sample_requires_consumption_and_actual_request_map(self):
        r,s,t=self.executed();t.before_sample()
        with self.assertRaisesRegex(RuntimeError,'NOT_CONSUMED'):t.after_sample(NS(req_ids=['b','a']))
        r.execute_model_state=None
        with self.assertRaisesRegex(RuntimeError,'REQUEST_MAPPING'):t.after_sample(NS(req_ids=['a','b']))
    def test_unknown_runner_cannot_install(self):
        r,s,t=self.fixture();observer=SourceBoundTextObserver(r,t.expected)
        with self.assertRaisesRegex(RuntimeError,'EXACT_REVIEWED'):observer.install()
        self.assertEqual(observer.snapshot()['owned_wrappers'],0)

if __name__=='__main__':unittest.main()
