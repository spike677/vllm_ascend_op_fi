import unittest
from types import SimpleNamespace as NS
import torch
from demoncat.runtime.vllm_ascend.worker_extension.v026_generation_observer import FirstDecodeTrace

class FirstDecodeTests(unittest.TestCase):
    def prepared(self):
        expected={str(i):[10+i,20+i] for i in range(8)};ids=list(reversed(expected))
        sampled={rid:100+int(rid) for rid in ids};tokens=torch.tensor([sampled[rid] for rid in ids]);pos=torch.tensor([[2]*8]*3)
        runner=NS(model=object(),supports_mm_inputs=False,execute_model_state=None,
            input_batch=NS(req_ids=ids,req_id_to_index={k:i for i,k in enumerate(ids)},num_computed_tokens_cpu=[2]*8),
            requests={rid:NS(req_id=rid,prompt_token_ids=p,output_token_ids=[sampled[rid]],mm_features=[],prompt_embeds=None,lora_request=None) for rid,p in expected.items()},
            input_ids=NS(cpu=tokens,gpu=tokens),mrope_positions=NS(cpu=pos,gpu=pos))
        scheduler=NS(num_scheduled_tokens=dict.fromkeys(ids,1),total_num_scheduled_tokens=8,
            scheduled_new_reqs=[],scheduled_cached_reqs=NS(req_ids=ids),scheduled_encoder_inputs={},scheduled_spec_decode_tokens={})
        t=FirstDecodeTrace(runner,expected);t.rows=[{'phase':'prefill'}];t.sampled=sampled;t.enter(scheduler)
        return runner,scheduler,t
    def test_reordered_real_decode_and_device_bytes(self):
        r,s,t=self.prepared();t.before_preprocess(s,8,None)
        result=(r.input_ids.gpu,None,r.mrope_positions.gpu,None,{},None);t.after_preprocess(result)
        kwargs=dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),result[:4]))
        t.model_enter(r.model,(),kwargs)
        self.assertEqual(t.input_values['request_ids'],list(reversed([str(i) for i in range(8)])))
    def test_reject_wrong_prefix_computed_and_schedule(self):
        for mutation in (lambda r,s:r.requests['0'].output_token_ids.append(1),
                         lambda r,s:r.input_batch.num_computed_tokens_cpu.__setitem__(0,1),
                         lambda r,s:s.num_scheduled_tokens.update({'0':2})):
            r,s,t=self.prepared();mutation(r,s)
            with self.assertRaises(RuntimeError):t.before_preprocess(s,8,None)
    def test_reject_cpu_device_disagreement(self):
        r,s,t=self.prepared();t.before_preprocess(s,8,None)
        r.input_ids.gpu=r.input_ids.cpu.clone();r.input_ids.gpu[0]=999
        result=(r.input_ids.gpu,None,r.mrope_positions.gpu,None,{},None);t.after_preprocess(result)
        kwargs=dict(zip(('input_ids','inputs_embeds','positions','intermediate_tensors'),result[:4]))
        with self.assertRaisesRegex(RuntimeError,'DEVICE_INPUT'):t.model_enter(r.model,(),kwargs)
if __name__=='__main__':unittest.main()
