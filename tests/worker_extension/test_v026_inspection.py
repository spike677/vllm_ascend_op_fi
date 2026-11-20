import copy,json,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from demoncat.runtime.vllm_ascend.worker_extension.v026_inspection import (
    parallel_assessment, inspect_single_chunk, describe_qwen35_structure)

class V026InspectionTests(unittest.TestCase):
    def fixture(self):
        parallel=NS(prefill_context_parallel_size=1,decode_context_parallel_size=1,
                    tensor_parallel_size=2,pipeline_parallel_size=1,data_parallel_size=1)
        runner=NS(vllm_config=NS(parallel_config=parallel,
                    scheduler_config=NS(enable_chunked_prefill=True,max_num_batched_tokens=8)),
                  dcp_size=1,use_dcp=False,
                  input_batch=NS(req_ids=['b','a'],req_id_to_index={'a':1,'b':0},num_computed_tokens_cpu=[0,0]),
                  requests={'a':NS(prompt_token_ids=[1,2],mm_features=[]),
                            'b':NS(prompt_token_ids=[3,4,5],mm_features=[])})
        scheduler=NS(num_scheduled_tokens={'a':2,'b':3},total_num_scheduled_tokens=5,
                     scheduled_encoder_inputs={},scheduled_spec_decode_tokens={})
        return runner,scheduler,{'a':[1,2],'b':[3,4,5]}

    def test_current_group_required_despite_runner_dcp_fallback(self):
        r,_,_=self.fixture()
        report=parallel_assessment(r,pcp_group=None,dcp_group=None,tp_group=NS(world_size=2),expected_tp=2)
        self.assertEqual({x['field'] for x in report['unmet']},{'pcp_group.world_size','dcp_group.world_size'})
        self.assertFalse(hasattr(r,'pcp_size'))
        self.assertFalse(report['execution_authorized'])

    def test_config_and_live_group_must_agree(self):
        r,_,_=self.fixture();r.vllm_config.parallel_config.prefill_context_parallel_size=2
        p=parallel_assessment(r,pcp_group=NS(world_size=1),dcp_group=NS(world_size=1),tp_group=NS(world_size=2),expected_tp=2)
        self.assertEqual(p['status'],'refused')
        del r.vllm_config.parallel_config.prefill_context_parallel_size
        self.assertEqual(parallel_assessment(r,pcp_group=NS(world_size=1),dcp_group=NS(world_size=1),tp_group=NS(world_size=2),expected_tp=2)['status'],'pending')

    def test_actual_reordered_rows_single_chunk_without_authority(self):
        r,s,p=self.fixture();out=inspect_single_chunk(r,s,p)
        self.assertEqual(out['status'],'matched');self.assertFalse(out['execution_authorized'])
        self.assertFalse(out['visual_execution_excluded'])

    def test_config_budget_cannot_prove_no_chunk(self):
        for mutation in ('scheduled','computed','mapping','mm','encoder','spec','budget'):
            r,s,p=self.fixture()
            if mutation=='scheduled':s.num_scheduled_tokens['a']=1;s.total_num_scheduled_tokens=4
            elif mutation=='computed':r.input_batch.num_computed_tokens_cpu[1]=1
            elif mutation=='mapping':r.input_batch.req_id_to_index['a']=0
            elif mutation=='mm':r.requests['a'].mm_features=[object()]
            elif mutation=='encoder':s.scheduled_encoder_inputs={'a':[0]}
            elif mutation=='spec':s.scheduled_spec_decode_tokens={'a':[7]}
            elif mutation=='budget':r.vllm_config.scheduler_config.max_num_batched_tokens=4
            with self.subTest(mutation=mutation):self.assertNotEqual(inspect_single_chunk(r,s,p)['status'],'matched')

    def test_missing_actual_fields_not_defaulted(self):
        r,s,p=self.fixture();del r.requests['a'].mm_features
        out=inspect_single_chunk(r,s,p)
        self.assertEqual(out['status'],'pending')
        self.assertEqual(out['unmet'][0]['actual'],{'missing':True})

    def test_real_config_and_small_structural_fixture(self):
        config=json.loads((Path(__file__).parent/'fixtures/qwen38_public_config.json').read_bytes())
        full=describe_qwen35_structure(config,{})
        self.assertEqual(len(full['full_attention_layers']),16)
        self.assertEqual(len(full['linear_attention_layers']),48)
        self.assertEqual(full['rotary_dim'],64)
        small=copy.deepcopy(config);small['text_config']['num_hidden_layers']=4
        small['text_config']['layer_types']=small['text_config']['layer_types'][:4]
        result=describe_qwen35_structure(small,{'model.layers.0.linear_attn.out_proj.weight':'W8A8_DYNAMIC'})
        self.assertEqual(result['full_attention_layers'],[3])
        self.assertEqual(result['linear_attention_layers'],[0,1,2])
        self.assertEqual(result['quantization_counts'],{'W8A8_DYNAMIC':1,'FLOAT':0})
        self.assertFalse(result['actual_calls_observed'])
        self.assertFalse(result['execution_authorized'])
        small['text_config']['num_hidden_layers']=5
        with self.assertRaises(ValueError):describe_qwen35_structure(small,{})

if __name__=='__main__':unittest.main()
