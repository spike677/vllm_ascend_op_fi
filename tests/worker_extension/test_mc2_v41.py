import unittest
import hashlib,importlib.util,sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace as NS
import torch
from fault_injection.sites.mc2_v41 import Join,MC2Observer


class MC2Tests(unittest.TestCase):
    def test_original_source_method_keyword_boundaries(self):
        path=Path(__file__).parents[1]/'fixtures/mc2_v032_fragment.py'
        spec=importlib.util.spec_from_file_location('mc2_original_fixture',path)
        fixture=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fixture;spec.loader.exec_module(fixture)
        join,metadata,hidden,combine_args=self.fixture()
        inputs=join.input;inputs.quant=NS(dispatch_with_quant=False);inputs.routing.mc2_mask=join.kwargs['x_active_mask']
        dispatcher=fixture.TokenDispatcherWithMC2();dispatcher.global_bs=0
        dispatcher.get_dispatch_mc2_kwargs=lambda token_dispatch_input:dict(join.kwargs,expert_token_nums_type=0)
        dispatcher.get_combine_mc_kwargs=lambda hidden_states,combine_metadata:combine_args
        stack=[]
        class CpuNativeCalls:
            def __init__(self,operator):self.calls=[]
            def __enter__(self):stack.append(self);return self
            def __exit__(self,*args):stack.pop()
        def dispatch(**kwargs):stack[-1].calls.append((kwargs,join.output));return join.output
        def combine(**kwargs):
            result=torch.zeros_like(inputs.hidden_states);stack[-1].calls.append((kwargs,result));return result
        fixture.torch_npu.npu_moe_distribute_dispatch_v2=dispatch
        fixture.torch_npu.npu_moe_distribute_combine_v2=combine
        observer=MC2Observer(lambda d,i:join.context)
        with patch('fault_injection.sites.mc2_v41.SOURCE_SHA',hashlib.sha256(path.read_bytes()).hexdigest()),patch(
                'fault_injection.sites.mc2_v41.NativeCalls',CpuNativeCalls):
            observer.install(fixture.TokenDispatcherWithMC2)
            produced=dispatcher.token_dispatch(token_dispatch_input=inputs)
            result=dispatcher.token_combine(hidden_states=hidden,combine_metadata=produced.combine_metadata)
            self.assertEqual(result.shape,inputs.hidden_states.shape);self.assertEqual(len(observer.records),1)
            self.assertFalse(observer.pending);observer.close()

    def fixture(self,expert=1):
        x=torch.ones(2,4);ids=torch.tensor([[expert,2],[0,0]],dtype=torch.int32);weights=torch.ones(2,2)/2
        inputs=NS(hidden_states=x,topk_ids=ids,topk_weights=weights,routing=NS(expert_map=torch.tensor([-1,0,-1,-1]),global_redundant_expert_num=0))
        args=dict(x=x,expert_ids=ids,expert_shard_type=0,shared_expert_rank_num=0,moe_expert_num=4,global_bs=0,
            group_ep='actual_group',ep_world_size=8,ep_rank_id=1,x_active_mask=torch.tensor([True,False]))
        outputs=[torch.ones(2,4),None,torch.tensor([99]),torch.tensor([2]),torch.tensor([1]),torch.tensor([0]),None]
        context=dict(owner='scope',generation=1,invocation=3,rows=[dict(request_id='real',position=12),None])
        join=Join(inputs,args,outputs,context)
        metadata=NS(topk_ids=ids,topk_weights=weights,expert_map=inputs.routing.expert_map,
            assist_info_for_combine=outputs[2],ep_recv_counts=outputs[4],tp_recv_counts=outputs[5],expand_scales=outputs[6])
        join.bind_metadata(metadata)
        combine={k:v for k,v in args.items() if k not in ('x','x_active_mask')}
        hidden=torch.ones(2,4);combine.update(expand_x=hidden,expert_scales=weights.float(),assist_info_for_combine=outputs[2],
            ep_send_counts=outputs[4],expand_scales=outputs[6])
        return join,metadata,hidden,combine

    def test_dynamic_routes_actual_metadata_join_and_no_invented_padding(self):
        for expert in (1,3):
            join,meta,x,args=self.fixture(expert)
            out=join.combine(x,meta,args,torch.ones(2,4))
            self.assertEqual(len(out['routes']),2);self.assertEqual(out['routes'][0]['expert'],expert)
            self.assertEqual(out['routes'][0]['position'],12);self.assertEqual(out['source_ep_rank'],1)
            self.assertFalse(out['qualification_granted']);self.assertFalse(out['destination_expert_row_decode_proven'])

    def test_stale_opaque_assist_routing_and_metadata_rejected(self):
        join,meta,x,args=self.fixture();args['assist_info_for_combine']=args['assist_info_for_combine'].clone()
        with self.assertRaisesRegex(RuntimeError,'OPAQUE_ASSIST'):join.combine(x,meta,args,x)
        join,meta,x,args=self.fixture();args['expert_ids'][0,0]=3
        with self.assertRaisesRegex(RuntimeError,'ROUTING_CHANGED'):join.combine(x,meta,args,x)
        join,meta,x,args=self.fixture();args['expert_scales'].add_(1)
        with self.assertRaisesRegex(RuntimeError,'WEIGHTS_CHANGED'):join.combine(x,meta,args,x)

    def test_uncombined_calls_block_close(self):
        observer=MC2Observer(lambda *a:None);observer.pending[1]=object()
        with self.assertRaisesRegex(RuntimeError,'UNCOMBINED'):observer.close()
