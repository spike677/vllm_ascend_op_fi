"""CPU-only shared-module / opaque Triton-shaped callsite contracts."""
import inspect
import types
import unittest
import torch

from fault_injection.rope_census import NaturalRoPETrace
from fault_injection.tensor_observer import NaturalTensorObserver
from fault_injection.core import UnsupportedCall
from fault_injection.rope_sites import ModuleSites
from test_rope_census import Rope, Consumer, native_rotary


def kernel_signature(q_ptr,q_row_stride,k_ptr,k_row_stride,cos_ptr,cos_row_stride,
        sin_ptr,sin_row_stride,cos_sin_ptr,cos_sin_row_stride,pos_ptr,num_tokens,
        n_qh,n_kh,hd,rope_dim,pad_n_qh,pad_n_kh,pad_rope_dim,BLOCK_SIZE,IS_NEOX_STYLE,USE_COS_SIN):
    raise AssertionError('signature only; not a native kernel')


class Compiled:
    name='CPU_SIMULATION_NOT_NATIVE';hash='CPU_SIMULATION_NOT_NATIVE'
    def run(self,g0,g1,g2,stream,*values):
        q,_,k,_,_,_,_,_,cache,_,positions,_=values
        native_rotary(positions,q.view(q.shape[0],-1),k.view(k.shape[0],-1),4,cache,True)


class Jit:
    def __init__(self):
        self.fn=kernel_signature;self.pre_run_hooks=[];self.compiled=Compiled()
        self.params=[types.SimpleNamespace(name=name) for name in inspect.signature(self.fn).parameters]
        self.non_constexpr_indices=tuple(range(12));self.skip_launch=False;self.replace_position=False
    def run(self,*args,grid,warmup,**kwargs):
        bound_args=dict(inspect.signature(self.fn).bind(*args,**kwargs).arguments)
        if self.replace_position:bound_args['pos_ptr']=bound_args['pos_ptr'].clone()
        non_constexpr_vals=tuple(bound_args[self.params[i].name] for i in self.non_constexpr_indices)
        kernel=self.compiled;grid_0=grid[0];grid_1=grid_2=1;stream=0
        if not self.skip_launch:
            kernel.run(grid_0,grid_1,grid_2,stream,*non_constexpr_vals)
        return kernel


JIT=Jit()


def triton_reader(positions,query,key,cos_sin_cache,head_size,rotary_dim,is_neox_style,offsets=None):
    q=query.view(query.shape[0],-1,head_size);k=key.view(key.shape[0],-1,head_size)
    JIT.run(q,q.stride(0),k,k.stride(0),None,None,None,None,cos_sin_cache,
        cos_sin_cache.stride(0),positions,len(positions),q.shape[1],k.shape[1],head_size,
        rotary_dim,q.shape[1],k.shape[1],rotary_dim,BLOCK_SIZE=q.shape[1],
        IS_NEOX_STYLE=is_neox_style,USE_COS_SIN=True,grid=(len(positions),),warmup=False)
    return q.view(query.shape),k.view(key.shape)


OUTER=torch.library.Library('rope_shared_triton_fixture','DEF')
OUTER.define('opaque(Tensor positions, Tensor(a!) query, Tensor(b!) key, Tensor cos_sin_cache, int head_size, int rotary_dim, bool is_neox_style) -> (Tensor(a!), Tensor(b!))')
OUTER.impl('opaque',triton_reader,'CPU')


class SharedRope(Rope):
    def forward(self,positions,q,k):
        return torch.ops.rope_shared_triton_fixture.opaque(positions,q,k,self.cos_sin_cache,4,4,True)


class Layer(torch.nn.Module):
    def __init__(self,shared):super().__init__();self.rope=shared;self.consumer=Consumer()
    def forward(self,x,positions):
        q,k=self.rope(positions,x+1,x*2)
        return self.consumer(q,k)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();rope=SharedRope()
        self.layers=torch.nn.ModuleList([Layer(rope) for _ in range(3)])
        self.skip_last=False
    def forward(self,x,positions):
        for layer in list(self.layers)[:2 if self.skip_last else 3]:x=layer(x,positions)
        return x


class SharedRoPETests(unittest.TestCase):
    def setUp(self):JIT.skip_launch=False;JIT.replace_position=False
    def trace(self,model):
        return NaturalRoPETrace(model,triton_reader,{'layers.0.rope':model.layers[0].rope},{},
            simulation=True,outer_operator='rope_shared_triton_fixture::opaque',triton_kernel=JIT)
    def run_trace(self,model):
        trace=self.trace(model);positions=torch.arange(4);x=torch.arange(16.).reshape(4,4)
        baseline=model(x,positions)
        with trace,NaturalTensorObserver(model):
            for request in ('a','b'):
                trace.set_request(request,positions,list(range(4)),{'actual_request':request})
                self.assertTrue(torch.equal(model(x,positions),baseline));trace.end_request()
        return trace.report()
    def test_shared_object_has_three_real_parent_sites_and_six_launches(self):
        report=self.run_trace(Model().eval())
        self.assertTrue(report['complete'],report['errors'])
        self.assertEqual(report['structural_cache_owners'],['layers.0.rope','layers.1.rope','layers.2.rope'])
        self.assertEqual(len(report['triton_launches']),6)
        self.assertEqual(report['observed_native_rotary_calls'],0)
        self.assertFalse(report['nested_torch_rotary_joins_complete'])
        self.assertTrue(report['triton_reader_joins_complete'])
        self.assertTrue(report['outer_reader_joins_complete'])
        self.assertFalse(report['raw_device_launch_payload_observed'])
        self.assertEqual(len({r['module_object_id'] for r in report['readers']}),1)
        self.assertEqual({r['active_parent_path'] for r in report['readers']},{'layers.0','layers.1','layers.2'})
        self.assertTrue(all(r['simulation'] and r['kernel_source_sha256'] is None for r in report['triton_launches']))
    def test_missing_shared_site_does_not_pass_from_total_reader_count(self):
        model=Model().eval();model.skip_last=True;report=self.run_trace(model)
        self.assertFalse(report['structural_coverage_complete']);self.assertFalse(report['complete'])
    def test_jit_return_without_launcher_call_cannot_complete_join(self):
        JIT.skip_launch=True;report=self.run_trace(Model().eval())
        self.assertFalse(report['triton_reader_joins_complete']);self.assertFalse(report['complete'])
    def test_final_position_clone_not_inferred_equal_by_values(self):
        JIT.replace_position=True;report=self.run_trace(Model().eval())
        self.assertFalse(report['triton_reader_joins_complete']);self.assertFalse(report['complete'])
        self.assertFalse(report['triton_launches'][0]['launches'][0]['actual_argument_join']['positions_exact_object'])


class BypassParent(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embed=torch.nn.Identity();self.body=Model()
    def __call__(self,*args,**kwargs):
        # Matches the observed vLLM decorated-model call contract: this
        # ancestor does execute, but its nn.Module hooks are bypassed.
        return self.forward(*args,**kwargs)
    def forward(self,x,positions):return self.body(self.embed(x),positions)


class BypassRoot(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=BypassParent();self.head_alias=self.model.embed
    def forward(self,x,positions):return self.model(x,positions)


class ActualAncestorTests(unittest.TestCase):
    def setUp(self):JIT.skip_launch=False;JIT.replace_position=False
    def test_hook_bypassed_forward_disambiguates_tied_embedding_by_live_parent(self):
        root=BypassRoot().eval();positions=torch.arange(4);x=torch.ones(4,4)
        trace=NaturalRoPETrace(root,triton_reader,
            {'model.body.layers.0.rope':root.model.body.layers[0].rope},{},simulation=True,
            outer_operator='rope_shared_triton_fixture::opaque',triton_kernel=JIT)
        with trace,NaturalTensorObserver(root):
            trace.set_request('actual-wrapper',positions,list(range(4)),{})
            root(x,positions);trace.end_request()
        self.assertTrue(trace.report()['complete'],trace.errors)
        self.assertEqual([r['model_path'] for r in trace.rows],
            ['model.body.layers.0.rope','model.body.layers.1.rope','model.body.layers.2.rope'])
    def test_same_object_alias_without_live_bound_ancestor_stays_ambiguous(self):
        root=BypassRoot();sites=ModuleSites(root)
        with self.assertRaisesRegex(UnsupportedCall,'AMBIGUOUS_OR_FOREIGN'):
            sites.resolve(root.model.embed,[('<root>',root)])
    def test_fake_self_local_does_not_supply_a_bound_forward_frame(self):
        root=BypassRoot();sites=ModuleSites(root)
        def pretend(self):return sites.resolve(self.embed,[('<root>',root)])
        with self.assertRaisesRegex(UnsupportedCall,'AMBIGUOUS_OR_FOREIGN'):
            pretend(root.model)
    def test_intermediate_graph_and_forward_changes_are_refused(self):
        root=BypassRoot();sites=ModuleSites(root)
        root.model.body=torch.nn.Identity()
        with self.assertRaisesRegex(UnsupportedCall,'LIVE_MODULE_CHILD_REGISTRATION_CHANGED'):
            sites.verify_ancestry('model.embed')
        root=BypassRoot();sites=ModuleSites(root)
        root.model.forward=types.MethodType(lambda self,*args:args[0],root.model)
        with self.assertRaisesRegex(UnsupportedCall,'LIVE_ANCESTOR_FORWARD_CHANGED'):
            sites.verify_ancestry('model.embed')


if __name__=='__main__':unittest.main()
