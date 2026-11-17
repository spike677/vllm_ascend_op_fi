"""Opaque CPU fixture contracts; no native or pretrained qualification."""
import sys
import unittest
from unittest.mock import patch
import torch

from fault_injection.rope_census import NaturalRoPETrace
from fault_injection.tensor_observer import NaturalTensorObserver
from experiments.c2c3_closure_20261117.prepare_tensor_natural_cases import cases

NATIVE=torch.library.Library('rope_census_cpu_native','DEF')
NATIVE.define('rotary(Tensor positions, Tensor(a!) query, Tensor(b!) key, int head_size, Tensor cache, bool neox) -> ()')
OUTER=torch.library.Library('rope_census_cpu_outer','DEF')
OUTER.define('opaque(Tensor positions, Tensor(a!) query, Tensor(b!) key, Tensor cache, int head_size, int rotary_dim, bool neox) -> (Tensor(a!), Tensor(b!))')
CALLS=[]
CAPTURE=None


def native_rotary(positions,query,key,head_size,cache,neox):
    CALLS.append((id(positions),id(cache),None if CAPTURE is None else CAPTURE.active))
    cos,sin=cache.index_select(0,positions).chunk(2,dim=-1)
    cos=cos.repeat(1,2);sin=sin.repeat(1,2)
    for value in (query,key):
        half=value.shape[-1]//2
        value.copy_(value*cos+torch.cat((-value[:,half:],value[:,:half]),dim=-1)*sin)


NATIVE.impl('rotary',native_rotary,'CPU')


def actual_reader(positions,query,key,cos_sin_cache,head_size,rotary_dim,is_neox_style,offsets=None):
    query=query.contiguous().view(query.shape[0],-1)
    key=key.contiguous().view(key.shape[0],-1)
    torch.ops.rope_census_cpu_native.rotary(positions,query,key,head_size,cos_sin_cache,is_neox_style)
    return query,key


OUTER.impl('opaque',actual_reader,'CPU')


class Rope(torch.nn.Module):
    def __init__(self):
        super().__init__();frequency=torch.arange(16.)[:,None]*torch.tensor([.1,.2])[None,:]
        self.register_buffer('cos_sin_cache',torch.cat((frequency.cos(),frequency.sin()),-1))
        self.replace_cache=False;self.replace_positions=False
    def forward(self,positions,q,k):
        return torch.ops.rope_census_cpu_outer.opaque(positions+1 if self.replace_positions else positions,
            q,k,self.cos_sin_cache.clone() if self.replace_cache else self.cos_sin_cache,4,4,True)


class Consumer(torch.nn.Module):
    def forward(self,q,k):return q+k


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.rope=Rope();self.consumer=Consumer()
    def forward(self,x,positions):
        q,k=self.rope(positions,x+1,x*2)
        return self.consumer(q,k)


class RoPECensusTests(unittest.TestCase):
    def trace(self,model,**kwargs):
        return NaturalRoPETrace(model,actual_reader,{'rope':model.rope},{},simulation=True,**kwargs)
    def test_real_opaque_reader_inputs_native_aliases_and_consumer_are_observed(self):
        model=Model().eval();x=torch.arange(16.).reshape(4,4);positions=torch.arange(4)
        baseline=model(x,positions);before=len(CALLS)
        trace=self.trace(model)
        with trace,NaturalTensorObserver(model):
            for request in ('actual-one','actual-two'):
                trace.set_request(request,positions,list(range(4)),{'request_id':request})
                self.assertTrue(torch.equal(model(x,positions),baseline));trace.end_request()
        report=trace.report();self.assertTrue(report['complete'],report['errors'])
        self.assertEqual(len(CALLS)-before,2)
        self.assertEqual(report['observed_native_rotary_calls'],2)
        self.assertEqual(report['readers_per_request'],{'actual-one':1,'actual-two':1})
        for row in report['readers']:
            self.assertTrue(row['candidate_rows_different'])
            self.assertTrue(row['complete_cache_unchanged'])
            self.assertEqual(row['output_aliases_actual_qk'],[True,True])
            self.assertEqual(row['consumers'][0]['path'],'consumer')
        native=[row for row in report['native_operators'] if 'rotary' in row['schema']]
        self.assertTrue(all(all(row['reader_argument_join'].values()) for row in native))
        self.assertIsNone(sys.getprofile());self.assertFalse(report['qualification_granted'])
        self.assertFalse(report['cache_creation_observed'])
        self.assertTrue(all(not m._forward_pre_hooks and not m._forward_hooks for m in model.modules()))
    def test_changed_actual_positions_or_cache_owner_cannot_borrow_scheduler_or_buffer(self):
        for flag in ('replace_positions','replace_cache'):
            model=Model().eval();setattr(model.rope,flag,True);positions=torch.arange(4)
            with self.subTest(flag=flag),self.trace(model) as trace:
                trace.set_request('actual',positions,list(range(4)),{})
                model(torch.ones(4,4),positions);trace.end_request()
            self.assertFalse(trace.report()['complete'])
            self.assertTrue(any('OWNER_BINDING_MISMATCH' in error for error in trace.errors))
    def test_diagnostic_reads_pause_collection_but_native_reader_stays_observed(self):
        global CAPTURE
        from test_tensor_statistics_window import GuardWindow
        model=Model().eval();positions=torch.arange(4);CAPTURE=GuardWindow();reads=[]
        original=NaturalRoPETrace._cpu
        def checked(trace,value):
            self.assertFalse(CAPTURE.active);reads.append(True);return original(trace,value)
        try:
            with self.trace(model,diagnostics_context=CAPTURE.diagnostics) as trace:
                trace.set_request('real',positions,list(range(4)),{})
                with patch.object(NaturalRoPETrace,'_cpu',checked),CAPTURE(model,'statistics_baseline'):
                    model(torch.ones(4,4),positions)
                trace.end_request()
            self.assertTrue(trace.report()['complete'],trace.errors)
            self.assertTrue(reads);self.assertTrue(CALLS[-1][2])
        finally:CAPTURE=None
    def test_foreign_profile_is_preserved(self):
        foreign=lambda *args:None;sys.setprofile(foreign)
        try:
            with self.assertRaisesRegex(RuntimeError,'FOREIGN_ROPE_PROFILE'):self.trace(Model()).__enter__()
            self.assertIs(sys.getprofile(),foreign)
        finally:sys.setprofile(None)
    def test_bounded_zero_injection_worker_manifest(self):
        row,=cases('/home/17-fault-c2c3-20261117/source-new','rope22-census')
        self.assertIn('--rope-trace',row['command']);self.assertEqual(row['review']['expected_hits'],0)
        self.assertIn('Qwen3-0.6B',row['review']['model'])
    def test_unobserved_structural_cache_owner_prevents_complete_census(self):
        model=Model().eval();model.unused=Rope();positions=torch.arange(4)
        trace=NaturalRoPETrace(model,actual_reader,{'rope':model.rope,'unused':model.unused},{},simulation=True)
        with trace:
            trace.set_request('actual',positions,list(range(4)),{})
            model(torch.ones(4,4),positions);trace.end_request()
        self.assertFalse(trace.report()['structural_coverage_complete'])
        self.assertFalse(trace.report()['complete'])


if __name__=='__main__':unittest.main()
