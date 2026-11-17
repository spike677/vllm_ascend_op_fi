import unittest
from types import SimpleNamespace
import torch

from fault_injection.route8 import RestoreOffsetSession
from fault_injection.moe_restore import MoEPhaseObserver
from test_route8_index22 import MoeWorkload, LIB


class RecordingSession(RestoreOffsetSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs); self.consumed=[]
    def dispatch(self,func,args,kwargs):
        if getattr(getattr(func,'_schema',None),'name','') != 'npu::npu_moe_token_unpermute':
            return super().dispatch(func,args,kwargs)
        rows = self.consumed
        class Original:
            _schema = func._schema
            def __str__(self): return str(func)
            def __call__(self,*actual_args,**actual_kwargs):
                values=dict(zip(('permuted_tokens','sorted_indices','probs'),actual_args))
                values.update(actual_kwargs); rows.append((actual_args,actual_kwargs,values))
                return func(*actual_args,**actual_kwargs)
        return super().dispatch(Original(),args,kwargs)


class MoEPassthroughIdentityTests(unittest.TestCase):
    def inputs(self):
        return (torch.arange(24.).reshape(6,4),torch.arange(6,dtype=torch.int32),
                torch.ones(3,2)/2)
    def assert_original(self,row,inputs):
        for key,tensor in zip(('permuted_tokens','sorted_indices','probs'),inputs):
            self.assertIs(row[2][key],tensor)
            self.assertEqual(row[2][key].untyped_storage()._cdata,tensor.untyped_storage()._cdata)
    def test_baseline_and_unselected_fault_use_original_argument_objects_once(self):
        model=MoeWorkload().eval(); inputs=self.inputs()
        for arm in ('baseline','fault'):
            with self.subTest(arm=arm), RecordingSession(model,arm=arm,admission=lambda event:False,
                    strict_group_oracle=True,simulation=True) as session:
                model(*inputs)
                self.assertEqual(len(session.consumed),1)
                self.assert_original(session.consumed[0],inputs)
                self.assertEqual(session.temps,[])
                self.assertTrue(session.events[0]['native_inputs_original_objects'])
                self.assertEqual(session.events[0]['native_calls'],1)
    def test_once_admission_exhaustion_preserves_later_call_objects(self):
        model=MoeWorkload().eval();inputs=self.inputs(); remaining=[True,False]
        with RecordingSession(model,arm='fault',admission=lambda event:remaining.pop(0),
                strict_group_oracle=True,simulation=True) as session:
            model(*inputs);model(*inputs)
            self.assertEqual(len(session.consumed),2)
            self.assertIsNot(session.consumed[0][2]['sorted_indices'],inputs[1])
            self.assert_original(session.consumed[1],inputs)
            self.assertEqual([e['applied'] for e in session.events],[True,False])
            self.assertTrue(session.events[1]['native_inputs_original_storage'])
            self.assertEqual([e['native_calls'] for e in session.events],[1,1])
    def test_trace_close_failure_still_releases_independent_observer(self):
        closed=[]; observer=object.__new__(MoEPhaseObserver)
        def fail():raise RuntimeError('trace close fixture')
        observer.trace=SimpleNamespace(close=fail)
        observer.observer=SimpleNamespace(close=lambda:closed.append(True))
        with self.assertRaisesRegex(RuntimeError,'trace close fixture'):observer.close()
        self.assertEqual(closed,[True])


if __name__ == '__main__': unittest.main()
