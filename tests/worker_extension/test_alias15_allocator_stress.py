import gc
import unittest
from experiments.worker_core_v11_20261118.alias15_allocator_stress import prime


class StressTests(unittest.TestCase):
    def test_only_freed_seed_is_written(self):
        trace=[]
        class Buffer:
            def __init__(self, number):self.number=number
            def data_ptr(self):return 256
            def numel(self):return 8
            def element_size(self):return 2
            def fill_(self,value):trace.append(('write',self.number))
            def __del__(self):trace.append(('free',self.number))
        def allocate():
            n=sum(t[0]=='alloc' for t in trace)
            trace.append(('alloc',n));return Buffer(n)
        output,facts=prime(allocate,lambda:trace.append(('complete',0)),(),{})
        self.assertEqual(trace,[('alloc',0),('write',0),('complete',0),('free',0),('alloc',1)])
        self.assertTrue(facts['reused'])
        self.assertEqual(output.number,1)
