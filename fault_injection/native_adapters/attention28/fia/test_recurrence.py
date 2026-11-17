"""Independent scalar recurrence checks; not NPU acceptance."""
import math
import random
import unittest

def online(scores,values,replay):
    maximum=float('-inf');den=0.0;num=0.0
    for index,start in enumerate(range(0,len(scores),128)):
        newmax=max(maximum,max(scores[start:start+128]))
        scale=math.exp(maximum-newmax)
        if replay and index==1:den*=2
        den=den*scale+sum(math.exp(x-newmax) for x in scores[start:start+128])
        num=num*scale+sum(math.exp(x-newmax)*v for x,v in zip(scores[start:start+128],values[start:start+128]))
        maximum=newmax
    return num/den

class RecurrenceTests(unittest.TestCase):
    def test_closed_form_multiple_rescalings(self):
        rng=random.Random(42)
        for n in (1,127,128,129,256,257,1024,2048):
            scores=[rng.uniform(-8,8)+i/80 for i in range(n)];values=[rng.uniform(-1,1) for i in range(n)]
            m=max(scores);weights=[math.exp(x-m) for x in scores]
            num=sum(w*v for w,v in zip(weights,values));den=sum(weights)
            self.assertAlmostEqual(online(scores,values,False),num/den,places=13)
            extra=sum(weights[:128]) if n>128 else 0
            self.assertAlmostEqual(online(scores,values,True),num/(den+extra),places=13)

    def test_all_uniform_queries(self):
        for n in range(1,2049):
            self.assertEqual(online([0]*n,[1]*n,False),1)
            self.assertEqual(online([0]*n,[1]*n,True),1 if n<=128 else n/(n+128))

    def test_replay_is_not_constant_output_scaling(self):
        a=online([0]*256,[1]*256,True)
        b=online([-4]*128+[4]*128,[1]*256,True)
        self.assertGreater(b,a+.3)

if __name__=='__main__':unittest.main()
