import unittest,torch
from experiments.worker_core_v11_20261118.replay_wy_reference import reference
class ReferenceTest(unittest.TestCase):
    def test_segment_tails_and_grouped_heads(self):
        # Identity A makes each result local: another sequence must not contribute.
        k=torch.arange(1,17,dtype=torch.float64).reshape(1,8,1,2)
        v=torch.arange(32,dtype=torch.float64).reshape(1,8,2,2)
        beta=torch.full((1,8,2),.5,dtype=torch.float64)
        g=torch.full((1,8,2),-1.,dtype=torch.float64)
        A=torch.zeros((1,8,2,4),dtype=torch.float64)
        for start,end in [(0,3),(3,7),(7,8)]:
            for row in range(start,end):A[0,row,:,row-start]=1
        inputs=dict(k=k,v=v,beta=beta,g_cumsum=g,A=A,cu_seqlens=torch.tensor([0,3,8]))
        w,u=reference(inputs)
        torch.testing.assert_close(u,v*.5)
        torch.testing.assert_close(w,k.repeat_interleave(2,dim=2)*(.5*torch.exp(torch.tensor(-1.,dtype=torch.float64))))
if __name__=='__main__':unittest.main()
