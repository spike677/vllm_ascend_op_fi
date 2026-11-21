"""CPU-FP64 logical reference for private captured WY inputs; no model repair."""
import argparse,json,pathlib,hashlib,torch

def reference(inputs):
    k,v,beta,g,A=[inputs[n].double() for n in ('k','v','beta','g_cumsum','A')]
    B,T,Hg,K=k.shape;H,V=v.shape[-2:];BT=A.shape[-1]
    assert B==1 and H%Hg==0
    cu=inputs['cu_seqlens'].tolist()
    assert cu[0]==0 and cu[-1]==T
    w=torch.empty((B,T,H,K),dtype=torch.float64);u=torch.empty_like(v)
    expanded=k.repeat_interleave(H//Hg,dim=2)
    for bos,eos in zip(cu,cu[1:]):
        for start in range(bos,eos,BT):
            stop=min(start+BT,eos);n=stop-start
            aa=A[0,start:stop,:,:n].permute(1,0,2)
            vb=(v[0,start:stop]*beta[0,start:stop,:,None]).permute(1,0,2)
            kb=(expanded[0,start:stop]*beta[0,start:stop,:,None]*g[0,start:stop,:,None].exp()).permute(1,0,2)
            u[0,start:stop]=torch.bmm(aa,vb).permute(1,0,2)
            w[0,start:stop]=torch.bmm(aa,kb).permute(1,0,2)
    return w,u

def compare(actual,expected):
    a=actual.double();finite=torch.isfinite(a);reference_finite=torch.isfinite(expected)
    bad=(~finite).nonzero();diff=(a-expected).abs()
    report=dict(shape=list(a.shape),nonfinite=int((~finite).sum()),reference_nonfinite=int((~reference_finite).sum()),
        finite_max_abs_error=float(diff[finite&reference_finite].max()) if (finite&reference_finite).any() else None,
        first_nonfinite=[dict(index=index.tolist(),reference=float(expected[tuple(index)])) for index in bad[:16]])
    return report

def main():
    p=argparse.ArgumentParser();p.add_argument('captures',nargs='+');a=p.parse_args();out=[]
    for name in a.captures:
        path=pathlib.Path(name);cap=torch.load(path,map_location='cpu',weights_only=True)
        assert not cap['args'],'KEYWORD_INPUT_CAPTURE_REQUIRED'
        inputs=cap['kwargs'];expected=reference(inputs)
        out.append(dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            inputs_finite=all(bool(torch.isfinite(v).all()) for v in inputs.values() if isinstance(v,torch.Tensor)),
            outputs={name:compare(actual,ref) for name,actual,ref in zip(('w','u'),cap['output'],expected)}))
    print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':main()
