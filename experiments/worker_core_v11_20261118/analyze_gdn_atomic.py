"""Private captures -> derived atomic-store evidence; does not modify tensors."""
import argparse,hashlib,json,pathlib,torch

def pair_prediction(expected,actual,marker=1.25):
    assert expected.dtype==actual.dtype==torch.bfloat16 and expected.shape==actual.shape
    assert expected.shape[-1]%2==0
    words=expected.contiguous().view(torch.float32)
    initial=torch.full_like(expected,marker).contiguous().view(torch.float32)
    predicted=(words+initial).view(torch.int32)
    actual_bits=actual.contiguous().view(torch.int32)
    overwrite_bits=expected.contiguous().view(torch.int32)
    return dict(pairs=predicted.numel(),atomic_exact=int((predicted==actual_bits).sum()),
                overwrite_exact=int((overwrite_bits==actual_bits).sum()),
                unchanged=int((initial.view(torch.int32)==actual_bits).sum()))

def analyze(capture):
    k=capture['kwargs'];cu=k['cu_seqlens'].tolist();H=k['v'].shape[2];Hg=k['k'].shape[2]
    output=[]
    for name,tensor in zip(('w','u'),capture['output']):
        for request,(start,end) in enumerate(zip(cu,cu[1:])):
            matches=[]
            for h in range(H):
                for peer,(ps,pe) in enumerate(zip(cu,cu[1:])):
                    if peer==request or pe-ps!=end-start:continue
                    # Match every actual mathematical input, not just prompt text.
                    if all(torch.equal(k[n][:,start:end,head],k[n][:,ps:pe,head]) for n,head in
                           [('k',h//(H//Hg)),('v',h),('beta',h),('g_cumsum',h),('A',h)]):
                        row=pair_prediction(tensor[:,ps:pe,h],tensor[:,start:end,h])
                        matches.append(dict(head=h,reference_request=peer,**row));break
            output.append(dict(output=name,request=request,heads=matches,
                totals={n:sum(r[n] for r in matches) for n in ('pairs','atomic_exact','overwrite_exact','unchanged')}))
    return output

def main():
    p=argparse.ArgumentParser();p.add_argument('capture');a=p.parse_args();path=pathlib.Path(a.capture)
    cap=torch.load(path,map_location='cpu',weights_only=True)
    print(json.dumps(dict(schema='demoncat.gdn.atomic_store_evidence.v1',capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         marker=1.25,rows=analyze(cap)),indent=2,allow_nan=False))
if __name__=='__main__':main()
