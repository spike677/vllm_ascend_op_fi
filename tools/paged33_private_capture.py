"""Bounded private real-call capture; no arithmetic change or qualification.

Deploy only inside the reviewed Worker call boundary. Pairing with a launch
capture remains an explicit review gate; an API ordinal is not a tiling serial.
"""
import hashlib,json
from pathlib import Path
import torch


def unpack_actual(arguments,output,request_ids):
    def require(ok,code):
        if not ok:raise ValueError(code)
    a=arguments;q=a['query'];keys=a['key'];values=a['value']
    require(len(request_ids)==1 and isinstance(request_ids[0],str),'CAPTURE_SERIAL_REAL_REQUEST_REQUIRED')
    require(isinstance(q,torch.Tensor) and q.ndim==3 and q.shape[0]==1 and q.dtype==torch.bfloat16,'CAPTURE_B1_Q1_BF16_REQUIRED')
    require(a.get('input_layout')=='TND' and a.get('sparse_mode')==3,'CAPTURE_TND_SPARSE3_REQUIRED')
    # This control models all keys through the current final causal query.
    # A supplied mask needs separate semantic decoding, never silently ignored.
    require(a.get('atten_mask') is None,'CAPTURE_MASK_REQUIRES_SEPARATE_REVIEW')
    require(a.get('antiquant_scale') is None and a.get('antiquant_offset') is None,'CAPTURE_QUANTIZED_KV_UNSUPPORTED')
    def single(x):
        if isinstance(x,(tuple,list)):
            require(len(x)==1,'CAPTURE_ONE_CACHE_TENSOR_REQUIRED');return x[0]
        return x
    key,value=single(keys),single(values);table=a['block_table'];seq=a['actual_seq_lengths_kv']
    seq=seq.detach().cpu().tolist() if isinstance(seq,torch.Tensor) else list(seq)
    require(len(seq)==1 and type(seq[0]) is int and seq[0]>0,'CAPTURE_ONE_VALID_LENGTH_REQUIRED')
    length=seq[0];block=a['block_size'];kv=a['num_key_value_heads'];heads=a['num_heads']
    require(type(block) is int and block>0 and type(kv) is int and kv>0 and heads%kv==0
            and heads==q.shape[1],'CAPTURE_HEAD_BLOCK_GEOMETRY')
    require(isinstance(table,torch.Tensor) and table.ndim==2 and table.shape[0]==1,'CAPTURE_REAL_BLOCK_TABLE_REQUIRED')
    count=(length+block-1)//block;ids=table[0,:count].long()
    require(ids.numel()==count and ids.unique().numel()==count,'CAPTURE_REPEATED_OR_MISSING_PAGE')
    require(key.dtype==value.dtype==q.dtype and key.shape==value.shape and key.ndim in (3,4)
            and key.shape[1]==block and key[0].numel()==block*kv*q.shape[-1],'CAPTURE_CACHE_LAYOUT')
    require(int(ids.min())>=0 and int(ids.max())<key.shape[0],'CAPTURE_PAGE_OUTSIDE_CACHE')
    k=key.index_select(0,ids).reshape(-1,kv,q.shape[-1])[:length]
    v=value.index_select(0,ids).reshape(-1,kv,q.shape[-1])[:length]
    native=output[0] if isinstance(output,(tuple,list)) else output
    require(isinstance(native,torch.Tensor) and native.shape==q.shape and native.dtype==q.dtype,'CAPTURE_NATIVE_OUTPUT_LAYOUT')
    require(all(bool(torch.isfinite(t).all()) for t in (q,k,v,native)),'CAPTURE_NONFINITE')
    return dict(q=q.detach().cpu(),k=k.detach().cpu(),v=v.detach().cpu(),output=native.detach().cpu(),
                metadata=dict(length=length,heads=heads,kv_heads=kv,scale=a['scale'],block_size=block,
                              sparse_mode=3,input_layout='TND',mask=None,physical_pages=ids.cpu().tolist(),
                              request_ids=request_ids,logical_key_range=[0,length],each_key_once=True))


class PrivateCapture:
    def __init__(self,folder,*,rank,lengths):
        if type(rank) is not int or rank<0 or len(set(lengths))!=3 or any(type(v) is not int or v<129 for v in lengths):
            raise ValueError('CAPTURE_THREE_DISTINCT_DECODE_LENGTHS_REQUIRED')
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.rank=rank;self.lengths=set(lengths);self.saved={};self.ordinal=0
    def observe(self,arguments,output,request_ids):
        self.ordinal+=1;q=arguments.get('query');seq=arguments.get('actual_seq_lengths_kv')
        if not isinstance(q,torch.Tensor) or q.ndim!=3 or q.shape[0]!=1 or seq is None:return
        seq=seq.detach().cpu().tolist() if isinstance(seq,torch.Tensor) else list(seq)
        if len(seq)!=1 or seq[0] not in self.lengths or seq[0] in self.saved:return
        data=unpack_actual(arguments,output,request_ids);length=seq[0]
        path=self.folder/f'rank{self.rank}-kv{length}.pt'
        with path.open('xb') as handle:torch.save(data,handle)
        raw=path.read_bytes();self.saved[length]=dict(rank=self.rank,length=length,file=path.name,bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),api_ordinal=self.ordinal,request_ids=request_ids,
            launch_serial=None,launch_association_review_required=True)
    def finish(self):
        result=dict(schema='dcat.paged33.private_capture.v1',rank=self.rank,
                    complete=set(self.saved)==self.lengths,missing=sorted(self.lengths-set(self.saved)),
                    captures=[self.saved[k] for k in sorted(self.saved)],qualification_granted=False)
        (self.folder/'manifest.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        return result
