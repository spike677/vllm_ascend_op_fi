"""Summarize private ACL launch records without exposing pointer arguments.

Offsets are a reviewed public-host serializer projection, not proof of full
CANN ABI equivalence. Keep the original host arguments private on the server.
"""
import collections,hashlib,json,struct
from pathlib import Path

FIELDS={'numHeads':(0,'I',1),'embeddingSize':(4,'I',1),'embeddingSizeV':(8,'I',1),
 'blockSize':(16,'I',1),'kvHeads':(28,'I',1),'batch':(32,'I',1),'scaleValue':(96,'f',1),
 'sparseMode':(144,'I',1),'totalSplitNodeNum':(168,'I',1),'needCoreNum':(172,'I',1),
 'startN1':(304,'i',10),'startS2':(512,'i',10),'endS2':(928,'i',10),'splitNum':(1968,'i',2)}

def summarize(folder,*,capture_limit):
    processes={};variants=collections.Counter();hashes=[];decode=0;prefill=0
    for p in sorted(Path(folder).glob('launch-*.bin')):
        b=p.read_bytes()
        if len(b)<80:raise ValueError('TRUNCATED_CAPTURE')
        h=struct.unpack_from('<10Q',b)
        magic,pid,serial,key,blocks,size,count,width,rc,phbytes=h
        if magic!=0x444341544c415531 or len(b)!=80+size+phbytes or count*width!=phbytes or rc!=0:raise ValueError('INVALID_OR_FAILED_LAUNCH')
        processes.setdefault(pid,[]).append(serial);hashes.append((p.name,hashlib.sha256(b).hexdigest()))
        if key==5000000000000200203:prefill+=1;continue
        if key!=5100000000010200203 or size!=2960 or blocks!=10 or width!=8:raise ValueError('UNREVIEWED_DECODE_LAYOUT')
        args=b[80:80+size];holders=b[80+size:]
        if struct.unpack_from('<II',holders)!=(280,296):raise ValueError('TILING_PLACEHOLDER_CHANGED')
        values={}
        for name,(offset,fmt,n) in FIELDS.items():
            v=struct.unpack_from('<'+fmt*n,args,296+offset);values[name]=v[0] if n==1 else list(v)
        variants[json.dumps(values,sort_keys=True)]+=1;decode+=1
    if not processes:raise ValueError('NO_CAPTURE')
    expected=dict(numHeads=12,embeddingSize=256,embeddingSizeV=256,blockSize=128,kvHeads=2,batch=1,scaleValue=.0625,sparseMode=3,totalSplitNodeNum=2,needCoreNum=10,startN1=[0]*5+[1]*5,startS2=list(range(5))*2,endS2=list(range(1,6))*2,splitNum=[5,5])
    limits=[dict(records=len(v),max_serial=max(v),contiguous=sorted(v)==list(range(len(v))),limit_reached=max(v)+1>=capture_limit) for v in processes.values()]
    return dict(schema='demoncat.paged33.launch_projection.v1',prefill_launches=prefill,decode_launches=decode,
        decoded_template_matches=bool(variants) and all(json.loads(k)==expected for k in variants),
        process_count=len(processes),process_records=sorted(limits,key=lambda x:x['records']),
        all_within_capture_limit=all(not r['limit_reached'] and r['contiguous'] for r in limits),
        decoded_variants=[dict(fields=json.loads(k),calls=v) for k,v in variants.items()],
        ordered_capture_sha256=hashlib.sha256(json.dumps(hashes,separators=(',',':')).encode()).hexdigest(),
        projection_status='OBSERVED_FIELDS_CHECKED_NOT_FULL_ABI_PROOF',raw_arguments_private=True,qualification_granted=False)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--capture-limit',type=int,required=True);p.add_argument('--output');a=p.parse_args()
    r=summarize(a.folder,capture_limit=a.capture_limit);text=json.dumps(r,indent=2)+'\n'
    if a.output:Path(a.output).write_text(text)
    else:print(text)
