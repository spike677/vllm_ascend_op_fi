"""Read bounded host launch records; unknown layouts stay unobservable."""
import argparse,collections,hashlib,json,struct
from pathlib import Path


def inventory(folder,limit=8192):
    groups=collections.Counter();processes={};files=[]
    for p in sorted(Path(folder).glob('launch-*.bin')):
        raw=p.read_bytes()
        if len(raw)<80:raise ValueError('TRUNCATED_CAPTURE')
        magic,pid,serial,key,blocks,size,count,width,rc,phbytes=struct.unpack_from('<10Q',raw)
        if magic!=0x444341544c415531 or len(raw)!=80+size+phbytes or phbytes!=count*width or rc!=0:
            raise ValueError('INVALID_OR_FAILED_LAUNCH')
        processes.setdefault(pid,[]).append(serial)
        groups[(key,blocks,size,count,width)]+=1
        files.append([p.name,len(raw),hashlib.sha256(raw).hexdigest()])
    if not files:raise ValueError('NO_CAPTURE')
    complete=all(sorted(v)==list(range(len(v))) and len(v)<limit for v in processes.values())
    return dict(schema='dcat.paged33.inventory.v1',capture_complete=complete,process_count=len(processes),
        variants=[dict(tiling_key=str(k[0]),launch_blocks=k[1],host_arg_bytes=k[2],placeholder_count=k[3],placeholder_width=k[4],calls=v,
                       operator_association=None,s2_grouping=None) for k,v in sorted(groups.items())],
        files=files,qualification_granted=False,
        limitation='Launch blocks are not a decoded S2 grouping. Join profiler/operator records and review the exact host serializer; unknown remains null.')


def main():
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--capture-limit',type=int,default=8192);a=p.parse_args()
    if a.capture_limit!=8192:p.error('capture library limit is 8192')
    out=inventory(a.folder,a.capture_limit);print(json.dumps(out,indent=2))
    return 0 if out['capture_complete'] else 2


if __name__=='__main__':raise SystemExit(main())
