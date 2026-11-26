"""N2 launch/host transport without asserting a tiling boundary; stdout JSONL only."""
import argparse,base64,copy,hashlib,json,struct,sys
from pathlib import Path
from tools.extract_fault33_existing import extract
from tools.review_launch_capture_v2 import HEADER

CORE=('rank','prompt_length','api','module','phase','q_shape','launch_relation')
def canonical(v):return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(b):return hashlib.sha256(b).hexdigest()

def records(root,api,plan):
    if any(c.get('tiling') is not None for c in plan.get('calls',[])):raise ValueError('LAUNCH_ONLY_MUST_NOT_DECLARE_TILING')
    original=extract(root,api,plan);root=Path(root).resolve()
    blocking=list(original['issues']);rows=[];payloads=[]
    for index,row in enumerate(original['rows']):
        raw=(root/row['capture']).read_bytes();header=HEADER.unpack_from(raw)
        host=raw[HEADER.size:HEADER.size+header[5]];holders=raw[HEADER.size+header[5]:]
        missing=[x for x in row['issues'] if x!='TILING_BOUNDARY_UNPROVEN' and
                 (not x.startswith('MISSING:') or x.split(':',1)[1] in CORE)]
        if missing:blocking.append(dict(row=index,issues=missing))
        rows.append(dict(row=index,capture=row['capture'],capture_sha256=row['capture_sha256'],pid=row['pid'],serial=row['serial'],
            launch_key=row['launch_key'],block_dim=row['block_dim'],api=row['api'],pointers=row['pointers'],
            host_bytes=len(host),host_sha256=sha(host),holder_bytes=len(holders),holder_sha256=sha(holders),
            tiling=None,serializer_validated=False,qualification_granted=False,
            deferred_issues=[x for x in row['issues'] if x not in missing],blocking_issues=missing))
        for region,data in [('host',host),('holders_opaque',holders)]:
            for offset in range(0,len(data),768):
                chunk=data[offset:offset+768]
                payloads.append(dict(kind='bytes',row=index,capture_sha256=row['capture_sha256'],region=region,
                    offset=offset,total_bytes=len(data),region_sha256=sha(data),encoding='base64',data=base64.b64encode(chunk).decode(),chunk_sha256=sha(chunk)))
    summary=dict(kind='summary',schema='dcat.fault33.n2.launch_host.v1',source_sha256=original['source_sha256'],
        api_sha256=original['api_sha256'],plan_sha256=original['plan_sha256'],rows=len(rows),n2_launch_complete=not blocking,
        blocking_issues=blocking,field_extraction_complete=False,serializer_validated=False,ready_for_control_recompute=False,qualification_granted=False,
        meaning='Launch identity, unique API association and whole host bytes only. Not a tiling decode or 33 admission.',
        captures=[dict(pid=c['pid'],manifest_sha256=c['manifest_sha256'],complete=c['complete'],issues=c['issues']) for c in original['captures']])
    out=[summary]+[dict(kind='row',**r) for r in rows]+payloads
    for i,r in enumerate(out):
        r['record_index']=i;r['extract_sha256']=sha(canonical(r))
        if len(canonical(r))+1>4096:raise ValueError('N2_RECORD_EXCEEDS_4096_NO_TRUNCATION')
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--capture-root',required=True,type=Path);p.add_argument('--api',required=True,type=Path);p.add_argument('--plan',required=True,type=Path);a=p.parse_args()
    out=records(a.capture_root,a.api,json.loads(a.plan.read_bytes()))
    for r in out:sys.stdout.buffer.write(canonical(r)+b'\n')
    return 0 if out[0]['n2_launch_complete'] else 2
if __name__=='__main__':raise SystemExit(main())
