"""Download measured, digest-pinned OCI layers into a fresh owned cache.
This is a post-measurement image transfer, not the diskless size measurement.
No Docker calls, device calls, retries, extraction, or host configuration edits.
"""
import argparse,concurrent.futures,datetime,hashlib,json,os,shutil,urllib.request
from pathlib import Path
BASE='https://quay.io/v2/ascend/vllm-ascend/'
RANGE=16*1024*1024

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return 'sha256:'+h.hexdigest()

def fetch(digest,size,start,end):
    with urllib.request.urlopen('https://quay.io/v2/auth?service=quay.io&scope=repository:ascend/vllm-ascend:pull',timeout=60) as r:
        token=json.load(r)['token']
    request=urllib.request.Request(BASE+'blobs/'+digest,headers={'Authorization':'Bearer '+token,
        'Accept-Encoding':'identity','Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(request,timeout=60) as r:
        if r.status!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('EXACT_RANGE_REQUIRED')
        data=r.read(end-start+2)
    if len(data)!=end-start+1:raise EOFError('RANGE_LENGTH_MISMATCH')
    return data

def transfer(layer,path,reader=fetch):
    size=layer['size'];h=hashlib.sha256();written=0
    with path.open('xb') as stream,concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending=[];next_start=0
        try:
            while written<size:
                while len(pending)<4 and next_start<size:
                    end=min(next_start+RANGE,size)-1
                    pending.append(pool.submit(reader,layer['digest'],size,next_start,end));next_start=end+1
                data=pending.pop(0).result();stream.write(data);h.update(data);written+=len(data)
            stream.flush();os.fsync(stream.fileno())
        finally:
            for f in pending:f.cancel()
    if written!=size or 'sha256:'+h.hexdigest()!=layer['digest']:raise ValueError('COMPRESSED_DIGEST_OR_SIZE_MISMATCH')
    return dict(bytes=written,digest=layer['digest'],verified=True)

def main():
    a=argparse.ArgumentParser();a.add_argument('--manifest',type=Path,required=True);a.add_argument('--config',type=Path,required=True)
    a.add_argument('--measurement',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a=a.parse_args()
    m=json.loads(a.manifest.read_bytes());c=json.loads(a.config.read_bytes());measured=json.loads(a.measurement.read_bytes())
    assert digest(a.manifest)==measured['manifest_digest'] and digest(a.config)==m['config']['digest']
    assert c['architecture']=='arm64' and c['os']=='linux' and len(m['layers'])==len(c['rootfs']['diff_ids'])==len(measured['layers'])
    for layer,diff,proof in zip(m['layers'],c['rootfs']['diff_ids'],measured['layers']):
        assert layer['digest']==proof['digest'] and layer['size']==proof['size']
        assert all(proof[k] is True for k in ('digest_verified','size_verified','diff_id_verified')) and proof['measured_diff_id']==diff
    # Cache + loader staging + final unpacked image; reserve 10GiB builds and15GiB free.
    budget=sum(x['size'] for x in m['layers'])+2*sum(x['uncompressed_bytes'] for x in measured['layers'])+25*1024**3
    assert not a.out.exists() and shutil.disk_usage(a.out.parent).free>=budget,'FRESH_CACHE_OR_DISK_BUDGET'
    a.out.mkdir();report=dict(status='DOWNLOADING',started=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_digest=digest(a.manifest),config_digest=digest(a.config),measurement_sha256=digest(a.measurement),
        layers=[],retries=0,npu_calls=0,docker_calls=0,disk_layer_cache=True,maximum_parallel_ranges=4,budget_bytes=budget)
    def save():
        p=a.out/'receipt.tmp';p.write_text(json.dumps(report,indent=2)+'\n');p.replace(a.out/'receipt.json')
    try:
        for index,layer in enumerate(m['layers']):
            path=a.out/(layer['digest'][7:]+'.gz.part');save();print(json.dumps(dict(index=index,status='START',bytes=layer['size'])),flush=True)
            row=transfer(layer,path);final=path.with_suffix('');path.rename(final)
            report['layers'].append(dict(index=index,file=final.name,**row));save();print(json.dumps(dict(index=index,status='VERIFIED')),flush=True)
        report['status']='ALL_COMPRESSED_LAYERS_VERIFIED_NOT_LOADED'
    except BaseException as e:report.update(status='STOPPED_ON_FIRST_ERROR',error=repr(e))
    finally:
        report['ended']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    return 0 if report['status']=='ALL_COMPRESSED_LAYERS_VERIFIED_NOT_LOADED' else 2
if __name__=='__main__':raise SystemExit(main())
