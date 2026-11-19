"""Measure pinned OCI gzip layers in memory only. Never extracts or pulls.
Only small JSON receipts are written. Stops at the first error, with no retry.
"""
import argparse
import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path
import urllib.request

BASE = 'https://quay.io/v2/ascend/vllm-ascend/'
CHUNK = 1024 * 1024

class HashReader:
    def __init__(self, source):
        self.source, self.bytes, self.sha = source, 0, hashlib.sha256()
    def read(self, size=-1):
        if size < 0: size = CHUNK
        value = self.source.read(min(size, CHUNK))
        self.bytes += len(value)
        self.sha.update(value)
        return value

def measure(source, row, expected_diff):
    reader = HashReader(source)
    expanded, digest = 0, hashlib.sha256()
    try:
        if row['mediaType'] not in ('application/vnd.oci.image.layer.v1.tar+gzip',
                                    'application/vnd.docker.image.rootfs.diff.tar.gzip'):
            raise ValueError('UNSUPPORTED_LAYER_ENCODING')
        with gzip.GzipFile(fileobj=reader, mode='rb') as stream:
            while True:
                block = stream.read(CHUNK)
                if not block: break
                expanded += len(block)
                digest.update(block)
        while reader.read(CHUNK): pass
        row.update(compressed_bytes=reader.bytes, uncompressed_bytes=expanded,
            measured_digest='sha256:'+reader.sha.hexdigest(),
            measured_diff_id='sha256:'+digest.hexdigest(), expected_diff_id=expected_diff)
        row['digest_verified'] = row['measured_digest'] == row['digest']
        row['size_verified'] = reader.bytes == row['size']
        row['diff_id_verified'] = row['measured_diff_id'] == expected_diff
        if not all(row[k] for k in ('digest_verified','size_verified','diff_id_verified')):
            raise ValueError('LAYER_DIGEST_OR_SIZE_MISMATCH')
        row['status']='VERIFIED'
    except Exception as error:
        row.update(status='FAILED_STOP', error=type(error).__name__+': '+str(error),
            compressed_bytes=reader.bytes, uncompressed_bytes=expanded)
        raise
    return row

def token():
    url='https://quay.io/v2/auth?service=quay.io&scope=repository:ascend/vllm-ascend:pull'
    with urllib.request.urlopen(url, timeout=60) as r: return json.load(r)['token']

def open_blob(path):
    request=urllib.request.Request(BASE+path,headers={'Authorization':'Bearer '+token(),
        'Accept':'application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json',
        'Accept-Encoding':'identity'})
    return urllib.request.urlopen(request,timeout=60)

def metadata(path, expected):
    with open_blob(path) as r: data=r.read(4*CHUNK+1)
    if len(data)>4*CHUNK: raise ValueError('METADATA_TOO_LARGE')
    if 'sha256:'+hashlib.sha256(data).hexdigest()!=expected: raise ValueError('METADATA_DIGEST_MISMATCH')
    return json.loads(data)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    result=dict(schema='demoncat.oci.memory_stream_measurement.v1',manifest_digest=args.manifest,
        image='quay.io/ascend/vllm-ascend:v0.26.0rc1',started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        layer_payload_written_to_disk=False,docker_pull=False,retries=0,layers=[])
    out=Path(args.out)
    def save(): out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    try:
        manifest=metadata('manifests/'+args.manifest,args.manifest)
        config=metadata('blobs/'+manifest['config']['digest'],manifest['config']['digest'])
        if config['architecture']!='arm64' or config['os']!='linux':raise ValueError('PLATFORM_MISMATCH')
        diffs=config['rootfs']['diff_ids']
        if len(diffs)!=len(manifest['layers']):raise ValueError('LAYER_COUNT_MISMATCH')
        result.update(config_digest=manifest['config']['digest'],expected_layers=len(diffs),
            compressed_expected_bytes=sum(x['size'] for x in manifest['layers']))
        for number,(layer,diff) in enumerate(zip(manifest['layers'],diffs)):
            row=dict(layer,index=number,status='STARTED');result['layers'].append(row);save()
            print(json.dumps(dict(event='start',index=number,bytes=layer['size'])),flush=True)
            with open_blob('blobs/'+layer['digest']) as response: measure(response,row,diff)
            save();print(json.dumps(row),flush=True)
        result.update(status='ALL_LAYERS_VERIFIED',compressed_bytes=sum(x['compressed_bytes'] for x in result['layers']),
            uncompressed_tar_bytes=sum(x['uncompressed_bytes'] for x in result['layers']),
            size_semantics='Sum of uncompressed tar streams including deleted/overwritten entries; not exact overlay allocation')
    except Exception as error:
        result.update(status='STOPPED_ON_FIRST_ERROR',error=type(error).__name__+': '+str(error))
    result['ended_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
    print(json.dumps({k:v for k,v in result.items() if k!='layers'}),flush=True)
    return 0 if result['status']=='ALL_LAYERS_VERIFIED' else 2

if __name__=='__main__':raise SystemExit(main())
