"""Bounded read-only Native build text; build success is not deployment approval."""
import argparse,hashlib,json,sys
from pathlib import Path


def extract(receipt,library,identity=None):
    p=Path(receipt);raw=p.read_bytes();r=json.loads(raw);lib=Path(library);b=lib.read_bytes()
    sha=lambda x:hashlib.sha256(x).hexdigest()
    if r.get('schema')!='demoncat.native_regions.build.v1':raise ValueError('WRONG_NATIVE_RECIPE_RECEIPT')
    declared=r.get('library',{});issues=[]
    if declared!={'file':lib.name,'bytes':len(b),'sha256':sha(b)}:issues.append('LIBRARY_IDENTITY_MISMATCH')
    if r.get('status')!='BUILD_AND_CPU_TESTS_ONLY' or r.get('npu_calls')!=0:issues.append('BUILD_STATUS_NOT_PASSED')
    commands=r.get('commands',[])
    if not commands or any(c.get('returncode')!=0 for c in commands):issues.append('BUILD_COMMANDS_NOT_PASSED')
    versions=r.get('sdk_version_files',{})
    if not versions:issues.append('SDK_VERSION_UNOBSERVED')
    if not isinstance(r.get('sources'),dict) or not r['sources']:issues.append('SOURCE_FILE_HASHES_MISSING')
    if not r.get('compiler'):issues.append('COMPILER_VERSION_MISSING')
    d=identity or {}
    for field in ('image_id','repo_digests','cann_runtime','cann_runtime_source','source_release','source_commit'):
        if d.get(field) is None:issues.append('DEPLOYMENT_FIELD_MISSING:'+field)
    if d.get('cann_runtime') not in (None,'9.1.0'):issues.append('CANN_RUNTIME_NOT_QWEN_9_1_0')
    result=dict(schema='dcat.native.build.extract.v1',recipe='tools/build_native_regions.py:postwrite-stream-object-v2',
        receipt=[p.name,len(raw),sha(raw)],library=[lib.name,len(b),sha(b)],status=r.get('status'),machine=r.get('machine'),
        source_files=r.get('sources'),compiler=r.get('compiler'),sdk_version_files=versions,
        commands=[[Path(c.get('args',['UNKNOWN'])[0]).name,c.get('returncode')] for c in commands],
        deployment={k:d.get(k) for k in ('image_id','repo_digests','cann_runtime','cann_runtime_source','source_release','source_commit')},
        deployment_source='YELLOW_SUPPLEMENTAL_DECLARATION' if identity else 'UNOBSERVED',
        issues=issues,qualification_granted=False,review_required=True)
    canonical=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result['extract_sha256']=sha(canonical(result))
    if len(canonical(result))+1>4096:raise ValueError('BUILD_EXTRACT_EXCEEDS_4096_NO_TRUNCATION')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');p.add_argument('library');p.add_argument('--deployment-identity');a=p.parse_args()
    r=extract(a.receipt,a.library,json.loads(Path(a.deployment_identity).read_bytes()) if a.deployment_identity else None)
    sys.stdout.buffer.write(json.dumps(r,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8')+b'\n')


if __name__=='__main__':main()
