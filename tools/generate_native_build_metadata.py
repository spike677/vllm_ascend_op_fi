"""Reconstruct a Native sidecar from reviewed build evidence, not from a rebuild."""
import argparse,hashlib,json,subprocess
from pathlib import Path

def sha(b):return hashlib.sha256(b).hexdigest()

def generate(receipt,context,*,commit,release,recipe,sources):
    r=json.loads(receipt)
    if sha(receipt)!=context['build_receipt_sha256']:raise ValueError('RECEIPT_SHA_MISMATCH')
    if r['library']['sha256']!=context['library_sha256']:raise ValueError('LIBRARY_SHA_MISMATCH')
    if not commit.startswith(context['source_commit']) or len(commit)!=40:raise ValueError('SOURCE_COMMIT_MISMATCH')
    if sha(recipe)!=context['tool_sha256']:raise ValueError('RECIPE_SHA_MISMATCH')
    if {n:sha(b) for n,b in sources.items()}!=r['sources']:raise ValueError('SOURCE_FILES_MISMATCH')
    if r['schema']!='demoncat.native_regions.build.v1' or r['status']!='BUILD_AND_CPU_TESTS_ONLY' or any(c['returncode']!=0 for c in r['commands']):raise ValueError('BUILD_NOT_PASSED')
    versions=set()
    for v in r['sdk_version_files'].values():
        if sha(v['text'].encode())!=v['sha256']:raise ValueError('SDK_METADATA_SHA_MISMATCH')
        versions.update(line.split('=',1)[1] for line in v['text'].splitlines() if line.startswith('version='))
    if len(versions)!=1:raise ValueError('SDK_VERSION_AMBIGUOUS')
    if not r.get('compiler') or not release:raise ValueError('BUILD_IDENTITY_MISSING')
    return dict(source_release=release,source_commit=commit,source_files=r['sources'],compiler_version=r['compiler'],
        cann=versions.pop(),recipe_sha256=sha(recipe),receipt_sha256=sha(receipt),library_sha256=r['library']['sha256'])

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--receipt',required=True,type=Path);p.add_argument('--context',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    context=json.loads(a.context.read_bytes());rev=context['source_commit']
    commit=subprocess.check_output(['git','rev-parse',rev+'^{commit}'],text=True).strip()
    release=subprocess.check_output(['git','describe','--tags','--always',commit],text=True).strip()
    read=lambda path:subprocess.check_output(['git','show',commit+':'+path])
    receipt=a.receipt.read_bytes();r=json.loads(receipt)
    result=generate(receipt,context,commit=commit,release=release,recipe=read('tools/build_native_regions.py'),
        sources={n:read('fault_injection/native_adapters/postwrite/'+n) for n in r['sources']})
    data=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():
        if a.output.read_bytes()!=data:raise ValueError('REFUSE_OVERWRITE_DIFFERENT_METADATA')
    else:
        with a.output.open('xb') as f:f.write(data)
    print(json.dumps(dict(file=str(a.output),bytes=len(data),sha256=sha(data),qualification_granted=False)))

if __name__=='__main__':main()
