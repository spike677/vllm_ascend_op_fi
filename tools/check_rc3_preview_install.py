"""No-device installed-source verification. Does not initialize a Worker/NPU."""
import argparse,hashlib,importlib,importlib.metadata,json,sys
from pathlib import Path


def check(site,release):
    site=Path(site).resolve();release=Path(release).resolve();sys.path.insert(0,str(site))
    manifest=json.loads((release/'release-manifest.json').read_text())
    source_raw=(release/'source-review.json').read_bytes()
    issues=[]
    if hashlib.sha256(source_raw).hexdigest()!=manifest['source_manifest_sha256']:issues.append('SOURCE_MANIFEST_HASH')
    for f in manifest['files']:
        data=(release/f['path']).read_bytes()
        if len(data)!=f['bytes'] or hashlib.sha256(data).hexdigest()!=f['sha256']:issues.append('ARTIFACT_HASH:'+f['path'])
    source={r['path']:r for r in json.loads(source_raw)['files']}
    checked=[];versions={}
    for name,version in manifest['versions'].items():
        dist=importlib.metadata.distribution(name);versions[name]=dist.version
        if dist.version!=version or Path(dist.locate_file('')).resolve()!=site:issues.append('DISTRIBUTION_ORIGIN_VERSION:'+name)
        for file in dist.files or []:
            rel=str(file).replace('\\','/')
            if not rel.endswith(('.py','.json')) or '.dist-info/' in rel:continue
            candidates=[rel,'src/'+rel,'fault39/src/'+rel]
            match=next((p for p in candidates if p in source),None)
            if match is None:issues.append('INSTALLED_SOURCE_NOT_REVIEWED:'+rel);continue
            raw=Path(dist.locate_file(file)).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=source[match]['sha256']:issues.append('INSTALLED_SOURCE_HASH:'+rel)
            checked.append(rel)
    for name in ('fault_injection.rc3_preview_profile','demoncat.runtime.vllm_ascend.worker_extension.continuous_binding',
                 'demoncat.runtime.vllm_ascend.worker_extension.readonly_inspect','demoncat.runtime.vllm_ascend.worker_extension.config'):
        m=importlib.import_module(name)
        if not Path(m.__file__).resolve().is_relative_to(site):issues.append('IMPORT_ORIGIN:'+name)
    profile=site/'demoncat/runtime/vllm_ascend/worker_extension/v026_installed_source_review.json'
    if not profile.is_file():issues.append('SOURCE_PROFILE_NOT_PACKAGED')
    return dict(schema='demoncat.rc3_preview.install.v1',passed=not issues,issues=issues,versions=versions,
        source_manifest_sha256=manifest['source_manifest_sha256'],installed_files_checked=len(checked),
        worker_created=False,npu_initialized=False,qualification_granted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--site',required=True);p.add_argument('--release',required=True);a=p.parse_args()
    result=check(a.site,a.release);print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 2)
