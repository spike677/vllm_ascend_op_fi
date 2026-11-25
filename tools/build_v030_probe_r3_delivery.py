"""Package r3 observation delta; immutable r2 noise/capture dependencies."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sha=lambda b:hashlib.sha256(b).hexdigest()
CHANGED=['experiments/worker_core_v11_20261118/dsv4_observe_probe.py',
'experiments/worker_core_v11_20261118/dsv4_probe_targeted.py',
'experiments/worker_core_v11_20261118/dsv4_probe_smoke_r3.py',
'tests/worker_extension/test_dsv4_probe_targeted.py','tools/extract_dsv4_targeted_record.py']
DOCS=['v4_v030_probe_r3_handoff_20261125.md','v4_d1_targeted_capture_20261125.md',
'v4_public8_d1_review_plan_20261125.md','v4_d13_independent_release_20261125.md','v4_moe_v030_review_sources_20261125.json']
def build(destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('IMMUTABLE_DELIVERY_EXISTS')
    old=ROOT/'releases/v4-v030-tools-20261125-r2/v030-tools.zip'
    with zipfile.ZipFile(old) as z:base={n:z.read(n) for n in z.namelist() if n!='manifest.json'}
    blobs=dict(base)
    for name in CHANGED:blobs[name]=(ROOT/name).read_bytes()
    evidence=ROOT/'docs/worker_core_v1/v4_probe_r3_20261125'
    mraw=(evidence/'source_manifest.json').read_bytes();manifest=json.loads(mraw)
    for item in manifest['files']:
        assert sha(blobs[item['path']])==item['sha256'],item['path']
    cpu=json.loads((evidence/'cpu.json').read_bytes());a2=json.loads((evidence/'a2-review.json').read_bytes())
    assert cpu['passed'] and a2['status']=='PASSED'
    assert cpu['source_manifest_sha256']==a2['source_manifest_sha256']==sha(mraw)
    comparison=json.loads((evidence/'shared-byte-comparison.json').read_bytes())
    for name,b in base.items():
        if name.endswith('.py') and name not in CHANGED:assert b==blobs[name],name
    for name in DOCS:blobs['docs/worker_core_v1/'+name]=(ROOT/'docs/worker_core_v1'/name).read_bytes()
    for name in ('cpu.json','a2-review.json','isolation.json','shared-byte-comparison.json'):
        blobs[name]=(evidence/name).read_bytes()
    blobs['source_manifest.json']=mraw
    blobs['HANDOFF.md']=blobs['docs/worker_core_v1/v4_v030_probe_r3_handoff_20261125.md']
    # Remove copied r2 smoke evidence/docs in supplement-root aliases only; originals
    # under their dated documentation paths remain clearly labelled historical.
    manifest=dict(schema='dcat.v030.probe.tools.v3',base_tag='yellow-delivery-20261125-r4.1',
        base_zip_sha256=sha(old.read_bytes()),source_manifest_sha256=sha(mraw),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        qualification_granted=False,noise_capture_unchanged=True,
        files={n:dict(bytes=len(b),sha256=sha(b)) for n,b in sorted(blobs.items())})
    blobs['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.mkdir(parents=True);out=destination/'v030-tools-r3.zip'
    with zipfile.ZipFile(out,'w') as z:
        for n,b in sorted(blobs.items()):
            info=zipfile.ZipInfo(n,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for n,b in blobs.items():assert z.read(n)==b,n
    for name in ('manifest.json','source_manifest.json','cpu.json','a2-review.json','isolation.json','shared-byte-comparison.json'):
        (destination/name).write_bytes(blobs[name])
    (destination/'SHA256SUMS').write_bytes(''.join(sha(p.read_bytes())+'  '+p.name+'\n' for p in sorted(destination.iterdir())).encode())
    return dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out.read_bytes()),source_manifest_sha256=sha(mraw),files=len(blobs))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);print(json.dumps(build(p.parse_args().output)))

