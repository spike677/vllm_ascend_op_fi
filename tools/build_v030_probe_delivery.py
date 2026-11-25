"""Build immutable r2 probe supplement from r1 dependency closure plus explicit tool changes."""
import argparse,hashlib,json,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(raw):return hashlib.sha256(raw).hexdigest()
def build(destination):
    destination=Path(destination)
    if destination.exists():raise ValueError('IMMUTABLE_DELIVERY_EXISTS')
    old=ROOT/'releases/v4-v030-noise-tools-20261125-r1/v030-noise-tools.zip'
    with zipfile.ZipFile(old) as z:blobs={n:z.read(n) for n in z.namelist() if n!='manifest.json'}
    original={n:sha(b) for n,b in blobs.items()}
    files=['experiments/worker_core_v11_20261118/'+n+'.py' for n in (
        'dsv4_observe_probe','dsv4_noise_capture','dsv4_probe_support','dsv4_probe_journal','dsv4_probe_extension','dsv4_probe_smoke')]
    files += ['tests/worker_extension/'+n+'.py' for n in ('test_dsv4_probe_v2','test_dsv4_noise_capture','test_dsv4_observer')]
    files += ['tools/extract_dsv4_v030_record.py','docs/worker_core_v1/v4_v030_probe_r2_handoff_20261125.md',
        'docs/worker_core_v1/v4_probe_r2_20261125/cpu.json','docs/worker_core_v1/v4_probe_r2_20261125/a2-review.json',
        'docs/worker_core_v1/v4_probe_r2_20261125/isolation.json']
    for name in files:blobs[name]=(ROOT/name).read_bytes()
    blobs['HANDOFF.md']=blobs['docs/worker_core_v1/v4_v030_probe_r2_handoff_20261125.md']
    tracked=[n for n in files if subprocess.run(['git','cat-file','-e','a9ddc3f4:'+n],cwd=ROOT,capture_output=True).returncode==0]
    diff=subprocess.check_output(['git','diff','a9ddc3f4','--',*tracked],cwd=ROOT)
    # Added files are included verbatim in unified diff, not silently omitted.
    import difflib
    for n in files:
        if n not in tracked:
            diff+=''.join(difflib.unified_diff([],blobs[n].decode('utf-8-sig').splitlines(True),fromfile='/dev/null',tofile=n)).encode()
    blobs['tool-diff.patch']=diff
    comparison={n:dict(before=d,after=sha(blobs[n]),equal=d==sha(blobs[n])) for n,d in original.items()}
    assert all(v['equal'] for n,v in comparison.items() if n.startswith('fault_injection/'))
    blobs['shared-byte-comparison.json']=(json.dumps(dict(base='r1',files=comparison,mechanism_files_unchanged=True),indent=2)+'\n').encode()
    manifest=dict(schema='dcat.v030.probe.tools.v2',base_tag='yellow-delivery-20261125-r3',
        previous_zip_sha256=sha(old.read_bytes()),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        qualification_granted=False,files={n:dict(bytes=len(b),sha256=sha(b)) for n,b in sorted(blobs.items())})
    blobs['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    destination.mkdir(parents=True);out=destination/'v030-tools.zip'
    with zipfile.ZipFile(out,'w') as z:
        for n,b in sorted(blobs.items()):
            info=zipfile.ZipInfo(n,date_time=(2026,11,25,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    with zipfile.ZipFile(out) as z:
        assert z.testzip() is None
        for n,b in blobs.items():assert z.read(n)==b
    (destination/'manifest.json').write_bytes(blobs['manifest.json'])
    (destination/'SHA256SUMS').write_bytes((sha(out.read_bytes())+'  v030-tools.zip\n'+sha(blobs['manifest.json'])+'  manifest.json\n').encode())
    return dict(path=str(out),bytes=out.stat().st_size,sha256=sha(out.read_bytes()),files=len(blobs))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    print(json.dumps(build(p.parse_args().output)))
