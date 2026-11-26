"""Package reviewed development tools without modifying a frozen runtime release."""
import argparse, hashlib, io, json, subprocess, sys, tempfile, zipfile
from pathlib import Path

NAME = 'native-n2-tools-20261126-r1'
PINNED = {
    'dcat-native-build.json': 'c61f1a27a78e27d65cb714f786a5f1015d64754550460027097ad0b04a973ea9',
    'tools/extract_fault33_launch_host.py': '93d0504f82cbbd7a247fe23803dbeef4c7706afcffc50938ff86341ffaa426df',
    'tools/extract_fault33_existing.py': '2c84f71cae4ef00e6b98e94974e5bd57faa7f2edc5f144aa179a16ab4983606c',
    'tools/review_launch_capture_v2.py': '66ee176da78971091b2e2108d7f296123bc6abeb5b909862f99a0fb38268743f',
}
VERIFY = r'''"""Offline verification; no runtime installation, no NPU, no real input mutation."""
import base64, hashlib, io, json, subprocess, sys, unittest
from pathlib import Path
if sys.version_info < (3,10): raise SystemExit('PYTHON_3_10_REQUIRED')
sys.dont_write_bytecode = True
root = Path(__file__).resolve().parent
sys.path[:0] = [str(root), str(root/'tests')]
manifest = json.loads((root/'manifest.json').read_bytes())
for entry in manifest['files']:
    p = (root/entry['path']).resolve()
    assert p.is_relative_to(root), 'PATH_OUTSIDE_BUNDLE'
    raw = p.read_bytes()
    assert len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'], entry['path']
metadata = json.loads((root/'dcat-native-build.json').read_bytes())
assert metadata['library_sha256']=='d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274'
assert metadata['cann']=='9.1.0'
suite = unittest.defaultTestLoader.loadTestsFromName('test_fault33_launch_host')
log = io.StringIO(); result = unittest.TextTestRunner(stream=log).run(suite)
if not result.wasSuccessful(): raise RuntimeError(log.getvalue())
from test_fault33_existing_extract import ExistingExtractTests
fixture = ExistingExtractTests(); fixture.setUp()
try:
    for call in fixture.plan['calls']: call.pop('tiling')
    plan = fixture.root/'plan.json'; plan.write_text(json.dumps(fixture.plan), encoding='utf-8')
    # -I ignores inherited PYTHONPATH and site user settings; inject this bundle only.
    code = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); path=sys.argv.pop(1); runpy.run_path(path,run_name='__main__')"
    cmd = [sys.executable,'-I','-B','-S','-c',code,str(root),str(root/'tools/extract_fault33_launch_host.py'),
           '--capture-root',str(fixture.root),'--api',str(fixture.api),'--plan',str(plan)]
    first = subprocess.run(cmd,cwd=fixture.root,capture_output=True)
    second = subprocess.run(cmd,cwd=fixture.root,capture_output=True)
    assert first.returncode==second.returncode==0, first.stderr.decode(errors='replace')
    assert not first.stderr and not second.stderr and first.stdout==second.stdout
    rows = [json.loads(line) for line in first.stdout.splitlines()]
    assert rows[0]['n2_launch_complete'] is True and rows[0]['ready_for_control_recompute'] is False
    assert all(len(line)+1<=4096 for line in first.stdout.splitlines())
    cli = dict(runs=2,byte_equal=True,records=len(rows),max_line_bytes=max(len(x)+1 for x in first.stdout.splitlines()),
               stdout_sha256=hashlib.sha256(first.stdout).hexdigest())
finally: fixture.doCleanups()
print(json.dumps(dict(status='PASSED',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
    skipped=len(result.skipped),files_verified=len(manifest['files']),cli=cli,python=sys.version.split()[0],
    manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
    isolated_interpreter=True,third_party_dependencies=[],npu_used=False,qualification_granted=False),sort_keys=True))
'''

def sha(raw): return hashlib.sha256(raw).hexdigest()
def encode(value): return (json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True)+'\n').encode()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-commit',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    def git(*args): return subprocess.check_output(['git',*args],cwd=root)
    commit=git('rev-parse',a.source_commit+'^{commit}').decode().strip()
    mapping={
      'dcat-native-build.json':'docs/worker_core_v1/native_build_metadata_d9aa_20261126/dcat-native-build.json',
      'README.md':'docs/worker_core_v1/native_n2_transfer_20261126.md',
      'docs/native-build.md':'docs/worker_core_v1/native_build_metadata_d9aa_20261126/README.md',
      'docs/n2-launch-host.md':'docs/worker_core_v1/fault33_n2_launch_host_20261126.md',
      'docs/owner-review.md':'docs/worker_core_v1/yellow_r8_identity_owner_review_20261126.md',
      **{n:n for n in PINNED if n.startswith('tools/')},
      **{'tests/'+n:'tests/worker_extension/'+n for n in ('test_fault33_existing_extract.py','test_fault33_launch_host.py')},
    }
    payload={n:git('show',commit+':'+src) for n,src in mapping.items()}
    payload.update({'tools/__init__.py':b'', 'verify_bundle.py':VERIFY.encode()})
    for n,digest in PINNED.items(): assert sha(payload[n])==digest,n
    assert payload['tools/review_launch_capture_v2.py']==git('show','e38c4113:tools/review_launch_capture_v2.py')
    manifest=dict(schema='dcat.development.tools.manifest.v1',name=NAME,source_commit=commit,
      development_tools_not_r8_frozen=True,public_runtime_included=False,
      files=[dict(path=n,bytes=len(raw),sha256=sha(raw),source_file=mapping.get(n,'GENERATED_BY_PACKAGER')) for n,raw in sorted(payload.items())])
    payload['manifest.json']=encode(manifest)
    payload['SHA256SUMS']=''.join(f'{sha(raw)}  {n}\n' for n,raw in sorted(payload.items())).encode()
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,raw in sorted(payload.items()):
            zi=zipfile.ZipInfo(NAME+'/'+n,(2026,11,26,0,0,0));zi.compress_type=zipfile.ZIP_DEFLATED
            zi.create_system=3;zi.external_attr=0o100644<<16;z.writestr(zi,raw)
    temporary=root/'.codex-tmp';temporary.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='native-n2-package-',dir=temporary) as td:
        temp=Path(td).resolve();assert temp.is_relative_to(temporary.resolve())
        with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as z:
            for n in z.namelist(): assert (temp/n).resolve().is_relative_to(temp)
            z.extractall(temp)
        extracted=temp/NAME
        for n,raw in payload.items(): assert (extracted/n).read_bytes()==raw
        run=subprocess.run([sys.executable,'-I','-B',str(extracted/'verify_bundle.py')],cwd=temp,capture_output=True)
        if run.returncode: raise RuntimeError(run.stderr.decode(errors='replace'))
        cpu=json.loads(run.stdout);assert cpu['status']=='PASSED'
        assert cpu['manifest_sha256']==sha(payload['manifest.json'])
        # Integrity negative: a changed tool must be rejected before any extraction work.
        altered=extracted/'tools/extract_fault33_launch_host.py';altered.write_bytes(altered.read_bytes()+b'\n')
        bad=subprocess.run([sys.executable,'-I','-B',str(extracted/'verify_bundle.py')],cwd=temp,capture_output=True)
        assert bad.returncode!=0 and b'tools/extract_fault33_launch_host.py' in bad.stderr
        cpu['tampered_tool_rejected']=True
    output=root/'releases'/NAME
    artifacts={'README.md':payload['README.md'],'manifest.json':payload['manifest.json'],
      'cpu.json':encode(cpu),NAME+'.zip':archive.getvalue()}
    artifacts['DELIVERY-SHA256SUMS']=''.join(f'{sha(raw)}  {n}\n' for n,raw in sorted(artifacts.items())).encode()
    output.mkdir(parents=True,exist_ok=True)
    for n,raw in artifacts.items():
        target=output/n
        if target.exists(): assert target.read_bytes()==raw,'REFUSE_OVERWRITE_DIFFERENT_DELIVERY:'+n
        else:
            with target.open('xb') as f: f.write(raw)
    print(json.dumps(dict(directory=str(output),source_commit=commit,zip_bytes=len(archive.getvalue()),
        zip_sha256=sha(archive.getvalue()),manifest_sha256=sha(payload['manifest.json']),cpu=cpu),sort_keys=True))

if __name__=='__main__': main()
