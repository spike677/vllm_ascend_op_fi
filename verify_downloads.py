"""Verify and optionally restore the two original eager statistics groups."""
import argparse,hashlib,importlib.util,json,os,sys,tempfile,zipfile
from pathlib import Path,PurePosixPath

def extended(path):
    p=path.absolute()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def sums(p):
    rows={}
    for line in p.read_text().splitlines():
        digest,name=line.split('  ',1);assert name not in rows;rows[name]=digest
    return rows

def verify_data(data):
    expected=sums(data/'SHA256SUMS')
    actual={p.relative_to(data).as_posix():p for p in data.rglob('*') if p.is_file()}
    assert set(actual)==set(expected)|{'SHA256SUMS'}
    assert all(sha(actual[n])==h for n,h in expected.items())
    assert {p.name for p in (data/'cases').iterdir()}=={'baseline','fault'}
    for role in ('baseline','fault'):
        case=data/'cases'/role
        native=json.loads((case/'observation/step0/rank0/dump.json').read_bytes())
        assert native['task']=='statistics' and native['level']=='mix' and native['dump_data_dir'] is None
        assert len([n for n in native['data'] if n.startswith('NPU.npu_fused_infer_attention_score.')])==28
        assert {p.relative_to(case/'observation').as_posix() for p in (case/'observation').rglob('*') if p.is_file()}=={'step0/rank0/'+n+'.json' for n in ('dump','stack','construct')}
        config=json.loads((case/'config.json').read_bytes());assert config['step']==[0] and config['statistics']['summary_mode']=='statistics'
    source=json.loads((data/'source_mapping.json').read_bytes())
    for row in source['files']:
        path=data/row['exported'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
    collector=data/'source/collector/pytorch_processor.py'
    for role in ('baseline','fault'):
        p=json.loads((data/'cases'/role/'collector_provenance.json').read_bytes())
        assert sha(collector)==p['actual_loaded']['sha256']
    acceptance=json.loads((data/'reports/acceptance.json').read_bytes())
    assert acceptance['status']=='PASS_EAGER_PREFILL_WITH_SCOPE_LIMITS'
    lease=json.loads((data/'resource_lease/operation_summary.json').read_bytes())
    assert lease['restored'] and not lease['raw_credential_exported']
    return {'files':len(actual),'bytes':sum(p.stat().st_size for p in actual.values())}

def restore(root,destination):
    destination=extended(destination);assert not destination.exists()
    destination.mkdir(parents=True)
    manifest=json.loads((root/'manifest.json').read_bytes());seen=set()
    for archive in manifest['archives']:
        p=root/archive['path'];assert sha(p)==archive['sha256'] and p.stat().st_size==archive['bytes']
        with zipfile.ZipFile(p) as z:
            infos=z.infolist();assert len(infos)==archive['files'] and sum(i.file_size for i in infos)==archive['uncompressed_bytes']
            for info in infos:
                name=PurePosixPath(info.filename)
                assert not name.is_absolute() and '..' not in name.parts and name.parts[0]=='data' and '\\' not in info.filename
                assert not info.is_dir() and info.filename not in seen and info.file_size<256*1024**2
                seen.add(info.filename);target=destination/Path(*name.parts);target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(z.read(info))
    return destination/'data'

def verify(root,extract_to=None):
    root=extended(root)
    listed=sums(root/'SHA256SUMS');actual={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
    assert set(actual)==set(listed)|{'SHA256SUMS'} and all(sha(actual[n])==h for n,h in listed.items())
    manifest=json.loads((root/'manifest.json').read_bytes());assert manifest['group_names']==['baseline','fault'] and manifest['actual_steps']==[0]
    if extract_to is None:
        with tempfile.TemporaryDirectory(prefix='fault28-eager-data-verify-',dir=root.parent) as temp:
            assert Path(temp).resolve().parent==root.parent.resolve()
            data=restore(root,Path(temp)/'restored');details=verify_data(data);check_localizer(data,root)
    else:
        data=restore(root,extract_to);details=verify_data(data);check_localizer(data,root)
    return {'status':'PASS_TWO_ZIPS_ORIGINAL_BYTES_AND_INDEPENDENT_RECHECK',**details,
        'actual_steps':[0],'main_groups':['baseline','fault'],'complete_tensors_present':False}

def check_localizer(data,root):
    # Only copied original statistics and the disclosed collector metadata feed
    # this recheck. No injector/host execution report chooses its first record.
    previous=sys.modules.get('statistics_values');old_path=list(sys.path)
    try:
        directory=data/'source/analysis';sys.path.insert(0,str(directory))
        spec=importlib.util.spec_from_file_location('statistics_values',directory/'statistics_values.py')
        values=importlib.util.module_from_spec(spec);sys.modules['statistics_values']=values;spec.loader.exec_module(values)
        spec=importlib.util.spec_from_file_location('restored_eager_independent_localizer',directory/'compare_statistics.py')
        localizer=importlib.util.module_from_spec(spec);spec.loader.exec_module(localizer)
        new=localizer.compare(data/'cases/baseline/observation/step0/rank0/dump.json',data/'cases/fault/observation/step0/rank0/dump.json')
        assert new['status']=='ALIGNED_OBSERVED_API_DIFFERENCE'
        old=json.loads((root/'first_difference.json').read_bytes())
        assert new['first_observed_api_difference']==old['first_observed_api_difference']
        assert new['all_aligned_records']==old['all_aligned_records'] and new['unobservable_fields']==old['unobservable_fields']
        assert new['evidence']==old['evidence']
    finally:
        sys.path[:]=old_path
        if previous is None:sys.modules.pop('statistics_values',None)
        else:sys.modules['statistics_values']=previous

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).parent)
    p.add_argument('--extract-to',type=Path);a=p.parse_args();print(json.dumps(verify(a.root,a.extract_to)))

if __name__=='__main__':main()
