"""Allowlisted original statistics/meta export for the two separate attempts.

Full log, raw prompt/token manifest, private reference tensors and logits stay
remote. Native window hashes are retained, not silently presented as public.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile

ROOT=Path('/home/fault28-m-only-model-20261109')
FORBIDDEN={'.pt','.pth','.bin','.npy','.npz','.safetensors','.so','.o','.a','.run','.log'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


class Export:
    def __init__(self,root):
        root.mkdir(parents=True,exist_ok=False);self.root=root;self.files=[];self.private=[]

    def copy(self,path,name,expected=None):
        relative=PurePosixPath(str(name))
        assert not relative.is_absolute() and '..' not in relative.parts
        assert not any(p in ('private_output','private_reference','inputs','__pycache__') for p in relative.parts)
        assert path.suffix.lower() not in FORBIDDEN and path.suffix.lower() in ('.json','.jsonl','.py','.sh','.txt','.patch','.md')
        assert path.is_file() and not path.is_symlink() and path.stat().st_size<256*1024**2
        observed=sha(path);assert expected is None or expected==observed
        target=self.root/str(relative);assert not target.exists() and target.resolve().is_relative_to(self.root.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target);assert sha(target)==observed
        self.files.append({'source':str(path),'published':str(relative),'bytes':target.stat().st_size,
                           'sha256':observed,'original_bytes_preserved':True})

    def retain(self,path,reason,expected=None):
        observed=sha(path);assert expected is None or expected==observed
        self.private.append({'path':str(path),'sha256':observed,'bytes':path.stat().st_size,
                             'reason':reason,'included_in_public_bundle':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    acceptance=json.loads((a.review/'execution_acceptance.json').read_bytes())
    assert acceptance['status']=='PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE' and acceptance['totals']['actual_strategy_switches']==112
    suite=json.loads((a.review/'suite.json').read_bytes())
    export=Export(a.destination);source_frozen=None;public_cases=[]
    for spec in suite['cases']:
        folder=Path(spec['case_dir']);attempt=folder.parent;name=folder.name
        result=json.loads((folder/'result.json').read_bytes());assert result['status']=='CAPTURE_COMPLETED'
        numeric=a.review/('numeric_b1.json' if name.endswith('b1') else 'numeric_b8.json')
        numeric_report=json.loads(numeric.read_bytes())
        assert numeric_report['status']=='PASS_EXACT_OBSERVATION_INTEGRITY'
        assert numeric_report['case_result_sha256']==sha(folder/'result.json')
        assert numeric_report['phase_manifest_sha256']==sha(folder/'phase_manifest.json')
        prefix=Path('cases')/name
        public_cases.append({'case_dir':prefix.as_posix(),
            'framework_report':f'reports/{name}/framework.json',
            'native_census_report':f'reports/{name}/native_census.json',
            'statistics_report':f'reports/{name}/statistics/summary.json',
            'completion':(prefix/'controller/completion.json').as_posix()})
        for file in ('result.json','receipt.json','config.json','collector_provenance.json','model_config.json',
                     'module_inventory.json','module_events.jsonl','phase_manifest.json'):
            export.copy(folder/file,(prefix/file).as_posix())
        for file in ('completion.json','source_manifest.json','selected_cases.json','before.json','before_disk.json',
                     'container_identity.json','resource_policy.json','resource_checks.jsonl','disk_checks.jsonl',
                     'launch_command.json','reservation_settle.json'):
            export.copy(attempt/file,(prefix/'controller'/file).as_posix())
        export.copy(attempt/(name+'.vendor_launch.json'),(prefix/'controller/vendor_launch.json').as_posix())
        manifest=json.loads((attempt/'source_manifest.json').read_bytes())
        if source_frozen is None:
            source_frozen=manifest
            for relative,expected in manifest.items():
                path=attempt/'source'/relative
                if relative=='inputs/manifest.json':
                    export.retain(path,'Private frozen request/token source',expected);continue
                export.copy(path,('source/'+relative),expected)
        else:
            assert manifest==source_frozen,'Both batches must execute identical source and case generation'
        for kind in ('framework_report','native_census_report','statistics_report'):
            file=Path(spec[kind])
            target=Path('reports')/name/('statistics' if kind=='statistics_report' else '')/file.name
            export.copy(file,target.as_posix())
            if kind=='statistics_report':
                for report in file.parent.glob('*.json'):
                    if report!=file:export.copy(report,(Path('reports')/name/'statistics'/report.name).as_posix())
        phases=json.loads((folder/'phase_manifest.json').read_bytes())['phases']
        for phase in phases:
            for relative in (phase['request_mapping_file'],phase['cache_observation_file']):
                export.copy(folder/relative,(prefix/relative).as_posix())
            for kind in ('dump','stack','construct'):
                relative=f"observation/step{phase['step']}/rank0/{kind}.json"
                export.copy(folder/relative,(prefix/relative).as_posix())
            export.retain(folder/phase['private_logits_file'],'Complete logits privately compared',phase['logits']['full_sha256'])
        for item in result['calls']:
            file=folder/item['file'];export.copy(file,(prefix/item['file']).as_posix(),item['sha256'])
            call=json.loads(file.read_bytes())
            if call['stage']=='prefill':
                relative=call['native_record_file'];record_file=folder/relative
                export.copy(record_file,(prefix/relative).as_posix(),call['native_record_sha256'])
                record=json.loads(record_file.read_bytes());profile=record_file.parent
                for extra in ('host_event.json','host_events.original.jsonl','public_pod_schema.json'):
                    export.copy(profile/extra,(prefix/profile.relative_to(folder)/extra).as_posix())
                export.copy(file.parent/'host_events.jsonl',(prefix/file.parent.relative_to(folder)/'host_events.jsonl').as_posix())
                export.retain(profile/'native_window.log','Original SDPA log window retained remotely; raw/typed public projection is included',
                              record['native_launch']['window_sha256'])
        export.retain(attempt/(name+'.log'),'Whole DEBUG log required for independent launch census; retained privately')
        export.private.append({'path':str(folder/'private_reference'),'reason':'Native full Attention and KV baseline references remain private',
                               'included_in_public_bundle':False,'individual_sha_recorded_in':'original numeric snapshot metadata'})
    for file in a.review.glob('*.json'):export.copy(file,('reports/'+file.name))
    for name in ('formal_b1_a1','formal_b1_a2','formal_b1_a3'):
        export.copy(ROOT/'outputs'/name/'completion.json','earlier_attempts/'+name+'/completion.json')
    for file in sorted((ROOT/'analysis').glob('*.py')):
        export.copy(file,'analysis/'+file.name)
    for file in sorted((ROOT/'analysis').glob('*cpu_a*.json')):
        export.copy(file,'cpu_checks/'+file.name)
    collector=Path('/home/fault28-fia-graph-20261106/collector_fp64_a1')
    for name in ('stage_manifest.json','source_manifest.json','collector.patch'):
        export.copy(collector/name,'collector/'+name)
    export.copy(collector/'packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py','collector/pytorch_processor.py')
    export.copy(ROOT/'native_vendor_stage_a1/vendor_manifest.json','build/vendor_manifest.json')
    save(a.destination/'public_suite.json',{'schema':'nes.model_coverage.acceptance_inputs.v1','cases':public_cases,
         'scope':'Public statistics and hash review; full native acceptance additionally requires remote-only log and private data.'})
    save(a.destination/'manifest.json',{'schema':'fault28.m_only.public_evidence.v1','files':export.files,
         'remote_only':export.private,'public_subset_can_rerun_full_log_acceptance':False,
         'full_F_attention_and_KV_tensors_not_retained':True,'B_F_all_elements_compared_by_common_runtime_observer':True})
    files=[p for p in sorted(a.destination.rglob('*')) if p.is_file()]
    assert all(p.suffix.lower() not in FORBIDDEN for p in files)
    assert sum(p.stat().st_size for p in files)<1024**3
    (a.destination/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(a.destination).as_posix()+'\n' for p in files))
    archive=a.destination.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as stream:
        for file in sorted(a.destination.rglob('*')):
            if file.is_file():stream.add(file,arcname=file.relative_to(a.destination),recursive=False)
    print(json.dumps({'status':'EXPORTED_STATISTICS_AND_METADATA','files':len(files)+1,'archive':str(archive),
                      'sha256':sha(archive),'bytes':archive.stat().st_size}))


if __name__=='__main__':main()
