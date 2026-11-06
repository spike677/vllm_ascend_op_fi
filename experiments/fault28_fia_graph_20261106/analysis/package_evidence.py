"""Allowlisted statistical evidence export; never copies full tensors/binaries/logs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from audit_case import ROOT

FORBIDDEN={'.pt','.pth','.bin','.npy','.npz','.safetensors','.o','.so','.run','.log'}
ALLOWED={'.json','.jsonl','.csv','.txt','.patch','.h','.cpp','.py','.sh'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


class Export:
    def __init__(self,destination):
        self.destination=destination;self.mapping=[]
        destination.mkdir(parents=True,exist_ok=False)

    def copy(self,source,relative):
        path=self.destination/relative
        assert source.is_file() and not source.is_symlink()
        assert source.suffix.lower() in ALLOWED and source.suffix.lower() not in FORBIDDEN
        assert source.stat().st_size<=32*1024**2
        assert not path.exists() and path.resolve().is_relative_to(self.destination.resolve())
        path.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,path)
        assert sha(path)==sha(source)
        self.mapping.append({'source':str(source),'published':str(relative).replace('\\','/'),
                             'bytes':path.stat().st_size,'sha256':sha(path),'original_bytes_preserved':True})

    def case(self,name,suffix,prefix):
        root=ROOT/'outputs'/(name+'_'+suffix);folder=root/name
        for file in ('result.json','case.json','inputs.json','config.json','graph_config.json','fx_graph_0.json',
                     'consumer.json','loaded_libraries.json','collector_provenance.json','statistics_receipt.json',
                     'host_events.jsonl','jit_compile_inventory.json'):
            if (folder/file).is_file():self.copy(folder/file,prefix/file)
        for step in (0,1):
            for name_json in ('dump','stack','construct'):
                relative=Path('observation')/f'step{step}'/'rank0'/(name_json+'.json')
                self.copy(folder/relative,prefix/relative)
        for kind in ('original','optimized'):
            files=list((folder/'msprobe_graph').rglob('dynamo_'+kind+'_graph_*.txt'))
            assert len(files)==1
            self.copy(files[0],prefix/'msprobe_graph'/('dynamo_'+kind+'_graph_1.txt'))
        # Only actual graph iterations; exclude statistics-collector eager ops.
        native=[]
        for path in (folder/'msprobe_graph').rglob('*.csv'):
            if path.parent.name in ('0','1','2') and path.parent.parent.parent.name=='graph_1':
                short='.'.join(path.name.split('.')[:2])+'.csv'
                dest=prefix/'msprobe_graph/raw/graph_1/1'/path.parent.name/short
                self.copy(path,dest);native.append(str(dest))
        assert len(native)>=3
        for file in ('completion.json','source_manifest.json','selected_cases.json','container_identity.json','resource_policy.json'):
            self.copy(root/file,prefix/'controller'/file)
        manual=ROOT/'outputs/manual_launch'/(name+'_'+suffix+'.json')
        if manual.is_file():self.copy(manual,prefix/'controller/prelaunch.json')
        launch=root/(name+'.vendor_launch.json')
        if launch.is_file():self.copy(launch,prefix/'controller/vendor_launch.json')
        # Original 1 KiB binary metadata is safe; the larger isolate record is
        # a labeled, field-specific excerpt with its original SHA, not raw dump.
        inv=json.loads((folder/'jit_compile_inventory.json').read_bytes())
        for item in inv['files']:
            if 'private_copy' not in item:continue
            path=folder/item['private_copy']
            assert sha(path)==item['sha256']
            if '_isolate_tiling_' in path.name:
                original=json.loads(path.read_bytes());info=original['tiling_info']
                keep=('block_num','task_ration','tiling_data','tiling_data_size','static_shape_flag',
                      'tiling_key','static_workspace_size','schedule_mode','tiling_info_completed')
                excerpt={'schema':'fault28.reviewed_compiler_tiling_excerpt.v1','original_file':item['file'],
                         'original_sha256':item['sha256'],'original_bytes':item['bytes'],
                         'optype':original['optype'],'is_static_shape':original['is_static_shape'],
                         'run_info':original['run_info'],'tiling_info':{k:info[k] for k in keep},
                         'generated_header_sha256':hashlib.sha256(info['file_content'].encode()).hexdigest(),
                         'scope':'Reviewed original-field excerpt; generated SDK header and unrelated static key registry omitted. Not an unmodified original isolate JSON or kernel argument read.'}
                save(self.destination/prefix/'compiler_tiling_excerpt.json',excerpt)
            else:self.copy(path,prefix/'compiler_binary_metadata.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--destination',type=Path,required=True)
    p.add_argument('--reports',type=Path,required=True);a=p.parse_args()
    acceptance=json.loads((a.reports/'acceptance.json').read_bytes())
    assert acceptance['status']=='PASS_CONTROLLED_FIA_GRAPH_CAUSAL_MECHANISM'
    neutrality=ROOT/'outputs/observer_neutrality_a1.json'
    assert json.loads(neutrality.read_bytes())['status']=='PASS_ALL_NUMERICAL_OUTPUTS_UNCHANGED'
    matrix=json.loads((ROOT/'setup/fp64_matrix_v1.json').read_bytes())
    export=Export(a.destination)
    for name in matrix['case_ids']:export.case(name,'fp64_a1',Path('cases')/name)
    for path in sorted(a.reports.glob('*.json')):export.copy(path,Path('reports')/path.name)
    export.copy(neutrality,Path('reports/observer_neutrality.json'))
    # A reproducible original failure, retained with native data and provenance.
    for name in ('g_m256_b1_r1','g_causal_m256_b1_r1'):
        export.case(name,'final_a1',Path('limited_fp32/cases')/name)
    for file in ('failure.json','causal_width256_b1_msprobe.json','causal_width256_b1_complete.json'):
        export.copy(ROOT/'outputs/final_acceptance_a1'/file,Path('limited_fp32/reports')/file)
    first=ROOT/'outputs'/(matrix['case_ids'][0]+'_fp64_a1')
    manifest=json.loads((first/'source_manifest.json').read_bytes())
    for name,digest in manifest.items():
        path=first/'source'/name;assert sha(path)==digest
        export.copy(path,Path('runtime_source')/name)
    build=ROOT/'build/evidence_a4'
    for name in ('build_completion.json','build_start.json','stage_completion.json','stage_start.json',
                 'policy_source_manifest.json','source_freeze.json','official_sources.patch','typed_fields.json',
                 'fia_abi_schema.json','fia_abi_compile.json','fia_abi_selftest.json','fia_schema_projection.h',
                 'selector_cpu_check.json','dependency_manifest.json','actual_offline_configure.json'):
        export.copy(build/name,Path('build')/name)
    export.copy(ROOT/'vendor_stage_a5/vendor_manifest.json',Path('build/vendor_manifest.json'))
    for name in ('stage_manifest.json','source_manifest.json','collector.patch'):
        export.copy(ROOT/'collector_fp64_a1'/name,Path('collector')/name)
    export.copy(ROOT/'collector_fp64_a1/packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py',
                Path('collector/pytorch_processor.py'))
    export.copy(ROOT/'outputs/collector_cpu_test_a2/collector_cpu_test.json',Path('collector/cpu_method_test.json'))
    save(a.destination/'source_mapping.json',{'schema':'fault28.original_statistical_file_mapping.v1',
         'files':export.mapping,'scope':'Original bytes preserved for mapped files; compiler_tiling_excerpt is explicitly derived. Long native filenames shortened without rewriting file contents.'})
    files=[path for path in sorted(a.destination.rglob('*')) if path.is_file()]
    assert all(path.suffix.lower() not in FORBIDDEN for path in files)
    assert sum(path.stat().st_size for path in files)<256*1024**2
    (a.destination/'SHA256SUMS').write_text(''.join(sha(path)+'  '+path.relative_to(a.destination).as_posix()+'\n' for path in files))
    archive=a.destination.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as stream:
        for path in sorted(a.destination.rglob('*')):
            if path.is_file():stream.add(path,arcname=path.relative_to(a.destination),recursive=False)
    print(json.dumps({'status':'EXPORTED_ALLOWLISTED_EVIDENCE','files':len(files)+1,'archive':str(archive),
                      'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size}))


if __name__=='__main__':main()
