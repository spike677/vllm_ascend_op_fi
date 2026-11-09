"""Reviewed statistics/graph/metadata export; full tensors and logs stay private."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from coverage import ROOT

FORBIDDEN={'.pt','.pth','.bin','.npy','.npz','.safetensors','.o','.so','.run','.log','.gz'}
ALLOWED={'.json','.jsonl','.csv','.txt','.patch','.h','.cpp','.py','.sh'}
PRIMARY={'model_b8_baseline','model_b8_fault_r1','model_b8_fault_r2'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


class Export:
    def __init__(self,destination):
        self.destination=destination;self.mapping=[];destination.mkdir(parents=True,exist_ok=False)

    def copy(self,source,relative):
        target=self.destination/relative
        assert source.is_file() and not source.is_symlink()
        assert source.suffix.lower() in ALLOWED and source.suffix.lower() not in FORBIDDEN
        assert source.stat().st_size<=64*1024**2
        assert not target.exists() and target.resolve().is_relative_to(self.destination.resolve())
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        assert sha(target)==sha(source)
        self.mapping.append({'source':str(source),'published':relative.as_posix(),'bytes':target.stat().st_size,
            'sha256':sha(target),'original_bytes_preserved':True})

    def case(self,name):
        attempt=ROOT/'outputs'/(name+'_a5');folder=attempt/name;prefix=Path('cases')/name
        names=('case.json','result.json','inputs.json','config.json','graph_config.json','fx_graph_0.json',
            'output_contract.json','model_config.json','loaded_parameters.json','parameters_receipt.json',
            'compiled_engine_bindings.json','collector_provenance.json','statistics_receipt.json',
            'host_events.jsonl','jit_compile_inventory.json')
        for n in names:self.copy(folder/n,prefix/n)
        for step in (0,1):
            for kind in ('dump','stack','construct'):
                rel=Path('observation')/f'step{step}'/'rank0'/(kind+'.json');self.copy(folder/rel,prefix/rel)
        for kind in ('original','optimized'):
            paths=list((folder/'msprobe_graph').rglob('dynamo_'+kind+'_graph_*.txt'))
            assert len(paths)==1;self.copy(paths[0],prefix/'msprobe_graph'/('dynamo_'+kind+'_graph_1.txt'))
        available=0;published=0
        for p in sorted((folder/'msprobe_graph').rglob('*.csv')):
            if p.parent.name not in ('0','1','2') or p.parent.parent.parent.name!='graph_1':continue
            available+=1
            if name not in PRIMARY and not p.name.startswith('FusedInferAttentionScore.'):continue
            # Preserve multiple observations of a shared Data node. Native
            # filename suffixes map through source_mapping; CSV bytes unchanged.
            short='.'.join(p.name.split('.')[:2])+f'.{published:05d}.csv'
            relative=prefix/'msprobe_graph/raw/graph_1'/p.parent.parent.name/p.parent.name/short
            self.copy(p,relative);published+=1
        assert published>=84
        save(self.destination/prefix/'public_ge_selection.json',{'available_actual_graph_csv':available,
            'published_csv':published,'selection':'all graph nodes' if name in PRIMARY else 'FIA nodes only; complete original statistics/graphs and private CPU checks retained',
            'repeated_csv_observations_preserved':True})
        for n in ('completion.json','source_manifest.json','selected_cases.json','container_identity.json',
                  'resource_policy.json','private_batch_storage.json'):
            self.copy(attempt/n,prefix/'controller'/n)
        self.copy(attempt/(name+'.vendor_launch.json'),prefix/'controller/vendor_launch.json')
        for source,target in [('native_lines.txt','native_lines.txt'),('provenance.json','native_provenance.json')]:
            self.copy(attempt/'native_extract'/source,prefix/target)
        inventory=json.loads((folder/'jit_compile_inventory.json').read_bytes())
        for row in inventory['files']:
            if not row.get('private_copy'):continue
            p=folder/row['private_copy'];assert sha(p)==row['sha256']
            if '_isolate_tiling_' in p.name:
                d=json.loads(p.read_bytes());info=d['tiling_info']
                fields=('block_num','task_ration','tiling_data','tiling_data_size','static_shape_flag',
                    'tiling_key','static_workspace_size','schedule_mode','tiling_info_completed','raw_run_info')
                save(self.destination/prefix/'compiler_tiling_excerpt.json',{
                    'schema':'fault28.reviewed_compiler_tiling_excerpt.v1','original_file':row['file'],
                    'original_sha256':row['sha256'],'original_bytes':row['bytes'],'optype':d['optype'],
                    'is_static_shape':d['is_static_shape'],'run_info':d['run_info'],
                    'tiling_info':{k:info[k] for k in fields},
                    'generated_header_sha256':hashlib.sha256(info['file_content'].encode()).hexdigest(),
                    'scope':'Field-specific original metadata excerpt, not an unmodified isolate JSON or device-argument read.'})
            else:self.copy(p,prefix/'compiler_binary_metadata.json')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reports',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    acceptance=json.loads((a.reports/'acceptance.json').read_bytes())
    assert acceptance['status']=='PASS_FULLMODEL_FIA_COVERAGE_FIRST_DIFFERENCE_AND_RESTORE'
    assert acceptance['first_token_logits_propagation'] and acceptance['all_eight_request_logits_changed']
    assert json.loads((a.reports/'byte_digest_controls.json').read_bytes())['status']=='PASS_DENSE_COMPARISONS_AND_ALL_REQUEST_BYTE_DIGESTS'
    matrix=json.loads((ROOT/'setup/matrix_v5.json').read_bytes());export=Export(a.destination)
    sources=None
    for name in matrix['case_ids']:
        export.case(name)
        directory=ROOT/'outputs'/(name+'_a5');record=json.loads((directory/'source_manifest.json').read_bytes())
        if sources is None:
            sources=record
            for rel,digest in record.items():
                source=directory/'source'/rel;assert sha(source)==digest
                export.copy(source,Path('runtime_source')/rel)
        else:assert record==sources,'Different frozen runtime among accepted model cases'
    for file in sorted(a.reports.rglob('*.json')):
        export.copy(file,Path('reports')/file.relative_to(a.reports))
    for file in sorted((ROOT/'analysis').glob('*.py')):export.copy(file,Path('analysis')/file.name)
    for name in ('analytic_tests_a3.json','analysis_repeated_csv_limit_a1.json','analysis_output_tree_limit_a1.json','log_storage_cpu_a1.json'):
        export.copy(ROOT/'outputs'/name,Path('diagnostics')/name)
    for version in ('a1','a2'):
        for name in ('inventory.json','completion.json','fx_code.py'):
            export.copy(ROOT/'outputs'/('cpu_export_'+version)/name,Path('diagnostics')/('cpu_export_'+version)/name)
    for version in range(1,5):
        source=ROOT/'outputs'/('model_b1_baseline_a'+str(version));prefix=Path('diagnostics')/('attempt_a'+str(version))
        for name in ('completion.json','source_manifest.json','post_failure_ownership_audit.json','model_b1_baseline/result.json'):
            if (source/name).is_file():export.copy(source/name,prefix/name)
        for name in ('controller.py','graph_probe.py','env.sh'):
            if (source/'source'/name).is_file():export.copy(source/'source'/name,prefix/'source'/name)
    for name in ('fia_abi_schema.json','fia_abi_compile.json','fia_abi_selftest.json',
                 'policy_source_manifest.json','official_sources.patch','fia_schema_projection.h'):
        export.copy(ROOT/'build/evidence_a4'/name,Path('build')/name)
    for name in ('device_grain_trace.json','device_grain_trace.patch'):
        export.copy(ROOT/'build'/name,Path('build')/name)
    export.copy(ROOT/'vendor_stage_a5/vendor_manifest.json',Path('build/vendor_manifest.json'))
    for name in ('stage_manifest.json','source_manifest.json','collector.patch'):
        export.copy(ROOT/'collector_fp64_a1'/name,Path('collector')/name)
    export.copy(ROOT/'collector_fp64_a1/packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py',
                Path('collector/pytorch_processor.py'))
    for name in ('model_manifest.json','tokenizer_manifest.json'):
        export.copy(ROOT/name,Path('build')/name)
    save(a.destination/'primary_BF.json',{'baseline':'cases/model_b8_baseline','fault':'cases/model_b8_fault_r1',
        'fault_repeat':'cases/model_b8_fault_r2','off_restore':'cases/model_b8_restore',
        'same_batch':8,'same_causal_rule':True,'independent_analysis':'analysis/graph_evidence.py'})
    save(a.destination/'source_mapping.json',{'schema':'fault28.original_statistical_file_mapping.v1',
        'files':export.mapping,'scope':'Original mapped file bytes preserved; explicitly labeled compiler field excerpts and raw-log line subsets. Full tensors, weights, buffers, binaries and raw logs private.'})
    files=[f for f in sorted(a.destination.rglob('*')) if f.is_file()]
    assert sum(p.stat().st_size for p in files)<512*1024**2
    assert all(p.suffix.lower() not in FORBIDDEN for p in files)
    (a.destination/'SHA256SUMS').write_text(''.join(sha(f)+'  '+f.relative_to(a.destination).as_posix()+'\n' for f in files))
    archive=a.destination.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as stream:
        for f in sorted(a.destination.rglob('*')):
            if f.is_file():stream.add(f,arcname=f.relative_to(a.destination),recursive=False)
    print(json.dumps({'status':'EXPORTED_FULLMODEL_FIA_STATISTICAL_EVIDENCE','files':len(files)+1,
        'bytes':sum(p.stat().st_size for p in a.destination.rglob('*') if p.is_file()),
        'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size}))


if __name__=='__main__':main()
