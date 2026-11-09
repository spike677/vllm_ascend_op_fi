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
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--chain',type=Path,required=True)
    p.add_argument('--device',type=Path,required=True);p.add_argument('--neutrality',type=Path,required=True)
    p.add_argument('--suffix',default='grain_a2');a=p.parse_args()
    acceptance=json.loads((a.reports/'acceptance.json').read_bytes())
    assert acceptance['status']=='PASS_CONTROLLED_FIA_GRAPH_CAUSAL_MECHANISM'
    chain=json.loads((a.chain/'acceptance.json').read_bytes())
    assert chain['status']=='PASS_CONTROLLED_FIA_GRAPH_MEASURED_CAUSAL_CHAIN'
    assert json.loads(a.neutrality.read_bytes())['status']=='PASS_ALL_NUMERICAL_OUTPUTS_UNCHANGED'
    matrix=json.loads((ROOT/'setup/fp64_matrix_v1.json').read_bytes());export=Export(a.destination)
    source_files=None
    for name in matrix['case_ids']:
        export.case(name,a.suffix,Path('cases')/name)
        attempt=ROOT/'outputs'/(name+'_'+a.suffix)
        source=json.loads((attempt/'source_manifest.json').read_bytes())
        if source_files is None:
            source_files=source
            for relative,digest in source.items():
                path=attempt/'source'/relative;assert sha(path)==digest
                export.copy(path,Path('runtime_source')/relative)
        else:assert source_files==source,'All accepted cases must use the same frozen capture source'
    for path in sorted(a.reports.glob('*.json')):export.copy(path,Path('reports/execution')/path.name)
    for path in sorted(a.chain.glob('*.json')):export.copy(path,Path('reports/measured_chain')/path.name)
    export.copy(a.neutrality,Path('reports/observer_neutrality.json'))
    for gate in ('bridge_gate_grain_a2.json','base_variant_gate_grain_a2.json'):
        export.copy(ROOT/'outputs'/gate,Path('reports/gates')/gate)
    device_summary=json.loads((a.device/'summary.json').read_bytes())
    export.copy(a.device/'summary.json',Path('device/summary.json'))
    remote_only=[]
    for row in device_summary['rows']:
        if row['status']=='STOCK_NO_DEVICE_OBSERVER':continue
        name=row['case_id'];folder=a.device/name;prefix=Path('device')/name
        export.copy(folder/'device_columns.json',prefix/'device_columns.json')
        receipt=json.loads((folder/'sdk_receipt.json').read_bytes())
        export.copy(folder/'sdk_receipt.json',prefix/'sdk_receipt.json')
        official=Path(receipt['saved_original_sdk_stdout'])
        assert sha(official)==receipt['decoder_stdout_sha256']
        export.copy(official,prefix/'official_sdk_stdout.txt')
        report=json.loads((folder/'device_columns.json').read_bytes())
        original=ROOT/'outputs'/(name+'_'+a.suffix)/(name+'.log')
        assert sha(original)==report['runtime_log_sha256']
        selected={r['line']:r for r in report['runtime_records']};parts=[];references=[];offset=0
        with original.open('rb') as stream:
            for line_number,payload in enumerate(stream,1):
                if line_number in selected:
                    assert hashlib.sha256(payload).hexdigest()==selected[line_number]['sha256']
                    parts.append(payload);references.append({'line':line_number,'byte_range':[offset,offset+len(payload)],
                        'sha256':hashlib.sha256(payload).hexdigest()})
                offset+=len(payload)
        assert len(parts)==len(selected)
        generated=ROOT/'outputs/public_extracts'/name;generated.mkdir(parents=True,exist_ok=False)
        literal=generated/'runtime_device_lines.txt';literal.write_bytes(b''.join(parts))
        provenance=generated/'runtime_line_provenance.json'
        save(provenance,{'kind':'verbatim_original_runtime_line_subset','original_log':str(original),
            'original_log_sha256':report['runtime_log_sha256'],'original_log_bytes':offset,'original_lines':references,
            'scope':'A literal line subset, not a complete DEBUG log. No scalar values generated or rewritten.'})
        export.copy(literal,prefix/'runtime_device_lines.txt')
        export.copy(provenance,prefix/'runtime_line_provenance.json')
        remote_only.extend({'path':r['file'],'sha256':r['sha256'],'bytes':r['bytes'],
                            'kind':'raw_device_scalar_buffer','published':False} for r in receipt['source_files'])
    for path in sorted((ROOT/'analysis').glob('*.py')):export.copy(path,Path('analysis')/path.name)
    for path in sorted((ROOT/'analysis/fixtures').glob('*.txt')):export.copy(path,Path('analysis/fixtures')/path.name)
    for name in ('analytic_tests_linux_a1.json','pilot_execution_a1.json','pilot_device_evidence_a2.json'):
        file=ROOT/'outputs'/name
        if file.is_file():export.copy(file,Path('reports/validation_or_limits')/name)
    # Preserve the original capture gap and its source rather than replacing it.
    early=ROOT/'outputs/g_causal_m128_b8_r1_grain_a1'
    for relative in ('completion.json','source_manifest.json','source/graph_probe.py',
                     'g_causal_m128_b8_r1/jit_compile_inventory.json'):
        export.copy(early/relative,Path('superseded_capture_a1')/relative)
    for name in ('policy_source_manifest.json','official_sources.patch','fia_abi_schema.json',
                 'fia_abi_compile.json','fia_abi_selftest.json','fia_schema_projection.h'):
        export.copy(ROOT/'build/evidence_a4'/name,Path('build')/name)
    for name in ('device_grain_trace.json','device_grain_trace.patch'):
        export.copy(ROOT/'build'/name,Path('build')/name)
    export.copy(ROOT/'vendor_stage_a5/vendor_manifest.json',Path('build/vendor_manifest.json'))
    for name in ('stage_manifest.json','source_manifest.json','collector.patch'):
        export.copy(ROOT/'collector_fp64_a1'/name,Path('collector')/name)
    export.copy(ROOT/'collector_fp64_a1/packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py',
                Path('collector/pytorch_processor.py'))
    save(a.destination/'primary_BF.json',{'baseline':'cases/g_causal_base512_b8_r1','fault':'cases/g_batch_b8_r1',
         'fault_repeat':'cases/g_batch_b8_r2','off_restore':'cases/g_restore_b8_r1',
         'same_batch':8,'same_causal_rule':True,'independent_analysis':'analysis/graph_evidence.py'})
    save(a.destination/'source_mapping.json',{'schema':'fault28.original_statistical_file_mapping.v1',
         'files':export.mapping,'remote_only_scalar_buffers':remote_only,
         'scope':'Mapped originals are byte preserved. Compiler excerpts and runtime line subsets are explicitly labeled. Private tensor/model/binary/complete-log files are excluded.'})
    files=[path for path in sorted(a.destination.rglob('*')) if path.is_file()]
    assert all(path.suffix.lower() not in FORBIDDEN for path in files)
    assert sum(path.stat().st_size for path in files)<256*1024**2
    (a.destination/'SHA256SUMS').write_text(''.join(sha(path)+'  '+path.relative_to(a.destination).as_posix()+'\n' for path in files))
    archive=a.destination.with_suffix('.tar.gz')
    with tarfile.open(archive,'x:gz') as stream:
        for path in sorted(a.destination.rglob('*')):
            if path.is_file():stream.add(path,arcname=path.relative_to(a.destination),recursive=False)
    print(json.dumps({'status':'EXPORTED_MEASURED_FIA_STATISTICS','files':len(files)+1,
          'bytes':sum(p.stat().st_size for p in a.destination.rglob('*') if p.is_file()),
          'archive':str(archive),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size}))


if __name__=='__main__':main()
