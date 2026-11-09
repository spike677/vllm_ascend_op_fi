"""Round-trip public bytes, raw ABI, scalar text, and statistics-only inference.

The blind view physically excludes host/control/result/numeric files. Complete
native execution acceptance and scalar buffers still require remote evidence.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

FORBIDDEN={'.bin','.pt','.pth','.npy','.npz','.safetensors','.o','.so','.run','.log','.gz'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def extended(path):
    path=path.absolute()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--blind-work',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=extended(a.bundle).resolve(strict=True);work=extended(a.blind_work)
    assert not work.exists() and not work.resolve().is_relative_to(root)
    work.mkdir(parents=True)
    sums={}
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);assert name not in sums and '..' not in Path(name).parts
        sums[name]=digest
    actual={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file()}
    assert set(actual)==set(sums)|{'SHA256SUMS'}
    for name,file in actual.items():
        assert not file.is_symlink() and file.resolve().is_relative_to(root)
        assert file.suffix.lower() not in FORBIDDEN
        assert not any(p in ('private_output','private_device_debug','inputs','__pycache__') for p in file.relative_to(root).parts)
        if name!='SHA256SUMS':assert sha(file)==sums[name],name
    mapping=json.loads((root/'source_mapping.json').read_bytes())
    for row in mapping['files']:
        file=actual[row['published']]
        assert file.stat().st_size==row['bytes'] and sums[row['published']]==row['sha256'] and row['original_bytes_preserved']
    assert len(list((root/'cases').iterdir()))==23
    for case in (root/'cases').iterdir():
        config=json.loads((case/'config.json').read_bytes())
        assert config['task']=='statistics' and config['level']=='mix' and config['statistics']['summary_mode']=='statistics'
        for step in (0,1):
            for kind in ('dump','stack','construct'):assert (case/f'observation/step{step}/rank0/{kind}.json').is_file()
    sys.path.insert(0,str(root/'analysis'))
    import device_trace,public_plan,graph_evidence
    matrix=json.loads((root/'device/summary.json').read_bytes());measured=0;scalar_records=0
    schema=json.loads((root/'build/fia_abi_schema.json').read_bytes())
    for row in matrix['rows']:
        if row['status']=='STOCK_NO_DEVICE_OBSERVER':continue
        name=row['case_id'];folder=root/'device'/name
        report=json.loads((folder/'device_columns.json').read_bytes())
        actual_plan=public_plan.verify(root/'cases'/name,schema)[0]
        assert report['actual_m']==actual_plan['decoded']['innerSplitParams.mBaseSize']
        assert report['actual_key']==actual_plan['actual_host_key']
        sdk=[]
        for line in (folder/'official_sdk_stdout.txt').read_text(encoding='utf-8').splitlines():
            parsed=device_trace.parse(line)
            if parsed:sdk.append(parsed)
        payload=(folder/'runtime_device_lines.txt').read_bytes();parts=payload.splitlines(keepends=True)
        provenance=json.loads((folder/'runtime_line_provenance.json').read_bytes())
        assert len(parts)==len(provenance['original_lines'])==report['canonical_scalar_records']
        assert provenance['original_log_sha256']==report['runtime_log_sha256']
        runtime=[];owner=json.loads((root/'cases'/name/'result.json').read_bytes())['owner_pid']
        for part,reference in zip(parts,provenance['original_lines']):
            assert hashlib.sha256(part).hexdigest()==reference['sha256']
            line=part.decode('utf-8');identity=device_trace.RUNTIME.search(line)
            assert identity is not None and int(identity[1])==owner
            parsed=device_trace.parse(line);assert parsed is not None;runtime.append(parsed)
        counts=Counter(map(device_trace.key,sdk))
        assert counts==Counter(map(device_trace.key,runtime)) and set(counts.values())=={3}
        unique={device_trace.key(r):r for r in sdk}
        points=device_trace.expand(list(unique.values()),report['actual_m'],int(report['actual_causal_variant']))
        assert points==report['query_tiles'] and report['individual_scalar_to_graph_call_pairing_proven'] is False
        measured+=1;scalar_records+=len(sdk)
    assert measured==19 and scalar_records==798
    roles=json.loads((root/'primary_BF.json').read_bytes());blind={}
    for role in ('baseline','fault','fault_repeat'):
        source=root/roles[role];target=work/source.name;target.mkdir()
        # These are the complete permitted inference inputs. No case/result,
        # host events, selected M, device traces, private tensors or hit files.
        for name in ('config.json','collector_provenance.json','fx_graph_0.json'):
            shutil.copyfile(source/name,target/name)
        shutil.copytree(source/'observation',target/'observation')
        shutil.copytree(source/'msprobe_graph',target/'msprobe_graph')
        blind[role]=target
    reviews=[]
    for role in ('fault','fault_repeat'):
        report=graph_evidence.compare(blind['baseline'],blind[role])
        assert all(r['status']=='ALIGNED_OBSERVED_FIRST_DIFFERENCE' and
                   r['first']['type']=='FusedInferAttentionScore' and r['first']['boundary']=='output'
                   and r['first']['index']==0 for r in report['independent_first_producer'])
        target=work/(role+'_blind_first.json')
        with target.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
        reviews.append({'role':role,'report_file':str(target),'sha256':sha(target),
                        'first':report['independent_first_producer']})
    result={'schema':'fault28.fia.public_measured_evidence_review.v1','status':'PASS_PUBLIC_BYTES_ABI_SCALARS_AND_BLIND_FIRST',
        'files':len(actual),'bytes':sum(p.stat().st_size for p in actual.values()),'mapped_originals':len(mapping['files']),
        'source_mapping_sha256':sha(root/'source_mapping.json'),'formal_statistics_triplets':46,
        'measured_private_cases':measured,'actual_scalar_records':scalar_records,'blind_reviews':reviews,
        'blind_inputs_exclude_control_and_device_metadata':True,'private_tensor_files_opened':0,'native_calls':0,
        'limits':['Public review does not reread private complete DEBUG logs, binary code, scalar buffers or target tensors.',
                  'Scalar records remain aggregate process/graph observations, not per-replay timing assignments.'],
        'analysis_source_sha256':sha(Path(__file__))}
    output=extended(a.output);assert not output.exists() and not output.resolve().is_relative_to(root)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({k:result[k] for k in ('status','files','bytes','mapped_originals','measured_private_cases','actual_scalar_records')}))


if __name__=='__main__':main()
