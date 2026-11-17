"""Deterministic registry views. --check refuses generated-file drift."""
import csv
import io
import json
from . import ROOT, definitions


def generated_files():
    from fault_injection.unified.config import example_config
    rows=definitions();files={}
    dump=lambda value:json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    fields=('canonical_id','legacy_id','name','family','primitive','backends','adapter_status','historical_c2','historical_c3')
    stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for row in rows:
        values={k:row[k] for k in fields if k in row};values['backends']='|'.join(row['backends'])
        values.update(historical_c2=row['historical']['c2'],historical_c3=row['historical']['c3']);writer.writerow(values)
        files['examples/'+row['canonical_id'].lower()+'.json']=dump(example_config(row['legacy_id']))
        card=['# '+row['name'],'','Generated from registry.yaml. Historical evidence is bounded to the recorded scope.','']
        contents={
            'Fault ID':row['fault_id'],'Mechanism ID':row['canonical_id'],'Compatibility ID':row['legacy_id'],
            'Fault Name':row['name'],'Fault Family':row['family'],'Root Cause':row['root_cause'],
            'Simulated Semantics':row['simulated_semantics'],'Injection Primitive':row['primitive'],
            'Injection Layer':', '.join(row['backends']),'Operator':row['operator_paths'][0]['family'],
            'Model Path':row['model_path'],'Trigger Condition':'After actual eligibility and target filtering; see common API contract.',
            'Parameters':json.dumps(row['parameters'],ensure_ascii=False),'Parameter Schema':row['schema'],
            'Observable Effect':row['legacy_details'][3],'Recovery':row['recovery'],
            'Evidence':row['historical']['evidence_paths'],'Limitations':row['legacy_catalog']['limitation'],
            'Unified Execution':row['adapter_status']+': '+row['adapter_reason']}
        for key,value in contents.items():card.extend(['**'+key+'**: '+value,''])
        files['generated/cards/'+row['canonical_id'].lower()+'.md']='\n'.join(card)
    files['generated/mechanism_matrix.csv']=stream.getvalue()
    files['generated/adapter_registry.json']=dump(dict(schema='demoncat.adapter_registry.v1',contract_version='1',
        mechanisms=[{k:r[k] for k in ('canonical_id','legacy_id','backends','adapter_status','adapter_reason','schema','operator_paths','compatibility')} for r in rows]))
    # Queue view describes current integration work, never generates a runnable NPU command.
    files['generated/npu_execution_queue.json']=dump(dict(schema='demoncat.unified_delta_queue.v1',
        jobs=[dict(mechanism=r['legacy_id'],adapter_status=r['adapter_status'],historical_scope=r['historical'],
            requires=['unified_backend_bridge','exact_model_sdk_abi','resource_approval','B/F/OFF/R','independent_audit'],
            executable=False,npu_qualified=False) for r in rows]))
    files['generated/mechanism_catalog.md']='\n'.join(['# Unified mechanism catalog','','Generated from registry.yaml. 17 registrations do not imply 17 executable bridges.','',
        '| Mechanism | Legacy ID | Unified execution |','| --- | --- | --- |']+
        ['| ['+r['canonical_id']+'](cards/'+r['canonical_id'].lower()+'.md) | '+r['legacy_id']+' | '+r['adapter_status']+' |' for r in rows])+ '\n'
    files['generated/api_reference.md']='\n'.join(['# API and configuration reference','','`import demoncat`; `demoncat.configure(mechanism, config)` returns an owner-scoped session.',
        '`python -m demoncat catalog|example|validate|inspect|generate --check` exports and checks definitions.',
        'All session operations return `demoncat.operation_result.v1` with ok/code/state/owner/generation.',
        'See ../docs/api.md for typed Request, lifecycle and Worker usage.','',
        '| Mechanism | Parameters | Schema |','| --- | --- | --- |']+
        ['| '+r['legacy_id']+' | '+', '.join(r['parameters'])+' | '+r['schema']+' |' for r in rows])+'\n'
    return files


def generate(*,check=False):
    files=generated_files();drift=[]
    for name,content in files.items():
        path=ROOT/name
        if check:
            if not path.is_file() or path.read_text(encoding='utf-8')!=content:drift.append(name)
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(content,encoding='utf-8',newline='\n')
    if drift:raise ValueError('GENERATED_VIEW_DRIFT:'+','.join(drift))
    return dict(files=len(files),mechanisms=len(definitions()),checked=check)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    print(json.dumps(generate(check=args.check)))
