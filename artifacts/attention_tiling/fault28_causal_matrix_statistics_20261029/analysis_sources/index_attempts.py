"""Index explicitly selected successful attempts for read-only analysis.

Creates only symlinks and a SHA-bound JSON mapping in a new directory; original
captures, logs, completion files and source snapshots are never copied/edited.
The symlinks do not themselves make their targets read-only. Consumers must
remain read-only and retain the source hashes. No device/library import occurs.
"""
import argparse
import json
import os
from pathlib import Path
import re

import profile_evidence as pe
from causal_core import require


def read(path):
    return json.loads(path.read_bytes())


def plan(matrix_path, attempts, profiles=None):
    matrix_path=matrix_path.resolve(strict=True)
    matrix=read(matrix_path)
    require(matrix['schema']=='fault28.causal.matrix.v1','Unrecognized frozen matrix')
    desired=[x for x in matrix['cases'] if not profiles or x['profile'] in profiles]
    require(desired,'No selected cases')
    by_id={x['case_id']:x for x in desired}
    require(len(by_id)==len(desired),'Duplicate matrix IDs')
    for x in desired:
        require(re.fullmatch(r'[A-Za-z0-9_-]{1,60}',x['case_id']) is not None,'Unsafe case ID')
        config=(matrix_path.parent/x['file']).resolve(strict=True)
        require(config.is_relative_to(matrix_path.parent) and pe.sha(config)==x['sha256'],'Frozen matrix case differs')
    seen={};source_sha=None;links=[];sources=[];unused=[]
    for attempt in attempts:
        attempt=attempt.resolve(strict=True)
        completion_path=attempt/'completion.json';completion=read(completion_path)
        require(completion['status']=='CAPTURE_COMPLETED' and completion.get('cleanup_errors')==[] and
                completion.get('owned_container_removed') is True and completion.get('device_lock_released') is True,
                'Attempt is not successfully completed and cleaned: '+str(attempt))
        frozen_path=attempt/'source_manifest.json';frozen=read(frozen_path)
        current_sha=pe.sha(frozen_path)
        require(source_sha is None or current_sha==source_sha,'Source snapshots differ across selected attempts')
        source_sha=current_sha
        selected={x['case_id']:x for x in read(attempt/'selected_cases.json')}
        cases=completion['cases']
        require(len({x['case_id'] for x in cases})==len(cases),'Duplicate cases in completion')
        for entry in cases:
            case_id=entry['case_id']
            require(entry['status']=='CAPTURE_COMPLETED_NOT_MECHANISM_ACCEPTANCE' and entry['rc']==0,'Incomplete case in successful attempt')
            if case_id not in by_id:
                unused.append({'attempt':str(attempt),'case_id':case_id})
                continue
            require(case_id not in seen,'Duplicate case across attempts; select an exact successful run')
            item=by_id[case_id]
            case_dir=(attempt/case_id).resolve(strict=True)
            require(case_dir.is_dir() and case_dir.parent==attempt,'Unexpected case directory')
            result_path=case_dir/'result.json';result=read(result_path)
            require(result['status']=='CAPTURE_COMPLETED' and result['case_id']==case_id and
                    pe.sha(result_path)==entry['result_sha256'] and result['source_case_sha256']==item['sha256'],'Runtime/completion/case SHA mismatch')
            relative=selected[case_id]['file']
            source=(attempt/'source'/relative).resolve(strict=True)
            require(source.is_relative_to((attempt/'source').resolve()) and frozen[relative]==item['sha256']==pe.sha(source),'Executed frozen config differs from requested matrix')
            seen[case_id]={'attempt':str(attempt),'result_sha256':pe.sha(result_path),'case_sha256':item['sha256']}
            links.append({'name':case_id,'target':str(case_dir),'is_directory':True,
                          'result_sha256':pe.sha(result_path)})
            for suffix in ('.log','.controller.json','.vendor_launch.json'):
                source_file=attempt/(case_id+suffix)
                require(source_file.is_file() or suffix=='.vendor_launch.json','Missing required sidecar '+suffix)
                if source_file.exists():
                    links.append({'name':case_id+suffix,'target':str(source_file.resolve(strict=True)),
                                  'is_directory':False,'sha256':pe.sha(source_file),'bytes':source_file.stat().st_size})
        sources.append({'attempt':str(attempt),'completion_sha256':pe.sha(completion_path),
                        'source_manifest_sha256':current_sha,'selected_cases_sha256':pe.sha(attempt/'selected_cases.json')})
    require(set(seen)==set(by_id),'Selected attempts do not cover every requested case: '+','.join(sorted(set(by_id)-set(seen))))
    return {'schema':'fault28.causal.attempt_index.v1','status':'VERIFIED_INDEX_PLAN_NOT_MECHANISM_ACCEPTANCE',
            'matrix_file':str(matrix_path),'matrix_sha256':pe.sha(matrix_path),'profile_filter':profiles or [],
            'cases':seen,'attempts':sources,'unselected_completed_cases':unused,'links':links,
            'original_evidence_modified':False,'symlink_does_not_enforce_readonly_target_permissions':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--attempt',type=Path,action='append',required=True)
    parser.add_argument('--profile',action='append')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    report=plan(args.matrix,args.attempt,args.profile)
    if args.plan_only:
        print(json.dumps(report,indent=2,ensure_ascii=True))
        return 0
    require(os.name=='posix','Create the symlink index on the Linux evidence host; Windows can use --plan-only')
    require(args.output is not None and not args.output.exists(),'Output index must be a new directory')
    output=args.output.resolve()
    require(all(not output.is_relative_to(Path(x['attempt'])) for x in report['attempts']), 'Do not place derived index inside original attempt')
    output.mkdir(parents=True,exist_ok=False)
    # No original file is written or chmodded; a partial failure leaves a
    # visibly incomplete new index without claiming materialization success.
    for item in report['links']:
        os.symlink(item['target'],output/item['name'],target_is_directory=item['is_directory'])
    report['status']='CREATED_READONLY_CONSUMER_INDEX'
    (output/'index_manifest.json').write_text(json.dumps(report,indent=2,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'status':report['status'],'cases':len(report['cases']),
                      'index_manifest_sha256':pe.sha(output/'index_manifest.json'),'output':str(output)}))
    return 0


if __name__=='__main__':raise SystemExit(main())
