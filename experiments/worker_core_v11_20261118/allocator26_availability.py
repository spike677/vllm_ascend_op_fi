"""Read-only physical-availability matrix; never admits or excludes a target.

The retained C1 review proves writer/consumer/source provenance separately.
This reader recomputes geometry from its original allocation files and keeps
all calls, including missing snapshots and the fault-contaminated request.
"""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath


def helper():
    path = Path(__file__).parents[1]/'c2c3_closure_20261117/review_native26_single_source54.py'
    spec = importlib.util.spec_from_file_location('_allocator26_reader', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = helper()


def geometry(snapshot, call):
    allocation, nonadjacent = B.allocation(snapshot, call)
    spans = []
    for index, segment in enumerate(snapshot):
        cursor = segment['address']
        for block in segment['blocks']:
            start = block.get('address', cursor)
            spans.append(dict(segment=index, segment_address=segment['address'],
                stream=segment.get('stream'), start=start, end=start+block['size'],
                requested_size=block.get('requested_size'), state=block['state']))
            cursor = start+block['size']
    matches = [[item for item in spans if item['state']=='active_allocated'
        and item['start'] <= call[name] and call[name]+call[name+'_bytes'] <= item['end']]
        for name in ('residual', 'output')]
    result = dict(status='ELIGIBLE_GEOMETRY' if allocation else
        'PROVEN_NONADJACENT' if nonadjacent else 'UNKNOWN_GEOMETRY', allocation=allocation,
        nonadjacent=nonadjacent, matches=matches, relation=None, signed_block_gap=None,
        intervening_blocks_by_state={})
    if all(len(items)==1 for items in matches):
        residual, output = (items[0] for items in matches)
        gap = output['start']-residual['end']
        same_segment = residual['segment']==output['segment']
        result.update(signed_block_gap=gap, relation='DIFFERENT_SEGMENTS' if not same_segment else
            'SAME_BLOCK' if residual==output else 'OUTPUT_BEFORE_RESIDUAL' if gap < 0 else
            'ADJACENT' if gap==0 else 'INTERVENING_BLOCKS')
        if same_segment and gap>0:
            result['intervening_blocks_by_state'] = dict(Counter(item['state'] for item in spans
                if item['segment']==residual['segment'] and residual['end']<=item['start']<output['start']))
    return result


def compare_sets(matrix, indices):
    sets = [{index for index, call in enumerate(matrix[row]['calls'])
             if call['geometry']['status']=='ELIGIBLE_GEOMETRY'} for row in indices]
    union = set.union(*sets) if sets else set()
    intersection = set.intersection(*sets) if sets else set()
    return dict(request_indices=indices, union=sorted(union), intersection=sorted(intersection),
        variable=sorted(union-intersection), counts=[len(items) for items in sets])


def review(root, result_sha, final_sha, expected_count):
    root = Path(root).resolve(strict=True)
    result_path = root/'probe/result.json'
    if B.file_sha(result_path)!=result_sha: raise ValueError('RESULT_CHANGED')
    driver = B.read(result_path)
    refs = driver['c1']['original_observations']['originals']
    ref = refs[-1]
    if ref['action']!='cleanup' or ref['sha256']!=final_sha: raise ValueError('FINAL_REFERENCE_CHANGED')
    relative = PurePosixPath(ref['path']).relative_to('/evidence')
    path = B.resolve_under(root, str(relative))
    if B.file_sha(path)!=final_sha: raise ValueError('FINAL_ORIGINAL_CHANGED')
    final = B.read(path); inv=final['inventory']; rows=inv['rows']; audit=B.Review()
    audit.check(inv['mechanism']=='26.residual_stomp' and len(rows)==8 and
        [row['phase'] for row in rows]==list(B.PHASES), 'ORIGINAL_SINGLE_SITE_PHASES')
    matrix=[]; artifacts=set(); first=rows[0]['candidates']
    for request_index, row in enumerate(rows):
        B.trace(audit,row,inv['lifetime'],row['phase'],mechanism=1)
        audit.check(len(row['candidates'])==expected_count and B.correlate(row,mechanism=1)==row['candidates'],
                    'FULL_NATIVE_DENOMINATOR_RECONCILED',request_index)
        calls=[]
        for ordinal, candidate in enumerate(row['candidates']):
            native=candidate['native']; cb=candidate['callback']; initial=first[ordinal]
            entry=dict(ordinal=ordinal, actual_call=native['call'], producer=native['producer'],
                consumers=[item['call'] for item in candidate['next_consumer']],
                original_execute_count=len(candidate['execution']), original_reasons=candidate['reasons'],
                target_key=cb.get('target_key') if cb else None,
                stable_formal_sha256=B.digest({k:native[k] for k in B.STABLE_NATIVE}),
                geometry=dict(status='NO_ALLOCATOR_OBSERVATION'))
            entry['same_formals_as_first']=all(native[k]==initial['native'][k] for k in B.STABLE_NATIVE)
            reference=cb.get('allocator_segments') if cb else None
            if reference:
                record=PurePosixPath(reference['path'])
                if not record.is_relative_to('/evidence'):raise ValueError('EXPLICIT_EVIDENCE_PATH_REQUIRED')
                artifact=B.resolve_under(root,str(record.relative_to('/evidence')))
                if B.file_sha(artifact)!=reference['sha256']:raise ValueError('ALLOCATOR_CHANGED')
                snapshot=B.read(artifact); actual=geometry(snapshot,native)
                audit.check(actual['allocation']==cb.get('allocation') and
                    (not cb.get('proven_nonadjacent') or actual['nonadjacent']==cb['proven_nonadjacent']),
                    'ACTUAL_ALLOCATION_MATCHES_RECORDED',str(request_index)+':'+str(ordinal))
                entry.update(geometry=actual,artifact_sha256=reference['sha256'],
                    original_full_snapshot_digest=reference['complete_snapshot_sha256'])
                artifacts.add(str(artifact))
            calls.append(entry)
        matrix.append(dict(index=request_index,request_id=row['request_id'],phase=row['phase'],
            mode=row['plan']['mode'],fault_request=row['plan']['mode']=='fault',calls=calls,
            geometry_counts=dict(Counter(item['geometry']['status'] for item in calls))))
    controls=[row['index'] for row in matrix if not row['fault_request'] and
        any(item['geometry']['status']!='NO_ALLOCATOR_OBSERVATION' for item in row['calls'])]
    stock=compare_sets(matrix,[0,1]); control=compare_sets(matrix,controls)
    changes=[]
    for index in stock['variable']:
        before,after=(matrix[i]['calls'][index] for i in (0,1))
        changes.append(dict(ordinal=index,target_key=before['target_key'],
            target_key_unchanged=before['target_key']==after['target_key'],
            formals_unchanged=before['stable_formal_sha256']==after['stable_formal_sha256'],
            statuses=[before['geometry']['status'],after['geometry']['status']],
            relations=[before['geometry']['relation'],after['geometry']['relation']],
            signed_block_gaps=[before['geometry']['signed_block_gap'],after['geometry']['signed_block_gap']]))
    return dict(schema='demoncat.writer26.allocator_availability_review.v1',**audit.summary(),
        result_sha256=result_sha,final_original_sha256=final_sha,expected_writers=expected_count,
        allocator_files_rehashed=len(artifacts),stock_geometry=stock,nonfault_observed_controls=control,
        stock_transitions=changes,matrix=matrix,read_only=True,actual_npu_calls=0,c2=False,c3=False,e5=False,
        limits=['Geometry availability is not lossless qualification or a live admission capability.',
            'All original calls and missing allocator observations remain in the matrix.',
            'The fault request is retained separately and excluded from healthy/control intersections.',
            'Only saved relevant segment bytes are rehashed; full allocator snapshot digest is a runtime receipt.',
            'Existing C1 review, not this geometric report, covers complete source/consumer/recovery provenance.'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-root',required=True,type=Path)
    p.add_argument('--result-sha256',required=True);p.add_argument('--final-sha256',required=True)
    p.add_argument('--expected-writers',type=int,required=True);p.add_argument('--out',required=True,type=Path)
    args=p.parse_args()
    result=review(args.evidence_root,args.result_sha256,args.final_sha256,args.expected_writers)
    with args.out.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('matrix','stock_transitions')}))
    return 0 if result['consistent'] else 2


if __name__=='__main__':raise SystemExit(main())
