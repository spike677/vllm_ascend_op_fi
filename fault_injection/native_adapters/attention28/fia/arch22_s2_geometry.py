"""Independent CPU checks of the actual arch22 SplitCore result and FD spans."""
from collections import Counter

VARIANT='tnd_page16_arch22_s2_256'
ARRAYS=('bN2End','gS1End','s2End','bN2IdxOfFdHead','gS1IdxOfFdHead','s2SplitNumOfFdHead',
        's2SplitStartIdxOfCore','gS1SplitNumOfFdHead','gS1LastPartSizeOfFdHead',
        'gS1IdxEndOfFdHead','gS1IdxEndOfFdHeadSplit')


def require(value,reason):
    if not value:raise ValueError(reason)


def validate_split(row):
    aic,kv,tile=(row[name] for name in ('aic','kv_length','s2'))
    require(all(type(x) is int for x in (aic,kv,tile)) and 16<=aic<=26
            and 1025<=kv<=2048 and tile in (256,512),'ARCH22_S2_CPU_SCOPE_DIFFERS')
    used=row['usedCoreNum'];fd=row['numOfFdHead'];vec=row['usedVecNumOfFd']
    require(all(type(x) is int for x in (used,fd,vec)) and 1<=used<=aic and 0<fd<=used,
            'ARCH22_S2_CORE_OR_FD_BOUND')
    for name in ARRAYS:
        width=2*aic if name in ('gS1IdxEndOfFdHead','gS1IdxEndOfFdHeadSplit') else aic
        require(type(row[name]) is list and len(row[name])==width
                and all(type(x) is int and x>=0 for x in row[name]),'ARCH22_S2_COMPLETE_ARRAY:'+name)
    loops=(kv+tile-1)//tile;previous=0;pieces={head:[] for head in range(8)};per_core=[]
    for core in range(used):
        head,g,s2=(row[name][core] for name in ('bN2End','gS1End','s2End'))
        require(g==0 and 0<=head<=8 and 0<=s2<loops and (head<8 or s2==0),
                'ARCH22_S2_OUTER_ENDPOINT')
        end=head*loops+s2
        require(previous<end<=8*loops,'ARCH22_S2_EMPTY_OVERLAPPING_OR_OUTSIDE_CORE')
        begin=previous
        for h in range(begin//loops,(end-1)//loops+1):
            pieces[h].append((core,max(begin,h*loops)-h*loops,min(end,(h+1)*loops)-h*loops))
        per_core.append((begin,end));previous=end
    require(previous==8*loops,'ARCH22_S2_INCOMPLETE_ALL_HEADS')
    heads=[h for h in range(8) if len(pieces[h])>1]
    require(fd==len(heads) and row['bN2IdxOfFdHead'][:fd]==heads
            and row['gS1IdxOfFdHead'][:fd]==[0]*fd
            and row['s2SplitNumOfFdHead'][:fd]==[len(pieces[h]) for h in heads],
            'ARCH22_S2_ACTUAL_FD_PRODUCER_PARTS_DIFFERS')
    for core,(begin,_) in enumerate(per_core):
        head=begin//loops
        expected=next(i for i,piece in enumerate(pieces[head]) if piece[0]==core) if head in heads else 0
        require(row['s2SplitStartIdxOfCore'][core]==expected,'ARCH22_S2_FD_PRODUCER_WORKSPACE_SLOT')
    # Q=1,G=2,M=512 and original FD M=8 require one final two-row chunk per
    # reduced head. Interpret the vector's inclusive endpoints independently.
    require(row['gS1SplitNumOfFdHead'][:fd]==[1]*fd
            and row['gS1LastPartSizeOfFdHead'][:fd]==[2]*fd
            and 1<=vec<=2*aic,'ARCH22_S2_FD_ROW_GEOMETRY')
    consumed=[];previous=-1
    for index in range(vec):
        end=row['gS1IdxEndOfFdHead'][index]
        require(row['gS1IdxEndOfFdHeadSplit'][index]==0 and previous<end<fd,
                'ARCH22_S2_FD_CONSUMER_ENDPOINT')
        consumed.extend(range(previous+1,end+1));previous=end
    require(consumed==list(range(fd)),'ARCH22_S2_FD_CONSUMER_COVERAGE')
    for name in ARRAYS:
        count=(used if name in ('bN2End','gS1End','s2End','s2SplitStartIdxOfCore')
               else vec if name in ('gS1IdxEndOfFdHead','gS1IdxEndOfFdHeadSplit') else fd)
        require(not any(row[name][count:]),'ARCH22_S2_UNUSED_ARRAY_NOT_ZERO:'+name)
    require(row['maxS2SplitNum']==max(map(len,pieces.values())),'ARCH22_S2_MAX_SPLITS_DIFFERS')
    # Complete element counts, not byte offsets borrowed from FAInferTilingData.
    mm1=2*tile;mm2=2*128;accum=aic*2*512*128;lse=2*aic*2*512*8
    require(sum(len(pieces[h]) for h in heads)<=2*aic,'ARCH22_S2_FD_WORKSPACE_CAPACITY')
    return dict(complete_split_arrays_checked=True,covered_q_heads=16,covered_logical_kv=kv,outer_tiles=8*loops,
        used_core_count=used,fd_head_count=fd,fd_producer_parts=sum(len(pieces[h]) for h in heads),
        workspace_elements=dict(mm1ResSize=mm1,mm2ResSize=mm2,fdAccumOutSize=accum,fdLogSumExpSize=lse),
        workspace_excluding_libapi=2*aic*(mm1*6+mm2*8)+(accum+lse)*4,
        npu_executed=False,qualification_granted=False)


def validate_matrix(rows):
    expected={(aic,length,tile) for aic in range(16,27) for length in range(1025,2049) for tile in (512,256)}
    seen=set();profiles=Counter()
    for row in rows:
        key=tuple(row[k] for k in ('aic','kv_length','s2'))
        require(key in expected and key not in seen,'ARCH22_S2_MATRIX_DUPLICATE_OR_UNEXPECTED')
        checked=validate_split(row);seen.add(key)
        profiles[(row['aic'],row['s2'],checked['used_core_count'],checked['fd_head_count'])]+=1
    require(seen==expected,'ARCH22_S2_MATRIX_INCOMPLETE')
    return dict(schema='demoncat.paged33.arch22_s2_cpu_matrix.v1',cases=len(seen),
        complete_split_and_fd_arrays_checked=True,actual_worker_platform_observed=False,
        profiles=[dict(aic=k[0],s2=k[1],used_cores=k[2],fd_heads=k[3],cases=v) for k,v in sorted(profiles.items())],
        npu_executed=False,qualification_granted=False)
