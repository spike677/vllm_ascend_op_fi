"""Independent CPU algebra for the exact TND denominator candidate; no authority."""
import math

ARRAYS=('bN2End','gS1End','s2End','bN2IdxOfFdHead','gS1IdxOfFdHead','s2SplitNumOfFdHead',
    's2SplitStartIdxOfCore','gS1SplitNumOfFdHead','gS1LastPartSizeOfFdHead',
    'gS1IdxEndOfFdHead','gS1IdxEndOfFdHeadSplit')


def require(ok,reason):
    if not ok: raise ValueError(reason)


def derived_key(*,m,causal=False,replay=False):
    """c77 GenTilingKey digits; FP16/TND/TND, perf3, no PA/FD/tree/quant."""
    require(type(m) is int and m in (128,512) and type(causal) is bool and type(replay) is bool
        and (not replay or causal) and (not causal or m==128),'TND28_UNREVIEWED_TEMPLATE')
    digits=(3,0,0,0,0,4 if m==128 else 0,0,3,0)
    native=10**17+3*10**15+sum(d*10**i for i,d in enumerate(digits))
    return native+(16*10**15 if replay else 6*10**15 if causal else 0)


def inner_width(flat,rows,block,base=128,native_width=128):
    require(all(type(n) is int for n in (flat,rows,block,base,native_width))
        and 0<=flat<4096 and 0<rows<=128 and flat+rows<=4096
        and base==128 and 0<native_width<=128 and 0<=block<16,'TND28_CAUSAL_COORDINATE')
    end=(flat+rows-1)//2+1
    return min(native_width,((max(0,end-block*base)+31)//32)*32)


def online_scalar(scores,values,*,replay=False):
    """Float64 oracle only. Repeat first-block mass before the second rescale.

    The caller supplies one causal row, so there are no artificial -inf masks.
    The numerator is calculated by the same recurrence in both arms.
    """
    require(type(replay) is bool and 0<len(scores)<=2048 and len(scores)==len(values)
        and all(math.isfinite(float(v)) for v in (*scores,*values)),'TND28_ORACLE_INPUT')
    maximum=-math.inf;denominator=numerator=0.;replays=0
    for i,start in enumerate(range(0,len(scores),128)):
        chunk=scores[start:start+128];new_max=max(maximum,max(chunk))
        if replay and i==1:
            denominator*=2;replays+=1
        scale=math.exp(maximum-new_max)
        denominator=denominator*scale+sum(math.exp(x-new_max) for x in chunk)
        numerator=numerator*scale+sum(math.exp(x-new_max)*v for x,v in zip(chunk,values[start:start+128]))
        maximum=new_max
    return dict(value=numerator/denominator,numerator=numerator,denominator=denominator,
                maximum=maximum,replays=replays,npu_executed=False)


def validate_split(row):
    cores=row['aic'];m=row['m'];s2=row['s2']
    require(all(type(x) is int for x in (cores,m,s2)) and 16<=cores<=26 and (m,s2)==(128,128),
            'TND28_SPLIT_EXACT_SCOPE')
    used=row['usedCoreNum']
    require(type(used) is int and 0<used<=cores and type(row['numOfFdHead']) is int
        and row['numOfFdHead']==0 and type(row['usedVecNumOfFd']) is int
        and row['usedVecNumOfFd']==0 and type(row['maxS2SplitNum']) is int
        and 0<=row['maxS2SplitNum']<=1,'TND28_SPLITKV_REFUSED')
    for name in ARRAYS:
        size=cores*2 if name in ARRAYS[-2:] else cores
        require(type(row[name]) is list and len(row[name])==size
            and all(type(v) is int and v>=0 for v in row[name]),'TND28_COMPLETE_SPLIT_ARRAY:'+name)
    spans=[];previous=0
    for i in range(used):
        head,flat_tile,block=(row[n][i] for n in ARRAYS[:3])
        require(head<=8 and flat_tile<32 and block==0 and (head<8 or flat_tile==0),
                'TND28_PARTIAL_OR_NONCANONICAL_REDUCTION')
        end=head*32+flat_tile
        require(previous<end<=256,'TND28_SPLIT_EMPTY_OVERLAP_OR_OUTSIDE')
        spans.append((previous,end));previous=end
    require(previous==256,'TND28_INCOMPLETE_HEAD_QUERY_COVERAGE')
    for name in ARRAYS:
        require(not any(row[name][used:] if name in ARRAYS[:3] else row[name]),'TND28_NONEMPTY_UNUSED_OR_FD_ARRAY:'+name)
    # Independently enumerate every (KV head,query,group,key) coverage count.
    covered=0;replayed_rows=0
    for begin,end in spans:
        for index in range(begin,end):
            flat=(index%32)*128
            last=(flat+127)//2
            blocks=(last+1+127)//128
            for row_index in range(flat,flat+128):
                query=row_index//2
                intervals=[]
                for b in range(blocks):
                    width=inner_width(flat,128,b)
                    if b*128<query+1: intervals.append((b*128,min(query+1,b*128+width)))
                require(intervals[0][0]==0 and intervals[-1][1]==query+1
                    and all(a[1]==b[0] for a,b in zip(intervals,intervals[1:])), 'TND28_CAUSAL_PAIR_GAP')
                covered+=sum(b-a for a,b in intervals)
                replayed_rows+=query>=128
    require(covered==16*2048*2049//2 and replayed_rows==16*(2048-128),'TND28_CAUSAL_PAIR_COUNT')
    return dict(aic=cores,used_cores=used,complete_row_reductions=256,
        covered_unmasked_pairs=covered,replay_eligible_rows=replayed_rows,
        actual_worker_platform_observed=False,npu_executed=False,qualification_granted=False)


def validate_matrix(rows):
    summaries=[];seen=set();refused=[]
    for row in rows:
        cores=row['aic']
        require(cores in range(16,27) and cores not in seen,'TND28_DUPLICATE_OR_UNEXPECTED_CORE_PROFILE')
        seen.add(cores)
        try: summaries.append(validate_split(row))
        except ValueError as e: refused.append(dict(aic=cores,reason=str(e)))
    require(seen==set(range(16,27)),'TND28_MATRIX_INCOMPLETE')
    return dict(schema='demoncat.attention28.tnd_denominator_cpu_matrix.v1',cases=len(seen),
        supported_core_profiles=[r['aic'] for r in summaries],refused_profiles=refused,profiles=summaries,
        status='CPU_SPLIT_PROFILES_RECORDED_NOT_NATIVE_ACCEPTANCE',
        npu_executed=False,qualification_granted=False)
