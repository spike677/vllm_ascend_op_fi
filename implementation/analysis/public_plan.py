"""Independently decode final public host raw; no device-argument ABI claim."""
import hashlib
import json
from pathlib import Path
import struct

FORMATS={'bool':'?','uint8_t':'B','int8_t':'b','uint16_t':'H','int16_t':'h',
         'uint32_t':'I','int32_t':'i','uint64_t':'Q','int64_t':'q','float':'f','double':'d'}


def require(ok,text):
    if not ok:raise ValueError(text)


def decode(schema,raw):
    require(schema['schema']=='fault28.fia_graph.public_serializer_layout.v1' and
            schema['type']=='FusedInferAttentionScoreTilingData' and schema['size']==len(raw)==1576,
            'Wrong schema or raw size')
    used=set();result={}
    for f in schema['fields']:
        name,dtype,offset,width,count=(f[k] for k in ('name','dtype','offset','width','count'))
        require(dtype in FORMATS and width==struct.calcsize('<'+FORMATS[dtype]) and count>0 and
                name not in result,'Invalid typed serializer field')
        positions=set(range(offset,offset+width*count))
        require(not positions&used and min(positions)>=0 and max(positions)<len(raw),'Overlapping/out-of-range public field')
        used|=positions
        values=struct.unpack_from('<'+str(count)+FORMATS[dtype],raw,offset)
        result[name]=values[0] if count==1 else list(values)
    require(used==set(range(len(raw))),'Public schema does not cover all bytes, including SDK padding')
    return result


def causal_plan(fields,prune):
    """Source-derived plan, explicitly not a measured device trace.

    Outer no-SplitKV partition edges must fall between complete M blocks.
    Full legal key coverage is checked for every target query/group. Returned
    widths describe the actual selected branch's deterministic arithmetic.
    """
    length=fields['baseParams.s1Size'];kv=fields['baseParams.s2Size']
    m=fields['innerSplitParams.mBaseSize'];n=fields['innerSplitParams.s2BaseSize']
    require(length==kv==2048 and m in (32,64,128,256,512) and n in (32,64,128,256,512),
            'Unsupported public geometry')
    require(n==512 or n==m,'Unexpected independent S2 intervention')
    require(fields['baseParams.gSize']==2 and fields['fdParams.numOfFdHead']==0,'SplitKV/geometry mismatch')
    cores=fields['baseParams.usedCoreNum']
    require(1<=cores<=16,'Invalid used core count')
    ends=list(zip(fields['outerSplitParams.bN2End'][:cores],fields['outerSplitParams.gS1End'][:cores],
                  fields['outerSplitParams.s2End'][:cores]))
    require(all(z==0 for _,_,z in ends),'A core boundary splits S2; no-FD claim insufficient')
    total=fields['baseParams.bSize']*fields['baseParams.n2Size']
    per_head=2*length//m
    flattened=[b*per_head+g for b,g,_ in ends]
    require(flattened==sorted(flattened) and flattened[-1]==total*per_head and
            flattened[0]>0 and len(set(flattened))==len(flattened),'Incomplete/duplicate M partition boundaries')
    widths={};partial_queries=set();plans=[]
    for q in range(length):
        first=(q//m)*m;end=first+m
        segments=[];covered=[]
        for start in range(0,end,n):
            width=min(n,((end-start+31)//32)*32) if prune else n
            require(0<width<=n and width%32==0 and start+width<=kv,'Illegal inner reduction width')
            valid=list(range(start,min(start+width,q+1)))
            covered+=valid;segments.append(width)
        require(covered==list(range(q+1)),'Lost/duplicated unmasked key')
        if any(w<n for w in segments):partial_queries.add(q)
        for w in segments:widths[str(w)]=widths.get(str(w),0)+1
        if q in (0,63,127,128,255,256,383,384,511,512,1023,2047):
            plans.append({'query':q,'m_block_start':first,'m_block_rows':m,'inner_N_widths':segments})
    return {'m':m,'s2_base':n,'pruned':prune,'used_cores':cores,'outer_end_tuples':ends,
            'outer_complete_m_blocks_covered':total*per_head,'target_queries_verified':length,
            'g_groups':2,'unmasked_key_coverage':'all target queries score/PV keys exactly once; see actual key for deliberate denominator replay',
            'target_queries_with_partial_inner_width':len(partial_queries),'per_query_inner_width_histogram':widths,
            'examples':plans,'evidence_kind':'source-derived from independently decoded public host raw and executed kernel variant; not device trace'}


def verify(folder,schema):
    events=[json.loads(line) for line in (folder/'host_events.jsonl').read_text().splitlines()]
    require(events,'Missing public host evidence')
    result=json.loads((folder/'result.json').read_bytes())
    rows=[]
    plan_cache={}
    for event in events:
        raw=bytes.fromhex(event['raw_hex']);fields=decode(schema,raw)
        require(len(event['typed'])==63 and event['typed'].keys()<=fields.keys(),'Wrong typed field coverage')
        require(all(fields[name]==value for name,value in event['typed'].items()),'SDK public raw disagrees with typed getters')
        require(event['raw_size']==1576 and event['serializer_roundtrip_matches'] and
                event['raw_unchanged_by_observer'] and event['kernel_argument_raw_observed'] is False,'Invalid raw observer')
        require(event['eligible'] and len(event['checks'])==23 and all(event['checks'].values()),'Eligibility guard did not pass')
        require(event['aic']==16 and event['aiv']==32,'Changed core budget')
        require(event['owner_pid']==result['owner_pid'] and event['owner_starttime_ticks']==result['owner_starttime_ticks']
                and event['token']==result['scope_token'],'Wrong graph owner identity')
        ancestry=event['scope_ancestry']
        require(ancestry[0]['pid']==event['pid'] and ancestry[-1]['pid']==event['owner_pid'] and
                ancestry[-1]['starttime_ticks']==event['owner_starttime_ticks'] and
                len({a['pid'] for a in ancestry})==len(ancestry),'Invalid recorded child ancestry')
        require(fields['innerSplitParams.mBaseSize']==event['selected_m'] and event['native_m']==512,
                'Selected M mismatch')
        key=event['tiling_key'];prune=event['causal_tile_pruning']
        # Validate branch constraints after reading real key; do not use a key
        # guessed from mode as execution evidence.
        require(key in ('103000000000000000','103000000000400000','109000000000000000','109000000000400000','119000000000400000'),
                'Unsupported actual kernel key')
        require((key.startswith('109') or key.startswith('119'))==prune,'Causal variant key disagrees with branch')
        require((key.endswith('400000'))==(event['selected_m']<=128),'Brcb variant mismatch')
        replay=key=='119000000000400000'
        require(not replay or (fields['innerSplitParams.mBaseSize']==128 and fields['innerSplitParams.s2BaseSize']==128),
                'Replay outside its compiled geometry')
        digest=hashlib.sha256(raw).hexdigest()
        if (digest,prune) not in plan_cache:plan_cache[digest,prune]=causal_plan(fields,prune)
        require(fields['workspaceParams.mm1ResSize']==fields['innerSplitParams.mBaseSize']*fields['innerSplitParams.s2BaseSize'],
                'Native BMM1 workspace does not follow selected M/N')
        require(fields['workspaceParams.mm2ResSize']==128*fields['innerSplitParams.mBaseSize'],
                'Native BMM2 workspace does not follow selected M')
        rows.append({'raw_sha256':digest,'sdk_fields':len(fields),'typed_fields_verified':63,
                     'actual_host_key':key,'denominator_replay':replay,'mode':event['mode'],'kernel_argument_raw_observed':False,
                     'decoded':fields,'plan':plan_cache[digest,prune],'owner_pid':event['owner_pid'],
                     'host_pid':event['pid'],'ancestry':ancestry})
    return rows
