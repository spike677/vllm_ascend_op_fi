"""Public v9.1.0 flash-decode candidate, B1/Q1 only; no grant.

Derivation: FAInferTiling::GetQNBlockTile, GetKSBlockTile,
ProcessCoreSplitInfo in source commit c77c0829. 512 is this candidate's
MAX_KV_STACK_LEN, not an inferred A3 hardware constant. Unknown cases refuse.
"""
import argparse,json
from pathlib import Path


def partitions(decoded):
    if (decoded.get('schema')!='dcat.fault33.candidate_decode.v1'
            or decoded.get('shape_consistent') is not True or decoded.get('issues')
            or decoded.get('key')!=5100000000010200203):
        raise ValueError('CANDIDATE_FLASH_DECODE_KEY_OR_FIELDS_UNPROVEN')
    f=decoded['fields'];heads=f['numHeads'];kv=f['kvHeads'];length=f['maxKvSeqlen']
    if (f['batch']!=1 or f['maxQSeqlen']!=1 or f['sparseMode']!=3 or heads%kv
            or not 1<=heads//kv<=128 or length<1):raise ValueError('CANDIDATE_B1_Q1_GQA_RANGE_ONLY')
    group=heads//kv;tiles=(length+511)//512;parts=[[] for _ in range(kv)]
    nodes=[];lse=0;out=0;core_rows=[]
    def arr(name,i):return f[name][i]
    for core in range(decoded['block_dim']):
        if any(arr('coreInfo.'+p,core)!=0 for p in ('startBIdx','endBIdx','startS1Idx','endS1Idx')):
            raise ValueError('CORE_BATCH_QUERY_RANGE')
        first=arr('coreInfo.startN1Idx',core);last=arr('coreInfo.endN1Idx',core)
        if not 0<=first<=last<kv:raise ValueError('CORE_HEAD_RANGE')
        start=arr('coreInfo.startS2Idx',core);end=arr('coreInfo.endS2Idx',core)
        if not 0<=start<=tiles or not 0<=end<=tiles:raise ValueError('CORE_S2_RANGE')
        first_offset=None
        for head in range(first,last+1):
            a=start if head==first else 0;b=end if head==last else tiles
            if a>b:raise ValueError('CORE_REVERSED_SPAN')
            if a==b:continue
            span=[a*512,min(b*512,length)]
            if span[0]>=span[1]:raise ValueError('CORE_EMPTY_EFFECTIVE_SPAN')
            parts[head].append(span);core_rows.append(dict(core=core,kv_head=head,span=span))
            if b-a<tiles:
                if first_offset is None:first_offset=(lse,out)
                if not nodes or nodes[-1]['head']!=head:
                    nodes.append(dict(head=head,count=0,lse=lse,out=out))
                nodes[-1]['count']+=1;lse+=group;out+=group*f['embeddingSizeV']
        expected=first_offset or (0,0)
        if (arr('coreInfo.firstSplitKVTaskLseOffset',core),arr('coreInfo.firstSplitKVTaskOOffset',core))!=expected:
            raise ValueError('CORE_SPLIT_OFFSET_MISMATCH')
    for spans in parts:
        if (not spans or spans[0][0]!=0 or spans[-1][1]!=length
                or any(a[1]!=b[0] for a,b in zip(spans,spans[1:]))):raise ValueError('KEY_GAP_OVERLAP_OR_CORE_ORDER')
    if f['totalSplitNodeNum']!=len(nodes):raise ValueError('SPLIT_NODE_COUNT')
    for i,node in enumerate(nodes):
        expected=dict(batchIdx=0,headStartIdx=node['head']*group,headEndIdx=(node['head']+1)*group,
                      qStartIdx=0,qEndIdx=1,splitNum=node['count'],lseTaskOffset=node['lse'],oTaskOffset=node['out'])
        if any(arr('splitInfo.'+k,i)!=v for k,v in expected.items()):raise ValueError('SPLIT_NODE_FIELDS')
    if f['splitLseTotalSize']!=lse*4 or f['splitOTotalSize']!=out*4:raise ValueError('SPLIT_WORKSPACE_SIZE')
    return dict(schema='dcat.fault33.candidate_partitions.v1',capture_sha256=decoded['capture_sha256'],
                key=decoded['key'],candidate_s2_tile=512,per_kv_head_spans=parts,cores=core_rows,
                complete_key_set=True,source_self_consistent=True,
                installed_kernel_correspondence=False,qualification_granted=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('decoded',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.write_text(json.dumps(partitions(json.loads(a.decoded.read_text())),indent=2)+'\n')
