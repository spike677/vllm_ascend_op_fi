"""CPU contracts for a new FD branch; no claim of actual device execution."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fault_injection.native_adapters.attention28.fia import fai_fd_geometry as fd

FOLDER = Path(__file__).resolve().parents[1] / 'fault_injection/native_adapters/attention28/fia'
with patch.object(sys, 'path', [str(FOLDER), *sys.path]):
    spec = importlib.util.spec_from_file_location('fai_fd_patch_tests', FOLDER / 'apply_tnd33_fai_fd_policy.py')
    policy = importlib.util.module_from_spec(spec);spec.loader.exec_module(policy)


def fixture(length=1025, aic=24, tile=512, mask=1):
    """Independent serial chunk partition fixture, not a native-host substitute."""
    typed = dict(numHeads=16, kvHeads=8, batch=1, embeddingSize=128, embeddingSizeV=128,
        numBlocks=256, blockSize=16, maxQSeqlen=0, maxKvSeqlen=0, maxNumBlocksPerBatch=128,
        firstBatchTaskNum=8, totalTaskNum=8, maskType=mask, sparseMode=3 if mask else 0,
        scaleValue=0.0883883461356163, pseQ=0, pseKv=0, padding3=0, preToken=0, nextToken=0,
        mainLoopTaskNum=0, tailLoopTaskNum=0, tailStartBatch=0, tailStartN2=0, tailKvNBlockTile=0)
    typed.update({'coreInfo.' + name: [0]*26 for name in fd.CORE})
    typed.update({'splitInfo.' + name: [0]*26 for name in fd.SPLIT})
    chunks = (length + tile - 1)//tile;target = (16*length + aic - 1)//aic
    intervals=[];cursor=0
    while cursor < 8*chunks:
        start=cursor;cost=0
        while cursor < 8*chunks and cost < target:
            cost += 2*min(tile, length - (cursor % chunks)*tile);cursor += 1
        intervals.append((start, cursor))
    contributions=[]
    for index,(start,end) in enumerate(intervals):
        sh,sk=divmod(start,chunks);eh,ek=divmod(end,chunks)
        if ek==0:eh-=1;ek=chunks
        for name,value in dict(startN1Idx=sh,startS2Idx=sk,endN1Idx=eh,endS2Idx=ek).items():
            typed['coreInfo.'+name][index]=value
        contributions.append([h for h in range(8) if 0 < min(end,(h+1)*chunks)-max(start,h*chunks) < chunks])
    # Each split head has one output slice per contributing core, in core order.
    offset=0;nodes=[]
    for index,heads in enumerate(contributions):
        if heads:
            typed['coreInfo.firstSplitKVTaskLseOffset'][index]=offset*2
            typed['coreInfo.firstSplitKVTaskOOffset'][index]=offset*256
        for head in heads:
            if not nodes or nodes[-1][0]!=head:nodes.append([head,offset,0])
            nodes[-1][2]+=1;offset+=1
    for i,(head,start,count) in enumerate(nodes):
        for name,value in dict(headStartIdx=head*2,headEndIdx=head*2+2,qEndIdx=1,
            splitNum=count,lseTaskOffset=start*2,oTaskOffset=start*256).items():typed['splitInfo.'+name][i]=value
    unit=aic*128*tile*3;lse=offset*8;o=offset*1024
    typed.update(needCoreNum=len(intervals),totalSplitNodeNum=len(nodes),splitLseTotalSize=lse,splitOTotalSize=o,
                 mm1OutSize=unit*4,smOnlineOutSize=unit*2,mm2OutSize=unit*4,UpdateSize=unit*4,workSpaceSize=unit*14+lse+o)
    host=dict(schema='demoncat.paged33.fai_fd_host.v1', route='FAInferTiling',aic=aic,
        native_route_or_decode_flags_changed=False,flash_decode=True,decoding=False,tiling_sink=False,
        query_length=1,kv_length=length,context_max_q=0,context_max_kv=0,workspace=16*1024**2+unit*16+lse+o)
    args=dict(key=5100000000010200100+(3 if mask else 0)+(900000000000000000 if tile==256 else 0),
        block_dim=len(intervals),schedule_mode=1,arm='fault' if tile==256 else 'baseline',kv_length=length)
    return typed,host,args


class FaiFdContracts(unittest.TestCase):
    def test_live_core_budget_is_distinct_from_active_launch_blocks(self):
        for aic in (20,24,25):
            for length in (1025,1280,1536,1792,2048):
                for tile in (512,256):
                    for mask in (0,1):
                        with self.subTest(aic=aic,length=length,tile=tile,mask=mask):
                            typed,host,args=fixture(length,aic,tile,mask)
                            result=fd.validate_fd_geometry(typed,host,**args)
                            self.assertEqual(result['actual_aic'],aic)
                            self.assertEqual(len(typed),52)
        typed,host,args=fixture()
        self.assertEqual(args['block_dim'],17)
        self.assertEqual(typed['splitLseTotalSize'],192)
        self.assertEqual(typed['splitOTotalSize'],24576)
        self.assertEqual(typed['splitInfo.splitNum'][:8],[3]*8)
        for name,value in (('aic',16),('flash_decode',False),('decoding',True),('context_max_q',1)):
            bad=dict(host);bad[name]=value
            with self.assertRaises(RuntimeError):fd.validate_fd_geometry(typed,bad,**args)

    def test_missing_fd_key_fixed_blocks_or_workspace_without_reduction_refuse(self):
        typed,host,args=fixture()
        for changed in (dict(args,key=args['key']-10**17),dict(args,block_dim=24),dict(args,schedule_mode=0)):
            with self.assertRaises(RuntimeError):fd.validate_fd_geometry(typed,host,**changed)
        wrong=dict(host);wrong['workspace']-=typed['splitLseTotalSize']+typed['splitOTotalSize']
        with self.assertRaisesRegex(RuntimeError,'WORKSPACE_DIFFERS'):fd.validate_fd_geometry(typed,wrong,**args)
        changed=dict(typed);changed['maxKvSeqlen']=1025
        with self.assertRaisesRegex(RuntimeError,'BASE_GEOMETRY_DIFFERS'):fd.validate_fd_geometry(changed,host,**args)

    def test_every_core_interval_and_reduction_array_remains_bound(self):
        typed,host,args=fixture()
        for name,index in (('coreInfo.startS2Idx',1),('coreInfo.endN1Idx',0),
                           ('coreInfo.firstSplitKVTaskOOffset',1),('coreInfo.startBIdx',25),
                           ('splitInfo.splitNum',0),('splitInfo.lseTaskOffset',7),('splitInfo.oTaskOffset',25)):
            with self.subTest(name=name,index=index):
                bad=copy.deepcopy(typed);bad[name][index]+=1
                with self.assertRaises(RuntimeError):fd.validate_fd_geometry(bad,host,**args)

    def test_new_patch_uses_only_fd_templates_and_keeps_native_decision(self):
        old=policy.previous
        header='        uint64_t GetTilingKey();\n        uint32_t blockNum_;\n        uint32_t kSBlockTile = MAX_KV_STACK_LEN;\n'
        header+='static_cast<uint64_t>(blockNum_) * WORKSPACE_BLOCK_SIZE_DB *\n'*4
        header+='        return tilingKey;\n'
        decision='static void ComputeSeqLenAndDecodeFlagsForFAI() { native_decision(); }\n'
        host='#include "flash_attention_infer_tiling.h"\n    fai_tiling.SetCoreNum(aicoreNum);\n'
        host+='static_cast<uint64_t>(fai_tiling.GetCoreNum()) * WORKSPACE_BLOCK_SIZE_DB * 4U * 3U * 4U\n'
        host+='    context->SetTilingKey(fai_tiling.GetTilingKey());\n'
        host+='REGISTER_TILING_DATA_CLASS(FusedInferAttentionScore_5000000000000200100, FAInferTilingData)\n'
        host+=decision+'\nstatic ge::graphStatus ConvertContextToParamsFAI() {}\n'
        kernel=''
        for suffix in ('NOMASK','CAUSALMASK'):
            for tail in ('','_FD'):
                symbol='QF16_KVF16_OUTF16_NOLSEOUT_TND_PAGEDCACHE_'+suffix+'_SPLITFUSE_TILING'+tail
                kernel+='        TILING_KEY_IS('+symbol+');\n#elif TILING_KEY_VAR == '+symbol+'\n'
                kernel+='TEMPLATE_FLASH_DECODE_'+str(bool(tail))+'\n'
        original=dict(zip(policy.PATHS,(header,host,'    constexpr uint32_t MAX_KV_STACK_LEN = 512;',kernel)))
        updated=policy.patched(original)
        self.assertEqual(policy.native_decode_decision(updated[policy.PATHS[1]]),decision)
        for key in policy.FAULT_KEYS:
            self.assertEqual(updated[policy.PATHS[3]].count(f'TILING_KEY_IS({key})'),1)
            self.assertIn(f'_TILING_FD || TILING_KEY_VAR == {key}ULL',updated[policy.PATHS[3]])
        self.assertFalse(any(str(key) in '\n'.join(updated.values()) for key in old.FAULT_KEYS))
        self.assertEqual(len(policy.KEYS),len(set(policy.KEYS)))
        self.assertEqual(set(policy.KEYS),set(fd.KEYS))
        wrong=dict(original);wrong[policy.PATHS[3]]=kernel.replace('_TILING_FD','_UNKNOWN')
        with self.assertRaises(Exception):policy.patched(wrong)

    def test_old_six_key_bundle_cannot_gain_fd_admission(self):
        from demoncat.runtime.vllm_ascend.worker_extension import paged33_tnd as runtime
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);completion=root/'completion.json'
            manifest=dict(schema='demoncat.fia.fresh_build.v1',
                status='CPU_BUILD_ONLY_REQUIRES_NATIVE_QUALIFICATION',source_commit=runtime.COMMIT,
                variant=policy.previous.VARIANT,loading_allowed=False,npu_executed=False,
                previous_experiment_objects_used=False,compiled_keys=list(policy.previous.KEYS))
            for variant,keys in ((policy.previous.VARIANT,policy.previous.KEYS),
                                 (fd.VARIANT,policy.previous.KEYS),
                                 (fd.VARIANT,(*fd.KEYS,fd.KEYS[0]))):
                manifest.update(variant=variant,compiled_keys=list(keys))
                completion.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError,'INDEPENDENT_FRESH_BUILD_REQUIRED'):
                    runtime.verify_bundle(completion,runtime.sha(completion),root)


if __name__ == '__main__':unittest.main()
