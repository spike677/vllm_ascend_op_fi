"""CPU-only build of a distinct online-Softmax denominator replay template.

Only a new task-private vendor is writable. The global SDK and all preceding
experiment artifacts remain untouched. CPU build is not NPU acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

ROOT=Path('/home/fault28-reduce-replay-20261113')
IMAGE='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
KEYS=[103000000000000000,109000000000000000,109000000000400000,119000000000400000]
STAGE='vendor_visible_a1'
VENDOR='vendors/fault28_fia_graph_transformer'
IMPL='op_impl/ai_core/tbe/fault28_fia_graph_transformer_impl'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,v):
    with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def once(s,a,b):
    assert s.count(a)==1,(a[:100],s.count(a))
    return s.replace(a,b)

FAULT_CODE='''    // Controlled reduction-error variant, not historical legal-tiling drift.
    // On the second S2 block only, replay the first block's denominator mass.
    // Do not change max/exp/PV/numerator or the final RowDivs/CopyOut path.
    if constexpr (FIAT::fault28DenominatorReplay) {
        static_assert(SOFTMAX_WITH_BRC && !FLASH_DECODE, "replay requires no-SplitKV BRC template");
        if (!info.isFirstSInnerLoop && info.s2Idx == 1) {
            const uint32_t replayCount = dealRowCount * this->brcbNum;
            PipeBarrier<PIPE_V>();
            Muls(inSumTensor, inSumTensor, COMPUTE_T(2), replayCount);
            PipeBarrier<PIPE_V>();
        }
    }
'''

def patch(vendor):
    base=vendor/IMPL;records=[]
    paths=[('common/op_kernel/fia_public_define.h','trait'),('ascendc/common/fia_public_define.h','trait'),
           ('common/op_kernel/arch22/fia_block_vec_nonquant.h','vec'),('ascendc/common/arch22/fia_block_vec_nonquant.h','vec'),
           ('fused_infer_attention_score/op_kernel/fused_infer_attention_score_v3.cpp','dispatch'),
           ('ascendc/fused_infer_attention_score/fused_infer_attention_score_v3.cpp','dispatch')]
    for name,kind in paths:
        p=base/name;before=sha(p);text=p.read_text()
        if kind=='trait':
            text=once(text,'static constexpr bool fault28CausalPrune = false;',
                      'static constexpr bool fault28CausalPrune = false;\n    static constexpr bool fault28DenominatorReplay = false;')
        elif kind=='vec':
            anchor='    if constexpr (SOFTMAX_WITH_BRC) {\n        SoftmaxFlashV2<SOFTMAX_TYPE'
            text=once(text,anchor,FAULT_CODE+anchor)
        else:
            text=once(text,'template <class Base> struct Fault28CausalFIAType : Base { static constexpr bool fault28CausalPrune = true; };',
                '''template <class Base> struct Fault28CausalFIAType : Base { static constexpr bool fault28CausalPrune = true; };
template <class Base> struct Fault28ReplayFIAType : Fault28CausalFIAType<Base> {
    static constexpr bool fault28DenominatorReplay = true;
};''')
            start=text.index('#define INVOKE_FAULT28_FIA_PRUNED(')
            end=text.index('\n#define FIA_COPY_TILING_DATA',start)
            macro=text[start:end]
            replay=macro.replace('INVOKE_FAULT28_FIA_PRUNED','INVOKE_FAULT28_FIA_REPLAY').replace('Fault28CausalFIAType','Fault28ReplayFIAType')
            text=text[:end]+'\n'+replay+text[end:]
            text=once(text,'    TILING_KEY_IS(109000000000400000);',
                      '    TILING_KEY_IS(109000000000400000);\n    TILING_KEY_IS(119000000000400000);')
            start=text.index('#elif TILING_KEY_VAR == 109000000000400000')
            end=text.index('#elif ',start+6)
            branch=text[start:end]
            new=branch.replace('109000000000400000','119000000000400000').replace('INVOKE_FAULT28_FIA_PRUNED','INVOKE_FAULT28_FIA_REPLAY')
            text=text[:end]+new+text[end:]
        p.write_text(text,encoding='utf-8',newline='\n')
        target=ROOT/'source_kernel'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
        records.append({'path':name,'before_sha256':before,'after_sha256':sha(p)})
    save(ROOT/'build/kernel_a1/source_changes.json',{'files':records,'numerator_or_output_path_changed':False,
        'dtype_or_accumulation_precision_changed':False,'new_key':KEYS[-1],
        'fault':'Replay first-block online Softmax denominator at the second S2 block only'})

def register():
    root=Path('/audit');ev=root/'build/kernel_a1';stage=root/STAGE;vendor=stage/VENDOR
    assert json.loads((ev/'compile_completion.json').read_bytes())['returncode']==0
    name=json.loads((ev/'input_param.json').read_bytes())['op_list'][0]['bin_filename']
    candidates=list((ev/'output').rglob(name+'.json'));assert len(candidates)==1
    meta=candidates[0];data=json.loads(meta.read_bytes());binary=meta.with_suffix('.o')
    assert data['sha256']==sha(binary) and data['binFileName']==name
    assert {int(x['tilingKey']) for x in data['kernelList']}==set(KEYS)
    sdk=Path('/usr/local/Ascend/cann-9.1.0/opp/built-in/op_impl/ai_core/tbe/kernel')
    config=sdk/'config/ascend910b/ops_transformer/fused_infer_attention_score.json'
    selected=[x for x in json.loads(config.read_bytes())['binList'] if Path(x['binInfo']['jsonFilePath']).stem==name]
    assert len(selected)==1 and selected[0]['outputs'][0]['dtype']=='float16'
    dest=vendor/'op_impl/ai_core/tbe/kernel';output=dest/'ascend910b/fused_infer_attention_score'
    assert output.is_dir()
    for f in (meta,binary):shutil.copyfile(f,output/f.name)
    selected[0]['binInfo']['jsonFilePath']='ascend910b/fused_infer_attention_score/'+meta.name
    registry=dest/'config/ascend910b/fused_infer_attention_score.json'
    registry.write_text(json.dumps({'binList':selected},indent=2)+'\n')
    receipt={'status':'CPU_REGISTERED_REQUIRES_NPU_VALIDATION','compiled_keys':KEYS,
        'kernel_symbols':[x['kernelName'] for x in data['kernelList']],
        'binary_sha256':sha(binary),'compiler_metadata_sha256':sha(meta),'sdk_matcher_sha256':sha(config),
        'source_changes_sha256':sha(ev/'source_changes.json'),'global_sdk_modified':False,'npu_executed':False,
        'injection':'online_softmax_denominator_replay','historical_fault28_root_cause_claimed':False}
    save(ev/'registration.json',receipt)
    manifest=json.loads((stage/'vendor_manifest.json').read_bytes())
    manifest.update(status=receipt['status'],generation='reduce_replay_a1',eager_kernel_registration=receipt,
        policy_kind='online_softmax_denominator_replay',kernel_binary_unchanged=False)
    manifest['host_extension']['kernel_binary_unchanged']=False
    manifest['artifacts']=[{'path':p.relative_to(vendor).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
        for p in sorted(vendor.rglob('*')) if p.is_file()]
    (stage/'vendor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'vendor_manifest_sha256':sha(stage/'vendor_manifest.json')}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');a=p.parse_args()
    if a.register:return register()
    ev=ROOT/'build/kernel_a1';ev.mkdir(exist_ok=False);stage=ROOT/STAGE;vendor=stage/VENDOR
    manifest=json.loads((stage/'vendor_manifest.json').read_bytes())
    for row in manifest['artifacts']:assert sha(vendor/row['path'])==row['sha256']
    patch(vendor)
    param=Path('/home/fault28-fia-graph-20261106/source_a4/ops-transformer-v9.1.0/build/binary/ascend910b/gen/FusedInferAttentionScore_8cd36e66e4ceb2d60ef96db29a147347_param.json')
    shutil.copyfile(param,ev/'input_param.json')
    script='''set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source /work/venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2
export ASCEND_GLOBAL_LOG_LEVEL=0 ASCEND_SLOG_PRINT_TO_STDOUT=1 ASCENDC_LOG_LEVEL=0
export ASCEND_CUSTOM_OPP_PATH=/audit/vendor_visible_a1/vendors/fault28_fia_graph_transformer
cd /audit/build/kernel_a1
mkdir output
python /usr/local/Ascend/cann-9.1.0/python/site-packages/opc_tool/opc.py /audit/vendor_visible_a1/vendors/fault28_fia_graph_transformer/op_impl/ai_core/tbe/fault28_fia_graph_transformer_impl/dynamic/fused_infer_attention_score.py --main_func=fused_infer_attention_score --input_param=/audit/build/kernel_a1/input_param.json --soc_version=Ascend910B4 --output=/audit/build/kernel_a1/output --impl_mode=high_performance,optional --simplified_key_mode=0 --op_mode=dynamic --tiling_key=103000000000000000,109000000000000000,109000000000400000,119000000000400000 --op_debug_config=dump_cce
'''
    (ev/'compile.sh').write_text(script)
    cmd=['docker','run','--rm','--network','none','--cpus','4','--memory','12g','--pids-limit','512',
        '--name','dcat-f28-replay-kernel-a1','--mount','type=bind,src='+str(ROOT)+',dst=/audit',
        '--mount','type=bind,src=/home/tik-constfold-cann91-overnight-20261022,dst=/work,readonly',
        '--entrypoint','bash',IMAGE,'/audit/build/kernel_a1/compile.sh']
    save(ev/'start.json',{'command':cmd,'keys':KEYS,'devices':[],'npu_executed':False,'started_at':time.time()})
    with (ev/'compile.log').open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=900).returncode
    save(ev/'compile_completion.json',{'returncode':rc,'finished_at':time.time(),'log_sha256':sha(ev/'compile.log'),'npu_executed':False})
    if rc:raise SystemExit(rc)
    subprocess.run(['docker','run','--rm','--network','none','--cpus','1','--memory','1g','--pids-limit','128',
        '--mount','type=bind,src='+str(ROOT)+',dst=/audit','--entrypoint','python3',IMAGE,
        '-B','/audit/build_replay_kernel.py','--register'],check=True,timeout=60)

if __name__=='__main__':main()
