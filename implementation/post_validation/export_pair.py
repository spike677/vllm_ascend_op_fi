"""Export only the accepted new-mechanism B/F raw statistical pair.

Post-run packaging, not part of execution or the first-difference comparator.
Do not publish private tensors, temporary logs or additional capture groups.
"""
import argparse,hashlib,json,shutil,subprocess,sys,zipfile
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.root;out=a.output;state=read(r/'search_state.json')
    formal=read(r/'reports/formal/review.json');restart=read(r/'reports/restart/review.json');controls=read(r/'reports/controls.json')
    assert formal['accepted_before_controls'] and restart['accepted_before_controls']
    assert controls['status']=='PASS_RESTART_OFF_AND_FIXED_QKV_BATCH_CONTROLS'
    assert read(r/'reports/final_acceptance.json')['status']=='PASS_NEW_DENOMINATOR_REPLAY_VARIANT'
    contract=read(r/'preregister.json')
    for name,digest in contract['source_files'].items():assert sha(r/name)==digest
    assert sha(r/'vendor_visible_a1/vendor_manifest.json')==contract['vendor_manifest_sha256']
    out.mkdir(parents=True,exist_ok=False);copied=[]
    def copy(src,dst):
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
        assert sha(src)==sha(dst)
        copied.append({'path':dst.relative_to(out).as_posix(),'source':str(src),'sha256':sha(dst),'bytes':dst.stat().st_size})
    for arm,letter in [('baseline','B'),('fault','F')]:
        src=r/state['cases']['formal_'+letter]['folder'];dst=out/arm
        assert sha(src/'result.json')==state['cases']['formal_'+letter]['result_sha256']
        captured=read(src/'statistics_receipt.json');assert captured['actual_steps']==[0,1,2]
        for row in captured['steps']:
            assert row['records']==1616 and len(row['FIA_records'])==28
            for name,digest in row['files'].items():
                assert sha(src/f"observation/step{row['step']}/rank0/{name}.json")==digest
        for name in ['config.json','collector_provenance.json','case.json','inputs.json','execution_config.json',
                     'statistics_receipt.json','parameters_receipt.json','versions.json','result.json',
                     'host_events.jsonl','loaded_native_libraries.json']:
            copy(src/name,dst/name)
        completion=read(src.parent/'completion.json')
        save(dst/'completion.json',{
            'schema':'controlled_reduction.task_only_completion_summary.v1','generated_summary':True,
            'original_controller_completion_sha256':sha(src.parent/'completion.json'),
            'attempt_directory':src.parent.name,
            **{k:completion[k] for k in ['status','card','probe','capture','start_epoch','end_epoch',
                'device_lock_acquired','device_lock_released','owned_container_removed','cleanup_errors']},
            'cases':[{k:c.get(k) for k in ['case_id','rc','result_sha256','status']} for c in completion['cases']],
            'scope':'Task-owned completion facts only; foreign process/container inventories remain private. The original statistics_receipt.json is separately copied byte-for-byte.'})
        copy(src.parent/'source_manifest.json',dst/'capture_source_manifest.json')
        for step in range(3):
            for name in ['dump.json','stack.json','construct.json']:
                rel=Path(f'observation/step{step}/rank0')/name;copy(src/rel,dst/rel)
    for name in ['compare_statistics.py','statistics_values.py','accept_statistics.py','test_statistics.py']:
        copy(r/'analysis'/name,out/'analysis'/name)
    copy(r/'build/audit_statistics_reads.py',out/'analysis/audit_statistics_reads.py')
    copy(r/'build/verify_downloads.py',out/'verify_downloads.py')
    (out/'.gitattributes').write_text('* -text\n*.zip binary\n',encoding='utf-8',newline='\n')
    # Each independent report is regenerated strictly from the exported raw pair.
    sys.path.insert(0,str(out/'analysis'))
    from compare_statistics import compare
    from accept_statistics import accept
    first=[]
    for step in range(3):
        value=compare(out/f'baseline/observation/step{step}/rank0/dump.json',out/f'fault/observation/step{step}/rank0/dump.json',rtol=1e-5,atol=1e-8)
        gate=accept(value);assert gate['accepted']
        save(out/f'reports/independent_first_difference_step{step}.json',value)
        save(out/f'reports/statistics_gate_step{step}.json',gate)
        first.append({'step':step,'first':value['first_observed_api_difference'],'qualifying_fields':gate['qualifying_fields'],
            'unobservable_fields':value['unobservable_fields'],'all_aligned_records':value['all_aligned_records'],
            'native_API_records':value['native_API_records']})
    save(out/'reports/first_difference.json',{'status':'PASS_NEW_REDUCTION_ERROR_VARIANT_ONLY','steps':first,
        'rtol':1e-5,'atol':1e-8,'reference':'baseline','minimum_relative_change':.001,
        'historical_fault28_root_cause_reproduced':False,'runtime_or_private_arrays_used_for_first':False})
    subprocess.run([sys.executable,'-B',str(out/'analysis/audit_statistics_reads.py'),'--baseline',str(out/'baseline'),
        '--fault',str(out/'fault'),'--source',str(out/'analysis'),'--output',str(out/'reports/statistics_only_access_audit.json')],check=True)
    for name in ['analytic.json','calibration_gate.json','real_qkv_mechanism.json','controls.json','final_acceptance.json','logit_distribution.json']:
        copy(r/'reports'/name,out/'reports'/name)
    for group in ['formal','restart']:
        for name in ['review.json','runtime_baseline.json','runtime_fault.json','logits_comparison.json']:
            copy(r/'reports'/group/name,out/'reports'/group/name)
    for name in ['test_recurrence.py.log','test_inventory_client.py.log','comparator_tests.log']:
        copy(r/'build'/name,out/'reports/CPU_tests'/name)
    for name in ['preregister.json']:
        copy(r/name,out/'implementation'/name)
    for name in contract['source_files']:
        copy(r/name,out/'implementation'/name)
    for p in (r/'build/host_a1/source').iterdir():
        if p.is_file():copy(p,out/'implementation/host'/p.name)
    for name in ['source_identity.json','exact_commands.json','completion.json','stage_completion.json']:
        copy(r/'build/host_a1'/name,out/'implementation/build_identity/host'/name)
    for name in ['source_changes.json','registration.json','compile_completion.json','input_param.json','compile.sh']:
        copy(r/'build/kernel_a1'/name,out/'implementation/build_identity/kernel'/name)
    copy(Path('/home/fault28-fia-fullmodel-20261109/build/evidence_a4/fia_abi_schema.json'),
         out/'implementation/build_identity/fia_abi_schema.json')
    copy(r/'vendor_visible_a1/vendor_manifest.json',out/'implementation/build_identity/vendor_manifest.json')
    vendor=r/'vendor_visible_a1/vendors/fault28_fia_graph_transformer'
    vendor_manifest=read(r/'vendor_visible_a1/vendor_manifest.json')
    vendor_hashes={x['path']:x['sha256'] for x in vendor_manifest['artifacts']}
    for name in ['fia_kernel_nonquant.h','fia_block_cube_nonquant_gqa.h','fault28_causal_tile.h']:
        relative='op_impl/ai_core/tbe/fault28_fia_graph_transformer_impl/ascendc/common/arch22/'+name
        assert sha(vendor/relative)==vendor_hashes[relative]
        copy(vendor/relative,out/'implementation/inherited_kernel'/name)
    collector=Path('/home/fault28-fia-fullmodel-20261109/collector_fp64_a1')
    provenance=read(out/'baseline/collector_provenance.json')
    assert sha(collector/'stage_manifest.json')==provenance['manifest_sha256']
    for name in ['collector.patch','stage_manifest.json','source_manifest.json',
                 'packages/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py']:
        copy(collector/name,out/'collector'/name)
    for name in ['export_pair.py','reference_real_qkv.py','finalize.py','logit_distribution.py']:
        copy(r/'build'/name,out/'implementation/post_validation'/name)
    save(out/'ORIGINAL_BYTES.json',{'raw_triplets_changed':False,'public_groups':['baseline','fault'],'files':copied})
    (out/'downloads').mkdir()
    for arm in ['baseline','fault']:
        with zipfile.ZipFile(out/'downloads'/(arm+'.zip'),'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for path in sorted((out/arm).rglob('*')):
                if path.is_file():z.write(path,'data/cases/'+arm+'/'+path.relative_to(out/arm).as_posix())
    # README is generated from actual verified values, not a prewritten PASS.
    s=first[0];qualified=s['qualifying_fields'];ln=formal['logits']
    rows='\n'.join('| '+str(x['path'])+' | '+repr(x['B'])+' | '+repr(x['F'])+' | '+str(x['relative_delta_B']*100)+'% |' for x in qualified)
    text=f'''# 受控归约错误：Online Softmax 分母首块重复累计

**新机制验证通过；不代表历史故障28的合法切分精度问题已复现。** 旧的合法切分实现和原始证据保留不变。本分支仅提供本次 B/F 两组原生 statistics/mix。

## 实际故障与调用链

B/F 同一 Qwen3-0.6B、8条自然请求、FP16、2048-token prefill、真实 FIA eager，使用同一私有构建包。两组均为 M128/S2=128。B 是健康的受控合法切分路径，不是原厂默认 M512 路径。

F 在第二个 S2 块的原生 Online Softmax 更新之前，将上一块的累计分母乘2，模拟首块分母贡献被重复计入一次。Q/K/V、权重、causal mask、FP32累加类型、PV分子及最终除法/输出拷贝均保持原实现。没有修改 FIA/logits 的返回张量。

```text
Qwen3 FullModelEager → 每层 Attention → torch_npu.npu_fused_infer_attention_score
→ native ACLNN → FiaTilingNonQuant::DoOpTiling
→ 健康 key 109000000000400000 / 错误 key 119000000000400000
→ FiaKernelNonQuant → FiaBlockVecNonQuant::SoftmaxFlashV2Compute
→ 第二个S2块：inSumTensor *= 2（仅F）→ 原生 Softmax / PV / RowDivs → logits
```

关键代码：[kernel 注入](implementation/source_kernel/ascendc/common/arch22/fia_block_vec_nonquant.h)、[模板分发](implementation/source_kernel/ascendc/fused_infer_attention_score/fused_infer_attention_score_v3.cpp)、[实际 host 边界](implementation/host/fia_tiling_nonquant.cpp)、[条件选择](implementation/source_policy/fault28_fia_policy.h)、[完整模型入口](implementation/model_adapter.py)。

对 query q，正确结果为 N/D；错误结果为 N/(D+D0)，D0是合法keys 0..127在最终最大值尺度下的分母贡献。q<128没有第二块，不触发；其余匹配query每次触发一次。每个合法score/PV key仍参与一次，分母的首块贡献故意参与两次，Reduce轴不跨核。该错误不是“每个输出统一乘常数”。

## 独立统计结论

固定判定 `abs(F-B) > 1e-8 + 1e-5*abs(B)`，baseline 为参考，不加 epsilon。三个真实步骤均定位到 `{s['first']['B_node']}` 的 `output[0]`。这些步骤是三次相同 prefill，**不是连续 decode token**。

| 达标字段 | B 原值 | F 原值 | 相对变化 |
| --- | ---: | ---: | ---: |
{rows}

这里的相对变化是 `abs(Norm_F-Norm_B)/abs(Norm_B)`，不等于完整张量的相对L2误差。0.1%是统计可观测性目标；数学错误的独立证据来自解析算子诊断。

每步对齐 {s['all_aligned_records']} 条记录，其中 {s['native_API_records']} 条叶子API；保留 {s['unobservable_fields']} 个不可观测字段。Mean不用于0.1%幅度验收。全batch统计不能区分每条请求或证明张量完全相等。统计首差之前的可观测字段均未超阈值。

[独立首差报告](reports/first_difference.json) 与 [严格文件读取审计](reports/statistics_only_access_audit.json) 只读取 B/F 的 dump、stack、construct、config 和采集器身份；禁止读取注入日志、策略、输出数组或联网。

```bash
python -B analysis/compare_statistics.py --baseline baseline/observation/step0/rank0/dump.json --fault fault/observation/step0/rank0/dump.json --rtol 1e-5 --atol 1e-8 --output recomputed-step0.json
python -B analysis/accept_statistics.py --report recomputed-step0.json --output recomputed-gate.json
```

将 step0 替换为 step1/step2可复核其余步骤；输出文件必须不存在。

## 机制、覆盖和传播的辅助验收

- 124物理卡2，Ascend910B4；同一构建包含健康/错误模板。匹配源码提交 `c77c0829f1f0f0b8f882db1d9b051a758b605472`（CANN9.1.0），torch2.10.0、torch-npu2.10.0.post4、transformers5.5.4。
- 解析算子诊断：Q=K=0、V=1，33554432个元素全部精确等于公式的FP16舍入值；前128个query不变。此诊断独立于自然模型输入。
- 健康新模板与先前合法M128真实QKV回放精确一致。[筛选和桥接](reports/calibration_gate.json)
- 正式每组预热1次+采集3次，28层全部覆盖，每次FIA调用包含8个请求；每组112个实际FIA launch、896个层请求组合。三次采集共84个FIA API原始记录。匹配调用持续触发；前128个query不触发是设计条件。
- 同一首差层完整Q/K/V逐元素字节相等；所有返回中间数组/logits有限且组内重复精确一致。独立重启再次复核，OFF在固定M128/S2=128下关闭分母错误，全部113数组×8请求精确恢复。[完整模型复核](reports/formal/review.json)、[控制实验](reports/controls.json)
- 首token logits：{ln['changed_requests']}/8请求变化，最大绝对差 {ln['max_abs']}，Top-1变化 {ln['top1_changed_requests']}/8。逐请求数值在[logits报告](reports/formal/logits_comparison.json)，这些数组比较不参与统计首差选择。
- [补充logits分布观察](reports/logit_distribution.json)分开记录共同偏移、去均值差异和CPU-FP64 Softmax分布变化；这是验收后的解释性分析，不参与候选选择或通过判定。
- 固定真实QKV、固定健康/错误策略的B1/B8目标请求输出精确相等；batch门控下B1保持健康M512，B8选择M128错误模板。主B/F固定同一M，隔离的是归约错误，不能用AUTO对照宣称历史 Batch→M 根因闭合。

## 下载与原始字节

[baseline.zip](downloads/baseline.zip) 和 [fault.zip](downloads/fault.zip) 解压后分别位于 `data/cases/baseline`、`data/cases/fault`。仓库浏览路径为 `baseline/`、`fault/`，实际均有 step0–2 的完整 dump/stack/construct。原始JSON按字节复制，未改统计值、精度或阈值；见 `ORIGINAL_BYTES.json`、`SHA256SUMS`。

执行 `python -B verify_downloads.py --extract-to restored` 可核验所有文件、两个ZIP与网页原件逐字节一致，并解压至新目录。

`statistics_receipt.json`是原始采集回执；`completion.json`明确标记为本任务完成字段的派生摘要，保留原控制器回执SHA，不公开其他任务的进程/容器清单。

双方使用同一已披露 CPU-FP64 msprobe采集器，`task=statistics, level=mix, summary_mode=statistics, precision=high`，未增加算子过滤。它不是未经修改的默认msprobe；身份/原处理器/补丁SHA见两组 `collector_provenance.json`。私有完整张量、输入token清单、额外采集组及原始大日志不在交付中。重跑模型需要原本已固定的私有模型与输入清单；独立复算统计首差不需要它们。

## 限制

仅验收这一受控分母重复累计错误在固定几何下的机制和完整模型prefill传播。没有宣称合法tiling本身应产生该大差异，也未验收FIA图模式、任意形状/版本、连续生成或任务准确率。统计首差可独立定位到FIA输出，**仅靠统计不能唯一确定分母重复累计的物理根因**；该根因证据来自独立模板、源码、实际launch、解析诊断和关闭恢复。
'''
    (out/'README.md').write_text(text,encoding='utf-8',newline='\n')
    files=sorted(p for p in out.rglob('*') if p.is_file())
    assert not any(p.suffix.lower() in ('.pt','.npy','.bin','.safetensors') for p in files)
    (out/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in files),encoding='utf-8')
    print(json.dumps({'status':'EXPORTED_NEW_MECHANISM_PAIR','path':str(out),'files':len(files),'bytes':sum(p.stat().st_size for p in files)}))

if __name__=='__main__':main()
