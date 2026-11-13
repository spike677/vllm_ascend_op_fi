# 受控归约错误：Online Softmax 分母首块重复累计

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

固定判定 `abs(F-B) > 1e-8 + 1e-5*abs(B)`，baseline 为参考，不加 epsilon。三个真实步骤均定位到 `NPU.npu_fused_infer_attention_score.0.forward` 的 `output[0]`。这些步骤是三次相同 prefill，**不是连续 decode token**。

| 达标字段 | B 原值 | F 原值 | 相对变化 |
| --- | ---: | ---: | ---: |
| $[0].Norm | 660.572120626524 | 655.3088952891358 | 0.7967677068169776% |

这里的相对变化是 `abs(Norm_F-Norm_B)/abs(Norm_B)`，不等于完整张量的相对L2误差。0.1%是统计可观测性目标；数学错误的独立证据来自解析算子诊断。

每步对齐 1616 条记录，其中 1189 条叶子API；保留 416 个不可观测字段。Mean不用于0.1%幅度验收。全batch统计不能区分每条请求或证明张量完全相等。统计首差之前的可观测字段均未超阈值。

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
- 首token logits：8/8请求变化，最大绝对差 8.4716796875，Top-1变化 0/8。逐请求数值在[logits报告](reports/formal/logits_comparison.json)，这些数组比较不参与统计首差选择。
- [补充logits分布观察](reports/logit_distribution.json)分开记录共同偏移、去均值差异和CPU-FP64 Softmax分布变化；这是验收后的解释性分析，不参与候选选择或通过判定。
- 固定真实QKV、固定健康/错误策略的B1/B8目标请求输出精确相等；batch门控下B1保持健康M512，B8选择M128错误模板。主B/F固定同一M，隔离的是归约错误，不能用AUTO对照宣称历史 Batch→M 根因闭合。

## 下载与原始字节

[baseline.zip](downloads/baseline.zip) 和 [fault.zip](downloads/fault.zip) 解压后分别位于 `data/cases/baseline`、`data/cases/fault`。仓库浏览路径为 `baseline/`、`fault/`，实际均有 step0–2 的完整 dump/stack/construct。原始JSON按字节复制，未改统计值、精度或阈值；见 `ORIGINAL_BYTES.json`、`SHA256SUMS`。

执行 `python -B verify_downloads.py --extract-to restored` 可核验所有文件、两个ZIP与网页原件逐字节一致，并解压至新目录。

`statistics_receipt.json`是原始采集回执；`completion.json`明确标记为本任务完成字段的派生摘要，保留原控制器回执SHA，不公开其他任务的进程/容器清单。

双方使用同一已披露 CPU-FP64 msprobe采集器，`task=statistics, level=mix, summary_mode=statistics, precision=high`，未增加算子过滤。它不是未经修改的默认msprobe；身份/原处理器/补丁SHA见两组 `collector_provenance.json`。私有完整张量、输入token清单、额外采集组及原始大日志不在交付中。重跑模型需要原本已固定的私有模型与输入清单；独立复算统计首差不需要它们。

## 限制

仅验收这一受控分母重复累计错误在固定几何下的机制和完整模型prefill传播。没有宣称合法tiling本身应产生该大差异，也未验收FIA图模式、任意形状/版本、连续生成或任务准确率。统计首差可独立定位到FIA输出，**仅靠统计不能唯一确定分母重复累计的物理根因**；该根因证据来自独立模板、源码、实际launch、解析诊断和关闭恢复。
