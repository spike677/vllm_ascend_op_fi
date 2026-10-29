# 故障28 B/F：原生 FA + 高精度 statistics/mix

已在124物理卡5、910B4、TP1通过受控机理、真实模型传播与独立API边界定位。模型为原BF16 Qwen3-0.6B统一加载为FP16，B/F均为Batch8×2048-token、两次独立完整prefill。

**数据依赖任务私有msprobe 26.1.0高精度统计修正，并非未经修改的官方collector。** 官方默认FP16四统计未分辨SDPA输出，首差是下游pow输入；失败推断完整保留。新配置保持task=statistics、level=mix、summary_mode=statistics，precision=high。

[实现、调用链、模拟依据与范围](../../experiments/fault28_native_tiling_20261029/README.md) · [原始数据包](fault28_native_fa_high_statistics_bf_20261029/README.txt) · [最终验收](fault28_native_fa_high_statistics_bf_20261029/summaries/execution_acceptance.json)

| 核验 | 结果 |
|---|---|
| 实际原生路径 | HF SDPA → aclnnFlashAttentionScore，未重算/替换输出 |
| 同一构建 OFF/ON | M128、2×1024 keys → M64、4×512 keys；合法keys不变，无SplitKV |
| 多次触发 / 保护 / 恢复 | 两次合格调用均触发；Batch1不触发；同进程关闭后全部恢复 |
| 精确输入 | layer1全Batch Q/K/V逻辑哈希相同；参数和精度保持 |
| 独立msprobe首差 | 两步均为layer1 SDPA输出Mean，输入统计匹配 |
| 模型传播 | logits改变，Top-1仍3516；未证明业务质量下降 |

只给独立分析器四文件/arm：config.json及dump/stack/construct，它不读取命中、host profile或运行时日志来指定首差。[step0报告](fault28_native_fa_high_statistics_bf_20261029/summaries/independent_msprobe_step0.json)和[step1报告](fault28_native_fa_high_statistics_bf_20261029/summaries/independent_msprobe_step1.json)保留原数据SHA、调用栈和模块归属。首差前50项统计不可观测；统计匹配不证明逐元素相同。实际CANN执行/tiling因果由单独源码、profile及精确运行时证据核验。

包内主采集只有B/F两组，每组两个step的原生JSON；208文件、18,337,954字节，不含整批张量、模型权重、编译库或巨日志。包内JSON/JSONL里的raw_hex是tiling元数据。源码冻结、采集器原/改源码与补丁、513文件manifest、8项独立报告及SHA绑定验收均保留。

```sh
python artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029/tools/package_bf_evidence.py verify \
  --package artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029
```

manifest SHA256：`561221711e723c891f2feb2729720f9a761e5452e42402ebf1c27f697f229eb6`。根验收状态为 `PASS_FAULT28_NATIVE_TILING_AND_HIGH_STATISTICS`；打包器的mechanism_accepted=false只表示它不是科学验收器，真实范围见SHA绑定的execution_acceptance.json。

本结果不是历史38Bv2或vLLM FIA缺陷复现。Batch1/8自然运行的目标QKV在Attention前已有差异，本轮使用同Batch8 OFF/ON隔离注入因果。FP64参考不用于指定输出，也没有强求Top-1翻转。
