# 故障 28：M-only 全模型 statistics/mix 证据

2026-11-09，124 物理卡 5、Ascend 910B4，完整 HF Qwen3-0.6B FP16 的 B1/B8 验收通过。
**只将 M128 切为 M64，实际 nRatio 始终为 8，即可产生稳定的模型 logits 差异；OFF 精确恢复。**
这是[本次 M-only 实现](../../../experiments/fault28_m_only_model_20261109/README.md)的新采集，
不以旧 joint 模型实验替代 M 的归因，也不代表历史盘古 FIA 图模式已复现。

| 验收维度 | 实际结果 |
| --- | --- |
| 全覆盖 | 560 次真实 SDPA；112/112 次受控 prefill 实际切换；全部 28 层、B8 全部 8 请求、504 个层/请求映射对。 |
| 原生执行 | 从整日志重新解析 280 个 prefill 的实际 launch raw、205 项 POD、169 项执行计划；未遗漏或出现游离 attention launch。 |
| 传播 | 第 0 层 Attention 有差，但末 query 不变；第 1 层起末 query 和真实 KV cache 有差，最终全部请求 logits 有差。 |
| 稳定/恢复 | 两轮 F 的完整各层输出、末 query、KV、logits 精确一致；OFF 恢复 B 的完整值。 |
| 独立定位 | 统一 CPU FP64 high 统计下，两组两轮 F 的 msprobe 首差均为第 0 层 SDPA 输出 Mean/Norm。 |
| 有效边界 | decode Q1/KV2049 明确范围外透传；Top-1 全部不变，未评估答案质量；未唯一定位 kernel 内 Reduce 指令。 |

完整 28 层、固定 2048-token prefill 和一次固定后续 token decode；每轮 fresh KV cache。
两组分别为 `formal_b1_a4/main_b1` 和 `formal_b8_a1/main_b8`，均清理自有容器并释放锁。
全部五轮共 280 次 prefill，其中 168 次 calibration/native/off 原生执行、112 次 F 切换；
另有 280 次范围外 decode。B8 一次调用携带 8 个请求，映射对不当作 kernel 调用次数。

## 原始内容与配置

[data/manifest.json](data/manifest.json) 列出 2,222 个原文件的源路径、公开路径、字节数和 SHA256，
原字节保持不变；连同生成的 manifest、相对路径 suite 和 SHA 清单，`data` 共 **2,225 文件、163,049,942 字节**。
下载后本地逐文件 SHA/字节数核验通过，原始统计及 560 份调用元数据也已重新计算。

| 内容 | 路径 |
| --- | --- |
| 每组 10 阶段原生 dump/stack/construct、配置及回执 | [B1](data/cases/main_b1)、[B8](data/cases/main_b8) 下的 `observation/step0…9/rank0`。 |
| 实际层、调用、输入输出 SHA、请求映射、cache 数值观察、原生 profile/POD | 各 case 的 `native_calls`、`phases`、`module_*`、`controller`。 |
| 实际冻结执行源码、validated runtime、证书与模型/输入指纹 | [source](data/source)，私有完整输入 manifest 排除。 |
| 实际 CPU FP64 collector、patch、stage/source manifest | [collector](data/collector)。 |
| 独立源码分析器、原始 CPU 检查报告 | [analysis](data/analysis)、[cpu_checks](data/cpu_checks)。 |
| 启动门禁及环境失败原始回执 | [earlier_attempts](data/earlier_attempts)，不计入正式通过矩阵。 |

两组使用 `task=statistics`、`level=mix`、`summary_mode=statistics`、`precision=high`、同步采集，
配置仅输出路径不同。step0/1 为 calibration，2/3 为 B，4/5 和 6/7 为两轮 F，8/9 为 OFF；偶数 prefill、奇数 decode。
全部 20 套统计三件套原样保留，不只抽取预期故障节点。

本轮使用已明确披露的统一 **CPU FP64 high collector**，所有实际浮点张量采用相同只读统计，
没有按算子、模式或预期差异筛选。模型计算和张量 dtype 仍为 FP16，完整私有张量不由该统计任务导出。
processor SHA `6688aa3e90849027b12458add4c9da6ac4b334b57e835b742df8403c2fc474b2`，
stage manifest SHA `06ce36bae3463f44a143047d184f53915dbea4dd04b39d79960518e6878d8010`。
这不是未修正默认 msprobe 统计精度下的首差结论。

## 结果报告

- [完整原生策略与覆盖验收](data/reports/execution_acceptance.json)：`PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE`，SHA256 `9056f5b95764938291ec418eeb92190e9fbfc74016f8db106a6054ec40b5252e`。
- [因果链汇总](data/reports/causal_chain.json)、[B1 数值](data/reports/numeric_b1.json)、[B8 数值](data/reports/numeric_b8.json)：完整 Attention、末 query、cache 和 logits 的共同观察检查。
- [B1 完整 logits 离线复核](data/reports/logits_b1.json)、[B8 完整 logits 离线复核](data/reports/logits_b8.json)：直接读取私有原字节重算数值和 Top-1，报告不含完整张量。
- [B1 独立统计](data/reports/main_b1/statistics/summary.json)、[B8 独立统计](data/reports/main_b8/statistics/summary.json)：只用原始统计和源栈找首差，未读取 host、命中、profile 或私有张量。
- [跨 Batch 目标请求审阅](data/reports/cross_batch_target.json)：B1/B8 的 request_00 在原生第 0 层 Q/K/V 已不同，单列该事实，不把本轮写成历史同 Q/K/V 的跨 Batch FIA 故障复现。
- [本地公开包字节核验](local_bundle_verification.json)、[本地统计及哈希重算](local_statistics_hash_review.json)：不执行 NPU；后者 SHA256 `aabace5a52499416554fb0268110d1b9f3cefb6256645d875f0a6bcdd758d806`。

| 最后位置 logits | 变化元素数 | 最大绝对误差 | Top-1 |
| --- | ---: | ---: | --- |
| B1 prefill | 138,344 / 151,936 | 0.03515625 | 不变 |
| B1 固定 token decode | 121,320 / 151,936 | 0.0234375 | 不变 |
| B8 prefill | 1,015,664 / 1,215,488 | 0.046875 | 8 请求均不变 |
| B8 固定 token decode | 1,095,593 / 1,215,488 | 0.046875 | 8 请求均不变 |

两次 F 同结果；OFF 的差异计数全部为 0。
B1 第 0 层有 33 个 Attention 元素变化，B8 有 324 个，末 query 均为 0；
第 1 层末 query 分别有 29、543 个变化。decode 未注入，两组独立首差均从第 1 层 SDPA 的已变化 K/V 输入出现。
模型所有符合条件的层同时切换，后续差异包含传播及后层自己的 M-only 执行，不能唯一归给第 0 层。

20 个阶段对照中 2 个 calibration/native prefill 因源栈不同保留不可对齐，其余 18 个完整对齐。
每对保留 122 项 null/不可观测字段；摘要相等不证明张量相等。
本次矩阵未见 SDPA 输出摘要碰撞，不能将它扩展为所有输入均可由四统计定位。

## 公开包能复算什么

可直接验证 [SHA256SUMS](data/SHA256SUMS)，并使用 [public_suite.json](data/public_suite.json) 重新执行
[review_formal_summary.py](data/analysis/review_formal_summary.py)，复算原始统计首差、完整层/请求元数据和输出 SHA 的对应关系。
报告与原文件均不覆盖；例如输出到新的 `NEW_REVIEW.json`：

```bash
python -B data/analysis/review_formal_summary.py --inputs data/public_suite.json \
  --acceptance data/reports/execution_acceptance.json --output NEW_REVIEW.json
```

manifest SHA256 为 `b8f0ce7dc3b1e55508c03774d312675f7324605d0d0982a572403ac2a2433512`。
传输归档 SHA256 为 `7d8b2a8bbd56a81c684c441856085f4d037043f0b0f478e5c314140f299fc6c3`，
归档仅用于下载；Git 中保存可审阅原文件。`.gitattributes` 禁止行尾转换。

manifest 另列 **305 项远端保留记录**：完整私有输入、原始窗口/整日志、完整 logits，以及私有 B Attention/KV 参考目录。
F Attention/KV 未保存整张量文件，运行时共同 CPU 观察器对其所有元素比较并保留计数、逐请求/逐 query 分布和必要小样本。
公开元数据不能离线重做未保留的 F Attention/KV 逐元素比较，也不能替代依赖私有输入、完整 logits 和整日志的原生联合验收。
不上传模型、`.pt/.bin` 张量、编译库或 DEBUG 大日志；CPU 核验成功也不替代上述实机验收。
