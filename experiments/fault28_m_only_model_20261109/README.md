# 故障 28：M-only 完整模型传播补验

2026-11-09，124 物理卡 5 已通过 **M128→64、实际 nRatio 始终为 8** 的完整模型传播验收。
B1/B8 均出现稳定 logits 差异，OFF 后完整 Attention、末 query、KV 和 logits 精确恢复。
联合策略验收为 `PASS_MODEL_ALL_ELIGIBLE_NATIVE_STRATEGY_COVERAGE`；两组数值结果均为
`M_ONLY_FULL_MODEL_LOGITS_PROPAGATION_OBSERVED`。本轮补齐了 M-only 到模型 logits 的因果证据。
本轮独立于[旧 joint 全模型实验](../fault28_model_coverage_20261030/README.md)，
也独立于[真实 FIA 图子图实验](../fault28_fia_graph_20261106/README.md)。

目标是补齐“只变 M 能否经过完整模型影响 logits”的证据。单层最后一个 query 不变并不预设模型最终结果；
本轮允许 logits 变化，也允许中间张量变化但 logits 不变。数值变化不作为策略覆盖通过的必要条件。

[实现、调用链和关键代码](analysis/IMPLEMENTATION.md) · [离线工具与观察限制](analysis/README.md)
· [本次原始 statistics/mix 与审计证据](../../artifacts/native_execution_strategy/fa_m_only_model_statistics_20261109/README.md)

联合报告 SHA256：`9056f5b95764938291ec418eeb92190e9fbfc74016f8db106a6054ec40b5252e`。

| case | 全部 SDPA | 受控 prefill 实际切换/符合条件 | 受控 decode 透传 | 受控层/请求映射对 |
| --- | ---: | ---: | ---: | ---: |
| main_b1 / formal_b1_a4 | 280 | 56/56 | 56 | 56 |
| main_b8 / formal_b8_a1 | 280 | 56/56 | 56 | 448 |
| 合计 | 560 | 112/112 | 112 | 504 |

全部五轮共 280 次 prefill：112 次受控切换、168 次 calibration/native/off 原生执行；
另有 280 次明确范围外 decode。全部 28 层和 B8 的 8 个请求都在分母内，
没有缺失、重复、UNKNOWN 或游离 attention launch。B8 一次 kernel 承载 8 个请求，映射对不等于调用次数。
最终从完整日志重新解析全部 280 个 prefill 的实际 raw/POD，并独立枚举 560 次 attention 执行。

## 实际执行与唯一干预

原始 HF Qwen3-0.6B FP16，28 层、Q heads16/KV heads8、D128。
固定 8 个不同自然文本请求，各 2048 个 prefill token 和 1 个 teacher-forced 后续 token。
B1 使用 request_00；B8 使用 request_00…07。各轮实际 token/position、模型与安装源码均核 SHA。

```text
calibration → native baseline → M-only → M-only → OFF
```

每轮创建新 KV cache，完成完整 prefill 和一次 Q1/KV2049 decode。各轮采用同一个固定后续 token，
不让自由生成 token 分叉改变后续输入。每次原始 SDPA 调用一次，返回原对象；Q/K/V 和输出不被注入器改写。

| 项目 | B/OFF | F（两次 controlled） |
| --- | --- | --- |
| 适配器及选择 | 同一个 validated FA/S1S2、`query_tile` | 同左 |
| 实际 M/nRatio | M128 / R8 | M64 / R8 |
| 派生计划 | 原生 host 生成 | 同一 host 按 M64 重新生成 |
| 符合证书的 prefill | 原生观察 | 每次匹配均切换 |
| Q1/KV2049 decode | 原 API | 原 API，明确范围外 |

控制只发生在原生 host 的合法 factor 选择边界，不直接覆盖已序列化 tiling、不修改 kernel。
`m_only_contract.py` 同时约束模型入口和验收器，joint 或实际 nRatio4 会失败。
本轮沿用原 23 文件 runtime、vendor 和证书，没有重新设计底层机制或扩大支持范围。

## M-only 数值传播

完整 Attention、最后 query、真实 KV cache 和 logits 均按共同逻辑观察。
B1 第 0 层 prefill Attention 有 **33** 个元素不同，最后 query **0** 个不同；
第 1 层 Attention 的最后 query 已有 **29** 个元素不同，第 1–27 层 cache 和最终 logits 也出现差异。
这说明单层最后 query 不变不能推出完整模型最后位置不变。

| 最后位置 logits | 变化元素数 / 全部元素数 | 最大绝对误差 | Top-1 |
| --- | ---: | ---: | --- |
| B1 prefill | 138,344 / 151,936 | 0.03515625 | 不变，1119 |
| B1 固定 token decode | 121,320 / 151,936 | 0.0234375 | 不变，5662 |
| B8 prefill | 1,015,664 / 1,215,488 | 0.046875 | 8 个请求均不变 |
| B8 固定 token decode | 1,095,593 / 1,215,488 | 0.046875 | 8 个请求均不变 |

B8 第 0 层 Attention 有 **324** 个元素不同，逐请求为 `[39,50,44,42,37,34,36,42]`，
末 query 在全部请求中仍无差异；第 1 层末 query 合计 **543** 个元素不同。
两组 prefill 的全部 28 层 Attention 均有变化，真实 cache 的第 1–27 层均有变化；
未注入的 decode 第 0 层保持一致，第 1–27 层及全部请求 logits 承接传播。
此前单算子回放的 27 个变化元素来自另一层的固定 Q/K/V，不能与本轮自然模型第 0 层计数混用。

两轮 M-only 的完整各层 Attention、末 query、KV 和 logits 哈希一致，OFF 逐字节恢复 native。
完整私有 logits 另行读取原字节、核 SHA 并复算逐请求差异，与运行时计数一致。
本轮对全部符合条件的层同时注入，不能将后层每一项差异唯一归因于第 0 层。

B1/B8 的两次 F，原始 msprobe 独立首差均为 `Functional.scaled_dot_product_attention.0.forward` 的输出 Mean/Norm。
该节点 Q/K/V 的 12 项有限统计一致，完整 Q/K/V SHA 由独立运行元数据核对。
decode 不注入，首差从第 1 层 SDPA 输入出现，符合接收已变化 cache/隐藏状态的传播方向。
统计推断只读取原始 dump/stack/construct，不读取策略、命中或私有张量来指定首差。

20 个阶段对照中，2 个 calibration/native prefill 因手动 scope 与 Session 源栈不同而保留不可对齐，
其余 18 个对照完整对齐。每个阶段对照的 122 项 null/不可观测字段保留；
本矩阵没有 SDPA 输出统计碰撞。公开原始统计及 560 份调用元数据重新计算通过。

## 跨 Batch 结果必须分开解释

同一 request_00 在 B1/B8 使用相同 token 和位置，但其原生第 0 层 Q/K/V 完整逻辑 SHA 已不同；
原生目标 logits 有 138,940 个元素不同。这不能作为“相同 Q/K/V 经 FA 后首先变差”的跨 Batch 证据。
本轮的 M-only 因果比较是**各自 Batch 内**的 B/F：首个受影响 Attention 的 Q/K/V 完整 SHA 一致，
仅实际 M 选择及其原生派生计划改变，随后输出、cache 和 logits 改变。
跨 Batch 逐请求审阅另列于 `cross_batch_target.json`，未对不同大小的整 batch 统计强行对齐。

## 采集和可复核性

两组配置：`task=statistics`、`level=mix`、`summary_mode=statistics`、同步 `precision=high`。
本轮从开始统一使用已披露的 **CPU FP64 high collector**：对所有实际浮点张量采用相同的只读统计，
不按模式、算子或预期首差过滤；模型计算仍为 FP16。
processor SHA 为 `6688aa3e90849027b12458add4c9da6ac4b334b57e835b742df8403c2fc474b2`，
stage manifest SHA 为 `06ce36bae3463f44a143047d184f53915dbea4dd04b39d79960518e6878d8010`。
这不是未修正默认统计精度的结论；原生 null 保留为不可观测。

每阶段保留原生 `dump.json`、`stack.json`、`construct.json`、共同配置和完成回执。
主交付不含 `.pt`、完整模型、私有完整张量或 DEBUG 大日志。
另存全部层/请求映射、实际 raw/POD、输入输出 SHA、逐元素观察摘要和真实 cache 元数据。

B Attention/KV 私有参考和各轮完整 logits 保留在远端，F Attention/KV 未保存整张量；
运行时共同观察器比较了所有元素。公开包可重算统计首差和检查完整哈希绑定，
不能离线重做未保留的 F Attention/KV 逐元素比较，也不包含原生联合验收所需的全部私有输入和日志。

## 设备、版本与尝试记录

124 的物理卡 5、Ascend 910B4，CANN 9.1（`V100R001C11SPC001B243`），
PyTorch 2.10.0、torch-npu 2.10.0.post4、Transformers 5.5.4。
固定模型、分词器、安装源码及私有 vendor 指纹；本轮输入 manifest SHA 为
`ed6b27940c3336efe8cd941fda9da191edcbd8257c34587396345f24d7994fe1`。
受控证书只支持已验证的 FP16 BNSD/B1/B8/2048-token causal FA 形状。

使用共享资源锁、每次调用前门禁和运行期间周期门禁；单次最多一个 case、16 GiB 内存、8 CPU，
日志上限 16 GiB、case 上限 3600 秒。只映射卡 5，不抢占卡 2。
按完整身份复核的两个既有容器可空闲共存，不能将此称为所有设备独占。

| attempt | 结果 |
| --- | --- |
| formal_b1_a1 | 外部 ATK 映射卡 5，被启动门禁拒绝，未形成模型采集。 |
| formal_b1_a2 | 嵌套只读 collector 挂载目标缺失，OCI 启动失败；原 unresolved cleanup 回执保留，随后独立核查无遗留自有容器。 |
| formal_b1_a3 | 模型初始化期间出现外部 NPU 初始化探针，严格门禁中止；没有完成 SDPA，自有容器已清理。 |
| formal_b1_a4 | 10 阶段、280 次 SDPA 完成，自有容器清理通过。 |
| formal_b8_a1 | 10 阶段、280 次 SDPA 完成，自有容器清理通过。 |

正式前补充了 CPU-only 的容器预留后空闲稳定等待；外部进程存在时不执行模型/NPU 调用，
未放宽后续门禁，也未停止外部服务或容器。捕获源码在两组正式 case 间保持冻结。

本轮结论仅适用于上述固定 FA 路径、输入和一个固定 token decode。
历史盘古 FIA 图模式、Batch 原生触发 M 的条件、某条核内 Reduce 指令的唯一归因及答案质量仍需分别验证。
