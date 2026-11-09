# M-only 全模型证据工具

本目录包含明确的实机启动入口 `run_case.py` 和离线分析工具。其余报告工具不执行 NPU，
只读原始采集，不更改策略或输入。最终 NPU 结论见[实验 README](../README.md)。

| 工具 | 输入与职责 |
| --- | --- |
| `run_case.py` | 启动一个有资源锁、设备门禁和时间/日志/内存预算的实际模型 case；每次新 attempt。 |
| `progress_cpu.py` | 只读阶段回执和逐调用 sidecar，不能据此认定验收通过。 |
| `run_analysis.py` | 按顺序运行框架分母核验、原始统计独立首差和整日志 attention census。 |
| `audit_model_coverage.py` | 从实际模型配置、无条件模块 hook、msprobe SDPA/construct 建立完整分母。 |
| `compare_model_statistics.py` | 仅按原始 `dump/stack/construct` 的真实模型 API 顺序寻找首个可观测差异；保留 null、无差异和无法对齐。 |
| `native_launch_census.py` | 扫描完整 DEBUG 日志，独立枚举 prefill/decode attention 执行，拒绝游离调用或窗口错配。 |
| `prepare_review.py` | 将已完成的 B1/B8 及各自分析绑定到新的 `suite.json`，不覆盖旧结果。 |
| `finalize_review.py` | 两组均完成后顺序执行全部离线联合检查，不等待设备、不启动模型、不自动发布。 |
| `accept_model_execution.py` | 重新读取整日志、核原始 raw、205 项 POD、169 项计划、公共输入、证书及每层/请求映射；严格拒绝 nRatio4 或 joint。 |
| `summarize_numerics.py` | 验算共同 CPU 观察器的完整 Attention、末 query、KV 和 logits 元数据，重哈希私有 B 参考，检查重复和恢复。 |
| `review_exact_logits.py` | 直接读取完整私有 logits 字节，独立复算逐请求差异、最大误差和 Top-1，并核原始 SHA。 |
| `review_formal_summary.py` | 从公开原始统计重新计算首差，再单独核对完整输出 SHA；不读取 host 命中或私有张量。 |
| `review_causal_chain.py` | 连接已验收策略、独立统计及数值传播报告；允许 logits 不变，不预设首差层。 |
| `compare_cross_batch_target.py` | 按 request_00 对齐 B1/B8 的逐请求 SHA 和完整 logits，单独报告跨 Batch 上游差异。 |
| `package_evidence.py` | 原字节导出统计及元数据；私有完整张量、输入、模型、完整日志和二进制不进入公开包。 |
| `verify_public_bundle.py` | 下载后逐文件核 SHA、字节数、20 套统计三件套与私有数据排除清单。 |

本轮固定 `query_tile`：B/OFF 是 M128/R8，F 是 M64/R8。每个 case 的五轮顺序为
`calibration → native → controlled → controlled → off`；每轮独立 cache，包含 prefill 和一次固定 token decode。
模型进程仅对合法的 prefill 应用策略；Q1/KV2049 decode 记录为范围外并原调用透传。

离线验收命令在 124 上使用 `PYTHONPATH=/home/fault28-m-only-model-20261109/runtime:/home/fault28-m-only-model-20261109`。
这只是导入冻结解析器，CPU 工具不初始化设备。例如：

```bash
python3 -B analysis/run_analysis.py --attempt formal_b1_a4 --case main_b1 \
  --output /home/fault28-m-only-model-20261109/outputs/analysis_b1_a1
python3 -B analysis/summarize_numerics.py \
  --case-dir /home/fault28-m-only-model-20261109/outputs/formal_b1_a4/main_b1 \
  --output /home/fault28-m-only-model-20261109/outputs/numeric_review_a1/numeric_b1.json
```

路径和输出目录必须对应实际 attempt；已有结果不覆盖。CPU 成功不能替代 NPU 验收。

msprobe 使用统一 CPU FP64 `high` 统计观察器，公开 processor、patch、stage manifest 及采集 provenance；
所有实际浮点张量均采用相同观察逻辑，没有按算子、模式、预期差异过滤。原始张量 dtype 和模型计算保持 FP16。
统计比较保留默认不可观测项，摘要相同不证明张量相等。

完整 B Attention/KV 私有参考和各轮完整 logits 保留在远端；F Attention/KV 不落整张量文件，
共同运行时观察器逐元素比较后保存计数、逐请求/逐 query 计数、末 query SHA 和有限小样本。
因此公开元数据可以重算统计首差和检查哈希绑定，不能离线重做未保留的完整 F Attention/KV 逐元素比较，
也不能仅凭公开子集重新执行依赖完整日志、私有输入和 logits 的原生联合验收。
