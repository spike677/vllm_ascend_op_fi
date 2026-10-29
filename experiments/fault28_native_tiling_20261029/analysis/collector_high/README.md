# msprobe 26.1.0 任务本地高精度统计修正

本目录保存故障 28 实验使用的 **任务本地修改版 collector** 的来源、最小补丁、验证记录和部署入口。它使用官方公开的 `precision="high"` 配置，在采集器内部对 FP16/BF16 临时数据计算 FP32 统计，保持 `task="statistics"`、`level="mix"`、`summary_mode="statistics"`、`tensor_list=[]`。它不改变模型、权重、attention 实现、原生 API 参数或返回张量，也不保存完整激活张量。

发布或引用依赖该修正的数据时，必须保留这一采集器依赖说明，不能标为“未经修改的官方 msprobe”。高精度统计可减少标量舍入掩盖差异的情况，但四项聚合统计仍可能碰撞，不能保证任意故障都能定位。

## 为什么需要修正

官方配置文档说明 `precision="high"` 使用 Float32 统计，并适用于 statistics 和 mix。当前安装的 26.1.0 包接受这个配置，但其 `PytorchDataProcessor.get_stat_info()` 没有对应的 FP32 转换；反而只在 `precision != high` 时计算非标量浮点张量的 Mean/Norm。因此，仅切换配置会使这些统计缺失，不能获得文档所述的精度。

已核对的官方仓库固定提交中的同一函数 AST 与安装版本一致，存在相同行为。该上游提交的版本声明是 26.0.0，这项交叉检查不代表安装 wheel 与某个上游 26.1 tag 完全相同。来源和行号见 [precision_review.json](precision_review.json)。

- [官方 high 配置文档，固定提交](https://github.com/Ascend/msprobe/blob/e924267f08362b9b053067faa928b0dd6ab52f96/docs/zh/user_guide/dump/config_json_introduct.md)
- [官方 collector 实现，固定提交](https://github.com/Ascend/msprobe/blob/e924267f08362b9b053067faa928b0dd6ab52f96/python/msprobe/core/dump/data_dump/data_processor/pytorch_processor.py)

原始 `bf_statistics_a2` 使用官方默认低精度采集器，独立首差验收未通过：SDPA.1 的输出统计在 B/F 中相同，首个可观测差异出现在后续 `Tensor.pow.7` 的输入。真实 attention 输出已有差异，但 FP16 Mean/Norm 舍入未显示它。该次原始 B/F、dump/stack/construct 和失败推断全部保留；本修正不重写旧数据，也不把后续算子的差异追认成原始采集已定位成功。

## 补丁范围

[collector.patch](collector.patch) 只改变 `get_stat_info()`，模块其余 AST 完全相同：

1. 在既有空张量、特殊格式、float8/float4、checksum 等分支之后，`precision="high"` 对 FP16/BF16 的 `detach()` 临时变量转换为 FP32。
2. 移除 Mean/Norm 分支中错误的 `precision != high` 排除，保留浮点且非 FP64 的原限制。FP32 的 high 模式因此也能正常生成 Mean/Norm。

相同策略统一应用于所有模块、API、层、forward 和 B/F。补丁不接收或检查 arm、layer、target、故障启用状态，也没有调用次数门槛。low 模式保持原函数行为；FP64、bool、整数、complex、空数据等既有规则保持。未引入 FP64 配置或新的采集模式。

`_analyze_tensor()` 仍从原张量读取 dtype，从原 common tensor 读取 shape。统计临时值不写回原张量，也不替代模型输出。现有 hook manager 在统计采集期间设置线程内重入保护；collector 内部转换不会作为新的模型 API 节点写入 dump。实机 pilot 另行核验了节点边界，未仅依赖这项静态推断。

## 来源与不可变绑定

原始 wheel 归档在已检查的机器路径和固定镜像中不可用；**本 stage 不是从 wheel 重新恢复的安装**。`direct_url.json` 记录了原 wheel SHA，构建器逐文件核验已安装完整 `msprobe` 包及其发行元数据与 `RECORD` 的对应关系，之后复制到新 stage，只修改一份 collector 源文件。`RECORD` 自身按其规则无自校验哈希，由 [source_manifest.json](source_manifest.json) 单独记录 SHA。

| 对象 | SHA-256 |
| --- | --- |
| 安装原 collector | `e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011` |
| 任务本地 patched collector | `3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141` |
| stage manifest | `743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af` |
| direct_url 记录的原 wheel | `c004abe1123629ecdfb9938eee25cc8b554694d545c5123ae801c1846e4cfa68` |

新 stage 为主机 `/home/fault28-native-tiling-20261029/collector_high_a1`，容器只读路径为 `/audit/collector_high_a1`，manifest 共列出 513 个文件。原安装目录没有改动。stage 保存原 collector、补丁、来源清单和完整包；文件权限及实验挂载均只读。`RECORD` 保留原安装数据，修改后的文件由新的 stage manifest 明确标识，未伪装成原发行文件。

[build_collector_stage.py](build_collector_stage.py) 创建全新的 stage，拒绝覆盖已有目录。[collector_stage.py](collector_stage.py) 在任何 `msprobe` 导入前验证完整清单与 case 批准的 manifest SHA，选择完整包路径；导入后再次核对实际 module 路径、`get_stat_info.__code__.co_filename` 和文件 SHA。case 的选择字段为：

```json
{
  "statistics_variant": "task_local_fp32_high_v1",
  "approved_collector_manifest_file": "/audit/collector_high_a1/stage_manifest.json",
  "approved_collector_manifest_sha256": "743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af"
}
```

HF 集成只增加采集器选择、`precision="high"` 配置和证据记录。每组 `result.statistics_collector`、`receipt.statistics_collector` 与 `collector_provenance.json` 记录实际加载路径及 SHA；receipt 还绑定 provenance 文件 SHA。标准 low case 不使用该 stage。

## 已完成验证及边界

[CPU v3 检查](cpu_checks_v3.json) 在无设备、无网络的临时容器中通过 23 项真实 PyTorch CPU 测试。测试覆盖 FP16/BF16/FP32/FP64 的向量、非连续、空张量、标量，以及 bool、整数、complex、NaN、Inf。每项核对原对象 id、data_ptr、dtype、shape、stride、device、storage_offset、版本计数和全部字节保持不变；low 输出与原函数 AST 的实际执行一致。

有限浮点输入另与已表示数值的 CPU FP64 Mean/L2 参考比较，容差为 `rtol=2e-6, atol=2e-7`。实际 `_analyze_tensor`、官方 DataWriter 的 flush、placeholder 替换及 JSON 往返验证了原 dtype/shape 保留，并且 FP32 统计不会按 FP16 元数据回舍入。例如两个 FP16 输入在 low 下 Mean/Norm 同为 `1/64`，high 下分别为 `1/64` 与 `1.000000238418579/64.00001525878906`。这是一项统计精度测试，不是故障定位通过的替代证据。

`high_statistics_smoke_a1/high_smoke` 已完成 stock FA、无注入、同进程两次完整 prefill 的实机透明性验收。独立审查将两次 forward 的 28 层分别与旧 stock B8 和旧 low-statistics 数据比较，共 112 次 target 调用对比：目标 slot 0 的全部 heads/positions 的 Q/K/V 前后值、attention 输出及 logits 哈希一致。这不是全 batch 所有元素逐一重算的声明。

每个 step 的 1621 个 dump 节点、425 个 module 节点、28 个 SDPA 节点，与 low 的顺序、非统计元数据、construct 和规范化后的模型调用栈一致；规范化只处理已审计 probe 源文件行号变化。未出现 collector 转换伪节点，513 个 stage 文件及实际加载的 patched source 均通过核验。见 [实机透明性报告](../../source_matching/fa_policy/build_evidence_a2/high_collector_transparency_review/summary.json)，SHA 为 `393c0a7138ae8ff88eb0e6f2a98217bc0f7130b30ef1f7334f265b4f1e7f35a0`；[独立审查汇总](../../source_matching/fa_policy/build_evidence_a2/collector_high_independent_final_review.json) SHA 为 `80714ee3c7c95ec855ccef92f58165fba03650975f40e2278be16a6bf0be5587`。

正式 `bf_statistics_high_a1` 已完成独立验收：B/F 两个 step 都首先观测到 layer1 `Functional.scaled_dot_product_attention.1.forward` 输出 Mean 不同，输入统计匹配；单独运行指纹检查确认模型值路径与诊断 OFF/ON 一致。首差前的不可观测字段继续保留。见 [正式证据与范围](../../../../artifacts/attention_tiling/fault28_native_fa_high_statistics_bf_20261029.md)。该结果依赖本目录披露的任务私有修正，不把官方默认 low 的下游首差改写为成功。

## 许可证与归属

[original_pytorch_processor.py](original_pytorch_processor.py) 保留原始 MindStudio/Huawei 版权头。修正后的 stage 文件同样保留该头；新增注释和补丁只说明本任务的局部修改，不替换上游归属。

本次安装包及 stage 未包含许可证全文，源码版权头引用的 [Mulan 官方页面](https://license.coscl.org.cn/MulanPSL2) 在获取时超时。因此 [LICENSE.MulanPSL-2.0](LICENSE.MulanPSL-2.0) 按字节复制自官方 SPDX license-list-data 的固定版本 [v3.26.0 原文](https://raw.githubusercontent.com/spdx/license-list-data/v3.26.0/text/MulanPSL-2.0.txt)，没有重写条款。许可证文件 SHA-256 为 `eb7a1d713eb919b146787629e22e4c975cb701f529a65d4d7e0fcd417558bf1c`。
