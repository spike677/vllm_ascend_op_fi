# Fault39 蓝区124验收

本次只实现 `FLOAT_MATMUL_CANCELLATION`。注入在真实 CANN 8.5.1
`OpExecCache::Run` 输入地址替换边界，保留原模型参数和输出地址。
临时 BF16 输入有限，每列正负项严格抵消，数学结果为零；实际 MatMul 的归约
可以产生 Inf。这里验证的是有限范围及归约行为，不能据此识别某条硬件累加指令。

内源框架不可访问，状态仍为 `INNER_FRAMEWORK_INTEGRATION_PENDING`。
本包没有冒充已经修改黄区的 catalog、bindings、CLI 或生产服务。

| 阶段 | 当前验收 |
|---|---|
| P0 / P1 | 历史源码溯源、独立核心/文档契约；89项CPU/Mock测试与aarch64核心/ABI检查通过 |
| P2 / P3 | 真实 MatMul metadata/cache/live slot 证明、三种无损替换与恢复通过 |
| P4 | 五种真实算子模式及完整Qwen五模式模型响应检查通过 |
| P5 | Qwen3与Llama两种真实架构通过发现、无损、注入和恢复；统计对齐范围见下文 |
| P6 | 真实TP2 rank过滤、正常同步清理、重复启停和Native owner拒绝通过；内源多fault组合及故障恢复压力另行验收 |

## 原生机制与动态尺寸

| 实验 | 实际结果 |
|---|---|
| 只读发现 | BF16 `aclnnMatmul`，typed GWS → formal metadata → cache → Run 三个 live storage 地址全部对应 |
| self / mat2 / dual 无损替换 | 完整算子输出逐字节相同，原输入 backing 哈希未变，OFF 精确恢复 |
| M1/K12800/N5120，A=2^58，half | finite=1024（其中 zero=1024）、+Inf=2048、−Inf=2048、NaN=0 |
| 相同尺寸与 A，block256/512/1024、alternating | 四组均为 5120 个零；合法命中，未观察到 range anomaly |
| M3/K8192/N80，auto A=2^58 | +Inf=144、−Inf=96；原输入未变，OFF 精确恢复 |
| 活动 owner 隔离 | foreign configure/disable/clean 返回 −4，stale generation disable 返回 −5；状态完整不变，原 owner 随后正常执行 |

`finite` 包含 `zero`，不能相加。历史 A2/A3 分布作为参考，不是跨硬件固定答案。
证据见 [P2](native_discovery.json)、[P3](lossless_validation.json)、
[五模式矩阵](cancellation_matrix.json) 和
[动态尺寸/owner 实测](../evidence/124_20261116/native/dynamic_owner.json)。

## Qwen3 原始 msprobe 独立首差

使用本轮未改动、以源码 SHA-256 固定身份的 msprobe `statistics/mix`，显式固定
`abs(F-B) > 1e-8 + 1e-5*abs(B)`，只读 B/F 原始三件套、配置、collector identity。
比较器不读 native 命中、目标签名、策略日志、生成响应或完整张量。

| 项目 | 结果 |
|---|---|
| 模型 | 真实 Qwen3-0.6B，BF16，vLLM 0.18 eager，单请求、TP1 |
| 实际采集 | 两组均 step0–7；本次分别对应一次 prefill 和七次 decode |
| step0 | 无可观测超阈值差异；6102 个 null 字段保留，不宣称所有张量相等 |
| 独立首差 | step1，`Functional.linear.3.forward/output/0`，shape `[1,1024]` |
| Max | **0.76953125 → +Inf** |
| Min | −0.640625 → +Inf |
| 首差前不可观测字段 | step1 有 258 项；Mean/Norm 为 null，未替换成零 |
| Python 输入 | `[1,3072]` 和 `[1024,3072]` 的可观测极值相同；Mean/Norm 不可观测 |

[基线 dump](../evidence/124_20261116/model_qwen/baseline/observation/step1/rank0/dump.json) ·
[故障 dump](../evidence/124_20261116/model_qwen/fault/observation/step1/rank0/dump.json) ·
[独立报告](../evidence/124_20261116/model_qwen/independent_report.json)

原始 construct 将该 API 归到第0层 `mlp.down_proj`；stack 的第36项记录
`down_proj → AscendRowParallelLinear → unquantized_gemm → functional.linear`。
这是定位后的模块归属，不是注入配置中的层名。继续向下才是 native 证据确认的
`aclnnMatmul → OpExecCache::Run`，因此 Python 统计里显示 `linear` 是正常观测边界。

独立定位结束后，再结合实际 native 证据：自动选择的 cache 几何为
M1/K3072/N1024，auto A=2^59，真实 MatMul 输出的 1024 项全部为 +Inf，
一次命中、真实 scheduler 请求映射、同步释放及 OFF 恢复均通过。
输入替换发生在 Python 之下，所以 msprobe 保留原调用方输入是预期观测边界。
统计首差本身不能唯一证明内部机制，请求映射也由独立 runtime 证据补充。

正常 token 为 `[7281,11,10339,3170,432,374,2989,311]`，故障为
`[7281,0,0,0,0,0,0,0]`；文本从 ` Also, explain why it is important to`
变为 ` Also!!!!!!!`。关闭后恢复正常 token/text。后续 step 已使用不同生成输入，
其 `index_select` 首差不是新注入根因。此处不验收问答质量或全部模型中间张量一致性。

## 完整模型五模式对照

Qwen3 使用同一实际几何 M1/K3072/N1024，五组 auto 幅值均为 A=2^59。

| Pattern | 实际1024项MatMul输出 | 本次8-token响应 |
|---|---|---|
| half | 全部 +Inf | 改变 |
| block256 | 全部零 | 与 baseline 相同 |
| block512 | 全部零 | 与 baseline 相同 |
| block1024 | 全部 +Inf | 改变 |
| alternating | 全部零 | 与 baseline 相同 |

五组实际命中、请求映射、同步释放及 OFF 恢复均通过。
block1024 与历史大 K 算子结果不同，说明不能把固定输出分布当成验收标准。
“没有 range anomaly”仍是有效控制结果；token 相同也不证明 logits 或全部中间张量相同。
[模型矩阵及原始回执路径](model_cancellation_matrix.json)

## 第二架构：真实 Llama 实现

使用 HuggingFaceTB/SmolLM2-135M-Instruct 的固定 revision 和真实权重，
`architectures=LlamaForCausalLM`。同一适配器从实际调用自动选择
M20/K1536/N576，auto A=2^60，未增加模型名或层名分支。
三种无损替换、请求映射、真实 half 注入和 OFF 恢复均通过。
实际输出11520项全部为 +Inf。

正式重采的 baseline 只有 **step0**，fault 为 **step0–7**。
比较器如实返回 `STEP_SET_MISMATCH`，不宣称整段 decode 对齐。
共同 prefill 的 step0 单独按同一固定容差复核，首差为
`Functional.linear.3.forward/output/0/Max`：**30.5 → +Inf**。
其之前的 null 仍不可观测。

本次普通 completion 提示下 baseline 提前返回 EOS token `[2]`；fault 返回
八个 token 0。两组解码文本均为空，因此结论是 **token/停止行为改变，文本未变**。
OFF恢复 `[2]`，不将此输入作为问答质量验收。

[真实模型与权重身份](multi_model_validation.json) ·
[共同step0独立报告](../evidence/124_20261116/model_llama/step0_independent_report.json) ·
[完整步数不匹配报告](../evidence/124_20261116/model_llama/independent_report.json)

旧采集曾把提前EOS后的OFF调用带入baseline。原件保留在私有实验目录，正式数据
使用请求结束立即 `finish_capture` 的版本重采，并保存 `CAPTURE_FINISHED` 回执。
不删改旧统计来伪造正确步数。详见 [被替代/失败attempt](superseded_attempts.json)。

## TP2 实机过滤与恢复

124 物理卡2/5、真实 rank0/1 的无损三模式及 half cancellation 均通过。
只选 rank0：命中1次、替换1次；rank1 命中0次、替换0次。
实际 rank0 几何 M1/K1536/N1024，auto A=2^60，输出1024个 +Inf，
生成 token 从第二项开始改变，OFF恢复 baseline。每轮都建立新的请求/KV缓存。
这不升级为 TP4、混合多请求或内源框架集成验收。
[TP2 无损回执](../evidence/124_20261116/tp2/lossless_validation.json) ·
[TP2 故障回执](../evidence/124_20261116/tp2/cancellation.json) ·
[恢复验收](recovery_validation.json)

## 离线复核

从仓库根目录运行，报告输出到新的文件：

```bash
python fault39/tools/compare_model_capture.py \
  --baseline fault39/evidence/124_20261116/model_qwen/baseline \
  --fault fault39/evidence/124_20261116/model_qwen/fault \
  --rtol 1e-5 --atol 1e-8 --output /tmp/fault39-qwen-comparison.json
```

算子级独立证据另见 [operator 报告](../evidence/124_20261116/operator/independent_report.json)：
`Torch.matmul.0.forward/output/0/Max` 为 −0.3359375 → +Inf。算子级与模型级不能互相替代。

每个 `operator`、`model_qwen`、`model_llama` 目录都只包含 baseline/fault 两组采集。
对应 `baseline.zip` / `fault.zip` 的成员路径为 `data/cases/baseline` / `data/cases/fault`，
解压字节与原始 JSON 一致；不同模型请解压到各自目录。
全部公开证据可用 `tools/verify_evidence.py` 和
[`checksums.json`](../evidence/checksums.json) 离线核验。

```bash
python fault39/tools/verify_evidence.py \
  --manifest fault39/evidence/checksums.json --root fault39/evidence
```

[执行版本映射](execution_manifest.json) 将各组原始证据对应到实际 attempt、源码快照和构建。
早期 P2/P3/P4 使用 build-a9；正式算子 B/F 使用最终 build-a10，并在每组内部重新通过
发现、真实地址证明、三种无损替换和 OFF 门禁，不能把两个构建的回执混称为同一次运行。

## 验收边界

- 自动选择来自实际 API、shape、stride、dtype、format、cache；没有模型名或层名分支。
  严格 `unique` 在 Qwen 的五种候选上明确拒绝。实测模型显式启用可选
  `max_k_min_m_unique`，仍拒绝最终候选歧义和不稳定。
- 唯一性是结构/cache 级，不是唯一模型层。本轮每次 enable 最多一次命中，
  不宣称所有层或所有请求持续覆盖。
- 已验证零 offset 的真实输入路径；其他合法 stride/offset 的通用计算与越界检查
  有 CPU 测试，尚未穷尽 NPU 实机覆盖。
- 正常同步停止和零 leases 有实测；设备同步/释放失败、超时及不可达 worker
  的失败恢复仍属模拟/契约验证；Run 与 disable 真正并发的压力尚未实机覆盖。
  真实内源多 fault 共存尚待黄区接线验收。
- TP2 已用独立双卡实测验证 rank0 选择和 rank1 零替换；TP4 只有模拟控制测试。
- 不公开权重、完整 tensor 文件、vendor 库或头文件。统计 JSON 原字节保留，
  包括 null、Inf、无差异和不能完整对齐的结果。

源码及构建身份见 [构建指纹](../evidence/124_20261116/build_manifest.json)、
[Native 源代码一致性](native_source_consistency.json)、
[最终执行器源码一致性](runner_source_consistency.json)、[CPU/Mock 测试](cpu_validation.json)。
黄区接线见 [实施文档](../docs/yellow_integration.md)。
