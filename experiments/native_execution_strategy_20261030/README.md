# Native execution strategy：124 验证 harness

状态：第一期正式矩阵通过 `PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1`：20 case、56次真实 API 调用。
已签发当前源码/构建的 `cann9.1-s1s2-v1` 证书；另4次 `validated` 别名检查通过，结果单独记录。
达到固定 FA 路径的执行策略与数值现象验收，模型级影响未测。旧 Fault28 结果没有替代本轮新构建验收。

本实验执行真实 `torch_npu.npu_fusion_attention_v3`，每个 step 只调用一次原 API。
输入来自此前 Qwen3-0.6B FP16 模型 layer 1 的实采 Q/K/V；这里执行的是算子 replay，
不是模型 forward。所有 batch 槽位均复制原实采 batch 0，不额外制造离群值，不改变 mask、
scale、精度属性或输出。V 保留原来的非连续 stride。

## 固定矩阵

| 阶段 | case | 每 case 调用 | 合计 | 支持范围 |
|---|---:|---:|---:|---|
| 原生 discovery | B1/2/4/8 × 两次独立重复 = 8 | 1 | 8 | B2/4 只标记 `observation_only` |
| 受控策略 | query_tile/key_group/joint × B1/8 × 两次独立重复 = 12 | 4 | 48 | 新桥接以显式 experimental 完成验收，再发证 |

受控 case 在同一进程、同一输入、同一策略实例下依次执行：

1. `native`：从真实 native host/raw 证据获得当前策略。
2. `controlled`：通用 `Policy.evaluate` 从适配器给出的合法候选中选择。
3. `controlled`：再次求值并触发，验证 `every_match` 没有首调用或次数限制。
4. `off`：同一 vendor 内停用控制，核验原生选择与输出恢复。

控制范围不使用旧 `batch_m` 特判。B1 和 B8 都需要通过通用 context/policy/adapter 路径。
步骤 0–3 只是算子调用序号，不是模型 token 或层号。56 是实际调用预算；discovery 的
`max_cases` 配置不能代替全实验预算。

## 实际结果与验收边界

正式输入、版本和策略范围固定，不把发现扩展成跨形状支持。124物理卡5，Ascend910B4，
torch 2.10.0 / torch-npu 2.10.0.post4，CANN `V100R001C11SPC001B243`，
官方源码 `c77c0829f1f0f0b8f882db1d9b051a758b605472`，FA S1S2 key `22619644976`。
正式执行为 `discovery_a4`、`controlled_query_tile_a1`、`controlled_key_group_a1`、`controlled_joint_a1`。

| 受控目标 | 每 case 的实际 M/nRatio | 完整目标 b0 改变元素 | 改变 query | 最大绝对差 |
|---|---|---:|---:|---:|
| query_tile | 128/8 → 64/8 → 64/8 → 128/8 | 27 | 23 | 2^-13 |
| key_group | 128/8 → 128/4 → 128/4 → 128/8 | 83,176 | 1,536 | 2^-11 |
| joint | 128/8 → 64/4 → 64/4 → 128/8 | 83,181 | 1,539 | 2^-11 |

这些完整目标差异在 B1/B8、两次独立进程重复和连续两次触发中一致，无非有限输出。
64项要求相等的完整目标/计划比较全部通过，包括 stock/private 原生透明性、关闭后逐字节恢复
和重复稳定性。24次受控调用真实切换，12次关闭恢复；返回 tuple 的对象身份保持不变，
每 step 只执行一次原 API。逻辑输入 SHA 不变；首次原生调用可观察到共同的格式0→2转换，
分析只接受该受限单向变化，不把其他 metadata 变化静默忽略。

原生 B1/2/4/8 的 M/nRatio 均为128/8，未发现这两个选择器的 Batch 切换。
完整执行计划有6项随 Batch 改变，涉及 bOuterSize、core 分配和 transpose 区段；
同一 b0 的完整输出仍逐字节一致。因此本轮不能证明原生 Batch→M→Reduce 的历史故障因果链。

独立 msprobe 固定比较88对：24次原生→受控比较的输入统计无差，首个主 attention 输出差异
均位于 `NPU.npu_fusion_attention_v3.0.forward` 的输出；12次 off 主输出摘要恢复。
同进程和跨进程重复均稳定。41对仅辅助返回有差，11对无可观测返回差，
12对 stock/private 因完整调用栈不同保留 `INCONCLUSIVE`，不丢帧强行对齐。
76对可对齐比较每对均保留20项全局不可观测字段，主 attention 输出不可观测字段为0。
seed/offset 不作为主输出故障；有限统计相等不等于张量相等，物理 kernel 内唯一首错指令未定位。

正式运行时清单 SHA `6982e9b83717b03349fa57707f635dfcfc3bc46ae778d676442c03bc734b6e7a`；
新 vendor 清单 SHA `42173485d5d3c86b57cd8a847dbce2dfd1fc42004c1a451bb282b7aa211d9ef1`；
最终验收报告 SHA `3399b779234d84b6f58d4e23a5a8d15aa18d08fcee737f4c25ace33122e93579`；
独立88对统计报告 SHA `1afea9252d85365404338c07fd8e50248e90216d1419dbd7d9d5207ac3fcd285`。
证书同时绑定当前15份 Python 依赖、34项公共输入、166项已解码执行字段和三个完整64字节区段。
205字段与192字节未展开区段合计绑定完整1464字节；仅语义比较排除已审计的5项 padding，原 raw 保留。

`validated_alias_a1` 的4步使用 `allow_experimental=false`，实际解析
`validated → cann9.1-s1s2-v1`，证书 SHA
`be58b4e77098a8911ab97ad8711144a5a5999dc3742080726c594ad88e8fcc5b`。
其完整输出/169项计划/34项输入均与正式 query_tile B8对应步骤一致，关闭精确恢复。
独立4对 msprobe 同样在两次受控调用的 FA 主输出发现首差，并确认 off 摘要恢复。
完整 canary 复核报告 SHA `c04ffcfcf03124f361a6cfaa44ded15bdaf96a091637df281c4f2fd9f593110f`，
独立统计报告 SHA `7acf5caa5b08f2169f18269fef70df11827b17ae4ad6a36906383cfaf559e513`。
canary 运行时包含新证书/报告/目录，清单 SHA
`ca0bf188bdc031ea3e4249c03ac13a9a8f751cd5ae8485655d9048c9f6cdba3b`。
FA 的15份 Python 依赖未变；采集 harness 只新增显式布尔 experimental 开关支持、
将该值传给证书加载器，并将 vendor inventory 结果与证书状态分开记录。
正式56次与 canary 采用各自不可变源快照，未把后者合并发证覆盖范围。

公开原生统计、完成回执、源码快照、分析和 gzip 日志见
[本轮证据](../../artifacts/native_execution_strategy/fa_phase1_statistics_20261030/README.md)。
公开包清单 SHA `dcb27c99df5d22fa32557bebb35962b09458bc17fedf632c50d63fe6265332d0`，
1111份受清单绑定的文件另加清单本身，本地副本已逐文件复核。
完整 Q/K/V、目标输出和 FP64 参考值留在124；报告中的精确比较可以在124重算，公开统计本身不足以代替它。

准备阶段 `discovery_a1` 在 Docker 挂载前失败（0调用）；`discovery_a2` 在第5次调用后因
CANN 字符分片被错误逐片分词而停止（1024可分成102与4）。修复后先完成前代 `discovery_a3` 的8次发现，
随后补齐192字节区段绑定并重冻完整正式矩阵。上述13次前代调用不计入正式56次；失败回执未改为成功。
分析器接口误报也单独保留，没有改写原始采集来适配分析结论。

## 资源与冻结

`controller.py` 仅允许物理卡 5。它复用已审查的有界资源门禁、自有随机 token 标签清理、
共享 flock、STOP、前后空闲检查及退出过程身份核验；不停止或删除任何外部容器。
两个已知后台设备容器只有在显式传入完整 ID、元数据持续匹配且没有外部 NPU 进程时才能共存，
不会因此声称设备独占。初次只读检查见 `preparation.json`；运行前仍必须重新检查。

远端任务根为 `/home/native-execution-strategy-20261030`。任务根只读挂载到 `/audit`，
当次 evidence 目录单独可写；依赖的高精度 collector 从旧实验只读挂载。
单 case 最长 300 秒、单次 controller 最长 1800 秒、每次最多 8 个 case。

每 step 在 debugger start 前，使用已核对安装源码的公开 `reset_peak_memory_stats` 清零峰值；
原 API 同步返回、debugger stop 后，通过 `memory_stats` 与 `max_memory_allocated` 记录 allocator
峰值字节。该值包含已经存活的输入与采集分配，是本进程 allocator 指标，不能代表整卡 HBM。
计入 `device_seconds` 的是 enter/exit monotonic 标记间的保守设备占用窗口时间，包含 Python 和
采集开销，不是纯 kernel 时间。原始测量和调用信息写入独立 `step_metadata.json`，SHA 绑定到
native observation；缺接口、缺统计 key 或读取失败保留 unknown，不将默认 0 当成实测值。
controller 的超时、16 GiB 主机内存限制、设备门禁仍独立执行，采集指标本身不证明硬预算通过。

`runtime_manifest.json` 使用 `nes.runtime_snapshot.v1`，`files` 是相对 `runtime/` 的
文件名到 `{sha256, bytes}` 的映射，另含非空 `native_components_sha256`。
每轮检查精确 inventory、SHA 与无符号链接，并将清单和实际全部运行源码复制到当次 source 快照。
`runtime/native_execution_strategy/` 包括实际 Python 和非代码证书 JSON，不从工作目录隐式导入旧实现。

`freeze_inputs.py` 使用标准库核对旧输入清单 SHA、每份实际输入 bytes/SHA 和原 source-call SHA，
复制到新输入目录，派生 B1/2/4/8 的完整逻辑重复输入 SHA。它不导入 torch，不调用 NPU，不写旧目录。

## 三项验收分别记录

- **策略切换**：实际 host callback、typed 参数、native raw 与 launch 相互绑定；配置或请求标签不能代替证据。
- **数值变化**：完整 target 输出和 auxiliary 返回分别比较，独立 msprobe 首差只读原生统计、stack、construct；允许没有差异。
- **模型影响**：本 harness 不执行模型，因此保持 `null` / 未测，不能从算子差异直接填为成功。

msprobe 采用 `task=statistics`、`level=mix`、`summary_mode=statistics`、空 `tensor_list`，
保留每 step 的原始 dump/stack/construct、配置及回执；明确披露任务本地 FP32 high collector：
stage manifest `743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af`，
实际 processor `3aa7ede049bc6cb4d54a572a85f907214a98890a3c9544ed7f75e25df1306141`。
这不是未修改的官方 collector；FP32 临时归约只用于采集，元数据和模型/算子原张量不变。
相同字节的[collector 补丁与清单](../../artifacts/attention_tiling/fault28_causal_matrix_statistics_20261029/collector/collector.patch)
已归档于上一轮公开证据；本轮重新核对其实际加载路径与 SHA，没有重新定义采集标准。
真实 `ReplayModule` 注册为外层持有模块的子模块，以满足当前 msprobe 跳过根模块的行为。

完整 target FP16 和边界 query FP64 参考值只保存在 `private_output/`，不进入公开 statistics 包或 GitHub。
读取哈希、私有数值比较均在 debugger stop 后进行，不把诊断节点伪装成原生 API 首差。

## 初始准备

```bash
python3 freeze_inputs.py \
  --source-inputs /home/fault28-causal-20261029/inputs \
  --output /home/native-execution-strategy-20261030/inputs
```

此命令只准备数据。正式 case 构造、实验性新 vendor、runtime manifest 和适配器证书完成后，
才允许执行 controller 的 `--prepare-only`；真正 NPU 执行由统一运行方串行启动。

## 独立 msprobe 比较

受控 case 的四步统一从同一源码位置进入 `StrategySession.execute`。Session 负责策略评估、
原生作用域和证据解析；harness 的同一个闭包包含一次真实 API 调用及 msprobe 采集边界。
两次 controlled 都使用 step 0 的真实 native 观测缓存，off 使用同一 adapter 和禁用 policy 的
session。每步保存 `session_execution.json`，核验原 tuple 对象身份不变；观测缺失或解释失败
会使 case 失败并保留原始错误。Stock discovery 直接调用原 API，不构造私有 Session 证据。

当整个 attempt 成功且清理完成后，同一受控 case 内分别比较 step 0→1、0→2、0→3：

```bash
python3 analysis/compare_replay_statistics.py \
  --baseline outputs/ATTEMPT/CASE --fault outputs/ATTEMPT/CASE \
  --baseline-completion outputs/ATTEMPT/completion.json \
  --fault-completion outputs/ATTEMPT/completion.json \
  --baseline-step 0 --fault-step 1 \
  --output outputs/analysis/CASE_step0_vs_step1.json
```

此工具只用 config、所选 step 的 dump/stack/construct 定位，核验整份 receipt、
collector 披露和 controller completion 的完整性；不读取 host/hit/profile 日志、策略意图、
完整张量或 replay 输入输出哈希来指定首差。它允许没有差异，并把 null 保留为不可观测。
Attention 主返回值与 softmax/seed/offset 等辅助返回分开报告。跨 B1/B8 的整个张量统计不能
自动对齐目标 request；元数据不匹配时不作首差断言。

`cpu_checks.py` 检查资源边界、source snapshot、原生 launch 字节窗口、调用者与 CANN worker
不同 TID 的合法场景，以及四步采集门槛；`analysis/test_compare_replay_statistics.py`
包含 20 个纯 CPU 统计比较用例。它们不代替真实 SDK / 新 vendor 的实机验收。
