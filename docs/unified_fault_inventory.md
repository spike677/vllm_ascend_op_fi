# 本地故障能力清单与自动适配边界

[文档索引](README.md) · [9171f3b 全局审计](#9171f3b-全局审计)

核查日期：2026-11-16。本文是本地源码与既有验收记录的整理，不是一次新的 NPU 验收。历史冻结源码、采集文件和编号不在本轮改写。

以下资产表保留旧实现/证据的边界。本轮新增适配器（含17/18重新实现）及当前接入状态单独列在[统一机制与验收范围](unified_mechanisms_20261116.md)，不能把新CPU验证回填成旧NPU验收。

## 必须分开的四种状态

| 层级 | 可以确认什么 | 不能由此推出什么 |
| --- | --- | --- |
| 注册 | 故障名称、适配协议、机制与入口可以被统一查询 | 当前模型会调用这个入口 |
| 模型结构发现 | 实际实例、Attention 注册调用、投影/量化方法/消费者关系被识别 | kernel 支持该形状或缓存语义 |
| 调用支持 | 实际 Q/K/V、dtype、mask、布局、版本、源码/构建指纹通过适配器约束 | 注入实际命中、数值一定变化 |
| 运行验收 | 真实调用、命中与拒绝分母、B/F/OFF、msprobe 和数值影响均有证据 | 其他模型、形状、框架或历史事故也已经验收 |

用户输入故障名称与模型对象即可。模块路径、layer_idx、query/KV 头数、输入布局和消费者必须由框架从模型结构及真实调用发现。发现失败时返回具体缺口，不将填写内部模块名转嫁给用户。对运行时地址、别名、分页和生命周期有要求的故障，模块结构只能提供候选，不能替代动态证明。

## 资产来源

原本地整理工作区包含较早的完整实现和证据；本次源码分支不复制其全部实验/数据目录：`fault20_fast_validation/source/`、`experiments/`、`artifacts/`。较新的工作区位于：

```text
C:/Users/admin/.codex/worktrees/batch-outlier-msprobe-publish/demoncatpfi/.publish/fault28-causal-20261029
```

下表中的 `NEW/` 指这个已存在的目录，而不是当前工作区的导入路径。独立已发布的分母重复累计数据分支位于其同级 `f28-denominator-replay-data-20261113`，提交 `f879af9bd19a4c1ca099c81c515e4c33cbee20c7`。跨工作区路径用于资产定位，不应硬编码到统一运行接口。

## 已找到的故障与变体

| 统一目录机制 ID | 编号/类别 | 实际注入边界及本地入口 | 现有约束、验收与统一接入状态 |
| --- | --- | --- | --- |
| `8.restore_offset` | 8 | `npu_moe_token_unpermute` 的配对 indices/top-k weight 输入替换；[证据说明](../artifacts/fault8/systemic_restore_offset_20260916/INJECTION_LOGIC.md) | Qwen3-VL MoE、TP2 固定 runner；只对多 token prefill 映射有效。当前目录只有证据，缺完整注入源码。可登记发现需求，不能宣称可直接执行。需识别真实 MoE 路由与恢复映射，不能将任意 Linear 当作目标。 |
| `15.alias_detach` | **15**，旧目录曾误称27 | `AliasInjector.dispatch`（原件路径 `experiments/fault15_alias_detach_allhit_20261028/alias_injector.py`，未随本源码分支复制）：format cast 后调用原 `as_strided` | 机制级通过；NZ29、BF16 `[16,16]`，view `[8,8]`/stride `[16,1]`/offset17，强引用 base、PID、forward、callsite。6次匹配全部触发。自然模型中的“view→copy_→原 base 消费”尚未证明。仅 Python `torch.as_strided` 包装不能覆盖 native 内部调用。 |
| `17.spatial_padding` | 17 | `visual.patch_embed.proj` 的真实 Conv3d 几何参数；[说明](../artifacts/fault17/spatial_padding_hw7_20260920/INJECTION_LOGIC.md) | 固定视觉 patch encoder、padding `[0,0,0]→[0,7,7]`、输出形状固定；当前目录是证据与说明，未交付完整源码。自动发现须基于实际 Conv3d 参数和视觉分支消费，不能仅匹配属性名 `proj`。 |
| `18.api_half` | 18 | RMSNorm 原生返回与消费者之间的输出增益代理；[证据](../artifacts/fault18/systemic_api_half_20260915/README.md) | Qwen3-VL、vLLM 固定 wrapper 组合；不是真实 Reduce 统计量错误。msprobe 可先记录原生正常输出，再在下游观察到代理差异。当前目录无完整 wrapper 源码。 |
| `20.buffer_substitute` | 20 | [OpExecCache Run preload](../fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp) 和 [typed metadata proxy](../fault20_fast_validation/source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp) | QMM/MatmulWeightNz/Matmul、INT2/4-equivalent/BF16截尾等已在固定ABI环境验证。可复用控制面、native slot 解析及临时buffer生命周期，但不能仅依据 Python 模块名猜 addresses 索引。ABI握手、真实formal tensor→runtime slot证明仍是门槛。 |
| `22.wrong_row` | 22 | 原生 IndexSelect 返回后、RoPE消费前；[说明](../artifacts/fault22/wrong_row_output_proxy_steps012_20260929/INJECTION_LOGIC.md) | 固定 `model.layers.0.self_attn.rotary_emb`、首次prefill第13行→源第0行，单次。不是32位地址溢出或 native kernel替换。当前主要为证据；应通过真实cache/index/消费者识别并验证所选合法行，不能把层号固化到统一入口。 |
| `25.reduce_scatter` | 25 | 对话存在方案；本次扫描的当前实现目录未找到可运行且验收的专属适配器 | 保留为待接入，不因存在同名规划而注册成已支持。需要真实 collective group、rank、dtype、通信实现和调用链。 |
| `26.residual_stomp` | 26 变体A | [QMM output underflow runtime](../fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_qmm_output_underflow.cpp) | 真实 allocator 相邻关系与 live residual 生存期证明后更换写入地址；此前231次受控命中。自动发现须由只读拓扑/分配追踪找到 producer→consumer，不能通过固定层路径或邻近地址猜测。 |
| `26.fia_destination` | 26 变体B | [FIA writer proxy](../fault20_fast_validation/source/fault26_fia_writer_proxy_v1/fault26_fia_writer_lossless.cpp)、[公开机制边界](../artifacts/fault26/fia_residual_stomp_bf8_20260916/INJECTION_LOGIC.md) | FIA 输出重定向到 live residual，再复制回原输出并等待消费者后恢复；不是自然OOB。旧B/F中447次 applied仅441次有完整 consumer证明，应保留缺口，不升级成全覆盖。 |
| `quant.batch_shared_scale` | 大batch量化归约范围错误 | `model_hook`（原件路径 `artifacts/quantization/batch_shared_scale_v2_bf_statistics_20261028/source/model_hook.py`，未随本源码分支复制）、同目录私有后端控制 | 将 DynamicQuant 内部描述符 `[T,H]→[1,T*H]`；实际写出Q/scale。现模型绑定 `model.layers.31.mlp.down_proj`、TP2、特定runner SHA，且需调度行映射。应发现实际动态量化方法/算子及行语义；batch不能直接等同于shape[0]。历史公开JSON来自 `summary_mode=md5` 且原采集含张量，不能标成纯 statistics 重采通过。 |
| `28.fa.legal_tiling` | 28 FA 路线 | `NEW/native_execution_strategy/{registry,session,adapters/flash_attention}.py`、`native/fa_bridge.h` | 已有统一执行策略框架，FA S1S2、CANN9.1、910B4，Q16/KV8、S2048、D128、FP16、B1/B8。56次算子矩阵+4次validated检查；另有Qwen3模型覆盖。可复用既有版本/证书/原生证据协议，证书没有自动扩展到FIA/新模型形状。 |
| `28.fia.legal_tiling` | 28 FIA合法切分 | `NEW/experiments/fault28_fia_graph_20261106/`、`fault28_eager_20261110/`、`fault28_statistics_visible_20261113_r2/` | 图模式/eager有不同实验，必须分开。真实M/causal核内S2分组；适配器仍固定Qwen3结构与2048 prefill。合法切分搜索没有达到最新0.1%统计目标，不能改名成失败的kernel或复用新错误的PASS。 |
| `28.fia.denominator_replay` | 28探索后产生的**新错误变体** | `NEW/experiments/fault28_reduce_replay_20261113/model_adapter.py`、`source_policy/fault28_fia_policy.h`、`source_kernel/.../fia_block_vec_nonquant.h` | 第二S2块重复首块分母贡献，B/F固定M128/S2=128。Qwen3-0.6B、28层、B8、2048 prefill的3步统计、重启/OFF已验收，Norm变化0.7968%。不是历史28合法tiling误差；NPU kernel仍固定Q16/KV8、D128、FP16、B1/B8、causal。统一接口可自动发现符合结构的模块，不能给GLM/DeepSeek授予旧NPU通过状态。 |
| `33.paged_decode` | 33 | `NEW/experiments/fault33_paged_tiling_20261105/{model_adapter.py,source_policy/fault33_pa_policy.h}` | 分页增量 IFA、固定B8/Q长度1/KV16384、128-token物理页、Q16/KV8/D128；SInner8192→1024，循环2→16。Qwen3实际KV prefix经分页重排/逆映射证明，28层一步decode通过。与28 prefill M轴不同；不能用静态Attention注册完成分页cache适配。 |
| `39.matmul_range` / `39.output_proxy` | 39 | 重用 [L4 runtime](../fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/)，见 [复现范围](../fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md) | 必须拆开有限输入→真实Matmul非有限、排列相关中间范围失败、返回后非有限值代理。没有证据唯一定位 accumulator类型。版本/slot/上下文保持精确门槛。 |
| `41.tail_column` | 41 | [设计记录](../fault20_fast_validation/reports/fault41/README.md) | DESIGN FEASIBLE / IMPLEMENTATION BLOCKED。旧preload未拦截到真实dispatcher；不能列为PASS或自动启用。先闭合真实cache writer/storage/slot/consumer。 |

## 与本轮自动接口的关系

上述 ID 对齐本轮 `fault_injection.catalog`，不等于全部拥有可执行插件。统一目录应同时返回 `catalogued`、`source_available`、`auto_discovery`、`runtime_supported`、`validation_scope`，缺失值不自动设为真。没有完整源码的证据目录应标为 `legacy_evidence_only`；需要额外 native 执行桥的源码应标为 `native_bridge_required`。

现有 `native_execution_strategy` 提供 `ExecutionStrategyAdapter`、`AdapterRegistry`、`StrategySession`、版本证书和实际launch证据协议。它可以作为执行层复用；模型结构发现属于上层，负责生成真正的调用上下文，而不改写旧冻结实验。

Attention 的公共注册与模型支持分开检查：

1. 使用 `transformers.AttentionInterface` 公共注册入口；不从 Qwen模块借名导入共享表。
2. 扫描 `named_modules()`，识别实际Attention对象、config、forward中共享注册入口及请求参数；不要求 `model.model.layers` 或固定 `self_attn` 属性名。
3. 实际回调再次读取Q/K/V的shape、dtype、device、scale、mask、cache及请求行映射。模型config仅是静态线索，不能覆盖实际张量事实。
4. Qwen、GLM或其他结构只要真实使用公共入口，就可进入同一发现流程。DeepSeek MLA的Q/K维度与V维度、缩放、mask、缓存若超出当前FIA合同，应明确报告哪个约束不满足，不能偷偷pad/crop、伪造mask或转成默认scale。
5. CPU结构/接口测试覆盖某模型，只表示自动路由和拒绝规则通过；只有新NPU测试才可扩展该模型的验收范围。

## 自动选择的共同要求

- 不以模型名白名单代替结构识别；也不把同名字段当成语义等价。
- 不静默遗漏符合条件的层。报告发现总数、实际进入回调数、条件匹配数、受控执行数、透传与拒绝原因；动态分支未运行时保留未观测。
- 按实例/作用域绑定，停用与异常退出恢复原配置和注册关系，避免影响同进程其他模型。
- 实际API调用必须发生在可采集的公开边界，不能缓存msprobe安装前的函数绕过采集。
- `statistics/mix`原件、首差比较、native命中/launch、完整私有张量辅助验证分开。先找统计首差，后检查执行证据；不得让命中日志指定首差。
- 若只发现结构但没有安全合法的native适配器，返回“不支持/待适配”及证据，不通过任意输出扰动伪装该故障已接入。

## 编号与证据维护

`fault27_alias_detach_20261028` 是故障15的历史单次资产别名，不新增一个不同的“27能力”。正确编号在目录索引中更正即可，原schema、hash、源码字节保持原样。

故障28的合法切分控制与 `28.fia.denominator_replay` 分别注册、分别展示验收；33的分页decode不能复用28的prefill合同。新通用接口的版本、结构测试及未来NPU报告独立保存，不回填历史冻结manifest或把旧模型通过记录推广到新的模型。

## 9171f3b 全局审计

生产接线分支新增机器可读的 `audit`，沿用同一机制目录、宿主导出和拒绝路径。每条记录列出源文件、合同、实际部署入口、证据级别与作用域及缺口标签。详见[本分支对账与验收](../experiments/production_wiring_20261116/README.md)。

### 审查范围

| 对象 | 本轮定位 |
| --- | --- |
| Git 公共目录 | `C:/Users/admin/Documents/demoncatpfi/.git` |
| 发布审计工作树 | `.publish/native-provider-adaptation-20261116`，`codex/native-provider-adaptation`，HEAD `9171f3bacd78b1550cc39cc74e126e3a13cfdb07`。审计前已有未跟踪实验记录，已跟踪内容与提交一致。 |
| 本地原型 | `.publish/auto-adaptation-20261116`，`codex/auto-adaptation-20261116`，HEAD `052db5c1db1ab9ea6b85eb9ce5020d8a95e3ac4d`，有未提交实现和证据。包含 QualifiedNativeSession、MatMul/QMM 回调、CandidateSet 和覆盖报告，具备本地 CPU/主机测试。 |
| 父工作树 | `fault28-native-fa-publish/demoncatpfi`，`codex/unified-model-injection-20261116`，HEAD `1eba3308c0cd274825fd777d0a4b6e703560f510`，有未提交实现和实验记录。 |

本轮检验为本地审计。发布提交、相关部署工作树和可见的本地项目副本均未找到黄区当前的 `src/demoncat`。正式宿主版本、现有 CLI 命令及真实 Controller/Worker API 是实施输入。蓝区 CLI 提供 `catalog` 和 `host-manifest` 两个只读命令。[发布 CLI](../fault_injection/__main__.py) · [宿主合同](remaining_faults_integration_20261116/README.md#内源项目接线契约)

17 个目录变体对应 15 条 `model_fault.<机制 ID>` 统一宿主合同、1 条 `fault39.native.matmul` 独立宿主合同及 1 个缺合同的输出代理。15 条统一合同中，5 条由会话构造适配器，5 条经部署 Provider，5 条没有统一可执行会话。两条 `39.*` 均在发布版 `ModelFaultBackend` 配置入口之外。所有宿主合同的正式接线状态为 `INNER_FRAMEWORK_INTEGRATION_PENDING`。[统一导出](../fault_injection/integration.py) · [Fault39 独立合同](../fault39/src/demoncat_fault39/contracts.py)

### 面板 A · 17 条机制能力矩阵

I = `IMPLEMENTED`，C = `CPU_VERIFIED`，N = `NATIVE_VERIFIED`，M = `MODEL_VERIFIED`。这些标记仅属于单元格所指路径。17 条的 `PRODUCTION_INTEGRATED` 均待验收。新统一入口的 `COVERAGE_VERIFIED` 需有独立分母、全量命中和整组恢复。目前只有下表所列的限定场景证据。

| 变体 | 发布范围 | 历史或独立实机 | 恢复与本阶段缺口 |
| --- | --- | --- | --- |
| `8.restore_offset` | 合同，证据入口 | 历史 N/M | 历史新进程恢复。缺可复用 worker、真实路由与请求归属。P5 |
| `15.alias_detach` | I/C，公开视图 API | 历史 N，机制级 | 退出包装。缺自然模型中的写入、原 base 消费和 native 内部调用。P5 |
| `17.spatial_padding` | I/C，标准 Conv3d | 历史 N/M，独立视觉试验 | CPU 清理后输出恢复。缺新入口真实视觉链与 NPU 恢复。P5 |
| `18.api_half` | I/C，标准 RMSNorm | 历史 N/M，API 增益代理 | CPU 清理后输出恢复。旧实机只确认 wrapper 恢复，缺新入口的实际调用和 OFF 数值。P5 |
| `20.buffer_substitute` | I/C/N/M，发布 MatmulWeightNz | 历史 QMM 另有证据 | 发布双模型同进程 OFF。P1 生产接线，P3 新 QMM 独立实测。 |
| `22.wrong_row` | 合同，证据入口 | 历史 N/M，固定 RoPE | 旧 wrapper 移除后新请求恢复。缺复用 worker、缓存和消费者绑定。P5 |
| `25.reduce_scatter` | 合同，待实现 | 待验 | 缺真实 collective 执行、进程组、rank 对齐和 drain。P5 |
| `26.residual_stomp` | I/C/N/M，发布 QMM | 旧邻接原生试验 | 发布 45 次/rank，消费、恢复完成。rank1 step0 独立首差受 NaN 阻断。P3 |
| `26.fia_destination` | I/C/N/M，发布 FIA V3 | 旧 writer 试验 | 发布 126 次/rank 有消费与恢复，首层按条件透传。仍缺新布局、跨阶段和失败场景。P3 |
| `quant.batch_shared_scale` | I/C/N/M，发布固定 decode | 旧量化后端证据 | 同进程 OFF 精确。缺动态调度及多 eligible 模块的行映射。P3 |
| `28.fa.legal_tiling` | 独立策略 I/C/N/M，统一会话待接 | 固定 FA S1S2 证书 | 旧 scope 恢复。缺普通模型 forward 到策略、请求和 launch 的绑定。P5 |
| `28.fia.legal_tiling` | I/C，统一 Attention 路由 | 历史 N/M，合法候选数值差异 | 旧固定模型 OFF。新路由缺 NPU launch，对应旧模型 0.1% 统计阈值未达标。P5 |
| `28.fia.denominator_replay` | I/C，统一 Attention 路由 | 历史 N/M，独立算术变体通过 | 旧重启及 OFF。需复查新路由实际 replay key 和恢复。P5 |
| `33.paged_decode` | I/C/N/M，发布固定 teacher decode | 旧固定布局及模型实验 | 28 层 × 3 次切分，独立 launch 审计与 OFF。当前独立首差在下游。P3 |
| `41.tail_column` | 合同，dispatcher 待查 | 旧尝试未命中目标 | 缺实际 writer、slot、storage、consumer 与恢复。P5 |
| `39.matmul_range` | 独立 I/C/N/M，统一接线待做 | 具体四版本算子，部分模型 | 独立包 OFF。未发布 QualifiedNativeSession 仅本地 CPU/主机验证。P2 |
| `39.output_proxy` | 通用桥，部署和宿主合同缺失 | 历史 N/M，返回代理 | 历史单点后新请求恢复。缺独立 Provider、写入合同和生产清理。P5 |

### 逐项能力卡

以下统一合同 ID 为 `model_fault.<变体>`。8、22、25、28 FA、41 的合同允许描述能力和拒绝原因。两条 39 的绑定见各自卡片。

#### 8.restore_offset

历史专用 worker 在 Qwen3-VL MoE TP2 eager 首次多 token prefill 中绑定真实请求和 owner 线程。实际 `npu_moe_token_unpermute` 使用配对替换的恢复 indices 和 top-k 权重，设备读回、同步和临时对象释放分别核对。历史 token combine 观察 48 次/rank，恢复使用新进程。发布通用入口缺 Worker、请求/路由映射和消费者证据。[历史原件说明](../artifacts/fault8/systemic_restore_offset_20260916/INJECTION_LOGIC.md)

#### 15.alias_detach

`attach` 使用 `AliasSession` 自动识别当前模型 owner 线程里的公开 `torch.as_strided`。F 将 NZ29 BF16 base 的 16×16 存储转成独立 format2 临时存储，目标 view 为 8×8、stride 16/1、offset 17。真实 `copy_` 后原 base 可丢失写回。旧试验 6 次目标、4 次透传。关闭还原包装，使用新 base 验证写回。缺模型自然写链、原 base 消费者和内部 native API 覆盖。[适配器](../fault_injection/alias.py)

#### 17.spatial_padding

`ModuleFaultSession` 按实际标准 Conv3d 对象、16×16 patch 及输入几何选点，调用原 `Functional.conv3d` 时将 padding (0,0,0) 改为 (0,7,7)，沿用原权重与 bias。当前在单 owner 模型 forward 内工作并移除 hook。CPU 真 Conv3d 回归已过。历史 Qwen3-VL TP2 视觉证据使用独立入口，需重新验证新入口真实图像、输出消费及 OFF。[适配器](../fault_injection/module_faults.py) · [历史视觉结果](../artifacts/fault17/spatial_padding_hw7_20260920/README.md)

#### 18.api_half

`ModuleFaultSession` 按标准 `nn.RMSNorm` 对象选点，在模块返回处乘 0.5，生成新张量。owner forward 结束后可撤销挂钩，CPU 已验证。历史 Qwen3-VL TP2 用 API 返回代理，native RMSNorm 记录可能先看到健康值。需在真实 NPU 路径区分算子返回与下游消费证据。[适配器](../fault_injection/module_faults.py) · [历史采集](../artifacts/fault18/systemic_api_half_20260915/README.md)

#### 20.buffer_substitute

发布路径是 `attach` + `Buffer20Runtime`，由 actual Linear 对象、当前 forward、typed GWS 和 `OpExecCache::Run` 的 live slot 选定 `aclnnMatmulWeightNz`。F 在独立临时 self 中截断尾数，沿用原 mat2、out 和真实 Run。CANN 8.5.1 试验使用 BF16 ND 输入、NZ29 权重、专用同步 Worker。同进程撤销后完整输出精确恢复。本地另有隔离的 QMM x1 与 WeightNz self 回调。新回调的资源、故障参数和 ABI/profile 单独验收。[部署适配器](../fault_injection/deployments/buffer20.py) · [实测](../experiments/provider_adaptation_20261116/README.md#推理范围与恢复)

#### 22.wrong_row

历史 Qwen3-32B W8A8 TP2 eager 按请求定位首次 prefill 的真实 RoPE。原 `aten.index_select.default` 返回后、消费者读取前，选取 BF16 结果的第 13 行并填入第 0 行值。独立返回存储及原输入/索引保护，单次受控命中。旧移除 wrapper 后新请求 token/text 恢复。需实现可复用 Worker、真实 cache/索引/消费的确认及拒绝路径。[历史边界](../artifacts/fault22/wrong_row_output_proxy_steps012_20260929/INJECTION_LOGIC.md)

#### 25.reduce_scatter

当前只读目录保留机制位。需要先确认实际 ReduceScatter API、executing rank、通信组及所有参与者的顺序和 dtype，再实现 dedicated native 干预、同步、退出和同输入恢复。调用、存储和实机证据均待建立。[能力目录](../fault_injection/catalog.py)

#### 26.residual_stomp

`Residual26WorkerExtension` 经已加载模型及 Provider 选择本 forward 的 AddRmsNormBias 残差。两块 live allocation 相邻时才在实际 QMM output-base 写入点启用，保存正常 out/padding，实际消费者后恢复残差。TP2 逐 rank 记账，相邻或 producer 证据不足时透传。现有 OFF 核对同引擎的 token/text/logprobs，独立首差限制见面板 C。[Provider](../fault_injection/deployments/residual26.py) · [Worker 接线](../fault_injection/deployments/residual26_worker.py)

#### 26.fia_destination

同一 Worker 生命周期按实际 Attention 对象选点。真实 FIA V3 的目标改绑到本 forward 的 live residual，正常输出复制回原 out，消费者真正读取后恢复受害存储。TP2 每 rank 两次没有当前 producer 的回调透传。基线 126 次为无损演练，F 126 次才是故障写入。已验证作用域内消费和恢复，后续需证明更宽 layout、缺消费者时的清理和异常退出。[Provider](../fault_injection/deployments/residual26.py) · [正式推理日志摘要](../experiments/provider_adaptation_20261116/validation_summary.json)

#### quant.batch_shared_scale

`QuantWorkerExtension` 在实际 runner/量化方法中选择首个 eligible 模块。真实 scheduler 校验请求、行、前缀和健康输入 SHA，再允许 DynamicQuantV2 将逐行归约变为共享尺度。TLS 绑定 owner，保留原输入。TP2/B8 pure decode，每 rank 3 次，移除后同进程精确 OFF。动态 batch、请求迁移和多模块传播需分别验证。[Worker 和 scheduler 检查](../fault_injection/deployments/batch_scale.py)

#### 28.fa.legal_tiling

`NativeStrategyBridge` 通过 `StrategySession` 按真实 `npu_fusion_attention_v3` S1S2 的参数、布局、合法切分证书及请求映射工作。改 host M/nRatio 后启动实际 kernel，各调用作用域退出后恢复原策略。已有固定 910B4/CANN9.1 试验，缺跨模型 forward 的统一挂接和新的 launch 证据。[策略桥](../fault_injection/strategy.py)

#### 28.fia.legal_tiling

`AttentionSession` 从共享注册表识别真实 Attention 对象，将满足私有 build、mask/scale 和 BNSD FP16 几何的调用交给 `FiaBackend`。合法切分在 M/S2 512→128 的健康模板间切换。当前 Python 包证据为路由和 host selector，实际 launch 需独立关联。会话退出还原注册与 config，旧证据固定 Qwen3、pre-fill shape 和图/eager 条件。[FIA 适配器](../fault_injection/fia.py)

#### 28.fia.denominator_replay

同一 `FiaBackend` 在 M128/S2=128 选择 replay key，使真实 kernel 的第二块重复首块分母贡献。输入 Q/K/V 保持原路由，单进程 owner scope 后恢复正常 key。历史 Qwen3 28 层、B8 三步首差是 FIA output Norm，变化约 0.7968%，有 OFF 与重启证据。新接线需对 actual key、完整输出和恢复重新关联。[FIA 适配器](../fault_injection/fia.py) · [冻结历史证据](https://github.com/spike677/vllm_ascend_op_fi/tree/f879af9bd19a4c1ca099c81c515e4c33cbee20c7)

#### 33.paged_decode

`Paged33Runtime` 由真实共享 Attention 对象、当前 DynamicCache 存活关系及完整 page-table 逆映射选点。真实 IFA host 切 SInner 8192→1024，B8 每次包含 8 个请求，保留 KV 逻辑布局。在同进程固定 teacher decode，裁回同一 cache 前缀后按原策略执行 OFF，完整 logits 精确。连续生成、其他分页环境和独立 IFA 统计定位待验。[Provider](../fault_injection/deployments/paged33.py)

#### 41.tail_column

历史 preload 尚未观测到实际 dispatcher。先观察原生 cache writer、尾列映射、有效存储身份及后续 reader，再定义单点干预、同步和验证。当前无运行写入及 OFF 回执。[历史阻塞](../fault20_fast_validation/reports/fault41/README.md)

#### 39.matmul_range

独立 `fault39.native.matmul` 使用 NativeBackend 和 Worker RPC，按真实 Worker/rank、GWS、cache、BF16 ND self/mat2/out 的 live slot 自动选点。在有界临时存储写入有限正负数并调用原 `aclnnMatmul`，以原输入保护、输出设备读回、sync 和 lease 核对。旧故障回执保留 `input_device_readback_verified=false`。独立原生单点及同作用域 OFF 已通过，常用计划上限 1。统一接线需保留 v1 和真实请求归属。本地 QualifiedNativeSession 的 D2D/临时输入读回是新路径。[独立包证据](../fault39/reports/README.md)

#### 39.output_proxy

历史 Qwen3-32B W8A8、8.5.1、TP2 eager rank0 的 MatMul 真 Run 结束后，向已证明的 BF16 out slot 写入限定非有限值，再读回并 disarm。历史 12 个故障样本有新请求恢复。当前通用桥只声明 `output_proxy_bounds` 等证据要求。仍缺可部署 Provider、宿主合同、边界与异常 drain 验收。[历史单独验收](../fault20_fast_validation/reports/fault39_float_matmul_output_nonfinite_proxy_validation.md)

### 面板 B · 模型、算子与构建兼容性

表内未列明的模型实现、算子版本、形状、SoC、执行模式和 TP 均需重新审查。运行检查以精确构建指纹为键。跨模型的发现能力与具体算子支持分别登记。

| 模型和范围 | 实际位置及语义 | 环境及 Provider/build | 结论 |
| --- | --- | --- | --- |
| Qwen3-0.6B，SmolLM2-135M（Llama） | Linear 的 `aclnnMatmulWeightNz`，BF16 ND self ≤64KiB、NZ29 weight，B2 单 token forward | 124 / 910B4 / CANN 8.5.1 / 同步 eager / 单卡，`Buffer20Runtime`，`build20-a1` | 两模型通过 B/F/OFF |
| Distill-Qwen-32B-W8A8 | QMM output-base、当前 allocator 邻接与残差；FIA V3 destination、同 forward producer/consumer | 124 / CANN 8.5.1 / vLLM eager / TP2，`Residual26Runtime`，`build26-a2` + `provider-source-a11` | 两条独立模型回执 |
| Distill-Qwen-32B-W8A8 | 量化模块、DynamicQuantV2、真实 scheduler 行，B8 pure decode | 124 / CANN 9.1 / vLLM TP2，`quant-scoped-a1` TLS | 三轮 B/F/OFF |
| Qwen3-0.6B | 分页 IFA，FP16 B8/Q1/KV16384/H16/KV8/D128/page128 | 124 / CANN 9.1 / 单卡 eager，`Paged33Runtime` 固定审计 vendor | 固定 teacher decode B/F/OFF；下游统计首差 |
| 固定 BF16 ND MatMul，M1/K12800/N5120 | `aclnnMatmul`，typed self/mat2/out，独立包 | 124 / CANN 8.5.1、9.0.0、9.1.0、9.2.0 / rank0，四个独立 profile | 算子实测 |
| Qwen3-0.6B；旧 SmolLM2/Llama 另有模型回执 | 实际 BF16 MatMul Linear、选中投影形状和原生 slot | Qwen 当前匹配构建为 8.5.1 TP1/TP2、9.1 TP2；旧 SmolLM2 回执单列 | 独立包实测 |
| Qwen3、GLM、DeepSeek-V3、Llama 小型 CPU 模型 | 实际模块结构和 Attention 注册路由 | 既有 CPU 套件 | 结构与路由验证 |

Fault20 记录中的 `Buffer20Runtime`、probe20 使用 `provider-source-a3`，L4 Run 源码和桥使用 `provider-source-a1`。上述发布源经 Git blob 的 LF 规范化重新对比，**35 条审查记录无不一致**。Fault39 四个 profile 各有 13 条部署源码检查均匹配的回执。20 的 `build20-a1/build.json` SHA-256 为 `9fbf0e3a851a0505682b27ff9655549217ff7184a3bc57cbe65cf705553827a0`，包含五个 DSO 的哈希。新宿主仍需核对实际加载的 proxy/real DSO、构建配置和运行镜像。另一个工作树的 scoped WeightNz 采用 3 位尾数，发布试验参数为 0 位。[源码审查](../experiments/provider_adaptation_20261116/source_review.json) · [构建和验收摘要](../experiments/provider_adaptation_20261116/validation_summary.json)

### 面板 C · 效果、覆盖与可观测性

下表中的 20/26/量化调用计数来自该次试验的 Provider 回执。独立调用分母和计划目标分母分别记录，欠缺时留空。实际命中、首个可观测统计差异和生产覆盖资格各自有证据。

| 试验 | 目标与实际调用 | 透传、恢复与独立首差 |
| --- | --- | --- |
| 20 两模型 | Qwen3 197、SmolLM2 211 个发现的 Linear。分别 591、633 次/组，三轮同输入前向。 | 请求 token 张量哈希不变，同进程 OFF。三步首差为 `Functional.linear.0.forward/output/0/Max`，3.4375→2.296875 或 4.125→2.9375。 |
| 26 QMM `provider26-final-a2` | 每 rank 258 个结构候选，512 次回调。F 45 次写入且消费恢复，B 0 次故障写。 | 467 次/rank 透传（463 不邻接、4 缺 producer），pending=0，OFF 精确。rank0 step0 首差为 RoPE 输入 Max 7.5→7.46875。rank1 step0 双方 NaN，最早位置未证明。 |
| 26 FIA `provider26-final-a2` | 64 个结构候选，128 次回调/rank。每 rank 126 次有效 scoped writer，B 无损对照。 | 2 次/rank 缺当前 producer 透传，pending=0。4/8 生成 token/text 变化，OFF 精确。两 rank step0 同见残差消费输入 Max 23.5→23.25。step1 输入含已变化 token。 |
| 共享量化 `quant-model-a2` | 首个 eligible 模块，3 次/组/rank，8 个真实请求行；其余模块透传。 | 6/8 logprobs 变化，token/text 未变，OFF 精确。两 rank 三步首差 quant output0 Max 127→98。 |
| 33 `provider33-model-a2` | 28 层 × 3 轮，84 次 F 切分。独立 native 审计每组 140 个 launch，含 warm 28、采集 84、OFF 28。 | 固定 cache 前缀 OFF 精确，logits maxabs 0.02734375，Top-1 未变。三步固定容差首差在 `Tensor.mean.3.forward/output/0/Max`，IFA 唯一归因未取得。 |
| 39 独立包 | 四版本各自单点算子，模型采用已验证请求和 rank 范围。 | 算子 step0 `Torch.matmul.0.forward/output/0/Max` −0.3359375→+Inf。Qwen 的可观测首差出现在 step1 Linear 输出。 |

同一 run 的 build、用例、组别、rank、真实 step、比较结果、原件 SHA 关联在发布摘要。发布摘要索引 385 个私有采集文件、22 份比较结果与 41 条用例回执。原始私有采集位于 124 的 `/home/provider-adaptation-20261116/`，本轮校验发布摘要，不访问 124。使用本地留存的 Fault39 原始统计重新运行比较器，算子 rank0 step0 首差与上表一致，旧 Qwen rank0 step0–7 的首差为 step1 `Functional.linear.3.forward/output/0/Max` 0.76953125→+Inf。比较器 Git blob SHA-256 分别为 `8404ce2214e190556f953a70e28c7d30365ab082d36601a36a2ecd062417f6f8`（跨 step）、`28d9d87fa3bab5ed18c692b29e3ac880e86e46d31916397e0da040c4eb2e9bbe`（统计叶）。保留 null、NaN、Inf、缺步和不可比较字段。新统计原件保存在验证机。[统计与原件索引](../experiments/provider_adaptation_20261116/validation_summary.json) · [覆盖验收标准](fault_coverage_acceptance_standard.md)

### 缺口依赖与工量

估算单位为工程人日，从实际宿主源码、匹配镜像和已有模型均可用起算，实机排队另计。

| 优先级 | 主要缺口和依赖 | 工量估算 |
| --- | --- | --- |
| P1 · 20 WeightNz | 获取真实 CLI/Catalog/Binding/Controller/Worker 接口及隔离 IPC 的同步 Worker。第一条 native 正式接线。 | 2–3 |
| P2 · 39 MatMul | 复用 P1 控制面，映射 WorkerEndpoint、作用域与 v1 ABI。补实际 watchdog/部分失败恢复。 | 1–3 |
| P3 · 20 QMM | 匹配构建、实际 x1/scale/out 槽、D2D 和恢复。 | 1–3 |
| P3 · 26 两变体 | QMM rank1 原始统计可比性，writer/consumer 新场景和异常 drain。 | 1–2／变体 |
| P3 · 共享量化、33 | 动态请求/页映射或原生 IFA 可观测性和恢复。 | 1–3／变体 |
| P4 · 覆盖 | 20/39 从单点到有界多目标，补多 rank、独立分母和整组恢复。 | 3–5／试点 |
| P5 · 15、17、18、28 三变体 | 模块/逻辑层、实际 operator 和独立 kernel 或 consumer 证明。 | 1–3／变体 |
| P5 · 8、22、25、41、39 输出代理 | 可复用 Worker 或 Native Provider、实际语义、资源和恢复。25/41 先确认执行边界再拆工。 | 2–5／初次可运行验证，待边界确认 |

本轮已通过 85 个发布文件校验、35 条源码 LF 校验；发布工作树的 12 个统一合同、39 个 Fault39 合同测试通过。本地原型的 6 个覆盖及 9 个适配测试通过。后续代码触点和测试入口见[生产接线规划](unified_model_injection.md#生产接线规划)。

## 17条隔夜执行结果覆盖层

本轮从 bfdd6abc 继续，历史能力表不自动升级为新 Worker 验收。17项逐条的实际结果、G0–G9、模型/环境范围、B/F/OFF和缺口见[最终报告](fault_closure_report.md)、[机器矩阵](../experiments/overnight_20261116/fault_matrix.json)与[能力卡](overnight_fault_cards/)。新增15/17/18仅通过构造NPU模块机制级；共享Worker实际故障闭环仍待可信回放。
