# 本地故障能力清单与自动适配边界

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
