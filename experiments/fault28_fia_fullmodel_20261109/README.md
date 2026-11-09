# 故障28：完整Qwen3的真实FIA图模式覆盖与首token传播

**124卡5已通过完整Qwen3图的覆盖、独立首差及首token logits传播验收（2026-11-09）。** 10组、30次完整模型图执行，840次实际FIA节点执行全部核验。B8两组独立F均覆盖28/28层和8/8请求；独立msprobe首差为第0层FIA output0，其Q/K/V在全部请求上逐元素一致。差异经后续层传播至8/8请求的首token logits，Top-1均未变。

[B/F原始statistics/mix与公开复核](../../artifacts/attention_tiling/fault28_fia_fullmodel_statistics_20261109/README.md) · [联合验收](../../artifacts/attention_tiling/fault28_fia_fullmodel_statistics_20261109/data/reports/acceptance.json) · [分析方法](analysis/README.md)

## 接入方式与实际调用链

使用Transformers原始 `Qwen3ForCausalLM` 加载原checkpoint，统一FP16。默认HF路由是SDPA/FA；本轮通过其 `AttentionInterface` 注册真实FIA，保留原来的embedding、全部28层Q/K/V投影、QK norm、RoPE、o_proj、MLP、残差、最终norm和lm_head。**这是显式适配后的完整Qwen3 FIA图，不是HF默认FIA路由或历史盘古模型。**

```text
同一组固定prompt token IDs与positions
→ 原始完整HF Qwen3前向
→ 每层原始Q/K/V投影、QK norm、RoPE
→ 注册的真实 npu_fused_infer_attention_score.default
→ TorchAir fullgraph → 28个GE FusedInferAttentionScore节点
→ 原生host选择M、重新生成依赖tiling、实际JIT kernel
→ 每层真实o_proj、残差与MLP → 后续层
→ 原始最终norm / lm_head → prefill位置2047的next-token logits
```

模型自然生成各层Q/K/V，没有将后续层替换为固定QKV回放。首个差异点的输入一致性由完整张量比较验证；后续Q/K/V随正常前向自然变化。输入是此前冻结的8个不同自然文本prompt，每个2048 tokens；B1使用request_00，B8使用request_00–07。没有生成token反馈、padding或请求重排。

主B/F均为B8、相同模型权重和token输入，共享同一合法causal尾块规则；**B选择M512，F选择M128**。B1的自适应分支仍选择M512。规则、kernel与[前一轮设备粒度实验](../fault28_fia_grain_20261109/README.md)完全相同：M块末端决定去掉哪些所有query都会mask的未来列，合法key仍各参与一次，S2 base仍为512、无跨核SplitKV。没有改写激活、权重或返回值。

## 单图执行与取证中性

每组先以 `torch.compile(fullgraph=True, dynamic=False)` 真正编译完整模型并warmup一次，再调用首次真实后端及其同一组存活参数/输入对象两次。保存真实placeholder绑定，断言输入确实进入该引擎，验证参数和输入均未改变；实际GE输出CSV证明三次执行。这样避免msprobe钩子使Dynamo前端重新编译，不能把普通Python/eager回放算作图执行。

观测图额外返回真实logits及28层Q/K/V/FIA输出，共113个张量。实际FX返回顺序、GE NetOutput到每层FIA输出和输入的边都逐项核验。另编译B8的B/F两组**不返回中间别名**的完整模型图：各自所有请求logits与观测图逐元素一致，完整及逐请求字节SHA一致，观测没有制造本轮logits差异。

## 所有层、所有请求的覆盖

同形状的28个FIA节点实际共享一份JIT kernel，通常只有一次host选择回调。**host命中次数不是层覆盖数，也不是运行时调用数。** 每层均须有实际GE节点、kernel二进制绑定、静态任务、runtime实例化以及三次实际输出CSV；所有请求作为该真实batch节点的输入/输出行核验。

| 范围 | 图执行 | 实际FIA节点执行 | 层／请求／执行组合 |
| --- | ---: | ---: | ---: |
| 8组正式B1/B8、B/F/F/OFF | 24 | 672 | 3,024 |
| 2组B8无中间别名中性对照 | 6 | 168 | 1,344 |
| **全部10组** | **30** | **840** | **4,368** |
| 其中B8两组F、全部M128 | 6 | 168 | 1,344 |

每个图28/28层有证据；B8每层覆盖8/8请求、全部2048位置。840份实际FIA迭代CSV及逐层/请求映射随包交付。请求不是单独的一次FIA launch；不能把一次B8节点伪计为8次独立launch。当前固定形状内全部节点匹配，不能据此宣称所有模型、形状或decode路径已覆盖。

## 数值传播与控制结果

两组F结果完全相同。首差前，第0层全部请求Q/K/V的完整数组逐元素相同，完整及逐请求SHA一致；第0层Attention共改变1,429/33,554,432个元素，8个请求的末query均未改变。**全部请求的末query从第1层Attention开始变化**，最终8/8首token logits改变。

| 请求 | 第0层Attention差异元素 / 4,194,304 | 首token logits差异 / 151,936 | logits最大绝对差 | Top-1变化 |
| --- | ---: | ---: | ---: | --- |
| request_00 | 185 | 135,850 | 0.03125 | 无 |
| request_01 | 170 | 146,020 | 0.046875 | 无 |
| request_02 | 185 | 134,282 | 0.03125 | 无 |
| request_03 | 174 | 110,308 | 0.0234375 | 无 |
| request_04 | 174 | 139,807 | 0.0380859375 | 无 |
| request_05 | 193 | 134,409 | 0.03125 | 无 |
| request_06 | 177 | 135,274 | 0.0546875 | 无 |
| request_07 | 171 | 135,011 | 0.02734375 | 无 |

不是仅比较Top-k或四统计：首轮完整比较每层Q/K/V、Attention及所有请求完整logits；其余两次另核对同步后的整批/逐请求SHA。F/F重复、B1保护与B1/B8重新编译OFF的全部观测数组逐元素相同，字节SHA也一致。完整比较与首差推断是两条独立证据链。

这里证明数值差异到达完整模型的首token logits；Top-1未变，未证明错误答案、业务质量下降或超出算子精度标准。模型内同B8对照用于传播归因，不把自然B1/B8中其他算子的数值路径变化全部归给FIA；同QKV的Batch/M/Reduce因果回放见前一轮实验。

## msprobe独立首差及其边界

两边统一 `task=statistics, level=mix, summary_mode=statistics, precision=high, async_dump=false`。保留20套step0/1、rank0的原生dump/stack/construct、配置及完成回执。step0/1是同一prefill图的两次回放，不是两个decode token。

继续使用已披露、统一处理真实浮点张量的CPU FP64四统计增强，不按算子、shape、mode或故障位置选择张量。原始module记录为GraphBoundary；列表输出的原生JSON结构是 `[[logits,Q0,K0,V0,A0,...]]`。由实际张量叶子路径 `$[0][4]` 对应实际GE `FusedInferAttentionScore:0`，没有把module记录改名为eager FIA API。

独立分析不读case模式、host记录、期望profile、命中、覆盖或完整数值报告；按实际GE依赖排序候选，两步×两个F均得到 **第0层FIA output0 Mean/Norm首差**。没有因logits在tuple slot0就把logits当首差，也没有预设第0层必须变化。

原生GE CSV本轮首先观察到 `Mul_13` 输入1，之前有326个无唯一CSV的节点；这些融合/缺失观测保留为gaps。默认六位CSV不能直接分辨FIA首差，高精度module输出及真实图别名补足该观测。共享embedding/lm_head在同执行组重复dump：保留全部原件，只合并完整CSV行完全一致的观测；跨组或值冲突继续拒绝定位。null、空LSE和缺失项不判相等。

发布包在本地再次做**物理盲验**：只复制配置、采集器身份、FX/GE图、原始统计及CSV，移除case/result、host、策略和数值报告。两个F的两步均独立定位FIA output0。完整元素比较还确认该层真实输入相同、输出首先变化；统计定位仍不是对某条Reduce指令的唯一物理根因证明。

## 关键代码

- [model_adapter.py](model_adapter.py#L12)：通过原始AttentionInterface接入真实FIA；[完整模型包装](model_adapter.py#L40)只返回实际值的别名，不修改层计算。
- [graph_probe.py](graph_probe.py#L124)：实际后端、28层FX映射、返回顺序断言；[单图回放](graph_probe.py#L160)绑定真实warmup参数，保存输入/权重及全部输出核验。
- [原生host选择器](../fault28_fia_grain_20261109/source_policy/fault28_fia_policy.h)和[causal尾宽](../fault28_fia_grain_20261109/source_policy/fault28_causal_tile.h)：沿用已验收vendor，没有新增数值补丁；每个匹配编译上下文选择策略，同一图所有回放持续执行。OFF必须新进程重新编译。
- [coverage.py](analysis/coverage.py#L68)：真实节点/二进制/任务/三次CSV、请求映射、公开ABI及编译器最终raw/key；不是只看host日志。
- [graph_evidence.py](analysis/graph_evidence.py#L177)：实际输出树与GE别名、重复CSV、独立拓扑首差；[完整数值比较](analysis/compare_model.py#L62)单独读取私有完整数组。
- [analyze_matrix.py](analysis/analyze_matrix.py)、[字节摘要复核](analysis/verify_bitwise_controls.py)、[公开包盲验](analysis/verify_public_bundle.py)：联合验收、全部请求SHA和发布原字节复核。

## 环境、范围与保留记录

环境：124/910B4/物理卡5/TP1，CANN9.1.0，torch2.10.0、torch-npu2.10.0.post4、Transformers5.5.4；FP16/BNSD、Q16/KV8/G2/D128、S1=S2=2048、sparse2、inner_precise0，共同核预算16|32。公开serializer覆盖1,576字节、65个范围和63个逻辑字段；实际无SplitKV、有效S2分核边界为0。

| 身份 | 值 |
| --- | --- |
| 官方ops-transformer源码 | `c77c0829f1f0f0b8f882db1d9b051a758b605472` |
| 原checkpoint SHA-256 | `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b` |
| 固定请求清单 SHA-256 | `ed6b27940c3336efe8cd941fda9da191edcbd8257c34587396345f24d7994fe1` |
| 最终矩阵v5 SHA-256 | `9acb7d5cfc54c439065ba98339f3287a4af6b7e34b7b23b11e986ae074e28408` |

34项CPU分析/反例测试通过，含缺失/重复层、外来进程、末请求遗漏、末query、Top-1不变但logits变化、输出树未知路径、重复CSV冲突和导出逃逸。CPU成功不算NPU验收。

早期a1旧挂载创建失败、a2日志上限、a3日志独占创建及a4三图重编译失败均保留，未计入正式840次。a1原FAIL_CLEANUP未改写，后续只读复查未观察到本任务容器。正式a5均清理本任务容器并释放锁；本轮没有停止外部服务或进程。

当前验收限定完整模型**prefill**及已验证形状。不是历史盘古默认策略复现，也未验证decode、动态batch、TP2或任意模型。核内宽度同时影响BMM1/Softmax/BMM2，未唯一分离某条Reduce指令贡献。

完整批Q/K/V/Attention、原始日志与缓存位于124任务专用 `/tmp/fault28-fia-fullmodel-20261109`，属于重启易失的私有tmpfs；目标请求完整QKV/Attention及全部请求logits保存在 `/home/fault28-fia-fullmodel-20261109/outputs`。公开交付只有实现、统计、图与审查后的验算证据，没有完整张量、权重或编译库。
