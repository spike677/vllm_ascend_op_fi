# 故障28：完整Qwen3 FIA图的B/F statistics/mix证据

**2026-11-09，124卡5完整模型覆盖与首token传播验收通过。** 10组、30次完整28层Qwen3图执行、840次FIA节点执行，合计4,368个层/请求/执行组合。两组B8 F均覆盖全部28层、全部8请求并实际执行M128；独立msprobe首差为第0层FIA output0，差异到达8/8请求首token logits，Top-1均未变。

[实现、调用链、代码位置与范围](../../../experiments/fault28_fia_fullmodel_20261109/README.md) · [联合验收](data/reports/acceptance.json) · [字节摘要控制](data/reports/byte_digest_controls.json) · [发布包物理盲验](local_public_verification.json)

## 主B/F与完整数值

[primary_BF.json](data/primary_BF.json)指定同B8、同causal规则、同输入和模型的对照：

- [B：M512](data/cases/model_b8_baseline/)
- [F：M128](data/cases/model_b8_fault_r1/)与[独立F](data/cases/model_b8_fault_r2/)
- [新进程重新编译OFF](data/cases/model_b8_restore/)

第0层全部请求Q/K/V逐元素及字节SHA一致，Attention改变1,429/33,554,432个元素，全部请求末query此时不变。全部请求从第1层Attention开始出现末query差异，最终logits改变1,070,961/1,215,488个元素，最大绝对差0.0546875。原始私有完整数组比较、各请求指标和后续层变化见[fault_r1_complete](data/reports/fault_r1_complete.json)及[fault_r2_complete](data/reports/fault_r2_complete.json)。

F/F、B1保护、B1/B8 OFF的全部观测数组逐元素相同，完整及逐请求SHA一致；B8无中间别名图的B/F各自logits也与观测图完全一致。不能将Top-1未变解读为logits未变，也没有据此认定业务质量下降。

## 采集了什么

全部10组均保存20套step0/1、rank0原生dump/stack/construct、配置和完成回执。`task=statistics, level=mix, summary_mode=statistics, precision=high`，同步、无tensor筛选。step是相同prefill图回放，不是decode token。

原生GraphBoundary统计包含输入token IDs、position IDs和causal mask的描述/可观测统计，以及logits和全部28层实际Q/K/V/FIA输出的统计。完整模型额外输出别名用于观测；[统一CPU FP64增强的补丁和指纹](data/collector/)明确披露，没有按故障位置选择张量或改写返回值。

主B8 B/F/F2保存全部原生图CSV；其余7组保留每次执行全部FIA CSV、完整原始module统计、FX/GE图和私有全量验算报告。各组[public_ge_selection.json](data/cases/model_b8_baseline/public_ge_selection.json)明确可用/公开范围。所有840份实际FIA迭代CSV均公开，其他控制组未公开的完整算子CSV留在124。

每个图实际28节点共享一个JIT kernel；host选择回调次数不等于运行时覆盖。逐节点/请求证据见[coverage](data/reports/coverage/)：原始实际GE节点、编译二进制SHA、原生任务/runtime实例化、公开raw/key与三次实际输出CSV均交叉核验。

## 独立首差与盲验

推断只读原始统计、配置、采集器身份和真实FX/GE图/CSV，不读case模式、host命中、期望profile或数值报告。按图依赖排序，两个F的两步均首先观察到 **FIA output0 Mean/Norm**。原始module列表包装保持 `[[logits,Q0,K0,V0,A0,...]]`，A0叶子路径 `$[0][4]` 经真实NetOutput映射FIA输出；不是把module记录改名为eager API。

- [远端F独立推断](data/reports/fault_r1_msprobe.json)与[独立F推断](data/reports/fault_r2_msprobe.json)
- [本地物理盲验](local_public_verification.json)：完整源字节、public ABI、所有节点/请求计数和840份CSV复核通过
- [本地F首差](local_fault_first.json)与[本地独立F首差](local_repeat_first.json)

本地盲验目录物理排除case/result、host、策略和数值报告。原生六位GE CSV首差仍是Mul_13输入1，之前326个无唯一CSV节点保留为gaps；默认工具精度不能直接分辨FIA首差。null、空LSE、缺失项均保留不可观测。共享Data的重复记录完整保留，仅同执行组完整行相同才等价合并；冲突和跨组记录不任意择取。

```bash
python data/analysis/graph_evidence.py \
  --baseline data/cases/model_b8_baseline \
  --fault data/cases/model_b8_fault_r1 \
  --output /path/to/fresh_first_difference.json
```

## 原字节与交付边界

`data/`包含 **15,119个文件、232,861,380字节**。其中15,096份原始文件由[source_mapping.json](data/source_mapping.json)逐项映射原路径、公开路径、字节数与SHA；[SHA256SUMS](data/SHA256SUMS)覆盖除自身外全部导出文件。原始统计与CSV字节不变；编译器字段摘录及原始日志行子集均明确标注。目录外README和本地复核报告不属于远端导出清单。

| 项目 | SHA-256 |
| --- | --- |
| source_mapping.json | `5c63a4b68284cb520a223e59b2052cf90b433772df3f67624f775e11f98ea3ba` |
| reports/acceptance.json | `f4fd1debb41c79cf367d473bcd27a4e339d42ccc0c357c5ed83afc131fbbb117` |
| reports/byte_digest_controls.json | `e0c7daf173a8713145b29d1ba34718950a885f32bb2732c5798153b47e672126` |
| local_public_verification.json | `7ba0813da6a751e43b4359f549d66ad1466a3ecf49ad57c92ea8b9345f33adff` |
| 传输归档（未提交） | `c8cab92ec9e5f1b8109bf168fb5deec4c393d2fbbdb680c1a63e795da94e1320` |

[34项CPU测试](data/diagnostics/analytic_tests_a3.json)通过。旧挂载、日志限制、日志创建、msprobe导致三图重编译、重复CSV与输出树映射的初次限制均在[diagnostics](data/diagnostics/)保留。正式a5与这些失败分开；没有将失败、null或未完成采集改写为通过。

完整张量、权重、原始大日志和编译库不在GitHub。完整批张量/日志/缓存保存在124任务专用tmpfs，重启易失；目标请求QKV/Attention与全部请求logits留在 `/home/fault28-fia-fullmodel-20261109/outputs`。公开验证无法重读这些私有完整数据或机器码；完整数组比较通过报告与原始capture摘要绑定。

范围是当前CANN9.1、Qwen3-0.6B/FP16/TP1/B1/B8/2048-prefill的注册FIA完整模型图。不是HF默认FIA、历史盘古默认策略、任意shape或decode验收。M引起的宽度也影响BMM1/BMM2，尚未唯一归因某条Reduce指令。
