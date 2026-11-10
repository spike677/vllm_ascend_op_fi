# 故障28：完整模型FIA整图的B/F逐算子统计

这是2026-11-10在124物理卡2的新采集。仅保留`baseline`、`fault`两组；两组都是完整Qwen3-0.6B的真实单一FIA GE fullgraph，Batch8、2048-token causal prefill、FP16。原始statistics/mix保留真实step0、step1，不改写成单一step，也不将warmup冒充生成token。

用户已选择“原生GE逐算子统计＋明确标注为转换结果的逐算子JSON”。B/F统一关闭GraphFusion和UBFusion，提高逐算子可见度；这与旧版融合开启的采集有区别。TorchAir FX dump会回退eager，本次未使用。原始module dump仍是GraphBoundary记录，内部算子的来源是原生GE CSV，派生JSON明确标记`is_native_msprobe_dump_json=false`。

下载[baseline.zip](downloads/baseline.zip)及[fault.zip](downloads/fault.zip)。两个ZIP合并恢复到`data/cases/baseline`和`data/cases/fault`；baseline.zip同时包含共同分析源码、报告、构建/collector身份及校验清单。网页的小JSON是原始预览或明确派生的索引，完整内容均在ZIP中。

```text
data/cases/<baseline|fault>/
  observation/step0/rank0/{dump,stack,construct}.json  原生module统计
  observation/step1/rank0/{dump,stack,construct}.json
  msprobe_graph/raw/graph_1/1/<0|1|2>/*.csv            全部模型GE统计
  msprobe_graph/auxiliary/*.csv                      观察器辅助统计
  derived/step0/rank0/operators.json                 明确派生的逐算子详情
  derived/step1/rank0/operators.json
  derived/warmup/rank0/operators.json
```

## 首个可观测差异

独立复核的step0、step1首差均为`FusedInferAttentionScore`（`FusedInferAttentionScore`）的`output`，输出索引0。实际来源为`native_msprobe_statistics_mix_via_direct_GE_NetOutput_alias`，对应GE输出边`FusedInferAttentionScore:0`；不使用注入命中日志或期望profile指定首差。

| 原始字段/序列化路径 | baseline | fault |
| --- | --- | --- |
| `$[0][4].Mean` | `-0.002108110085222492` | `-0.0021081099433395423` |
| `$[0][4].Norm` | `660.5721472641185` | `660.572149645903` |

详见[独立首差报告](reports/independent_first_difference.json)。其完整逐项比较、不可观测项及原始文件SHA在ZIP的`data/reports/restored_independent_first_difference.json`中。高精度依据为统一披露的CPU-FP64 module collector及真实GE输出别名；原生GE CSV保持原来的舍入精度，不能单独声称高精度首差。

仅看原生GE CSV，本轮最早差异为`BatchMatMul_6`的输出0，`AvgValue`从`-0.00418699`变为`-0.00418698`。加入高精度module统计并通过实际GE输出边映射后，首个可观测生产者提前到FIA输出；两种来源的定位结果分别保留，没有改写原生CSV首差。FIA之前仍保留86个缺少原生CSV的节点、6处原生字段不可观测的边界，不能据此排除所有上游差异或唯一证明物理根因。

每次图执行的原生CSV对应2,097个GE节点，另有84个节点仅能通过直接输出别名观察；其余2,100个节点没有上述独立数值观测，包括1,359个Const。“完整”指保留全部实际采集内容并索引整图，不能理解为每个节点都有独立四项统计。

每组的优化图共4281个节点，每次图执行有2296份模型CSV；覆盖状态逐组记录于[manifest](manifest.json)。所有节点均索引，缺失/NA/null/空输出保持不可观测。`Const`、形状节点和冗余操作可能没有独立CSV，统计相等也不能证明完整张量相等。

## 实际机制与验证范围

B/F共用模型、请求、causal尾块规则、collector和融合配置。正常B选择M512，受控F选择M128，均由真实host生成依赖tiling并运行真实FIA kernel；不改写Q/K/V或输出。独立机制审查验证最终public raw/POD、编译元数据和真实节点/任务/迭代证据。host编译命中计数不等于模型运行覆盖。

两组共6次完整图执行、168次FIA节点执行，覆盖28层及全部8请求行。每组三次forward的全部113个返回张量重复一致。对完整8请求数组的独立精确比较确认首个变化Attention的Q/K/V一致，Attention差异传播到8个请求的首token logits；Top-1变化请求数为0。[完整数组比较摘要](reports/complete_output_comparison.json)仅提供计数和必要小索引，不上传完整张量。

本次按用户明确授权，在采集期间使用临时卡2Docker-CLI资源租约，保护本轮容器并拒绝新增卡2任务；实际运行中拦截的操作统计保存在ZIP的`data/resource_lease/operation_summary.json`。采集验收后已解除文件绑定，原Docker入口SHA256恢复为`e95b17c978781194139f48cc1998826ae16b6851fadd4699df677dcafedf8a65`。Docker daemon、SSH配置及模型数值路径没有修改。该租约针对已确认的CLI清理/派发路径；私有凭据和原始可执行文件不上传。

这是本轮B8受控策略和统计可见性的补采验收。没有重新测试默认SDPA等价性、B1/B8完整因果矩阵、连续生成、任务正确率或新的OFF组。不能唯一归因到某条Reduce指令，也不能认定复刻历史盘古版本缺陷。此前外部stop/kill中断的attempt未混入这两组数据。

## 本地核验

```bash
python verify_downloads.py --extract-to restored
python restored/data/analysis/graph_evidence.py --baseline restored/data/cases/baseline --fault restored/data/cases/fault --output independent_recheck.json
```

`verify_downloads.py`核对两个ZIP、完整清单、原始字节、网页预览及派生索引。完整私有张量、权重、编译二进制和多GB DEBUG日志不在公开数据中。ZIP中保留全部实际采集的原生CSV、两步原生triplets、真实图、派生逐算子JSON和独立报告，未按首差算子筛选输出。
