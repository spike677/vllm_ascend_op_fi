# 故障28：真实FIA eager完整模型B/F统计

2026-11-10在124物理卡2完成固定形状的eager机制、覆盖和首token logits验收。完整Qwen3-0.6B，8个固定不同请求、2048-token prefill、FP16、BNSD、causal sparse2、inner_precise=0，进程核预算16 Cube/32 Vector。B/F都直接执行公开 `torch_npu.npu_fused_infer_attention_score` → `aclnnFusedInferAttentionScoreV3`，模型没有使用TorchAir/torch.compile整图。ACLNN自己的算子执行器仍可使用GE相关内部组件。

这是独立eager数据分支，旧图模式分支保持不变。仅提供 `baseline` / `fault` 两组原始statistics/mix。每组真实只有 `step0`：三次相同输入前向中的第二次被采集；第一次是预热，第三次是未采集重复。它们都不是decode token步数。

下载[baseline.zip](downloads/baseline.zip)和[fault.zip](downloads/fault.zip)，合并解压到同一目录，得到 `data/cases/baseline`、`data/cases/fault`。baseline.zip还包含共同源码、构建/collector身份、独立报告、资源恢复证据和完整SHA清单。网页同时提供完整原始三件套、便利索引及报告；ZIP中的原始字节不变，单一算子摘录不替代完整dump。

```text
data/cases/<baseline|fault>/
  observation/step0/rank0/dump.json
  observation/step0/rank0/stack.json
  observation/step0/rank0/construct.json
  config.json, collector_provenance.json, statistics_receipt.json
  actual_runtime_coverage.json, host_events.jsonl, result.json
data/reports/independent_first_difference_full.json
data/source/  实际执行/分析代码、kernel源码及collector源码
```


| 采集组 | 控制策略 | 网页完整原始统计 | 完整原始数据 |
| --- | --- | --- | --- |
| baseline（B） | causal_base512，M512 | [step0三件套](baseline/observation/step0/rank0/) / [FIA摘录](baseline/first_fia_excerpt.json) | [baseline.zip](downloads/baseline.zip) |
| fault（F） | batch_causal_m128，M128 | [step0三件套](fault/observation/step0/rank0/) / [FIA摘录](fault/first_fia_excerpt.json) | [fault.zip](downloads/fault.zip) |

本次按故障33的模式补齐网页原始dump/stack/construct、完整首差报告、推理结果摘要、核验回执及校验清单。只是整理已有eager实机结果，没有新增NPU调用、重采或连续生成测试。两份ZIP保持原样；物理采集仍仅各step0/rank0。完整dump每组约2.54 MB，网页与ZIP字节一致。

[独立首差摘要](first_difference.json) · [完整独立首差报告](reports/independent_first_difference.json) · [推理结果摘要](inference_result.json) · [摘要一致性回执](inference_recheck_receipt.json) · [归档核验](verification.json) · [包装回执](packaging_receipt.json)

## 独立首差

首个可观测有限统计差异是 `NPU.npu_fused_infer_attention_score.0.forward` 的 `output[0]`，对应第0层FIA。比较器仅读取原始config/dump/stack/construct与collector身份，不读取注入器、host策略或期望profile来选择首差。

| 原始字段 | baseline | fault |
| --- | --- | --- |
| `$[0].Mean` | `-0.002108109542442449` | `-0.002108109450752238` |
| `$[0].Norm` | `660.5721173259877` | `660.572120054888` |

共对齐 1616 条原始记录，其中 1189 条API、其余为module。416 项null/空张量等统计保留为不可观测。[首差报告](reports/independent_first_difference.json)提供完整逐项比较，ZIP中也保留相同原报告；[负对照](reports/independent_localizer_controls.json)确认更早embedding统计发生变化时，定位器能够报告embedding。模块返回值不被当作其内部首个API的执行顺序。

采集是原生API/module statistics/mix，没有GE转换JSON。配置为 `task=statistics, level=mix, summary_mode=statistics`、同步、无算子/张量过滤，使用披露的统一CPU-FP64 high collector。该collector是任务私有的官方采集器修改版，所有真实浮点张量都使用同一计算逻辑；原始统计字节未改写。完整模型中的受支持API/module都保留，FIA融合kernel内部子算子不单独成为msprobe记录。

## 机制与覆盖

B/F使用同一模型、输入、执行源码、collector及同一份OPC编译机器码。真实host在B选择M512，在F每次匹配都选择M128；共同启用同一合法causal尾块规则，依赖tiling由host重新生成，没有改写Q/K/V或输出。实际public host raw/POD独立解码后与原生kernel launch一致；M128同时选择合法的M相关Brcb模板。源码推导确认causal核内Reduce宽度变化，不能将其冒充指令级设备trace，也不能唯一排除其他M相关模板效应。

每组3次完整eager前向、84次真实FIA调用、672个层/请求执行单元，覆盖全部28层和8请求行。B/F合计168次FIA、1344个层/请求单元。每组全部113项真实张量观测的完整值摘要在三轮中一致，采集开启/关闭未改变对应输出。

私有完整数组比较确认首差层Q/K/V在所有请求逐元素一致，FIA输出开始变化，并传播到 8 个请求的首token logits：共 1070967 个logit元素不同，最大绝对差 0.0546875，Top-1变化请求数 0。关闭后额外三次前向的全部113项、全部8请求逐元素精确恢复。[验收](reports/acceptance.json)、[完整比较摘要](reports/complete_output_comparison.json)及[机制报告](reports/mechanism_review.json)分别说明数值、覆盖和因果证据，完整私有数组不上传。

## 验证范围与限制

通过的是固定B8/2048-token形状下的eager受控策略、全层/请求覆盖、重复稳定、OFF恢复及首token logits传播。未在本轮重测eager原生B1/B8策略发现、历史盘古版本、任意形状/模型、连续生成、文本质量或默认SDPA等价性。私有二进制只包含当前三种key，不能作为任意eager FIA的通用替换包；适配器在支持范围外拒绝调用。

统计首差符合受控故障28路径的预期，但统计本身不能唯一证明某条Reduce指令就是物理根因；值摘要相同也不能替代完整张量相等。empty LSE及布尔/整数不支持的统计始终保留null。

早期挂载缺目录、图包缺eager binary/index、CPU构建依赖/输出dtype误选，以及第一次完整基线540秒超时均保留在私有失败attempt，未混入本两组。最终匹配包由官方OPC在Ascend910B4编译，并用匹配源码的官方生成器补齐ACLNN总索引；系统CANN与旧图包未修改。临时卡2Docker入口隔断分两次明确授权，分别最长45/20分钟，均已提前恢复原二进制并删除私有凭据；卡5任务未停止。


## 推理结果摘要的范围

8/8请求首token logits变化，Top-1 token ID相同；逐请求ID、变化数和最大误差见inference_result.json。本次只核对已发布的完整数组比较报告与三轮采集回执之间的一致性，没有重新读取私有logits或运行tokenizer解码，因此没有增加文本片段或连续生成的通过结论。故障33的单步分页decode与本报告的prefill阶段不同，文件结构统一不代表实验阶段相同。

网页的first_fia_excerpt.json明确标记generated_excerpt=true并绑定原始dump SHA；完整报告及原始三件套均同时提供。index.json是额外便利索引，可保留使用。两个ZIP合并解压到data/cases/baseline和data/cases/fault。

## 下载复核

```bash
python verify_downloads.py --extract-to restored
python restored/data/source/analysis/compare_statistics.py --baseline restored/data/cases/baseline/observation/step0/rank0/dump.json --fault restored/data/cases/fault/observation/step0/rank0/dump.json --output independent_recheck.json
```

核验器检查两个ZIP、原始统计和全部源码/元数据SHA，并重新独立计算首差。完整张量、权重、编译二进制、原始大DEBUG日志和私有输入token清单不在公开包内；本机重跑模型需要已固定的私有模型与输入清单。
