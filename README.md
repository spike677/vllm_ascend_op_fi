# 故障28：FIA正常/故障统计数据

本分支仅存放两组数据及核验材料，不继承实现仓库的历史。原始统计在网页直接查看，完整图和CSV放入两个下载包，避免打开数MB的图描述或浏览数千CSV文件。

| 组别 | 网页查看原始dump | 完整证据下载 |
| --- | --- | --- |
| 正常FIA B，M512 | [step0](baseline/observation/step0/rank0/dump.json) / [step1](baseline/observation/step1/rank0/dump.json) | [baseline.zip](downloads/baseline.zip)，4.43 MiB |
| 故障FIA F，M128 | [step0](fault/observation/step0/rank0/dump.json) / [step1](fault/observation/step1/rank0/dump.json) | [fault.zip](downloads/fault.zip)，2.74 MiB |

每份dump约34 KB。对应目录均含原生`stack.json`、`construct.json`、配置、采集器身份与完成回执。两组均显式接入真实FIA完整Qwen3图；B为共同causal规则下的受控M512基线，F为M128，不是默认SDPA与FIA的对比。

同模型、同B8输入、同2048-token prefill、同`task=statistics, level=mix, summary_mode=statistics, precision=high`配置。统计使用已披露的统一CPU FP64增强及真实图输出别名；相关补丁、指纹、完整图、全图CSV和完整比较报告均在下载包中。[独立首差摘要](first_difference.json)显示两步首差均为FIA output0；默认六位GE CSV更早节点存在观测缺口，不能单靠六位CSV得到这一结论。null和缺失项保持不可观测，四统计相同不证明逐元素相同。

## 下载与校验

在ZIP页面点击下载按钮，或下载本分支后运行：

```bash
python verify_downloads.py
python verify_downloads.py --extract-to /path/to/new-directory
```

第二条命令要求目标目录尚不存在，校验后将两包还原到同一个`data/`目录。也可将两个ZIP解压到同一目录；B包额外含公共分析/采集器/来源材料，F包仅含F组，两个包的路径不重叠。合计完整恢复9,486个文件、51,822,225字节，与原两组公开数据逐字节一致，没有删减完整图或CSV。

还原后可独立复核实际覆盖和首差：

```bash
python data/analysis/verify_public_bundle.py --primary-bf-only \
  --bundle data --blind-work /path/to/new-empty-review \
  --output /path/to/new-verification.json
```

覆盖28/28层、8/8请求，合计6次完整图执行、168次FIA节点执行。差异传播到8/8首token logits，Top-1未变。范围仍为当前版本FP16/TP1/B8/2048-prefill的受控实验，未唯一归因到某条Reduce指令，也未据此验收图模式KV-cache decode或历史盘古原始缺陷。

## 来源

采集日期：2026-11-09，124卡5；整理日期：2026-11-10。本次只重新包装已完成的数据，未重新调用NPU，未改写原始统计字节，未增加其他采集组。[来源提交6dad85c](https://github.com/spike677/vllm_ascend_op_fi/tree/6dad85c480255829b0a4e74657a7e2ff2ce390b3/artifacts/attention_tiling/fault28_fia_fullmodel_statistics_20261109) · [文件及ZIP校验清单](manifest.json) · [本次归档复核](verification.json)。
