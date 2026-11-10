# 故障33：两组完整模型 statistics/mix 数据

本分支只保留 **`model_baseline` 与 `model_fault_r0`** 两组采集，附独立首差报告、推理结果复核及必要来源材料。分支使用独立根提交，不继承实现仓库历史。

| 采集组 | 真实控制 | 网页预览 | 完整原始数据 |
| --- | --- | --- | --- |
| `model_baseline`（B） | OFF，KV计算分块8192、2轮 | [首个IFA原始记录的摘录](model_baseline/first_ifa_excerpt.json) | [model_baseline.zip](downloads/model_baseline.zip)，0.55 MiB |
| `model_fault_r0`（F） | inner1024，KV计算分块1024、16轮 | [首个IFA原始记录的摘录](model_fault_r0/first_ifa_excerpt.json) | [model_fault_r0.zip](downloads/model_fault_r0.zip)，0.33 MiB |

原始完整dump每组约2.2 MB，连同原始`stack.json`、`construct.json`、配置、采集器身份及完成回执保存在ZIP中，字节未改。网页摘录明确标记`generated_excerpt=true`并绑定源dump的SHA，**不是完整dump**。每组只有step0/rank0；这里的step0是一次decode的采集编号。

[独立首差摘要](first_difference.json) · [完整独立首差报告](reports/independent_first_difference.json) · [推理结果复核](inference_result.json) · [CPU作业回执](inference_recheck_receipt.json) · [归档与复算校验](verification.json) · [文件清单](manifest.json)

## 连续生成补测状态（2026-11-10）

按用户最新指令，**本轮停止等待卡5，不执行新的实机补测，直接提交当前方案与状态说明**。[32-token连续生成补测方案](planned_continuous_validation.json)仅为待实施设计，连续生成适配和验收实现尚未完成，尚无新的NPU调用或msprobe采集。

当前B/F两组原始数据、独立首差及下述推理比较均保持原有单步decode范围，**不能据此判断连续生成token或完整文本是否变化**。方案拟在现有a4支持的KV长度16,384处核验全部28层实际命中，再沿各组自己生成的token继续decode；后续超出匹配范围的调用必须单独记录为未匹配。此流程尚未实测，不作为通过结论；恢复补测需用户重新指示。

## 注入前后的推理结果

2026-11-10在124复读两组正式实机采集对应的完整FP16 logits，逐文件及逐请求校验SHA，再用该模型同一tokenizer解码贪心下一token。结果为：

- 8/8请求的logits均变化；1,035,472 / 1,215,488个元素改变，最大绝对差0.1171875。
- 8/8请求的贪心下一token相同，对应解码片段也相同；没有将logits变化解释为已经出现错误文本。
- 两侧均有限值，每个请求的最大logit唯一。完整Top-5、逐请求误差、token及tokenizer指纹见推理报告。

| 请求 | B token ID | B片段 | F token ID | F片段 |
| --- | ---: | --- | ---: | --- |
| request_00 | 19415 | ` compiler` | 19415 | ` compiler` |
| request_01 | 15311 | ` titles` | 15311 | ` titles` |
| request_02 | 46876 | ` algebra` | 46876 | ` algebra` |
| request_03 | 729 | ` function` | 729 | ` function` |
| request_04 | 323 | ` and` | 323 | ` and` |
| request_05 | 279 | ` the` | 279 | ` the` |
| request_06 | 6008 | ` plant` | 6008 | ` plant` |
| request_07 | 6439 | ` worked` | 6439 | ` worked` |

这些是**一个预测token的文本片段**，不是完整回答。原始输入是此前冻结的8条长重复种子文本，先作16,383-token SDPA prefill，再喂入相同已知teacher token，比较长度16,384处的一步decode。没有测试新的短聊天输入或连续自由生成。

本轮是已有NPU结果的CPU复核，**没有重新启动NPU推理或重新采集msprobe**；源实机采集于2026-11-06完成。卡5当时由其他PTQ任务预留，未停止外部任务。最终CPU复核为`cpu_inference_a3`，无NPU设备挂载、无网络，容器已清理；早期CPU依赖检查失败回执保留在124，未充作通过结果。

## 独立首差及故障原理

只把两组原始dump/stack/construct、配置、采集器身份与统计回执复制到物理隔离目录，排除case/result、mode、host命中和数值验收报告。比较全部2,510节点后，首个可观测API差异为：

**第0层`model.layers.0.self_attn`的`NPU.npu_incre_flash_attention.0.forward`输出0。** 包含诊断API的未过滤原始顺序也得到同一首差。

| 字段 | B | F |
| --- | ---: | ---: |
| `$[0].Mean` | -0.001310580875724554 | -0.0013106090482324362 |
| `$[0].Norm` | 14.949091911315918 | 14.94908618927002 |

Max=1.328125、Min=-0.75927734375两侧相同。该节点14项有限输入统计相同；int32块表的Mean/Norm为null，保持不可观测。整份trace的178项不可观测字段均保留，四统计相同不证明完整张量相同。

另一条取证链独立解码两组各28份最终公共host serializer raw：B实际KV计算分块8192/2轮，F为1024/16轮，物理`block_size`均为128。它不参与首差选择。原始28层调用映射、实际launch摘录、输入哈希和公共raw随包保留。

预期机制是：相同物理输入下，KV计算分组及softmax更新轮次改变，导致有限精度数值差异，首先在分页IFA输出可见，再传播到`o_proj`和logits。**当前B/F验证了这类受控机制；没有在这对B/F中改变物理页大小。** 原生blockSize变化的stock发现和历史事故需要各自证据，未纳入这两组交付。

## 采集配置、覆盖及限制

- 两组均为`task=statistics`、`level=mix`、`summary_mode=statistics`、同步且无张量筛选，使用已披露的任务私有high FP32四统计归约采集器。补丁、原始/修改后源码、许可证与指纹随ZIP提供；它不是故障28后续使用的CPU FP64采集器。
- 真实Qwen3-0.6B、FP16、TP1，124/910B4/CANN9.1；8请求、Q16/KV8/D128、16,384长度、128-token物理页。两组各28层分页IFA decode，共56次实际调用、448个请求/层组合；F组28/28匹配调用均有受控记录。
- 模型保留原SDPA prefill，decode通过明确的分页IFA适配器接入。它不是故障28的FIA整图prefill路径，也不是未经适配的自然MindIE/vLLM调度。
- msprobe可定位首个可观测的算子输出边界；不能独立唯一确定某条Reduce指令或内部tiling根因。公共host raw仅为72字段/3,000字节serializer投影，`kernel_argument_raw_observed=false`。
- 公开包不含完整张量、权重、原始大日志或编译库。完整logits差值来自124上私有二进制复读，公开哈希不能独立重算这些差值；统计首差可用公开文件重新计算。
- 当前两组未附F1、OFF恢复、stock预检或算子级其他组；这些历史对照留在[原发布提交c1dcb0f](https://github.com/spike677/vllm_ascend_op_fi/tree/c1dcb0f22438362bfdd7100f834b0be62cd425c9/artifacts/attention_tiling/fault33_paged_statistics_20261106)。Top-1未变不保证后续生成或业务准确率不变，也没有证明历史MindIE RC2 B080事故已复刻。

## 下载及复算

下载两个ZIP后解压到同一目录；B包含公共核验材料，F包只含F组，两个包路径不重叠。完整恢复71个文件、13,293,841字节。也可下载本分支并执行：

```bash
python verify_downloads.py
python verify_downloads.py --extract-to /path/to/new-directory
```

目标目录须尚不存在。脚本检查ZIP SHA、全部还原文件、原始来源和网页摘录。还原后，从新目录独立复算首差：

```bash
python data/analysis/compare_model.py \
  --baseline data/model/model_baseline/observation/step0/rank0/dump.json \
  --fault data/model/model_fault_r0/observation/step0/rank0/dump.json \
  --output /path/to/new-first-difference.json
```

`data/source_mapping.json`逐项映射原文件路径、长度和SHA，标明2026-11-10新增CPU复核材料；`data/SHA256SUMS`覆盖全部数据文件。完整三件套、统计配置及原有回执均按原字节保留。
