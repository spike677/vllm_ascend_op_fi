# 33 方向 A：真实 launch 控制重算检查点（2026-11-22）

本轮限额从13:00 UTC起计：6小时工作、NPU窗口合计不超过7200秒。99%逐位控制门槛不下调。冻结r5不包含此新增路线，标签和包不改动。

## 当前范围

124 A2 / CANN9.1.0 / vLLM0.26.0、vllm-ascend0.26.0rc1 / 社区Qwen3.8 W8A8 / TP2 / BF16 / 2304-token单chunk prefill + 4生成token。只针对真实paged decode：Q形状 `[1,12,256]`，KV头2，head_dim256，scale0.0625，实际KV长度2305–2307，page128。16个真实full-attention层，linear-attention不适用；所有rank上的每次符合条件的decode调用都纳入验收。

## 从真实执行取得的依据

只读拦截公开ACL的函数查找及host参数launch，参数原样转发，不读写额外设备内存。观察到decode key `5100000000010200203`、blockDim10、2960字节host参数，placeholder指向偏移296的tiling数据。首轮完整记录192个decode launch，其可解析字段一致：每KV头5个S2分组、合计10核、每组512 key，最后一组按实际长度取尾块；core起止索引分别0→1…4→5。

布局解析依据既有公开host serializer投影，**不是完整CANN二进制ABI等价证明**。原始host参数含地址，只留蓝区；公开摘要只保留去地址字段和整体哈希。完整模型复核还需检查未触及有界采集上限、实际launch均落在已审查模板。`tools/summarize_paged33_launch.py` 对截断、launch失败、placeholder变化及模板变化分别拒绝或标记。

当前SDK选中的是 `FAInfer` 的 `IS_FD` 路线（`flash_attention_regular.h`），不是另一个独立的 `FAInferDecoding` 类。控制模型按实际语义执行：BF16 Q/K矩阵输入 → FP32 score → scale/max/exp → FP32分母；exp转BF16后与BF16 V相乘，输出FP32；各核部分输出和LSE为FP32，按split顺序做LSE权重合并，最后舍入BF16。每个合法key恰好参与一次。

相关实际镜像源码指纹：

| 文件 | SHA256 |
| --- | --- |
| fused_infer_attention_score.cpp | d34068f64e515b4c8ff75f12a2222f22cf9a397fb4e5812dd3dea1c5c573f787 |
| flash_attention_regular.h | 7047e1af14d0724c10fdbaf55690acef247c07c3ce982cda1cf4bb12bd9eb693 |
| flash_attention_interface.cpp | 53802885a1dfb87a1d624cf240494899e436eee87aa1fbbebbe3ed279594dee4 |
| CombineScale.hpp | 9864ecfdc901df1034057778bd357f4f23bfe083fbd3e7cac5a36bcf2c770df8 |
| online_softmax_row_ops.inc.hpp | 7250d8eee873c8eed4d030a477f48e046c6085473ad81cd6337681060f6169e1 |

## 控制修正与失败保留

初版把BF16矩阵先转FP32再相乘。首层固定QKV的NPU控制达3067/3072和3071/3072，但完整模型rank1第47层（KV2307）只有3040/3072（98.9583%）；换CPU参考后该输入达3070/3072，却在另一个实际调用（rank0第51层、KV2306）降至3034/3072（98.7630%）。两种旧控制都未通过全模型门槛，失败回执保留。

修正：使用匹配栈实际支持的 `torch.mm(BF16, BF16, out_dtype=torch.float32)`，使QK及PV的操作数dtype与原生kernel相同；FP32累加和输出、最终BF16舍入保持不变。四份固定真实输入的NPU控制结果依次为3072/3072、3072/3072、3071/3072、3072/3072。不是调整阈值或按输出挑选元素。

对fault只改归约分组512→256，控制及fault共用上述运算边界，完整目标行被重算；不按“哪个元素恰好匹配”筛选写入。逐调用记录控制剩余差异、分组差异、原生输出差异与完整key覆盖。不要求token改变，也不声称全部控制指令与原生逐位等价。

门槛失败仍计为eligible失败，原始输出透传且不写入，待所有rank完成真实前向后的既有query/collect边界汇总失败；不能把失败记为正常不适用。此修改避免条件性异常使另一rank停在模型通信中。

## 公共入口与验收状态

- 实验包装器两作用域已通过独立复核：每rank每作用域48次eligible全部applied/changed/consumed/restored，failed=0，OFF/health精确恢复。最低控制一致率rank0为3070/3072（99.9349%），rank1为3071/3072（99.9674%）；每作用域累计控制残余元素8/2，如实保留。两作用域均有hidden/logits变化，耗时倍数1.9013/1.7546。实际1152次decode、384次prefill launch，2个进程，未达到采集上限，已解析模板字段均匹配。见 [实验独立复核](paged33_direction_a_20261122/experiment_review.json)。
- 开发线新增可信binding配置 `paged_partition=[512,256]`、`arithmetic_device="input"`，直接选择 `NativeBoundarySplitFuse33Site`，无需替换Python类名。仅接受已测长度与后端；原legacy路线保留。
- 公共宿主依旧使用 normal/fault/off/health、同组Worker两个owner作用域。每作用域每rank应观察64次full-attention调用，48次decode eligible、16次prefill不适用；不要求用户给层名。
- 组合身份新增 `paged_reference`，区分旧控制与BF16输入/FP32输出的新控制；没有验收/可信许可时保持关闭。33未加入冻结r5候选列表。
- 公共入口实机独立复验通过，见 [正式复核](paged33_direction_a_20261122/public_review.json)。同组Worker两作用域、每rank48/48次全部应用/消费/恢复，failed=0；实际消费者为 `AscendRowParallelLinear.forward`，hidden/logits均变化，OFF/health精确，许可/租约/钩子/在途请求清零。仅将实际两个rank身份写入开发线 BLUE_COMBINATION 登记，仍需可信作用域许可；不授予产品HTTP、A3、TP1或其他形状资格。

## A3限制及后续取证

A2结论不直接沿用到A3。A3须重新记录实际镜像/驱动/CANN及模型制品身份、实际decode key/tiling/合并模板、该环境的BF16→FP32矩阵接口；固定真实QKV重新达到99%，随后按全部匹配调用、真实消费者、两作用域和三项恢复验收。模板或形状不匹配时拒绝，不套用A2的512分组。无需为此改动现役服务或冻结r5。

历史与检查点摘要见 [checkpoint.json](paged33_direction_a_20261122/checkpoint.json)。完整私有QKV和host参数不上传GitHub。

## 复核入口与源码身份

正式验收驱动为 `experiments/worker_core_v11_20261118/continuous_public_probe.py --mechanism 33 --paged-control splitfuse_a2 --paged-length 2304 --arithmetic-device input`。实际模型、镜像、设备锁与有界窗口由部署侧提供；该选项只选择已审查的控制模型，不授予组合资格。运行时仍要有可信许可。

独立复核只读原始回执及原始launch，不依赖注入器的“通过”标记：

```sh
PYTHONPATH=. python tools/review_continuous_public.py /private/continuous_public.json --require-negative-creation --output /private/review.json
python tools/summarize_paged33_launch.py /private/launch-args --capture-limit 8192 --output /private/launch-summary.json
```

公共回执必须 `acceptance_scope=PUBLIC_WORKER_RUNTIME`。实验包装器需要显式 `--reference-variant A2_BF16_SPLITFUSE_512_TO_256`，不能冒充公共入口。控制记录条数必须等于eligible次数；每条从整数bitwise_equal/elements重算99%门槛。launch摘要还必须确认两进程、完整未截断、模板字段匹配。

运行源清单SHA `ac10df03ab77fe0f0064d457893ecadf0c958d834bdb1f5ab940e4b64f65aec5`，与提交 `6dbcf6c6` 核对329文件，其中325个原始字节相同，4个只有CRLF换行符差异且AST相同。实际运行SHA逐文件固定，见 [源码审查](paged33_direction_a_20261122/source_reconciliation.json)；不宣称完整checkout字节相同。只读观察器源码SHA、编译命令、镜像digest和二进制SHA也在该文件中。

控制与原生仍有少量差异，因此“完整重算后模型变化”不能百分之百归因于分组切换：逐调用单独比较control→alternate，证明分组本身产生变化；native→alternate用于实际传播验收。该区分保留，不把≥99%写成100%历史kernel复刻。

## 正式公共验收结果

|项|rank0（每作用域）|rank1（每作用域）|
|---|---:|---:|
|observed / eligible|64 / 48|64 / 48|
|applied / changed / consumed / restored|48 / 48 / 48 / 48|48 / 48 / 48 / 48|
|failed|0|0|
|最低control逐位一致率|99.9348958%|99.9674479%|
|累计control残余元素|8|2|
|control→alternate变化元素 / 有变化调用|7262 / 48|6764 / 48|
|native→alternate变化元素|7261|6765|

两作用域以上数值相同。耗时分别1.95192×、1.74387×，仅记录，不作性能门禁。原始回执、完整QKV、host指针参数留蓝区；公开复核含原件SHA和字节数、实际量化描述/索引SHA、三个调度参数。部署参数4096/4096/1。

正式窗口14:12:39.648610–14:20:23.565179 UTC，463.916598秒；10个方向A窗口共2621.496682秒（43.69分钟，包括失败、加载与清理），在7200秒限额内。13:00之前的初始census另记529.438秒，不混入新预算。所有窗口隔离已恢复，未停止外部容器；逐项时间与操作见 [windows.json](paged33_direction_a_20261122/windows.json)。

测试回执内的故障资源全部清零；模型服务进程退出仍有vLLM强制结束Worker及Python shared_memory清理警告，保留为独立的第三方关闭限制，不包装成整个服务关闭无告警。
