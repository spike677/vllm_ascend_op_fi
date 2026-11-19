# RC2 普通任务建设与蓝区验收（尚未冻结发行）

RC1、RC1.1冻结包和标签不变。本页是当前开发线结果；蓝区、黄区测试部署、黄区现役部署分别验收。
原始失败与旧机制证据保留，不能把机制实验、CPU成功或容器清理当作真实普通任务通过。

## 准入与统一宿主协议

`ordinary_admission.ordinary_route_report`供能力查询和创建共同使用。部署验收状态分为：
`ACCEPTED_DEPLOYMENT`、`IMPLEMENTED_NOT_ACCEPTED_FOR_DEPLOYMENT`、`NOT_IMPLEMENTED`。
安装binding或结构兼容不授予验收；未匹配组合默认返回`ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT`。
没有普通适配器的项返回`ORDINARY_TASK_ADAPTER_PENDING`，不回退C1/group/coverage。
研究机制仅保留显式`create_research_provider`，公共FaultConfig没有研究或蓝区旁路参数。

组合匹配核对14项实际信息：torch/torch_npu/vLLM/vllm-ascend/CANN、dtype、TP/PP/DP、
eager、模型配置SHA、实际模型类和内存配置与文件一致、A2/A3硬件族。各机制的源码、ABI、权重、调度和
实际调用检查仍独立生效。Batch按真实动态W8A8/INT8结构发现候选，不按模型名白名单；
新Qwen3.5/3.8、DeepSeek-V4等仍属于未验组合，默认关闭。

可信蓝区验收器可安装限定Worker/model、机制、进程/线程、attempt及源码清单SHA的许可。
它不创建accepted记录、不绕过Native或实际调用检查。只有结果审查完成后才登记精确组合。
实际CPU替身使用显式测试身份/许可，不修改生产版本常量，也不授予NPU资格。

普通任务宿主采用`advance(actual_internal_id) → 等待真实请求完成 → query`。
Batch使用同义的`advance_batch(actual_internal_ids)`；无需用户填写层名，也无需按机制插入
capture/collect。query只读，不能运行模型或推进阶段。
17在真实logits返回后完成证据；Batch在实际首decode进入前核对真实prefill token后开启观察，
真实logits返回后完成收集/Native回收。两者均没有隐藏模型调用。

Batch保留Native所需的六个真实批次normal/reference/lossless/calibration_off/fault/off，
其余当前四请求路线为normal/fault/off/health。统一宿主动作不等于抹掉机制必需的校准阶段。
正常完成finish→disable（适用时）→unregister→detach→移除binding；失败保留首错，
recover/abort后按实际状态完成OFF/健康或保留所有权，不能以卸载/重启冒充恢复。

## 本轮蓝区验收与范围

|能力|当前普通任务及蓝区范围|黄区状态|
|---|---|---|
|18|既有Source126四请求；共享回归通过|用户已确认RC1.1合入且两作用域通过；RC2组合登记/现役待验|
|39 output_proxy|既有Source116四请求；RC1.1诊断维护，新增Native错误汇总|同上；不得套用为现役支持|
|8|124 a5新构建、两个独立任务各4请求；单一实际MoE链|本轮仅蓝区验收，不要求黄区准备|
|17|124 a5统一回调、两个独立任务各4请求；单视觉投影Torch padding|本轮仅蓝区验收，不要求黄区准备|
|22|124 a1、两个独立任务各4请求；首个source-bound fused RoPE|未验|
|Batch|124 a1/a3两个独立作用域，每次六B8批及前后健康；单动态W8A8投影|黄区待配置，不要求本轮准备|
|20、26两变体、33 Source105、41|历史机制保留，普通任务仍待接入/验收|默认不可执行|
|28 denominator|真实TND F/OFF未完成|关闭；legal变体仅历史保留|

15当前部署不适用，25不支持。所有通过项只授予对应版本/模型/形状/调度组合，不代表新模型、
新CANN、TP2或全目标覆盖。8/17/22本轮没有新增msprobe/E5结论；未上传新统计或完整张量。

### 8 / 17 a5

源`16f40f1f25b9f5d5d516ce266deb2bbd01ee79b2`，清单
`bcc6759f1a1c9f8a25a825a6379d2e4b0193aadcb69fc7fc76627c0af30956fe`。
[审查总结](../../experiments/worker_core_v11_20261118/ordinary8_17_a5_20261119_summary.json)。
124卡5、CANN8.5.1、TP1/BF16/串行eager，标准vLLM Worker与调度器。

- 8：DeepSeek-V2-Lite-W8A8，正常实际forward自动选择`model.layers.1.mlp.experts`。
  在原I/P上成组映射后执行一次真实unpermute，不改router或重放算子。
  26,592/28,672元素变化，max_abs_diff=2.859619140625；hidden/logits变化，首token/文本未变。
- 17：Qwen3-VL-2B-Instruct，实际`patch_embed.proj`的Torch Conv3d padding
  [0,0,0]→[0,7,7]，原算子一次；同一完整输入/权重/bias，输出/首消费者/logits改变。
  shape [3536,1024,1,1,1]；首token未变。只验收单投影Torch参数，不是ACLNN payload/Native ABI。
- 两项均完成同Worker两个独立作用域；OFF/health精确恢复，hooks/leases/inflight为0，binding移除。
  17真实RPC query为8次，collect_replay为0次；不能用缓存命中代替encoder实际执行。
- 8默认部署准入实测通过；17实测时默认关闭、使用限定蓝区许可，13项实际字段全部匹配，
  完成审查后才晋级。a4曾在首次普通绑定前拒绝，未故障；TorchVersion字符串子类规范化修复后
  a5通过。a4失败保留，不被新结果覆盖。

### 22 a1

源`4c56701f2fc87edd8faa4e343a4b3b3b3f3987e6`，清单
`15a6eb3895e59e27c8eeef5680eea8df73c5a53cf3cf0207000dcba6a8034f94`。
[审查总结](../../experiments/worker_core_v11_20261118/ordinary22_a1_20261119_summary.json)。
124卡5、Qwen3-0.6B BF16、TP1/串行eager；两任务共8真实请求。

自动选中首条`model.layers.0.self_attn.rotary_emb`，F在真实fused入口将末位置使用的合法行
13替换为0；原算子执行一次，独立最终reader确认变换、完整cache不变及真实Attention消费。
同一完整Q/K输入，非目标输出不变；Q/K、hidden/logits变化，首token/文本未变。
OFF/health精确恢复，所有权/资源清零。`cache_creation_observed=false`原样保留，
仅观察既有cache读取，不宣称cache生成、整数溢出、全部层或连续生成验收。
实测时使用blue许可，之后才将实际13项条件加入已验组合。

### Batch a1：统一宿主回调

源`32f87374cec5028a99fb5794aad2717f35969b8d`，清单
`d0c02dde6002ca45e4401e8a2961a757267d5f4339dd77f1e0e5efa34e300096`。
[审查总结](../../experiments/worker_core_v11_20261118/ordinarybatch_a1_20261119_summary.json)。
124卡5、Qwen3-0.6B动态W8A8、TP1/BF16、B8首decode，复用Native包
`b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`；未改Native ABI/机制。

实际选择`model.layers.0.self_attn.qkv_proj`，四次Native准入为3次校准+1次F；
588次实际callback完成，其中584次透传。F中Q改变5565/8192元素（最大绝对差58），
scale改变7/8元素（最大绝对差0.009165847674012184）。两者均与当前输入的Native参考核验，
真实QMM消费且decode logits改变；所有prefill logits一致，OFF hidden/logits精确恢复。
按实际请求映射对齐，只有逻辑请求7的两token序列变化；其余阶段token相同。不是按JSON键排序比较。

Host只有6次advance、6次query，无额外capture/collect；两次常规健康B8输出一致。
Native/Hook/patch/readback所有权均释放。默认门禁在实测前保持关闭、限定许可验收；
13项实际部署字段全部匹配后才更新精确组合。Source118历史记录保留。
a1历史记录仅一个NPU作用域；本轮a3补齐第二个实际独立Worker/owner作用域，见下节。
不宣称长生成质量、全投影、新模型或新msprobe/E5验收。

## 回执大小：已定位并设公共上限

8的a2失败在验收器`write(host_receipt.json, client.snapshot())`累计Host历史超过16MiB，
没有写出该文件；原单次state没有单独留存，不能虚报其精确大小。
[a3总结](../../experiments/worker_core_v11_20261118/ordinary8_17_20261119_summary.json)复核到
两个Host回执为15,950,324和15,951,092字节，单次state最大565,122、evidence最大282,111。
17的13,117,415字节验收文件主要是重复RPC历史，单次state最大150,357。

公共上限：evidence 512KiB、单次provider_state 2MiB，与旧Host累计16MiB限制分别生效。
超限返回`ORDINARY_EVIDENCE_LIMIT_EXCEEDED`/complete=false及原件字节数/SHA，保留私有原始
观察、所有权控制字段和恢复路径，不截断成成功，不放宽阈值。长历史可使用已有cursor/stream。

|新构建实测wire最大字节|8 a5|17 a5|Batch a1|
|---|---:|---:|---:|
|provider_state|565309|150543|113163|
|evidence|282111|75138|111197|
|核查state数量|30|30|18|
|超限数量|0|0|0|

wire按`json.dumps(sort_keys=True, allow_nan=False)`默认转义/分隔符的UTF-8长度计算；
文件缩进大小不混用。公共溢出/恢复CPU测试也通过。

## 三层身份诊断与基线关闭

公共三层为deployment_identity、mechanism_compatibility、actual_call。串行eager前提
一次汇总缺失/不匹配项；部署身份错误优先，不被“未观察到消费者”覆盖。
39 postwrite与Batch bundle均接`fault_injection.prerequisites`，保留原异常类型/错误码：

- postwrite：文件、批准SHA、必要符号、实际ABI、已加载inode/device、加载后字节；持有期间
  `read_native_binding`继续检查SHA/ABI/read_span。现有公开ABI不暴露编译器/SDK清单，
  不能从运行CANN版本反推编译来源，构建回执仍须另查。
- Batch：CDLL前聚合批准manifest、文件、当前生产源码、实际opapi/SDK及进程模式；
  加载后核对DSO/SDK/ctypes布局与映射。a1实测成功加载相同冻结包，不伪造新编译身份。
- Host/RPC返回有界32KiB prerequisites和2048字符消息，保留期望/实值。
  其他Native加载器随对应普通任务推进，不宣称全覆盖。

[基线关闭记录](merged_plan_baseline_20261119.md)：RMS暂停统计边界、旧清单、CPU dladdr借用
字节生命期已关闭；4项历史冻结SHA审计保留原pin，默认作为历史检查skip，可在历史checkout
显式重跑。定向136项=132通过+4历史skip；当前源码导出/重定位3项通过。
本轮普通/共享最终组合152项CPU通过；其他定向组合有重叠，不相加作为唯一总数。
原3295项全量基线仍原样保留，未重跑全量。CPU/安装/NPU三类结论分别记录。

## 黄区准备及剩余工作

[openchamber接入准备清单](openchamber_rc2_preparation.md)包含RC1.1回归、每能力模型/权重、
TP1串行eager、视觉输入、动态W8A8及bundle要求。蓝区不能直接联系黄区；文档供用户转交，
不声称已确认黄区可获取权重。无需等待外部反馈继续开发。

普通20、26两变体、33、41以及28真实TND仍未闭环。RC2尚未冻结三包发行物；不得把本轮
开发包当作黄区现役支持或17项全覆盖。黄区接线/product_closed仍须openchamber实际验收。

## RC2 第二个 Batch NPU 作用域（a3）

[独立复核总结](../../experiments/worker_core_v11_20261118/ordinarybatch_a3_20261119_summary.json)：
源码`f83d25cbf024038a6f6c1ac324f6747c3f47ac51`，清单`199b0d9a2154179c0f67839c92a3ff22ff5cfea04576a532afd32c1510724959`。
124卡5、同一模型/Native构建。a2因旧kubelet清单SHA变化在NPU调用前拒绝；a3重新按当前清单/API/PodResources核验无活动owner后执行，未修改分配或停止外部服务。

a3正常/F/OFF机制及输入与a1一致：Q 5565/8192、scale 7/8改变，最大绝对差58和0.009165847674012184；decode logits变化、仅逻辑请求7的decode Top-1变化；全部prefill一致。OFF hidden/logits精确恢复，前后健康token一致、模型文件未变。6次advance/6次query、零collect_replay；所有Native/hooks/patch/readback释放，Host DETACHED，binding移除，容器退出0且清理。单state最大114155字节，18条state审计均为有效BLUE_COMBINATION/A2身份。

a1/a3属于两个实际独立Worker/owner，各自完成同Worker OFF/health；不宣称两owner在同一个Worker中完成。旧a1原件不补造新增14字段或许可日志。新总结工具对于缺失旧字段保持不可观测，实际两作用域资格依据各自原始证据的人工复核。无新msprobe/E5、无全目标或长生成质量结论。

RC2新增[部署授权与登记](deployment_acceptance_rc2.md)，黄色来源与蓝色记录隔离；[最新黄区清单](openchamber_rc2_preparation.md)只要求18/39/22。此前本页黄区权重准备描述由此覆盖。
