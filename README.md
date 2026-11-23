## 2026-11-22 隔夜：r5冻结与33方向A通过限定验收

[r5交付单](releases/RC3-PREVIEW-R5-HANDOFF-20261122.md)：源码、三wheel、安装检查和SHA齐备；产品RPC契约、900秒正式预算及身份字段已发布。黄区回传缺身份时不登记，现有冻结物不改动。

[33方向A](docs/worker_core_v1/paged33_direction_a_20261122.md)已在A2/0.26/CANN9.1/Qwen3.8社区W8A8/TP2完成正式公共入口两作用域。BF16矩阵操作数、FP32输出的控制最低一致率99.9349%；512→256分组后，每rank每作用域48次decode全部应用、消费和恢复，hidden/logits变化，OFF/health精确。仅开发线限定BLUE登记，不是历史kernel逐位复刻，不在r5，也不授予产品模式或A3资格。

[隔夜汇总](docs/worker_core_v1/overnight_summary_20261122.md) · [DeepSeek噪声规则](docs/worker_core_v1/dsv4_noise_acceptance.md) · [V4只观测步骤](docs/worker_core_v1/dsv4_observe_handoff_20261122.md) · [AllGather参照](docs/worker_core_v1/allgather8_reference_20261122.md)。后续历史记录按其日期和范围理解。

## 2026-11-22 夜：Worker 自动发现真实 HTTP 请求

[产品接口](docs/worker_core_v1/rc3_product_usage.md)只支持开启期间全部请求。Worker从实际调度识别内部ID；宿主只控制attach/enable/status/disable/detach，用户直接走vLLM推理API，不再调用逐请求钩子，也没有next-N配额。

[本轮HTTP验收](docs/worker_core_v1/product_http_acceptance_20261122.md)与历史宿主钩子结果分别记录。A2/0.26/Qwen3.8社区W8A8/TP2结果不代表黄区产品闭环；`product_closed=false`，未登记组合默认关闭。

[黄区r4审查](docs/worker_core_v1/yellow_intake_r4_20261122.md)：五项通过是收到的文本结论，缺summary全文及逐组合参数，尚未登记。量化描述、全部权重索引SHA和三个调度参数是现役登记前提。[33方向A](docs/worker_core_v1/paged33_native_a_20261122.md)已取得实际launch key和核数，动态tiling与累加顺序仍未闭合，不扩大替换范围。[V4-0731核对](docs/worker_core_v1/dsv4_0731_adaptation_20261122.md)与V4.1历史准备分开。

冻结r4及既有发行物保持不变；耗时本轮只记录，不做优化。以下为历史检查点，不扩大当前接口或验收范围。

## 2026-11-22：Native 公共持续入口增量

A2／CANN9.1／vLLM0.26／Qwen3.8社区W8A8／TP2 上，[41 KV区域](docs/worker_core_v1/continuous_native41_20261122.md)、[26 FIA写入residual](docs/worker_core_v1/continuous_native26_20261122.md)、[20输入内容覆盖](docs/worker_core_v1/continuous_native20_20261122.md)均已完成公共入口同组Worker两个作用域，全部eligible调用消费与字节恢复、完整OFF/health精确恢复及资源清零。每rank每作用域分别为48、64、1216次；41的16次直接KV prefill调用不适用。17视觉公共验收也已通过，33因模型传播未达标保持关闭。

新增项仅在开发线，冻结r3仍为原六项；A3须独立构建和验收，不代表现役对外服务原样配置可用。[1023项CPU子集](docs/worker_core_v1/native_cpu_review_20261122.md)无失败或错误，2项旧研究入口跳过；[MC2接线准备](docs/worker_core_v1/mc2_dispatch_join_20261122.md)仍缺runner分片行与接收端expert归属，不登记DeepSeek。

## 2026-11-22：r3收尾与验收范围

[39 matmul_range / 33报告](docs/worker_core_v1/continuous_range39_paged33_20261122.md)：A2 / 0.26 / Qwen3.8社区W8A8 / TP2，39本地列分片投影公共两作用域和源码审查通过；每rank704次eligible调用全部消费恢复。33合法重排已有实际消费和局部变化，但最终模型传播门禁未过，保持关闭，不加入r3。

[RC3预览r3交付](releases/RC3-PREVIEW-R3-HANDOFF-20261122.md)已冻结六项：28、Batch、39 output_proxy、18、22、39 matmul_range。971项范围门禁及33项补充检查通过；124无设备安装核对358个源文件一致，Git发行字节已复核。由负责人转交，未直接向黄区发送。87的TP1、TP2各自验收登记。支持对象是eager、串行、无异步调度/prefix cache/MTP/KV connector的专用实例；现役对外服务的原样配置不在支持范围。

## 2026-11-21 晚：公共持续入口

已冻结 [RC3 预览 r2](releases/RC3-PREVIEW-REPLACEMENT-20261121.md)，仅 28/Batch；标签 `demoncatpfi-rc3-preview-20261121-r2`。三个 wheel、源码包、SHA、无设备安装检查及 A3 TP2 验收步骤齐备。原预览标签发生 Git 换行归一化导致清单失配，保留失败记录，不应使用。r2 从 Git 对象复核全部发行 SHA 一致。完整 RC3 与 A3 验收尚未完成。

[阶段1实施及验收](docs/worker_core_v1/continuous_runtime_20261121.md)：28 torch denominator_replay 已在 A2 / 0.26 / Qwen3.8 社区 W8A8 / TP2 通过公共 Runtime 同组 Worker 两作用域，逐次租约、消费者事件、OFF/health 完整恢复及真实无许可创建拒绝均通过。Batch NPU 与 CPU 公共入口两作用域完成，最终交付脚本真实创建负向检查通过。开发线新增39 output_proxy、18、22公共持续两作用域通过，消费者及恢复完整。它们未加入冻结r2。详见[增量候选](docs/worker_core_v1/rc3_continuous_candidates_20261121_night.md)。

[夜间汇总](docs/worker_core_v1/night_summary_20261121.md)：V2-Lite TP2的28 MLA、8、18、22、39 output_proxy已通过声明范围内的公共两作用域；18融合KV内部norm、MoE单组decode无新变化、39不适用输出均明确保留。Batch延迟核对实测约5.93–5.97倍，37项相关CPU测试通过；历史测试欠账仍未清零。19个实机窗口全部解除隔离，未冻结r3。

[12项进度总表](docs/worker_core_v1/continuous_progress_20261121.md)区分实验 binding 和公共入口；[RC3预览验收步骤](docs/worker_core_v1/rc3_preview_acceptance_20261121.md)仅含28/Batch，A3待验。Native28保留研究入口，公共入口继续关闭。以下旧记录保留历史适用范围，不覆盖本节新状态。

[2026-11-20最新实机进展](docs/worker_core_v1/npu_progress_20261120.md)：A2/0.18/Qwen3-0.6B的18、22、39、Batch、41 TP2已完成限定的同Worker双作用域闭环。新栈A2/0.26/CANN9.1/社区Qwen3.8 W8A8 TP2的18及22，已通过预热后双作用域的实际命中、全rank OFF/health精确恢复与资源清零；冷启动首请求有无注入差异，仍不在验收范围。22覆盖实际融合mRoPE目标Q/K行；39的社区量化out_proj不等价于黄区BF16目标。详见[新栈适配](docs/worker_core_v1/v026_adaptation_20261120.md)。

[2026-11-21无注入定位](docs/worker_core_v1/m2_b8_nan_localization_20261121.md)：新证据将“GDN请求块漏写”修正为错误累加：WY的23个可匹配head逐位符合把两个BF16作为FP32与旧存储相加，KKT也出现跨任务累加。零初始化及单独前置复位均不是完整修复；按负责人决定已停止继续追查和复位试验，保留为环境缺陷证据。冻结包与公共准入不变。

[28 torch 持续触发真实模型验收](docs/worker_core_v1/continuous28_real_model_20261121.md)：V2-Lite W8A8 和社区 Qwen3.8 W8A8 / 0.26 / A2 / TP2 已完成真实接线两作用域。每作用域每 rank 分别为27层108次、16个 full attention 层64次，全部实际改写、被消费者读取并恢复；OFF/health 完整 hidden/logits 精确恢复。公共 Runtime 准入尚未开放，不能将实验 binding 通过等同于已发行能力。

28已完成[真实TND两阶段Native研究闭环](docs/worker_core_v1/tnd28_fresh_values_20261120.md)：两个research owner、FIA与hidden/logits变化、完整OFF/health恢复。它不是普通Runtime B5验收，公共入口继续关闭；M128止损与其他限制保留。

[RC2.1](https://github.com/spike677/vllm_ascend_op_fi/tree/demoncatpfi-yellow-rc2.1-20261120)和[RC2.2](https://github.com/spike677/vllm_ascend_op_fi/tree/demoncatpfi-yellow-rc2.2-20261120)均已冻结发行，标签与包保持不变。RC2.1仅登记黄区测试部署的文本证明；RC2.2不包含多卡和新软件栈代码。A2结果不代表A3或黄区现役，尚未发行RC3。

# vLLM Ascend Operator Fault Injection

这是一个面向 vLLM-Ascend/CANN 的开发者模式原生算子故障注入原型。实现覆盖 CANN repeat executor-cache replay 边界、typed operator adapter 和 temporary device buffer 替换，并提供受控、可审计的输入数值退化、非有限输出代理与 allocation-boundary 写错地址代理。

本仓库保留原生实现和历史限定验收，并提供经过审查的独立集成包、CPU测试和统计证据。
不公开模型权重、完整私有张量或 vendor 库。下文旧能力表属于历史运行范围，不自动授予新接口验收。

## 黄区测试部署 RC1.1 维护

首批范围仍为 **39 output_proxy、Batch 普通 W8A8、18 普通单任务**。
RC1.1 修复39阶段用完时的诊断，公共39入口拒绝历史C1绑定，并补齐四请求、
真实内部ID、证据出口、环境与清理说明；不改OFF/health状态机。
交接入口：[openchamber 接线、安装与恢复](docs/worker_core_v1/yellow_rc1.md)；
发行目录：releases/demoncatpfi-yellow-rc1-1-20261119/。
原RC1标签、包和SHA不覆盖。
openchamber已报告黄区Qwen3-0.6B BF16/TP1/eager测试部署的18、39两轮独立任务通过；
Batch因缺少动态W8A8权重和bundle待配置。黄区现役多卡支持及product_closed仍待实际验收。
28/33等后续开发不进入本维护版本的完成声明。

## RC2 普通入口调整与124复核

[当前范围、准入及实测总结](docs/worker_core_v1/ordinary_tasks_rc2_20261119.md)：
8、17、22均完成新普通路线的两个独立作用域；Batch完成统一回调后的六批及前后健康复核。
宿主使用advance（Batch为advance_batch）→等待→query，无17/Batch专用capture/collect步骤。
实现、结构兼容和精确部署验收分开；未验组合默认不可执行，研究入口不能由用户配置打开。
39 postwrite与Batch bundle接入统一身份诊断；普通证据512KiB、单次state2MiB上限明确。
[RMS及历史检查关闭记录](docs/worker_core_v1/merged_plan_baseline_20261119.md)与
[openchamber模型准备](docs/worker_core_v1/openchamber_rc2_preparation.md)已补齐。
RC2六项子集已冻结；后继开发线41普通单点已完成124两个作用域，见[41验收边界](docs/worker_core_v1/kv41_ordinary_task.md)。20/26两变体/33普通路线及28 TND仍待完成，不宣称17项全通过或黄区现役支持。
RC2.1黄区测试部署登记已经冻结；master实际升级由负责人转交步骤后执行，不由本分支推定已升级。RC2原[补充单](releases/RC2-HANDOFF-20261119.md)保留历史口径，当前发行状态以页首链接为准。

## 当前统一接入

[2026-11-19 产品交付矩阵与实施状态](docs/worker_core_v1/product_delivery_20261119.md)：
本轮已把缩放、独立存储转换和行写入抽为共用实际动作，并将有界数值摘要接入18的
Worker查询。故障25明确不支持；39输出代理限定独立单次／每请求最多一个目标；
33只继续已验收的Source105单点；28只做真实TND分母重复累计，**该故障实现仍未完成**。
Batch先接普通Qwen W8A8投影，不等待routed GMM1。旧全目标研究、失败和统计原件保留。

新蓝区公共入口实测：39已完成单任务一个实际输出目标的F及同Worker OFF/health恢复；
普通W8A8已完成自动qkv选点、Q与scale共同变化及OFF精确恢复。结果与范围见上述矩阵，
不授予主项目接线、累计C3、E5或跨版本资格。查询现在区分当前请求效果和历史F效果，
新OFF/健康请求尚未执行时不会显示旧F数值。
18新增[普通单次运行时](docs/worker_core_v1/rms18_ordinary_task.md)：自动绑定首个实际RMS返回及消费者，
normal/F各观察一个目标，OFF/health透传；旧十请求覆盖实验保留。Source126已完成实机及独立复核：
14336个值减半，OFF/health完整hidden/logits恢复，首token不变；[限定结论](docs/worker_core_v1/review_rms18_ordinary_source126.md)。

**实际主项目尚未接通。** 当前蓝区没有黄区现役Catalog／Controller／推理调度及结果界面源码。
公共契约、包安装和CPU测试均不作为主项目验收。运行时与验收协议的拆分也尚未全部完成。
能力查询现在返回静态 `current_delivery`，并明确实际部署未评估、主项目效果和OFF未验收。
新鲜Worker仍必须经过现有对象、存储、构建及所有权门禁，不能用历史报告授予资格。

[公共 Worker Core V1](docs/worker_core_v1/README.md) 已统一 `FaultConfig`、真实 Worker 身份、
模型结构发现、实际调用盘点、provider 生命周期和源码交付包。终端用户只填写故障编号及变体；
部署适配器根据实际对象和调用选择位置。历史清单保留17条，16条有RPC工厂，25在创建前拒绝；
**工厂存在不代表已支持**。33有受限定的单点执行，28真实TND故障仍受未完成适配限制。
工厂存在、模型自动发现与实际故障验收分别记录。

以下是历史检查点与证据索引；其中旧的下一阶段研究要求由上述产品矩阵取代。
[Worker Core V1.1 当前实现与实机检查点](experiments/worker_core_v11_20261118/README.md)：
17的单投影Torch参数层、26 FIA累计注入的完整原件已独立复核；
26逐目标独立回放、Batch共享尺度全部请求行、39输出全目标发现及33正常桥接仍在补齐。
本检查点不声明17/17全部验收，也不继承E5或跨版本资格。
[当前逐项矩阵](experiments/c2c3_closure_20261117/C2_C3_MATRIX.md) ·
[本轮公共入口与新构建验证](experiments/c2c3_closure_20261117/CORE_PROVIDER_PROGRESS_20261118.md)。
[后续单目标效果、原件复核与失败边界](experiments/c2c3_closure_20261117/WORKER_INCREMENT_20261118.md)：
26 QMM/FIA 和39输出变体现已通过新的公共 Worker 单目标效果、OFF和原件独立复核；
8/18的公共入口也已单独完成 C2/C3 复核。39此前140个 MODEL_LOCAL 独立回放及持续注入失败记录保留，不转移资格。
[本轮交付报告](experiments/c2c3_closure_20261117/WORKER_CORE_DELIVERY_20261118.md) ·
[可复现源码包清单](experiments/c2c3_closure_20261117/integration_bundle_delivery.json)：
373个源码文件，最终包解压后的214条Core测试通过；三个分发包及installed-only测试已验证。
新公共26/39入口的diagnostic采集尚未形成独立statistics/mix窗口，E5明确未验收。
已有通过范围保留，新的原始统计、完整张量和 Native 库均留在124。
目前严格限定验收为 C2 6/17、C3 6/17；公共 Core22、Core41 的原件独立复核已通过，
覆盖、恢复与 msprobe 独立定位分别记账，不能由前两项推定 E5 通过。

[2026-11-17 17条工程整合与新增机制实测](experiments/engineering_20261117/README.md)：
从 `cae9cdf4` 继续建设算子专属构建器、真实边界身份、Tensor/MoE/RoPE/通信适配、
20自然QMM、26 residual、独立输出/缓存尾列Provider以及28新构建与33实际分页诊断。
构造机制、标准Worker、独立统计与完整覆盖分别记账，未授予公共 C2/C3。

[2026-11-17 P3 独立统计诊断与 P4 推进](experiments/p3p4_closure_20261117/README.md)：15/17/18 的原严格失败保留；26 FIA 新增真实 SmolLM2/GLM4 公共 Worker 的 writer→consumer→恢复，29/39次适用调用。真正 DeepSeek-V2-Lite 新增batch共享量化的公共Worker单点、逐请求Q/scale和OFF恢复；599项CPU测试通过。[FIA限定验收](experiments/p3p4_closure_20261117/FIA26_REPORT.md) · [量化首差与统计限制](experiments/p3p4_closure_20261117/QUANT_REPORT.md)

[2026-11-17 真实 Worker 正向闭环](experiments/worker_positive_20261117/README.md) · [可信 Serving Replay 接口](docs/worker_extension/host_driven_replay.md)

从 `dcbef7a0` 继续，Fault20 WeightNz 和 Fault39 MatMul 已在标准 vLLM 创建的临时 Worker、安装后的 wheel、完整 Qwen3-0.6B 上取得限定单点的真实注入和 OFF 恢复证据。Host 使用真实 enqueue/scheduler ID 逐阶段推进，Extension 不创建 Worker、不替换核心执行方法。Fault20 使用明确披露的 NZ 部署和 controlled PTA 路径；不能把它称为默认部署已通过。

Fault39 的独立 B/F 原始 statistics/mix 在固定容差下，首个可观测差异为 `Functional.linear.112.forward/output/0` 的 Max：17.625→+Inf；之前的不可观测项保留。GLM-4-9B-0414 有26 FIA限定故障证据；真正 DeepSeek-V2-Lite 的量化首decode可观测首差为 DynamicQuant scale输出 Min：0.13877953588962555→0.14271654188632965。后者完整基线自比仍有NaN，不能称为全段严格统计通过。新采集原件留在124，GitHub仅发布实现、总结及哈希。

[Fault33连续生成](experiments/p3p4_closure_20261117/FAULT33_REPORT.md)已在独立HF路径完成4次真实decode、KV增长、实际切分命中及OFF精确恢复，B/F各审计336次IFA launch。logits逐步变化而生成token不变；统计首差在下游，未取得IFA唯一定位或公共Worker资格。

[2026-11-16 的17条审计报告](docs/fault_closure_report.md) · [当时逐项矩阵](experiments/overnight_20261116/fault_matrix.json) · [当时源码与证据校验](experiments/overnight_20261116/checksums.json)

此前17条能力卡审计、445项CPU测试和15/17/18构造NPU模块B/F/OFF结论保留；其严格统计调用链对齐失败也保留。下文旧能力表均为历史范围。当时尚无公共 `all_eligible` 资格；后续已经审查的限定 C2/C3 范围见当前逐项矩阵，单输出投影验收仍不等于全模型所有层注入。

[文档索引](docs/README.md) · [生产接线分支的实现与验收](experiments/production_wiring_20261116/README.md) · [17 项能力与支持矩阵](docs/unified_fault_inventory.md#9171f3b-全局审计)

蓝区可独立安装 `demoncat-worker-extension`，由 vLLM-Ascend 的 `--worker-extension-cls` 加载到已有 Worker。共用入口管理 Worker 身份、附着、Provider 分发和原生状态查询。Fault20 WeightNz／QMM 与 Fault39 MatMul 分别沿用已有的原生实现和单点资格门禁。[接入与部署](docs/worker_extension/deployment.md) · [124 验收记录](experiments/worker_extension_20261116/README.md)

124 在 vLLM-Ascend 0.18.0、CANN 8.5.1 和 Qwen3-0.6B BF16 eager 下通过了 Worker 附着验收。TP1 验证了 Fault20／Fault39 预加载后的原生空闲状态、缺少可信回放时的注册拒绝，以及解除附着后的生成一致性。TP2 验证了两个 Worker 的 rank、逻辑卡和正常生成。Worker 重建后，旧租户和旧 owner 票据均失效。

上述是11月16日的附着与负门禁范围；11月17日新增的正向闭环见本节首个链接。旧回执保持原样。[附着验收总结](experiments/worker_extension_20261116/validation_summary.json) · [当时 SHA-256 校验清单](experiments/worker_extension_20261116/SHA256SUMS)

已有 `attach` 和 `ModelFaultBackend` 入口。显式登记合格 Worker Provider 后，Fault20 可走新的单目标回放门禁。Fault39 保留独立 ABI v1，并提供可选宿主资格适配。覆盖资格可在 CPU 核对完整目标集、两种分母、租约和恢复；真实执行按已有单点门禁和各机制发布范围。生产状态与已发布实机证据见实验记录。

- [5条部署 provider 与 Fault39 8.x/9.x 本轮实机总结](experiments/provider_adaptation_20261116/README.md)：
  自动选点、原生命中、B/F/OFF、独立统计首差及不可观测限制；本轮不公开新采集原件。
- [其余15个机制的统一接口与逐项状态](docs/remaining_faults_integration_20261116/README.md)：
  模型结构自动选择、注册默认关闭、真实 disable/clean、清理失败保留及文档宿主契约。
- [Fault39 独立原生接入及124实测](fault39/README.md)：保留两种真实架构、TP2及B/F证据，
  不与旧 `39.output_proxy` 混用。

黄区现役内源 `src/demoncat` 和 CLI、catalog、bindings 的接线仍待验证，宿主契约保留 `INNER_FRAMEWORK_INTEGRATION_PENDING`。

## 已验证能力

| 项目 | 状态 |
|---|---|
| `aclnnQuantMatmulWeightNz`，INT8 x1=`addresses[0]` | `PROVEN_E2E` |
| `aclnnMatmulWeightNz`，BF16 self=`addresses[0]` | `PROVEN_E2E` |
| INT2-equivalent / INT4-equivalent primitive | `PROVEN_E2E` |
| BF16 mantissa truncate | `PROVEN_E2E` |
| single/multi-hit、dual-rank、all-hit selector | `PROVEN_E2E` |
| QMM x1 extreme-value single/progressive/all-hit | `PROVEN_E2E`（CANN 8.5.1，最大 12,160 hits） |
| QMM x2/weight runtime slot 与 NZ storage metadata | `PROVEN_E2E_READ_ONLY`（CANN 8.5.1） |
| QMM x1+x2 双 temporary buffer（lossless / 127 extreme） | `PROVEN_E2E`（CANN 8.5.1，rank0 single-hit） |
| QMM x2_scale formal tensor、runtime slot 与 FP32 metadata | `PROVEN_E2E_READ_ONLY`（formal ordinal 3，`addresses[2]`） |
| QMM x2_scale temporary buffer（lossless / finite ×2） | `PROVEN_E2E`（CANN 8.5.1，rank0 decode single-hit） |
| `aclnnMatmul` BF16/ND slot mapping 与 self/mat2/dual lossless temporary buffer | `PROVEN_E2E`（CANN 8.5.1，rank0 decode single-hit） |
| `aclnnMatmul` 有限 BF16 双输入到真实 `+Inf` 输出 | `MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED`（`2^58` 首次 non-finite） |
| `aclnnMatmul` 有限 BF16 正负抵消、数学结果为零的排列对照 | `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED`（half non-finite；block/alternating 全零） |
| `aclnnMatmul` post-Run output `+Inf/-Inf/qNaN` proxy | `PROVEN_E2E`（1/5/51/512 elements，readback exact） |
| Fault26 QMM output-base underflow → live residual stomp proxy | `PROVEN_E2E`（rank0，allocation-boundary，231 all-eligible hits） |
| Fault26 fault request 输出分叉与 disarm 后精确恢复 | `PROVEN_E2E`（token 7 首次分叉，post-fault exact recovery） |
| Fault20 v17 metadata ABI handshake + immutable bundle | `PROVEN_E2E`（CANN 8.5.1，TP=2，C1/C2/C3/recovery） |

结论严格限定于已验证环境和算子，不代表任意模型、任意算子或跨版本生产能力。详细统计见[当前注入能力统计](fault20_fast_validation/reports/fault20_injection_capability_inventory_20260807.md)。

## Fault39 current capability

| 路线 | 已验证内容 | 结论 |
|---|---|---|
| QMM extreme | x1 single/progressive/all-hit；x1+x2 dual extreme | 输入替换与模型传播已闭环，真实 QMM output 全 finite |
| QMM x2_scale | formal ordinal 3、Run slot 2、lossless、有限 `×2` | scale 扰动传播已闭环，真实 QMM output 全 finite |
| Floating MatMul finite dual | self/mat2 temporary 均为有限 BF16 | `MATMUL_FINITE_INPUT_NONFINITE_REPRODUCED` |
| Floating MatMul cancellation | 理论结果为 0；half 与 block/alternating 排列对照 | `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED` |
| MatMul output proxy | post-Run `+Inf/-Inf/qNaN`，1/5/51/512 elements | 三种 `MATMUL_OUTPUT_*_PROXY_PROVEN` |

当前证据层级：

```text
Root-cause approximation
  finite dual BF16 -> real aclnnMatmul natural +Inf
  cancellation     -> theoretical zero, layout-dependent natural +/-Inf

Outcome proxy
  real finite output -> direct +Inf / -Inf / qNaN -> downstream anomaly
```

最强 A2 判定是 `MATMUL_INTERMEDIATE_RANGE_FAILURE_REPRODUCED`。它证明真实
`aclnnMatmul` 对 K 轴正负排列表现出不同的 non-finite 行为，但没有定位 kernel 内
accumulator 的类型、位置或首次 overflow 时刻，因此不能写成
`ACCUMULATOR_OVERFLOW_PROVEN`。完整证据见
[Fault39 A2 复现总结](fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md)。

## Fault26 current capability

Fault26 的当前最强能力不是“直接改坏 residual”，而是在已由运行时 allocator
只读证明的相邻 allocation pair 上，把真实 QMM output base 向后偏移，使真实
`aclnnQuantMatmulWeightNz` 写入跨过 allocation boundary 并覆盖 live residual；
随后恢复 padding 与正常 output，让精确的下一次 `aclnnAddRmsNormBias` 消费受控破坏，
并在 consumer 完成后恢复 residual。

```text
live residual allocation end == QMM output allocation begin
  -> real QMM Run with bounded backward-shifted output base
  -> allocation boundary crossed
  -> next exact AddRms consumes corrupted residual
  -> padding/output/residual byte-exact recovery
```

Phase 30 在 rank0 的三个精确、live-proven 地址对上完成一次请求内 231 次
all-eligible hit（`7 + 7 + 217`），失败/中止为 0；模型响应在 completion token 7
首次分叉，显式 disarm 后下一请求精确回到 baseline。`all eligible` 只表示命中控制面
列出的、已证明相邻且仍存活的地址对，不表示对未审计 QMM 地址做无条件写入。

该能力应准确命名为：

```text
QMM_OUTPUT_BASE_UNDERFLOW_RESIDUAL_STOMP_PROXY_PROVEN
ALL_ELIGIBLE_QMM_HITS_PROVEN
MODEL_RESPONSE_DIVERGENCE_PROVEN
POST_FAULT_EXACT_RECOVERY_PROVEN
```

它是本地开发者模式下用于诊断工具验证的受控算子故障代理，不证明健康 QMM kernel
会自然发生 OOB，也不证明历史 FIA workspace 缺陷的物理根因。

## 仓库内容

```text
fault20_fast_validation/
├── evidence/                            # 完整 Fault26 验证证据压缩包与校验说明
├── source/
│   ├── l4_buffer_substitute_allhit_v1/   # 最终 runtime、控制面和容器脚本
│   ├── l4_metadata_lifecycle_proxy/      # typed GWS metadata adapter
│   ├── fault26_readonly_trace_v1/        # residual/QMM trace、underflow runtime 与 worker 控制面
│   └── fault26_fia_writer_proxy_v1/      # FIA actual-writer / wrong-destination 分阶段代理
└── reports/
    ├── fault20_final_architecture.md
    ├── fault20_capability_matrix_final.md
    ├── fault20_fault_primitive_design.md
    └── fault20_injection_capability_inventory_20260807.md
```

## 已验证环境

```text
model       Qwen3-32B-w8a8-nopdmix (dense)
vLLM image  quay.io/ascend/vllm-ascend:v0.12.0rc1
vLLM pkg    0.12.0
torch_npu   2.8.0
CANN target 8.3.RC2
arch        aarch64
C++ ABI     _GLIBCXX_USE_CXX11_ABI=0
execution   TP=2, eager, repeat executor cache
```

CANN 8.5.1 / vLLM 0.18.0 / torch_npu 2.9.0.post1 已完成独立编译、
`libnnopbase.so` ABI smoke 和 Qwen3-32B TP=2 模型级 E2E。3/16/64/all-hit extreme
均已验证；all-hit 的 12,160 次真实 QMM output 全部 finite。随后已证明 x2/weight 为
`addresses[1]`，并完成 x1+x2 无损门禁及单次双极值 E2E。双极值将真实 QMM BF16 output
放大到 `max_abs=872`，但仍全部 finite，因此不构成 accumulator overflow 证据。
当前还证明了真实 QMM 的 `x2_scale` 为 formal ordinal 3、compact Run slot
`addresses[2]`、连续 FP32 `[5120]` storage。无损 temporary substitution 与模型响应/token
精确等价；有限 `×2` 令 5,120 个 scale 全部改变，并使生成 token 从 0-based index 4 开始
分叉，但真实 QMM BF16 output 仍全部 finite。该结论命名为 scale amplification，不命名为
accumulator overflow。

Fault20 v17 的最终 msprobe 交接版本增加了显式 metadata ABI descriptor 和 fail-closed
握手，并将 preload、control、proxy、真实 `libopapi`、manifest、校验器与参数化 rootfs
launcher 固定为只读 bundle。远端 bundle 是
`/home/haoran/fault20_fast_validation/runtime/cann851_semantic_slot_bundle_v17_hs2`，模型级
证据是 `/home/haoran/f20runs/v17h2`。交接前必须运行 bundle 内
`harness/verify_v17_bundle.sh` 并得到 `FAULT20_V17_BUNDLE_VERIFY_PASS`；不得把其中任一
`.so` 与旧版本混用。详细 hash、设备映射、C1/C2/C3 和 Dump-ON 交接步骤见
[CANN 8.5.1 使用说明](FAULT20_CANN851_USAGE.md)。

## 构建

必须在与推理服务相同的 vLLM-Ascend/CANN 镜像内构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/l4_allhit_v1
bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh

export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/l4_metadata_lifecycle_proxy
bash /fault20/source/l4_metadata_lifecycle_proxy/build.sh
```

CANN 8.5.1 独立构建必须显式锁定 toolkit 与真实库路径片段，并为每次构建使用新的
`OUTPUT_DIR`。构建只写入该目录内同名二进制，不会清理其他 runtime、logs 或 evidence：

```bash
export FAULT20_BUFFER_OUTPUT_DIR=/fault20/runtime/cann851_validation_v1
export FAULT20_ASCEND_TOOLKIT=/usr/local/Ascend/cann-8.5.1/aarch64-linux
export FAULT20_EXPECTED_LIBRARY_PATH_FRAGMENT=/cann-8.5.1/
bash /fault20/source/l4_buffer_substitute_allhit_v1/build.sh

export FAULT20_METADATA_PROXY_OUTPUT_DIR=/fault20/runtime/cann851_metadata_proxy_v17
export FAULT20_REAL_LIBOPAPI=/usr/local/Ascend/cann-8.5.1/opp/built-in/op_impl/ai_core/tbe/op_api/lib/linux/aarch64/libopapi.so
bash /fault20/source/l4_metadata_lifecycle_proxy/build.sh
```

runtime 构建脚本会检查 ABI、SONAME、目标导出符号、意外 CANN `DT_NEEDED` 依赖和 `LD_BIND_NOW` 加载。离线穷举测试及其结果属于本地验证资产，不在公开仓库中。

主要产物：

```text
/fault20/runtime/l4_allhit_v1/libfault20_buffer_substitute_abi0.so
/fault20/runtime/l4_allhit_v1/fault20_buffer_ctl
/fault20/runtime/l4_metadata_lifecycle_proxy/libopapi.so
```

## 容器参考

```bash
bash /fault20/source/l4_buffer_substitute_allhit_v1/create_allhit_container_v2.sh
```

该脚本包含实验机专属的模型路径、NPU 设备 2/5、端口 18005 和挂载路径，只应作为已验证配置参考；迁移前必须调整路径、设备和版本门禁。

关键加载顺序：

```bash
export LD_LIBRARY_PATH=/fault20/runtime/l4_metadata_lifecycle_proxy:$LD_LIBRARY_PATH
export LD_PRELOAD=/fault20/runtime/l4_allhit_v1/libfault20_buffer_substitute_abi0.so
```

control v17：

```text
magic   0x4632304255463137
version 17
SHM     /fault20_l4_buffer_substitute_ctl_v17
```

## 命中范围

v17 保留历史 `arm-int4` / `arm-int2` 的 arm-lifetime bounded 语义：`MAX_HITS`
是一次 arm 生命周期内的累计预算，候选窗口也跨请求累计。该行为用于可复现实验，未改变。

v17 继续支持 `arm-int4-observe` / `arm-int2-observe` 组合模式：先执行 QMM x1
数值退化，再在同一次 `OpExecCache::Run` 后同步并扫描真实 QMM output，输出
`qmm_output_observation` 事件。这样 Fault20 的 INT4/INT2 路径可以获得与 Fault39
cancellation 类似的 “fault + post-Run observe” 证据链。

在 CANN 8.5.1 的 `aclnnQuantMatmulWeightNz` 路径上，`OpExecCache::Run`
的 `addresses[]` 可能只包含输入/scale/bias 等 storage，不包含 output storage。v17 继续把
GWS 阶段捕获到的 formal output metadata 复制到 legacy metadata 头部；post-Run observe
优先使用已证明的 `addresses[out]`，否则使用 GWS formal `out.data_addr` 做只读 D2H
扫描。若两者都不可用则 fail-closed。

v17 新增 QMM x1 的 proof policy，但不改变历史命令语义：

```text
arm-int4
  -> proof_mode=STRICT_PRODUCER
  -> 仍要求前驱 AscendQuantV3 output 与 QMM Run slot 0 地址一致

arm-int4-slotproof
  -> proof_mode=SEMANTIC_SLOT
  -> 不要求前驱 producer 地址守卫
  -> 必须证明 formal tensor ordinal=0、compact Run slot=0、参数顺序/地址数/空槽规则精确、无共享 storage、x1 storage 唯一
```

两种策略都使用同一个 `ResolveSemanticTarget(QMM_X1)` 和同一个 INT4 transform/substitution 核心。候选计数只在目标 shape/dtype/layout/bytes 与所选 proof policy 全部通过后递增。GWS metadata 中的 `data_addr/storage_addr` 与 Run address 的直接匹配只写入日志作诊断，不能单独作为注入依据。未知 proof mode、槽位歧义、共享 storage、参数顺序或地址数量不一致均 fail-closed。

新增的 request-scoped 模式是独立功能，适用于本地开发者模式中的重复在线实验：

```text
arm-int[24]-request
  -> 每请求最多 MAX_HITS_PER_REQUEST 次
  -> 所有请求累计最多 MAX_HITS_TOTAL 次；到达后自动 disarm
request-begin REQUEST_ID
  -> 用单调 request id 开启下一请求并清空请求内 candidate/hit 计数
disarm
  -> 立即停止后续请求的注入
```

`request-begin` 必须在同一 Docker IPC namespace 中、且在目标 HTTP 请求进入模型前调用。
同一个 request id 可幂等重复提交；较小 id 被拒绝。TP worker 不得各自递增 id：控制端应为
同一 HTTP 请求写入同一个单调 id。native `OpExecCache::Run` 本身无法推断 HTTP 请求边界，
因此自动化接入需由 worker/controller 在请求边界调用该命令或等价控制 RPC。

## 控制命令

默认必须保持 disabled。所有 `arm-*` 命令及实际 runtime substitution 都要求进程环境中
`VLLM_SERVER_DEV_MODE=1`；否则 fail closed。示例中的 device mask `3` 表示已验证的两个逻辑设备 0/1。

```bash
fault20_buffer_ctl status
fault20_buffer_ctl reset
fault20_buffer_ctl disarm

# 历史兼容：STRICT_PRODUCER，参数语义不变
fault20_buffer_ctl arm-int4 0 1 5120 1 1 5120

# 新模式：SEMANTIC_SLOT；只在完整 formal-slot proof 通过后命中
fault20_buffer_ctl arm-int4-slotproof 0 1 5120 1 1 5120

# QMM all-hit
fault20_buffer_ctl arm-all-qmm-lossless 3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int2     3 1 5120 5120
fault20_buffer_ctl arm-all-qmm-int4     3 1 5120 5120

# 可重复在线 QMM 数值退化：每个请求 3 hit，所有请求共最多 96 hit。
# 首次 request-begin 后才会命中；REQUEST_ID 必须单调递增。
fault20_buffer_ctl arm-int4-request 0 1 5120 1 3 96 5120
fault20_buffer_ctl request-begin 1001
# 发送一次 HTTP 推理请求
fault20_buffer_ctl request-begin 1002
# 发送下一次 HTTP 推理请求；本请求的 candidate/hit 重新从 1 开始
fault20_buffer_ctl disarm

# QMM INT4 + post-Run output observe；最后一个 MAX_OUTPUT_BYTES 可省略，默认 65536。
fault20_buffer_ctl arm-int4-observe 0 1 5120 1 1 5120 65536

# request-scoped 版本：每请求最多 3 hit，所有请求共最多 96 hit，并记录每次 output stats。
fault20_buffer_ctl arm-int4-observe-request 0 1 5120 1 3 96 5120 65536
fault20_buffer_ctl request-begin 2001

# QMM extreme x1 bounded single-hit
# DEVICE M K CALL_INDEX MAX_BYTES VALUE RATIO_BPS PATTERN
fault20_buffer_ctl arm-extreme 0 1 5120 1 5120 127 100 positive_max

# QMM extreme x1 progressive（2..64 hits）
# DEVICE M K CALL_INDEX MAX_HITS MAX_BYTES VALUE RATIO_BPS PATTERN
fault20_buffer_ctl arm-progressive-extreme 0 1 5120 1 16 5120 127 10000 positive_max

# QMM extreme x1 all-hit；mask=1 只命中 rank0/device0
# DEVICE_MASK M K MAX_BYTES VALUE RATIO_BPS PATTERN
fault20_buffer_ctl arm-all-qmm-extreme 1 1 5120 5120 127 10000 positive_max

# QMM x2/weight 只读槽位观察
# DEVICE M K CALL_INDEX MAX_WEIGHT_BYTES
fault20_buffer_ctl arm-observe-x2 0 1 5120 1 268435456

# QMM x1+x2 双 temporary buffer；已验证 x2=[5120,5120]、26,214,400 bytes
# DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES
fault20_buffer_ctl arm-lossless-x1-x2 0 1 5120 5120 1 5120 26214400

# DEVICE M K X2_N CALL_INDEX X1_BYTES WEIGHT_BYTES VALUE PATTERN
fault20_buffer_ctl arm-extreme-x1-x2 0 1 5120 5120 1 5120 26214400 127 positive_max

# QMM x2_scale 只读证明、无损替换与有限倍率放大
# DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES
fault20_buffer_ctl arm-observe-x2-scale 0 1 5120 5120 1 20480
fault20_buffer_ctl arm-lossless-x2-scale 0 1 5120 5120 1 20480

# DEVICE M K SCALE_ELEMENTS CALL_INDEX MAX_SCALE_BYTES FACTOR
# FACTOR 仅允许 2、10、100、1000；已完成模型 E2E 的档位为 2
fault20_buffer_ctl arm-amplify-x2-scale 0 1 5120 5120 1 20480 2

# aclnnMatmul 精确 decode projection：
# self=[1,12800]、mat2=[12800,5120]、out=[1,5120]
# 仅支持 device/rank0 single-hit 的已验证路径
fault20_buffer_ctl arm-float-matmul-observe 0 1
fault20_buffer_ctl arm-float-matmul-lossless self 0 1
fault20_buffer_ctl arm-float-matmul-lossless mat2 0 1
fault20_buffer_ctl arm-float-matmul-lossless dual 0 1

# 有限 BF16 双输入幅值；EXPONENT 仅允许 48、56、57、58、59
# 首个 non-finite 会自动 disarm；已验证在 2^58 首次得到全 +Inf
fault20_buffer_ctl arm-float-matmul-finite 0 1 58

# A=2^58 正负抵消；每个 pattern 都保证 6400 正项和 6400 负项
# 仅支持 device/rank0、exact decode shape、single-hit
fault20_buffer_ctl arm-float-matmul-cancel half        0 1
fault20_buffer_ctl arm-float-matmul-cancel block256   0 1
fault20_buffer_ctl arm-float-matmul-cancel block512   0 1
fault20_buffer_ctl arm-float-matmul-cancel block1024  0 1
fault20_buffer_ctl arm-float-matmul-cancel alternating 0 1

# 真实 MatMul 正常执行后写回原 output storage；COUNT 仅允许 1/5/51/512
fault20_buffer_ctl arm-float-matmul-output-proxy pos_inf 0 1 1
fault20_buffer_ctl arm-float-matmul-output-proxy neg_inf 0 1 5
fault20_buffer_ctl arm-float-matmul-output-proxy qnan    0 1 51

# MatmulWeightNz all-hit
fault20_buffer_ctl arm-all-matmul-lossless 3 1 12800 25600
fault20_buffer_ctl arm-all-matmul-mantissa 3 1 12800 25600 bf16 3
```

完整 bounded/multi-rank 命令见 `fault20_buffer_ctl` usage 和[能力矩阵](fault20_fast_validation/reports/fault20_capability_matrix_final.md)。

控制器必须在 worker 所在 Docker IPC namespace 内执行，例如
`docker exec -e VLLM_SERVER_DEV_MODE=1 CONTAINER fault20_buffer_ctl ...`；宿主机与容器内的
同名 POSIX SHM 不是同一个控制块。progressive/all-hit 每次命中后扫描真实 QMM output，
首次发现 non-finite 或 observer/runtime 失败即自动 disarm。

## Fail-closed 边界

以下条件任一失败时，runtime 应原样调用真实 Run：

- exact API/cache/adapter 不匹配；
- `VLLM_SERVER_DEV_MODE` 不等于 `1`；
- device/dtype/format/shape/stride/offset/bytes 不匹配；
- producer address 或 Matmul storage guard 失败；
- ABI/runtime API/分配/拷贝/同步条件不满足；
- primitive 参数非法、元素未实际改变或选择条件不明确。

禁止依赖 private executor layout、长期缓存 activation 地址、创建新 `aclTensor` 填回 cache，或原地修改调用方 address vector。

## 文档

- [当前注入能力统计](fault20_fast_validation/reports/fault20_injection_capability_inventory_20260807.md)
- [最终能力矩阵](fault20_fast_validation/reports/fault20_capability_matrix_final.md)
- [最终架构](fault20_fast_validation/reports/fault20_final_architecture.md)
- [Primitive 设计](fault20_fast_validation/reports/fault20_fault_primitive_design.md)
- [CANN 8.5.1 使用说明](FAULT20_CANN851_USAGE.md)
- [Fault39 QMM 验证报告](fault20_fast_validation/reports/fault39_qmm_overflow_validation.md)
- [Fault39 Floating MatMul 路径清点](fault20_fast_validation/reports/fault39_float_matmul_path_inventory.md)
- [Fault39 Floating MatMul 有限输入到 Non-finite 输出验证](fault20_fast_validation/reports/fault39_float_matmul_finite_nonfinite_validation.md)
- [Fault39 Floating MatMul 正负抵消与中间数值范围验证](fault20_fast_validation/reports/fault39_float_matmul_cancellation_validation.md)
- [Fault39 Floating MatMul Output Inf/NaN Proxy 验证](fault20_fast_validation/reports/fault39_float_matmul_output_nonfinite_proxy_validation.md)
- [Fault39 A2 MatMul 故障复现总结](fault20_fast_validation/reports/fault39_matmul_fault_reproduction_summary.md)
- [Fault39 A3 验证交接](fault20_fast_validation/reports/fault39_a3_validation_handoff.md)
- [Fault26 allocation topology 与 single-hit underflow](fault20_fast_validation/reports/fault26_writer_allocation_topology_and_qmm_underflow_20260818.md)
- [Fault26 bounded multi-hit](fault20_fast_validation/reports/fault26_qmm_underflow_bounded_multihit_20260818.md)
- [Fault26 all-eligible-hit 与模型响应分叉](fault20_fast_validation/reports/fault26_qmm_underflow_all_eligible_20260818.md)

## 当前不支持

- 任意 ACLNN 算子自动适配；
- FP16 模型级 E2E；
- packed INT2/INT4 或 kernel/指令/硬件故障；
- prefill 大 shape、warmup/cold、non-repeat-cache；
- 并发、soak、异步 buffer pool、性能 SLA；
- 跨 CANN/torch_npu/vLLM 版本和其他架构；
- MoE、GroupedMatmul、MatmulAllReduce；
- QMM post-Run output Inf/NaN proxy（当前只完成了 floating `aclnnMatmul`）；
- 外部故障框架集成。

2026-11-21持续任务进展：[逐项审视](docs/worker_core_v1/continuous_task_audit_20261121.md)。Batch torch已在0.26/A2/TP2的Qwen3-0.6B与Qwen3.8完成B8、prefill+3decode、同组Worker两作用域真实site验收；采用CPU-FP32参考重算并交给原NPU QMM，详见[实机报告](docs/worker_core_v1/continuous_batch_real_model_20261121.md)。公共Runtime准入尚未完成，不继承为A3/现役通过。


本开发分支新增 [Fault15 B档产品验收](docs/worker_core_v1/alias15_product_20261123/README.md)：A2/0.26/Qwen3.8 W8A8/TP2，ND视图别名断开，不含私有格式触发条件；独立于冻结r7。A3须先审查现役普查，当前仍拒绝。
