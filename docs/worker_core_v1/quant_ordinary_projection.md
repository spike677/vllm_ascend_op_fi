# 普通 Qwen W8A8 的 batch_shared_scale 入口

当前是可供主项目接线的 Worker 适配实现。**Source118已完成该新入口的六阶段NPU效果/OFF及独立复核；黄区实际主项目接线仍未验收**。
当前结论见[独立原件总结](review_quant_ordinary_source118.md)；下文首轮失败、后继修复及准备步骤按历史顺序保留。
不继承旧 source64 C1 或 routed/GMM 观察的资格。本轮不处理135全目标、GMM1、MoE/专家行、C2/C3。

范围为普通 `qwen2`/`qwen3`、真实动态 W8A8、TP1/PP1/DP1、B8、eager、同步调度、无 prefix cache、
无 chunked prefill、无 speculative/CP、首个纯 decode。模型名称不能代替 actual INT8 weight、实际
`AscendW8A8DynamicLinearMethod`、scheduler 与 Native formals 检查。Qwen MoE 和未知模型类型拒绝。
每个真实 prompt 最多1024 tokens；每个完整读回最多16 MiB，超过则拒绝，不截断比较。

## 实际接入点

新增 `serving_quant_projection.install_quant_projection_binding(worker, root=..., build_sha256=...)`。
由可信部署适配层在真实 Worker/runner/model 已存在后安装 inert binding；此动作不装 Native owner、
不执行模型。参数是已审核 Native bundle 的本地路径及真实 build.json SHA，不能由用户输入层号或地址。
原 `QuantProvider`、`WorkerFaultRuntime`、`WorkerFaultClient`、`FaultConfig('quant', 'batch_shared_scale')`
继续管理 owner、generation、attempt、动作与恢复。没有新 runner、RPC forward、Host 状态机或配置变体。
部署环境仍要求 `VLLM_ENABLE_V1_MULTIPROCESSING=0`、`TASK_QUEUE_ENABLE=0`；真实 Native 依赖必须按原
bundle 启动前装载，保留 managed C1/group DSO 互斥与 source/profile/mapped identity 检查。

主项目的正常调度调用方消费 `serving.next_phase/next_action`：

1. 通过原公共 prepare/register/check/enable 建立本次 owner。
2. 原调度器 enqueue 八个真实新 request，调用 `client.advance_batch(actual_request_ids)`。
3. 原调度器正常执行一次 prefill；调用 `client.begin_decode_capture()`。
4. 原调度器正常执行该 B8 首次 decode；调用 `client.collect_replay()`，可随时纯 query。
5. 根据下一 phase 用相同逻辑输入、全新实际 request IDs 继续调度。OFF完成后 finish/disable/unregister/detach；
   最后可复用现有 `remove_quant_binding(worker)`，它仍要求公共 owner 已释放。

实际 prompt 从该 batch 到达 Worker 后的 `runner.requests` 读取。无预置实验 prompt文件，无伪造 scheduler ID。
prefill仅提供真实 first-decode 上下文；故障只在捕获的 pure decode 发生。方法本身从不创建或回放请求。

## 为什么首次作用域仍需六个真实 batch

保留当前已审核 group Native ABI。它要求单点 Reference/Lossless/OFF 和 seal，不能通过删除 Python
实验步骤、清零 `_attempted`、复用旧 target 指针或反序列化旧证明绕过。

| batch | phase | 实际行为 |
|---|---|---|
|1|normal|原始量化；自动绑定 scheduler→embedding→首个 RMSNorm 的原对象→当前普通投影，观察真实 Q/scale/QMM|
|2|reference|该单点 Native Reference；完整 Q/scale、input、hidden、logits 与 normal 对齐|
|3|lossless|该单点独立临时输出无损路径；同样完整比较|
|4|calibration_off|Native OFF 预验证；同样完整比较，然后才允许单点 seal|
|5|fault|一次实际调用：正确域 reference helper 与共享域 effect 两个真实 Native 子调用；发布 Q 与 scale|
|6|off|已停止、真实同步、drain/recover/remove 后，原量化与同 Worker 健康请求恢复|

这是**首次当前作用域校准的显式成本**，不是声称普通运行/验收已经彻底拆分或已成为三 batch API。
一次新任务创建新 session/owner；旧 spent 对象不能再 enable。若未来要免除新 owner 的这些校准，必须
独立修改并重新验收 Native 授权协议，本实现没有做该变更。总 batch 数固定，不静默增加自选层/多次 F 重试。

## 自动实际调用绑定及效果

仅一次结构枚举寻找真实普通动态投影，不做两次全模型 census、不启动全量 row graph、metadata/MoE
profile 或 msprobe。类型与当前 callable/source/weight identity 持续核对；instance forward 替换拒绝。
重用已经使用的直接 embedding→RMSNorm 前缀证明。首个正常化输出必须是投影实际输入的同一个 Tensor，
不能因同形状/同值承认 roll、专家重排或别的 module。轻量行 observer 只从这个 norm 输出到选中 QMM
返回，最多256 Torch调用；实际 Native admission 再核当前 row prefix、thread/device、调用和 lifetime。
输入 metadata/写 epoch 变化、当前 proof 失效、不同 logical prefix、不同 input 完整SHA均拒绝。

Normal 与 OFF 读独立 Native journal 的实际 parent 和4个 GWS/execute events。校准和 F 走现有
`public_begin/invoke/public_end`、owner/generation/target/actual-row lease、严格 typed completion 验证。
F明确保留两次 Native execute（helper+effect）；没有把它们伪报为整个 F 只有一次 Native execute。
非目标原方法/API仍执行一次，Native callbacks仅 pass；不把结构候选数当合格分母。

真正的 QMM必须消费该次实际 Q、pertoken scale 与当前 projection-owned weight/scale/bias，且原 QMM
只执行一次。消费前完整Q/scale读回须仍等于 Native 发布后的值。Native当前输入不变、当前正确 reference
与 effect 比较、准确 publication、资源释放均复用原 ABI硬门槛。

效果同时要求 Native `changed_quant_bytes > 0` **且** `changed_scale_bytes > 0`，已匹配输入的完整 CPU
Q/scale 两个比较也都发生变化；仅 Q 变或仅 scale 变不能交付。若本次输入不产生双效应，保留失败并恢复，
不修改用户输入或重试 spent F。`numeric_change` 复用公共函数，从已经取得的 CPU readback输出完整变化
元素数、有限最大差与至多4个样例；无 tensor 文件、全量dump或隐式统计实验。

`fault_effect`明确指向唯一已完成故障 batch；`current_request_effect`在 OFF/普通 batch 为 null，
不会把旧 F 的效果冒充当前 OFF。E5仍是 `NOT_ASSESSED_NO_STATISTICS_RUN`，不用于阻止安全功能闭环。

## 失败与恢复

首次错误不被后续 hook/collect错误覆盖。Native同步、未知pending、原接口身份或 hook所有权不确定时保留
owner/资源，不晚回写失效公开存储。F已尝试但尚无健康 OFF 时不能直接 unregister。

现有公共 `recover()` 可在真实 Native stop/drain/recover/remove成功后保留同一 owner和观察hooks，进入
`recovery_only`。它只允许一个实际 OFF batch，仍用正常 scheduler/advance/capture/collect；原错误、失败
request IDs和已消耗 F 保留。真实 OFF确认后可释放 owner，但失败功能不会因此改成通过。若 drain失败，
不能进入 recovery_only；重试只做恢复，不重复 F。

## CPU复现与实机 handoff

在仓库根执行（本地 Python路径按部署环境选择）：

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension'
$env:PYTHONDONTWRITEBYTECODE='1'
python -B -X utf8 -m unittest test_quant_projection_runtime.OrdinaryQuantTests -q
```

这些合同跑真实 CPU Torch module/API hooks、公共 Host/Runtime 和显式非CANN Native doubles，验证调度/身份/
故障/恢复协议，不作为 CANN数值、真实硬件或主项目资格。受影响旧 `test_quant_group_c1.QuantGroupC1Tests`
也应保留通过；Native源、ABI、数学路径及既有构建均未修改。

唯一下一实机由 root调度：经正式工具生成并记录config/weights身份的小Qwen动态W8A8，原真实 bundle的新
安装包，TP1/card映射、B8同步纯decode，六个相同逻辑输入且实际IDs全新的正常 batch。只用当前公共 Host
入口和真实调度，不拿旧七阶段实验或独立HF runner代替；记录实际 Native、双输出效果、同Worker OFF、
清理与普通健康请求。Native bundle须通过当前 source/profile一致性检查，不能仅按旧成功标签引用。
若模型、调度、直接前缀或 Native formals不匹配，原样保存拒因并停止；不得补 public 值到 Native descriptor。

Root导出闭包需增加 `serving_quant_projection.py`；其依赖（现有serving/row proof、Native binding、
`deployments.quant_group_c1`/`quant_group_inventory`及公共 `effect_summary.py`）已经属于当前源树。
本提交不编辑公共 registry/exporter/Host/Runtime，主项目接线无法访问的限制仍明确保留。

## 独立蓝区短验收 driver

`experiments/worker_core_v11_20261118/probe_quant_ordinary.py` 只用于 root 已有124队列的新入口验收，
**不是主项目接线，也不授 C1/C2/C3 或 E5**。它经标准 `DemoncatWorkerExtension`、上述 inert binding、
公共 Host/Runtime 和实际 enqueue/prefill/decode 调度完成六阶段；前后各一个普通健康 B8，共八批。
每批相同八个普通 prompt、两个生成 token、全新实际 request IDs；单点仅首个 decode。正常推理不在 RPC 内执行。

输入固定为 root 用官方 ModelSlim `3b3f5151521f0b2a0c1f29937fc735ba4938db43` CPU生成、再做精确
loader元数据兼容的独立 a5 目录：`/home/17-fault-c2c3-20261117/product-qwen06-w8a8-cpu-a5/model`。
原 a4 与其失败原件保留。a5仅将 `config.quantization_config` 改为实际独立description的完整字典；
其余原config字段不变，权重、独立description和其他文件字节不变。本次实际唯一权重文件名为
`quant_model_weight_w8a8_dynamic.safetensors`，没有 index；描述文件为
`quant_model_description_w8a8_dynamic.json`。生成器核实际 `W8A8_DYNAMIC` 描述、BF16普通 Qwen配置，
并按完整canonical JSON核验inline字典与该description一致，含所有字段/嵌套值/类型；不只比较若干选项。
记录所有顶层文件完整SHA与单文件真实loader选择；不改文件名、不改公共模型hash helper、不修改模型。
这里的模型身份记录本身不证明 vLLM 已成功加载或执行这个量化产物。

source116/a4首次实际尝试在 `LLM_CONSTRUCTION` 拒绝 `KeyError('model.embed_tokens.weight')`，
`result.json` SHA `9fb2116063bc40ad6314e4059cf730e9171e20fac34f9097738834768699b66b`。
当时 phases为空，无Host owner，已有Native只读诊断的GWS/execute/admission/resources均0。
固定image的真实 `weight_utils.py`（SHA `e7c4934a6fea0f6a3a5116d88221f24904bc34106890ac6e524051886239f46d`）
L267–305把非空HF `quantization_config` 直接传入 `from_config`；真实 `modelslim_config.py`（SHA
`cebb76def24d7da0de8a9ced18a00b07d875119c8000b4ae66b2998cdcdd12d4`）L364–366原样保存该字典，
L603–606因非空跳过独立文件加载，L292随后索引逐层weight键。旧 `w_bit/a_bit/is_dynamic` 生成选项
不是这个loader的逐层description契约。默认查找文件名实际为 `quant_model_description.json` (L48)。
因此本修复要求完整内联**已有**description；不补猜测层键、不改loader或权重、不声称NPU已通过。
取证来自固定镜像的自有 `docker create/cp/rm`、从未start或导入框架，原件在124
`product-quant-loader-source-a1`；controller SHA
`76f6b8272d8a7dfdc04c38ab749be942df7d31744ce9cfc4b2f3684345430c47`。a4失败原件未覆盖。

复用已审核 bundle `/home/17-fault-c2c3-20261117/native-quant-group-nd-flat-a2`，其 `build.json` SHA 为
`b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`。
生成器重新核全部 bundle 文件、3份生产 Native 源、实际SDK metadata ABI和固定镜像身份；Native编译commit
与新的 Worker/source commit分别记录。任何字节不匹配都拒绝复用，不因旧通过记录省略验证。
本增量不改 Native源、ABI或数学路径，不需要新编译。

Root在124冻结包含新模块/两个实验脚本的完整source和安装包后，以实际路径设置 `SOURCE`、`INSTALLED`、
`SOURCE_COMMIT`（完整40位），执行以下准备命令。准备器只校验本地输入并写case JSON，不启动容器或NPU：

```sh
R=/home/17-fault-c2c3-20261117
python3 "$SOURCE/experiments/worker_core_v11_20261118/prepare_quant_ordinary_case.py" \
  --source-root "$SOURCE" --source-commit "$SOURCE_COMMIT" --installed "$INSTALLED" \
  --quant "$R/native-quant-group-nd-flat-a2" \
  --expected-build-sha256 b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac \
  --model "$R/product-qwen06-w8a8-cpu-a5/model" --card 5 \
  --out "$R/product-quant-ordinary-a1-cases.json"
```

case固定实际镜像 `sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`，
沿原量化 `ASCEND_CUSTOM_OPP_PATH`/stock custom preload顺序，无 late load，无新增trace journal。
仅 `/evidence` 由现队列写入；model/source/installed/Native/probe均只读。case上限900秒、32GiB、8CPU；
从probe import前的开始时刻计660秒工作截止、840秒最终截止，F前至少保留180秒。
这只是预留和拒绝门槛，不是对未来耗时的保证；真实超时、未完成OFF或清理保留失败，不称容器删除为恢复。

实际启动只允许 root 经现有live资源门禁（需要的fresh allocation scope由root生成，不能拷贝旧scope）：

```sh
python3 "$SOURCE/experiments/c2c3_closure_20261117/run_queue_124.py" \
  --cases "$R/product-quant-ordinary-a1-cases.json" --attempt product-quant-ordinary-a1 --cards 5
```

每phase的原件独占写入 `phase-<name>.json` 并保存完整SHA，`result.json`保存实际model前后身份、
8条请求对应关系、逻辑输入、双效果、同Worker/OFF和清理；`host-core-receipts.json`保留公共动作原件。
实际enqueue后立即保存 `completed=false`，map失败或decode/collect失败不能抹去已发请求。
F后出错会先保存纯query原件，再走公共 recover；只有实际Native释放成功才能做真正OFF，随后释放owner。
错误/未完成drain/未知owner一律保留，F不重试，首错不被恢复或写盘错误改为成功。

新增harness短合同（真实CPU module hooks、Host/Runtime；明确非CANN Native double）与运行时合同：

```powershell
$env:PYTHONPATH='src;fault39/src;.;tests;tests/worker_extension'
python -B -X utf8 -m unittest test_quant_ordinary_probe test_quant_projection_runtime -q
```

停止条件：实际模型/库/方法或source身份不符、不是TP1 B8纯decode、自动前缀无法证明、Native/public formals
不符、Q与scale未同时改变、Native pending/owner不明、真实OFF/健康未恢复、预算不足、原件无法保存。
不为通过改变prompt、补Native字段、另选用户指定层、清空attempt或重复F。成功只产生这个新蓝区入口的
待审功能/OFF原件；主项目无法访问的接线缺口和C2/C3/E5未验收仍保留。

### source117 普通前缀 Native frame 修复

source117/a5实际 normal 完成，自动选择 `model.layers.0.self_attn.qkv_proj`：输入 BF16 `[8,1024]`，
输出Q为同几何INT8、scale为FP32 `[8]`；实际Native prepare/execute各一次，实际QMM消费一次。
`phase-normal.json` SHA `473488af83b13cb80add2802b77593adfe7d50607d9aa8c9bb14a11a2a223dc0`。
紧接的reference在admission前拒绝 `NATIVE_ROW_SOURCE_FRAME_CONTEXT_CHANGED`，外层为161001；
admitted/targets/F均0，公共recover/empty-owner清理完成且DETACHED，owned hooks/API/modes/资源全0。
`result.json` SHA `9663072fd76dcfa367d576770a87122c5b6a11b8f7801f17b1389654c6a0dc02`，原件保留。

根因由固定源码及真实CPU OpOverload复现确认：普通前缀没有 `SourceBoundRows`（它不需要MoE/MLA全量
source observer）；公开输入查询时外层operator/source-frame数均0，进入真实量化dispatcher后operator数1，
旧Native query仍从空source读取frame数0。该拒绝正确阻止了admission，但缺少普通dispatcher来源接线。

修复只在现 `RequestRowProvenance.__torch_dispatch__`、没有source adapter时，保留本次实际func、args、
kwargs和pending event的瞬时frame；原函数返回/异常后撤销，end要求frame数0。不回放算子、不构造虚假
source frame，也不把这个frame当任意opaque操作的行证明。原source-bound分支保持原source stack检查。
Native query仍要求同observer/forward/输入对象/写epoch、同外层stack、唯一实际已指定OpOverload和schema、
默认非写参数、对应pending event与实际mode栈。unknown、同shape换对象、非默认tensor参数、错误、epoch变化
或非pending event均拒绝且不消耗F；失败drain仍保留owner。

CPU合同先复现同一reference拒因，再跑真实Host/Runtime的六阶段与上述反例；Native仍为显式CPU double，
不会将此修复当新的NPU reference/F/OFF验收。新Worker/source包必须重新实测；Native ABI/build不变。

后续Source118已完成真实六阶段、双效果和同Worker OFF/健康恢复。
完整限定结论、各原件SHA及消费者取证限制见[独立复核](review_quant_ordinary_source118.md)。
同一原件内的pending OFF查询曾借用旧F效果；后续仅展示层修复已补CPU窗口测试，
原实测记录不重写，当前请求与历史F效果分开显示。
