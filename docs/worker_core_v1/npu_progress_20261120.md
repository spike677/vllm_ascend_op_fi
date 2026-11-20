# 2026-11-20 蓝区实机验收检查点

## 开发提交同步

开发线 `codex/product-integration-20261119`，本轮执行字节以清单而非旧 HEAD 名称认定。18/39/Batch/TP2-18 使用 `e85a886d` 上的审查中源码快照：`ea84055519103b91f970433c248fd47a1a639a36fae8f34e64a2c5ec88ff8da2`。新提交收录实现、独立审查器及小型总结；不提交本轮原始统计、完整张量、vendor 或权重。安装来源是已核对的源码目录，不冒称发行 wheel 安装验收。

## RC1.x / RC2.x 维护

冻结 RC1、RC1.1、RC2 的标签和包不变。未收到可登记的黄区 RC2 双作用域回传，不开 RC2.1 分支，不向黄区发送内容。RC2.2仍未冻结；多卡文件及接线只能进RC3，不能直接以当前开发 HEAD 打RC2.2。

## 逐项蓝区闭环

均为124 Ascend910B4/A2、CANN8.5.1、vLLM/vLLM-Ascend0.18、标准Worker、串行eager。下列总结通过重新读取原始请求及原生资源回读产生，不采信顶层PASS。

|能力|模型 / TP / 卡|实际执行与独立复核|边界|
|---|---|---|---|
|18 api_half|Qwen3-0.6B BF16 / TP1 / 2|同Worker两个owner，各normal/F/OFF/health；首RMS实际返回减半，14336元素变化；hidden/logits变化；全量hash精确恢复，资源0|token未变；不是全部目标覆盖|
|39 output_proxy|Qwen3-0.6B BF16 / TP1 / 5|同Worker两个owner，各四请求；实际MLP投影的一个BF16值变+Inf，SwiGLU消费；native open/apply/restore/release/live闭环；OFF/health精确，资源0|token变化；这是输出代理，不是matmul_range历史越界|
|Batch|Qwen3-0.6B W8A8 / TP1 / B8 / 5|同Worker两个owner；共12个阶段批、4个外部健康批、128请求；首decode qkv的Q 5565元素及scale 7元素变化；OFF Q/scale/hidden/logits精确，资源0|外部health只保留token和模型/Worker身份，没有完整hidden/logits健康捕获；不得称该层级也已验证|
|18 api_half|Qwen3-0.6B BF16 / TP2 / 2+5|同一组Worker两个owner，8请求/16个rank请求观测；rank0命中，rank1选中边界未变；两个rank hidden/logits均变，OFF/health精确，资源0|只验证实际Ascend use_all_gather=True分支；没有直接collective边界记录，不唯一归因某次all-reduce|

原件只留本地私有目录及远端专用attempt；公开摘要：
- [TP1 18](../../experiments/worker_core_v11_20261118/ordinary18_two_scopes_20261120_summary.json)
- [TP1 39](../../experiments/worker_core_v11_20261118/ordinary39_two_scopes_20261120_summary.json)
- [TP1 Batch](../../experiments/worker_core_v11_20261118/ordinary_batch_two_scopes_20261120_summary.json)
- [TP2 18](../../experiments/worker_core_v11_20261118/m1_rms18_tp2_20261120_summary.json)

`review_two_scope_requests.py`、`review_batch_two_scopes.py`、`review_m1_rms18.py`分别重算这些总结。TP2清理以Host历史的finish后分rank状态为准；完成health前后尚挂载的观察hook不能当作finish后泄漏，也不能略过真正清理回读。

### 失败保留与资源授权

初次TP1/M1曾因ATK容器保留设备命名空间触发驱动87/507899，在注入前失败；其他独立尝试不覆盖失败原件。旧门禁只识别`ptq_`，现补充`atk_q_`。用户授权仅停止本批Atan2在卡2/5的实例，包括重派实例；未修改Docker入口、未删除数据、未停止其他批次。原Kubernetes旧分配仍按授权豁免原样记录，不把原始idle=false改成true。

## 发行包交付

本次尚不新发RC，不修改部署准入登记。18/39的TP1双作用域补验可用于后续RC2.2收口；Batch的外部health观测范围需明确。未验收的模型/TP/硬件组合默认关闭。

## 多卡主线进度

M1的18已通过上述限定实际分支。39/22的TP2 provider和两个同Worker作用域现已实机通过，见下方追加复核。Batch/41多卡尚未完成，不能由18/22/39或TP1结果推导。M2受0.26资源/驱动门禁及权重预算限制；M3待A3专用实例回传；M4待DeepSeek实际运行证据，蓝区不做DeepSeek实测。

## 现役路径进度

本次没有新黄区材料；0.26完整流式解压总量、精确驱动配套依据未补齐，不拉取镜像或改宿主栈。A2/0.18小模型通过不代表A3、0.26、Qwen3.8或DeepSeek通过。

## 28 限时进度

B1于2026-11-20 02:26:37 UTC开始；真实8.5.1 SDK对象/CPU借用合同通过，新完整host/kernel构建成功，completion SHA `1b1ac86d6d85063ef7fa8a8f9d2ae99eae46d35560645de7bef3d4a482d4fe78`。构建约1.28GB，不占NPU。

首轮研究脚本仍经公共入口，在prepare被拒绝，未进入目标FIA观测请求。这是入口接线缺口，已改为实验目录内显式`create_research_provider`的零F驱动：仅一次observe、禁止公共owner共存、不授普通任务资格。公共Runtime/config拒绝保持不变。第二次真实正常借用在typed绑定前因研究owner不符合Native hex32合同被拒绝；失败及所有权未清理状态保留。已修复owner格式并保留绑定首个返回码，第三次以新进程/作用域验证中；完整stock/private或M128数值桥接、F/OFF以及第二作用域尚未通过。

4小时NPU、两个工作日限制不变。占卡时间须含初始化/失败/健康请求，后续按controller实际起止补记。20/26两变体/33仍为普通适配缺口，未将历史研究证明转为当前普通任务通过。


CPU验证：共享TP1/TP2调用链与设备门禁128项通过；28打包/typed/借用/研究入口与总结审查76项通过；owner/绑定返回码修复后13项专项通过。集合重叠，不累计为独立测试总数。

## 追加复核：22/39 TP2与28正常桥接

以上初始检查点保留，以下为新attempt复核，不覆盖既有失败。

|能力|组合|同Worker两个作用域结果|执行清单 SHA256|
|---|---|---|---|
|22 wrong_row|A2 / CANN8.5.1 / vLLM-Ascend0.18 / Qwen3-0.6B BF16 / TP2|8请求/16 rank请求观测；rank0第一层rotary源行4→0；rank1所选边界原值；两rank完整hidden/logits变化；OFF/health精确，资源0|c124d28dbd820394b6e19ed6906369bde64f3775a7126921608195e1073ceff4|
|39 output_proxy|同上|8请求/16 rank请求观测；目标rank单元素BF16→+Inf、真实消费者读到；guard不写；两rank组合hidden/logits与独立logits变化，OFF/health精确；native open/apply/restore/release/live全通过，资源0|ad25b9105850b927144f75cfac0023ffff05e2dff7f64564883e83569828e555|

[22独立总结](../../experiments/worker_core_v11_20261118/m1_rope22_tp2_20261120_summary.json)、[39独立总结](../../experiments/worker_core_v11_20261118/m1_output39_tp2_20261120_summary.json)。22审查直接比较B/F完整Q/K指纹，不采信guard的`same_complete_inputs_as_healthy`派生标记（原baseline分支该标记为false，实际完整Q/K相同）。对guard输入哈希和OFF哈希的内存负向修改均被审查器拒绝；原件未改。

39只保留组合hidden/logits哈希及独立logits哈希，不能单凭组合哈希宣称hidden自身一定变；没有直接collective边界归因。两项仅为实际`use_all_gather=True`路径，不代表所有多卡executor/模型。现有三项M1结果不授A3、0.26或现役资格，也不并入RC2.x。

补充的TP2统一recover路径要求：即使本rank是guard，或已经完成本地health，peer恢复时也重做实际OFF/health；CPU通过，最新源码实机矩阵正在补测。不能把早期通过清单自动等同于此后新增恢复分支的NPU验收。

### 28新构建的事实边界

A2 / CANN8.5.1 / vLLM-Ascend0.18 / Qwen3-0.6B FP16 / TP1 / TND2048。第三次正常借用经过两个typed阶段、真实child/parent执行及drain/clear，原始public对象、描述符、Slice/FIA实际launch与新ELF的关联通过独立审查。实际FIA key仍为`5000000000000200103`，不是预设M128 key。28层完整公开输入输出哈希、最终完整hidden与logits，与另一进程的stock请求精确一致。

- [正常借用总结](../../experiments/worker_core_v11_20261118/tnd28_borrow_20261120_summary.json)
- [完整stock/private数值桥接](../../experiments/worker_core_v11_20261118/tnd28_stock_private_20261120_summary.json)

stock census仍有`CURRENT_SCOPE_FUNCTION_BIND_NOT_UNIQUE`，不能以完整值相同填补该原生关联不可观测项。正常解除后的health仅token层级，未证明全hidden/logits健康。

M128控制对照连续两次在目标请求前被严格接线类型检查拒绝：第一次`TND_INPUT_EXACT_SINGLE_CAPTURE_NO_EXTERNAL_JOURNAL`，第二次`ATTENTION_REVIEWED_NATIVE_ADAPTER_REQUIRED`。两次均owner未保留、未发生目标控制FIA；初始化NPU用时仍计入预算。按同类接线问题两次止损，本轮不继续放宽检查重试。显式研究控制代码保留为**未接通、不可执行验收**；公共28入口仍关闭。

B1构建通过；B2仅正常借用及公开数值桥接通过，M128对照与stock原生关联仍缺；B3–B5（分母F/OFF及第二作用域）未通过。实际`authorize_fault`仍拒绝`TND_REPLAY_NATIVE_TWO_STAGE_ADMISSION_UNBOUND`，是实现缺口，不写成“只差测试”。本轮6次28容器尝试累计669.188377秒（含初始化、失败、stock、health）；4小时预算剩余13730.811623秒。两个工作日从2026-11-20 02:26:37 UTC实际B1开始计，未自动延长。

## 存储只读评估

受用户另行要求，独立agent完成[存储评估](storage_review_20261120.md)，已回传指定侧会话。两目录134.40GiB不等于可释放；9.82GiB只是已备份日志检查范围，绝大多数含原始证据线索，尚无确认可直接删除项。无新增删除。当前构建、vendor、source、模型及当天增量继续保护。


## 统一recover实机收尾

同一BLUE组合（A2、CANN8.5.1、vLLM-Ascend0.18、Qwen3-0.6B BF16、TP2）完成18、22、39各两个作用域，均由normal→fault→显式recover→off→health结束。每项8个真实请求、16条rank请求观测；两个rank均精确OFF/health恢复、最终next_phase=complete、资源清零并解除owner。三项使用清单`29b1658dce59090da612ca883592d0d1bf0976a1f650d2fbcb8c2548a36b12b1`，源码基线7ee7a752加已披露开发修改。

- [18恢复独立审查](../../experiments/worker_core_v11_20261118/m1_recovery_18_tp2_20261120_summary.json)
- [22恢复独立审查](../../experiments/worker_core_v11_20261118/m1_recovery_22_tp2_20261120_summary.json)
- [39恢复独立审查](../../experiments/worker_core_v11_20261118/m1_recovery_39_tp2_20261120_summary.json)

此前a2的18实测发现计划索引缺陷：恢复health已经精确，但next_phase仍为off。a3修复恢复终态索引，保留a2失败。独立审查器对旧终态off、缺rank、未恢复peer和recover后再次F均拒绝；39回执的serving是资源字段、实际请求在evidence中，曾因解析错位拒绝，修正解析并以同一原件重审通过，未改原始数据。

本轮验证显式recover调用，不宣称真实Worker崩溃或通信中断已经实机覆盖。三项跨rank传播不唯一证明all-reduce归因。开发线不授未登记部署资格，不修改RC1/RC1.1/RC2冻结标签或包。CPU最终大集合210项通过，之后恢复审查器7项和Batch健康观察/审查5项专项通过；集合有重叠，不相加。


## Batch完整health补齐

BLUE组合：A2、CANN8.5.1、vLLM-Ascend0.18、Qwen3-0.6B W8A8、TP1、B8，卡5。同一Worker两个owner作用域；每个作用域六批normal/reference/lossless/calibration_off/fault/off，加前后两个普通health批次，共16批、128请求。

新增观察器只在无故障health请求记录完整prefill/decode hidden与logits哈希，不修改输出、不安装故障hook，不保存完整张量。两作用域均自动选中首个实际qkv_proj的decode调用，Q有5565个元素变化、scale有7个变化；完整OFF量化/hidden/logits与基线相同，前后health的四项完整哈希精确一致、token一致、观察hook清零，Native与公共owner资源清零。

[独立总结](../../experiments/worker_core_v11_20261118/ordinary_batch_full_health_20261120_summary.json)以`--require-full-health`重算；旧[仅token健康报告](../../experiments/worker_core_v11_20261118/ordinary_batch_two_scopes_20261120_summary.json)保持原样，不将缺失健康张量观测追溯改为通过。新清单SHA为`401e54fa7876c90c9b640ed5c37792d8bfcef4a3e7ed89cf3ff976f28a005a17`，原Native构建SHA为`b4ac9a2b68ff58c3a5f6b54c1f052170ce5d1f6d1f8c53a6a571d175ed9328ac`。

量化入口、完整health及恢复审查CPU回归60项通过。旧CPU generate替身原先仅返回token，已改为实际CPU模型prefill/decode并检查观察器回收；不能把替身结果称为NPU验收。实机初始化的forkserver signal警告保留，模型实际加载及完整两作用域均已完成。

这些是TP1结果，Batch/41的TP2仍缺适配，20/26/33普通路线仍未闭合。0.26门禁、A3/现役组合没有新增通过。RC2.2尚未冻结，M1多卡成果不并入RC2.x；RC2.1仍仅等待黄区组合登记。

本轮按明确授权仅停止Abs指定批次的卡2/5实例，包括重派实例；临时有界清理器已结束。未改Docker入口、宿主驱动/CANN或其他任务数据。Git仅包含已审查实现、测试和总结，不提交新的statistics/mix原件、模型或完整张量；冻结RC1、RC1.1、RC2标签保持原提交。


## 晚间交付更新

RC2.1、RC2.2已冻结发布；41 TP2同Worker两作用域通过独立复核，Batch TP2因原生调用链未观测连续两次拒绝而止损；28 B3–B5仍未完成，M2镜像测量因EOF停止。具体组合、源码、资源恢复、存储删除与隔离记录见[晚间交付](evening_delivery_20261120.md)。本节取代上文“RC2.2尚未冻结/RC2.1待登记”的当前状态，保留历史记录。

## M1 Batch TP2：新 attempt 双作用域通过

源码 8c476f60f853e8a8373d7a46432f35b8e07b67a3，清单 SHA 20eb8b351342c41579eeeed6fe06ab738465ea8f0d13186ebdffce9873c02879，Native bundle 不变。A2/CANN8.5.1/vLLM-Ascend0.18/Qwen3-0.6B W8A8/TP2/B8。

无设备 dladdr 证实 stock libcust 的 DynamicQuantV2 来自其 libopapi 依赖。父进程完成原厂 platform 初始化后，在标准 spawn 创建 Worker 的窗口恢复 private-first vendor 顺序，创建完成即恢复父进程环境；未改 Worker 类或算子。新实测的 Native prepare/execute 记录证明两 rank 均命中原调用，非仅靠环境日志判通过。

同组 Worker、不同 owner 的两作用域均跑 normal/reference/lossless/calibration_off/fault/off 六批，前后各一批 health；共16批、128请求。两作用域均自动选择 model.layers.0.self_attn.qkv_proj 的实际 first-decode，rank0 Q 改变5686/8192元素、scale改变7/8；rank1原 Q/scale 不变，两 rank 完整 hidden/logits 均变化。两 token 生成序列在本输入均未改变。所有 rank 的 OFF Q/scale、hidden/logits及前后 health 精确恢复，资源清零。显式统一 recover 在F后调用并进入OFF。

独立脚本不导入 provider，不读 driver 的 independent_checks 或通过状态；核对实际请求映射、Native子调用/输入归约几何、Q/scale效果、守卫、完整返回哈希和清理字段。5种篡改回执均被拒绝。见[总结与原件SHA](../../experiments/worker_core_v11_20261118/m1_batch_tp2_20261120_summary.json)。原始回执保留在 D 盘，不提交。

### 抢占与隔离

UTC 10:44:25.053016–10:55:47.915877（北京时间18:44:25–18:55:47），窗口682.862899秒，probe内616.434005秒；这是整个测试窗口/进程墙钟，不是 kernel profiler 时间。只对卡2/5启用有界 docker-stop 清理器，未改调度器配置、Docker入口、驱动或CANN。停止卡5容器 `ptq_dcde1a8de6b13b34_list61-index-s0-01_dev5_slot0_eb2cd437`，镜像 `cann:9.1.0-910b-torch2.10.0_npu-atk26.7.8`，10:44:25.220761开始、10:44:26.832965返回0。未删除外部容器/镜像/数据。结束清理器已join；10:58:06观察到新的外部卡5实例，确认派发继续。完整操作记录保留在私有 isolation.jsonl。


## 新栈M2：18预热后TP2双作用域（15:24 UTC）

A2/910B4、vLLM0.26.0+empty、Ascend0.26.0rc1、CANN9.1.0、社区Qwen3.8-27B W8A8、TP2。源码c3d5f993，清单f4c4fb14ce6cccc7ad26ccf38aaaa00d3c66ace0b0aa625d8e9959ef7f089101。自动首个实际RMS返回为language_model.model.layers.0.input_layernorm；两个owner都只在rank0修改25600个BF16元素，rank1本地边界不变，两边完整hidden/logits变化，OFF/health精确恢复，所有hook/wrapper/lease/inflight归零，生成首token不变。

独立审查不读probe的checks或通过标志，重核两Worker身份、不同owner、真实请求/单chunk与设备完整token/position、实际消费者、恢复哈希和finish资源。基于原件改造的5个反例（OFF差异、guard写入、残留wrapper、错请求、预热不稳）全部拒绝。[原件SHA和精简总结](../../experiments/worker_core_v11_20261118/newstack_rms18_warmed_tp2_20261120.json)。不上传大回执或新统计原件。

明确限制：冷启动第一次与后续无注入前向不同；后续4次交替有/无RMS观察的完整输出相同。原因未唯一定位。因此正式实验先运行3个无注入请求，并要求后两次全量精确一致；仅接受预热后同Worker组合，不把冷启动或任意文本质量写为通过。首个未预热的18 attempt因OFF不精确失败被保留，绝不修改旧回执或降低阈值。

18容器窗口2026-11-20T15:19:34.878778Z—15:24:34.199955Z，299.321177秒。随后22加载模型被外部Docker signal15/9中断，exit137、OOMKilled=false；不计机制止损。共享窗口15:19:32—15:27:24，共472.584257秒，结束有界清理器已join。期间只停止一项明确映射卡2的外部容器，原名/镜像/绝对时刻保留isolation.jsonl；没有改Docker入口或其他卡。22使用同一源码新attempt继续。

新CANN9.1独立V2量化后端无设备编译通过，库SHA e7df992a1aec907b7f54ebc7cc8b2bdff0e2266aec289bc2e48767431e1766e5，清单04f1c8af72231aec3d98ec071882e28c6960ccd7852381c9ea65e5fa212a3aa9。实际V2调用和新Worker B8 decode仍未验收，不继承0.18/M1资格。[构建总结](../../experiments/worker_core_v11_20261118/quant_v2_cann91_build_20261120.json)。


## 22 新栈TP2独立复核与Batch原生算子验证（15:40 UTC）

A2/Ascend910B4、vLLM0.26.0+empty、vllm-ascend0.26.0rc1、torch_npu2.10.0.post4、CANN9.1.0、社区Qwen3.8-27B W8A8、TP2。源码c4119860，清单740645f768f2f7896baa7f4acae03d2db6e68b576d6980de18de519efb66a5b0。同组Worker两个owner各完成normal/fault/off/health，自动选中language_model.model.layers.3.self_attn的真实融合mRoPE。rank0末token的Q/K行变化，indexed cache变化123元素；非目标Q/K、原positions/cache和V/gate保持不变。rank1本地边界不变，但两卡完整hidden/logits均变化；OFF/health完整哈希恢复、所有拥有资源归零，生成token仍为11751（Paris）。

独立复核直接读取原始result/scope回执，核对映射、输入、实际融合消费者、效果和恢复；不信任probe通过状态或checks。原件修改的5个负例（OFF差异、guard写入、非目标Q/K变化、目标Q/K未变、残留wrapper）全部拒绝。正式验收先3次无注入请求、后两次全量精确一致，仍只覆盖预热后Worker；冷启动首请求差异根因未唯一定位，不宣称任意请求或A3通过。[独立总结](../../experiments/worker_core_v11_20261118/newstack_rope22_warmed_tp2_20261120.json)。原始回执仅留本地/124，不上传新统计原件。

a1在加载阶段被外部Docker stop/kill中断；a2正常阶段结构识别拒绝，未注入F。根因为0.26保留Qwen3NextAttention实例类，仅把forward替换成Ascend实现。c4119860同时审查原始类身份和已固定forward源码，未放松为按类名盲选；a3通过。历史失败原件不改。

本窗口2026-11-20T15:35:28.711144Z至15:40:50.395503Z，共321.684397秒；22容器15:35:31.553886Z至15:40:34.096511Z，302.542625秒。只有卡2/5有界清理器，无外部容器被本窗口停止，未改派发器/Docker入口；结束cleaner已join，isolation_removed=true。时间是墙钟窗口，不是kernel profiler时间。

随后CANN9.1新V2库实际通过B/F/OFF：Q变化112/128元素、scale变化7/8，输入不变，OFF精确一致，Native pending/execute/owned资源清零。每轮实际V2 prepare/execute各1，只有F的fault_executions=1。[算子总结](../../experiments/worker_core_v11_20261118/quant_v2_cann91_smoke_20261120.json)。该用例只有一个torch进程、logical0；调度review模板错误沿用TP2 worker/ranks字段，已在总结注明更正，原件不改。它不是QuantGroupNative六批校准后端的完整新栈验收，更不是B8模型TP2验收。

剩余：Batch的group Native构建及0.26真实B8 first-decode映射；17的视觉请求路径（不能用image/video=0文本门禁验收视觉注入）；41的真实KV布局/ScatterPaKvCache到消费者映射；39指定BF16 out_proj在社区权重不适用，需要匹配黄区制品。RC3未冻结，新组合不自动放行，冻结RC2.x不修改。


## 新栈B8无注入基线阻塞与Group后端构建（16:00 UTC）

新只读`v026_generation_observer.py`连接实际prefill→首decode，核对scheduler、内部ID、首个生成token、三轴位置、完整设备输入字节以及真实ExecuteModelState；它不授予故障资格。25项观察器CPU测试通过。A2/0.26/CANN9.1/社区Qwen3.8 W8A8/TP2/B8的两次独立启动都在无故障阶段遇到非有限输出，未执行任何Batch注入，也未完成首次decode验收。

第二次实读：prefill真实模型返回hidden_states，shape=[88,5120]，BF16，NaN=56320/450560，Inf=0。没有用预期shape或命中日志推定该结果。两次都是冷启动后的首个被观察B8请求；预热后的B8稳定性尚未验证，不能唯一归因到模型、kernel、制品、分配器或观察器。也没有完成固定同prompt的B1/B8因果对照。保留失败，不用已通过的B1预热后18/22结果代替B8。见[失败总结及日志SHA](../../experiments/worker_core_v11_20261118/newstack_b8_baseline_20261120.json)。

第一次窗口15:48:56.116639Z–15:53:27.393600Z，271.276989秒；第二次15:54:57.427257Z–15:59:43.759628Z，286.332399秒。EngineCore因观察到非有限基线失败后，由本项目主动docker stop自己的测试容器，不是外部抢占；这不构成同进程OFF恢复。两个窗口的有界清理器均退出、恢复派发，未遗留测试容器。本轮未修改Docker入口、驱动或CANN。完整隔离事件保留在总结。

普通Batch的group DSO另按CANN9.1重构建，清单05b1c1be096d5d80fd8972d38f23afbecb5138d58361ba1ef1c96c5f8b85e513。原8.5.1配方保持默认；新增9.1显式SDK选择、实际头文件/真实libopapi身份、ELF loader mock契约和真实无设备加载/ABI检查。未知版本、错SDK、默认8.5.1配置加载9.1包均拒绝。36项group构建/加载CPU回归通过；实际无设备容器load/query通过，未arm、未运行模型，NPU用时0。[构建与加载总结](../../experiments/worker_core_v11_20261118/quant_group_cann91_build_load_20261120.json)。首次准备缺少两个mock SDK头文件，补齐新源目录后重试通过，旧失败日志保留。

新group库尚未完成NPU六批校准和完整模型B/F/OFF；V2独立算子通过不替代此项。普通Batch入口仍默认旧配置，不自动迁移或放行。A3的[候选验收契约](rc3_a3_acceptance_candidate_20261120.md)已经准备，明确TP1/TP2、binding、许可、request_plan、双作用域、负向检查及效果字段；它不是已冻结候选包，也未向黄区发送。正式可信RPC、TP1候选宿主与发行安装清单仍未交付。
