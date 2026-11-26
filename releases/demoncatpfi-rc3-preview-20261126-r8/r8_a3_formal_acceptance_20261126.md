# r8 A3 正式测试执行单

负责人转交版，2026-11-26。冻结源码 `81643582d2983b574b97208ddb3530312b167f07`；标签 `demoncatpfi-rc3-preview-20261126-r8`。目标组合：A3 Ascend910_9382、v0.26.0rc1-a3、CANN9.1.0、现役 Qwen3.8 W8A8、TP2。实际版本、模型制品和设备身份必须读运行环境，不用本句补字段。

**r8 的 A3 登记表为空。本轮全部使用 PRODUCT_ACCEPTANCE_ONLY 和验收许可；不是免许可正式使用。** BLUE 18 登记不能借给 A3。通过结果由 codex 审查后进入只改登记表的 r8.1。33、公共8、V4及未列项关闭。现有黄区测试不因本单被强行终止；完成当前安全恢复后切换独立 r8 安装。

## 1. 安装、身份和启动

以下变量由部署方填入真实绝对路径，命令在容器内运行。`RELEASE` 是完整发行目录，`SOURCE` 是其中源码 tar.gz 解压后的目录，`SITE` 是新建的独立安装目录，不能覆盖 r7.4.1。Python3.12，使用镜像已有 torch/torch_npu，不联网换依赖。

```bash
cd "$RELEASE"
sha256sum -c DELIVERY-SHA256SUMS
mkdir -p "$SITE" "$SOURCE"
python -m pip install --no-index --no-deps --target "$SITE" "$RELEASE"/wheels/*.whl
tar -xzf "$RELEASE/demoncatpfi-rc3-preview-20261126-r8.tar.gz" -C "$SOURCE"
python "$SOURCE/tools/check_rc3_preview_install.py" --site "$SITE" --release "$RELEASE" > "$EVIDENCE/install-check.json"
export PYTHONPATH="$SITE:$SOURCE:$SOURCE/experiments/worker_core_v11_20261118"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

先创建 `EVIDENCE` 并确认无旧同名输出；所有命令失败即保留 stderr/traceback，不把旧文件当新结果。install-check 必须 passed=true、406个文件一致。运行时 module.__file__ 必须来自 SITE；不要让旧源码目录抢在 SITE 前。源码清单为发行目录 **source-review.json**，不是 release-manifest.json，也不是 A2 实验的 source_manifest.json：

`source_manifest_sha256 = 8568f8a4b59fbfc98060238bbd4acf526f0056d67ddd8eb2f91835dddcc311f5`。

启动参数：TP2/mp executor、eager、seed42、关闭异步调度/prefix cache/MTP/KV connector；max_model_len=1024、max_num_batched_tokens=1024、单请求一个chunk；普通 max_num_seqs=1，Batch=8。纯文本项限制 image=0/video=0；17和外部视觉项允许一张64×64图片、video=0，并用真实processor展开tokens。不能把纯文本启动参数照搬到视觉槽。

同一容器切测试项前，先关闭旧API server及其Worker，确认退出，再启动新实例。保存新API/Worker PID、启动时间、监听端口和健康请求对应关系；只看到 /health=200 不足以证明连的是新进程。**同一项的两个 owner 之间不重启 Worker、不重载模型。** 不在故障作用域内清缓存或重建模型以制造恢复。

## 2. 16项清单与四槽安排

公共项使用 `FaultConfig(fault_id, variant, implementation_version="1")`。外部项使用 `ExternalFaultConfig("external", variant)`，不能经公共 `FaultConfig.from_dict` 或公共部署JSON加载。

|序号|mechanism / profile建议名|fault_id|variant|槽|
|---|---|---|---|---|
|1|18.api_half|18|api_half|A1|
|2|18.variance_over4|18|variance_over4|A1|
|3|18.variance_under4|18|variance_under4|A1|
|4|22.wrong_row|22|wrong_row|A1|
|5|39.output_proxy|39|output_proxy|A0|
|6|39.matmul_range|39|matmul_range|A2|
|7|28.fia.denominator_replay|28|fia.denominator_replay|A2|
|8|17.spatial_padding|17|spatial_padding|A3|
|9|quant.batch_shared_scale|quant（字符串）|batch_shared_scale|A2|
|10|15.alias_detach|15|alias_detach|A1|
|11|41.tail_column|41|tail_column|A0|
|12|26.fia_destination|26|fia_destination|A0|
|13|20.buffer_substitute|20|buffer_substitute|A0|
|14|external.module_input_gain|external|module_input_gain|A3|
|15|external.dispatch_input_gain|external|dispatch_input_gain|A3|
|16|external.conv3d_output_gain|external|conv3d_output_gain|A3|

这里是13个公共命名项加3个外部示例，不是新增16种机制。18三档分别测试、分别回传；三个示例不代表黄区全部自有适配器通过。

|槽|物理卡|执行顺序|隔离|
|---|---|---|---|
|A0|0、1|39op→41→26fia→20|Native库、owner异常项单独保留完整控制记录|
|A1|2、3|18三档→22→15|每项新session；同项两个owner共享Worker|
|A2|4、5|Batch→39mr→28|Batch独立B8服务；切项恢复max_num_seqs=1|
|A3|6、7|17→视觉Conv3d→module input→DynamicQuant input|视觉/文本启动参数分开，不共享错误profile|

每槽独立API端口、socket、容器、输出目录；8–15卡留给B组。槽内串行，卡/内存条件不足时降低并发。保存 rank→实际芯片/逻辑设备映射，不能把槽号当设备身份。

## 3. 公共部署文件、可信bootstrap与许可

下面是18基础档的完整字段集合。其它公共项只按表替换 fault 和 profile；模型路径必须与实际 model_config.model 完全相等。非Native项两个native字段必须为null。

```json
{
  "schema": "demoncat.product.deployment.v1",
  "profile": "18.api_half",
  "fault": {"fault_id": 18, "variant": "api_half", "implementation_version": "1"},
  "model": "/actual/model/path",
  "tp": 2,
  "mode": "PRODUCT_ACCEPTANCE_ONLY",
  "source": "YELLOW_DEDICATED_INSTANCE",
  "source_manifest": "/actual/r8/release/source-review.json",
  "source_manifest_sha256": "8568f8a4b59fbfc98060238bbd4acf526f0056d67ddd8eb2f91835dddcc311f5",
  "native_library": null,
  "native_sha256": null,
  "canary": {"prompt": "Explain why the sky appears blue.", "image": null, "max_tokens": 4},
  "evidence_directory": "/actual/evidence/18-api"
}
```

Batch的canary.prompt为2–8条**互异**字符串数组，例如 `["Describe a river.","Explain a bicycle."]`；金丝雀发送逐条串行B1，正式用户请求才并发。17的image=null使用 `vision_input.deterministic_image()`，真实processor产生expanded token IDs；也可提供真实图片路径，但不能伪造图片SHA。保存实际像素SHA、image_grid和processor身份。

20/26fia/41必须填实际库绝对路径，`native_sha256`固定为本版接纳的 `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。不是任意“构建OK”的库。各rank同时证明实际映射 libascendcl.so/libruntime.so 链到CANN9.1.0；不接受9.2.1产物或仅toolkit版本文件。路径不存在、SHA不同、映射证据缺失即停止该项，保留configure traceback，不改准入。

Worker模型加载后，由黄区**可信初始化器**在各rank执行（worker、paths、canary_tokens、鉴权回调来自宿主，不从用户请求取得）：

```python
from demoncat.runtime.vllm_ascend.worker_extension.product_bootstrap import configure_product_bootstrap
configure_product_bootstrap(
    worker, source="YELLOW_DEDICATED_INSTANCE",
    authorize_operator=deployment_authorize_operator,
    paths=deployment_paths,
    canary_tokens=processed_canary_tokens_by_profile,
)
```

canary_tokens每个普通profile为实际token ID列表，Batch为列表的列表，视觉为真实processor展开后的列表。保持宿主鉴权，不使用示例里的恒真回调。每个Worker只初始化一次。宿主通过已有可信executor向**全部rank**调用官方RPC；以下 `ex` 是黄区会话已有executor，不是新开HTTP服务：

```python
configured = ex.collective_rpc("demoncat_product_bootstrap", args=(
    "configure", {"operator": OPERATOR}))
# 检查每rank required_permits 与所测mechanism、source、清单SHA一致。
permit = ex.collective_rpc("demoncat_product_bootstrap", args=("permit", {
    "operator": OPERATOR, "mechanisms": [MECHANISM],
    "attempt": ATTEMPT,  # 每项/轮次唯一，1..128字符；不是owner
    "source_manifest_sha256": "8568f8a4b59fbfc98060238bbd4acf526f0056d67ddd8eb2f91835dddcc311f5"
}))
# attach/enable/status/drain/disable/unregister/detach，两个新owner，共用同一Worker组。
removed = ex.collective_rpc("demoncat_product_bootstrap", args=(
    "remove_permit", {"operator": OPERATOR}))
```

许可返回 `acceptance_granted=false` 是正确的：允许验收，不授予产品资格。全部rank许可安装成功后才能enable；部分rank失败则撤销已安装许可、核对资源，不能继续。完成两个owner后先drain、disable/unregister、detach，资源归零后再remove_permit。remove返回PERMIT_REMOVED外，还需各rank原件证明许可不存在、lease/owner/binding/wrapper/inflight为0。`recovery_required=true` 时停止新流量，保留当前owner及控制证据；不盲删包装、不强行换owner。不能通过重启把失败标成精确恢复。

## 4. 三个外部示例的接线

源码和依赖已在源码包及发行目录examples中，使用本版文件，先核对source-review中逐文件SHA。逐rank导入、核对 `factory_sha256(factory)`（它是工厂所在模块文件SHA），保留依赖文件SHA。用黄区既有可信Worker初始化路径：

```python
from demoncat.runtime.vllm_ascend.worker_extension.external_provider import (
    ExternalSite, ExternalFaultConfig, factory_sha256, register_external_site)
from demoncat.runtime.vllm_ascend.worker_extension.deployment_acceptance import observe_deployment
from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
    configure_deployment_acceptance, install_deployment_acceptance_permit,
    remove_deployment_acceptance_permit, authority_for)
from demoncat.runtime.vllm_ascend.worker_extension.product_binding import configure_product_profiles
from experiments.worker_core_v11_20261118.external_input_examples import ExternalModuleInputSite
from experiments.worker_core_v11_20261118.external_quant_argument_example import ExternalQuantArgumentSite
from experiments.worker_core_v11_20261118.external_vision_example import ExternalVisionGainSite

# 部署选择其中一行；先审核文件SHA，再确认实际设备/软件/模型身份。
choices = {
    "external.module_input_gain": (ExternalModuleInputSite, "model"),
    "external.dispatch_input_gain": (ExternalQuantArgumentSite, "model"),
    "external.conv3d_output_gain": (ExternalVisionGainSite, "vision"),
}
factory, stage = choices[MECHANISM]
authority = authority_for(worker)
if authority is None:
    authority = configure_deployment_acceptance(worker,
        source="YELLOW_DEDICATED_INSTANCE", authorize_operator=deployment_authorize_operator)
spec = ExternalSite(MECHANISM, factory, factory_sha256(factory),
    observe_deployment(worker, worker.model_runner.model), stage=stage, region_kind="transient")
register_external_site(worker, spec, authority=authority, operator=OPERATOR)
configure_product_profiles(worker, profiles={MECHANISM: {
    "mechanism": MECHANISM, "mode": "PRODUCT_ACCEPTANCE_ONLY",
    "canary_prompts": [processed_canary_token_ids]}}, authority=authority, operator=OPERATOR)
install_deployment_acceptance_permit(worker, authority=authority, operator=OPERATOR,
    mechanisms=[MECHANISM], attempt=ATTEMPT, source_manifest_sha256=SOURCE_MANIFEST_SHA)
# 宿主ProductClient用 ExternalFaultConfig("external", MECHANISM.split(".",1)[1])。
# 两owner全部安全detach之后，各rank执行：
remove_deployment_acceptance_permit(worker, authority=authority, operator=OPERATOR)
```

上述是部署初始化契约，不是整段一次执行：remove在正式测试结束后执行。外部profile不能通过公共load_product_deployment加载；其部署侧配置另存 mechanism、factory文件/SHA及依赖SHA、stage、region_kind、完整observe_deployment、canary tokens/image、source清单SHA、mode、证据目录。不要扩展公共JSON字段。黄区CLI映射由黄区加入，命名变体保持表中值；用户不能传Python类、裸地址、rank或layer。重复owner使用同一已注册site，不重复register造成冲突。v1拒绝持久权重和KV区域。

## 5. 实际CLI与效果工具

入口仍为黄区 `python -m demoncat.cli`；CLI/helper不创建socket。由宿主建立可信session，`capabilities()`和product_contract决定可执行性，旧静态目录不授予资格。

```bash
python -m demoncat.cli fault list --socket "$SOCKET"
python -m demoncat.cli fault enable --socket "$SOCKET" --fault-id "$MECHANISM"
python -m demoncat.cli fault status --socket "$SOCKET"
python -m demoncat.cli fault disable --socket "$SOCKET"
```

正式使用下面效果工具时，以上操作由黄区已有可信effect adapter/session执行，**不要先手动enable再启动工具**。它必须走黄区CLI或保留等价CLI原件；只用ProductClient直连不能声称黄区CLI验收通过。`new_session`给同一Worker组创建新owner；既有 `tools.product_effect_adapter.ProductAdapter` 可复用证据读取，黄区保留本地CLI/HTTP接线，不修改公共运行时。

每项建立独立effect配置，包含实际宿主配置以及：

```json
{"prompts": ["Name a city with a river."], "temperature": 0, "max_tokens": 4}
```

工具固定seed42、logprobs5。每个owner自动执行 baseline→enable→fault→status→disable和金丝雀→off→detach；两个owner共六轮，不需要另跑旧验收矩阵。正式请求和金丝雀prompt固定保存原文/token IDs/SHA。Batch的prompts至少2条、最多8条不同文本，HTTP sender并发发出；其他项每次只发一个真实请求。视觉sender走chat真实图文请求，转换sampling为 `logprobs=true, top_logprobs=5`；保存实际图片像素SHA及输出，不能返回合成logprob。

```bash
# ADAPTER是黄区已审查的本地接线文件；ADAPTER_SHA是其完整SHA。
python "$SOURCE/tools/product_effect_probe.py" \
  --adapter "$ADAPTER" --adapter-sha256 "$ADAPTER_SHA" \
  --config "$EFFECT_CONFIG" --output "$EVIDENCE/effect-original.json" \
  > "$EVIDENCE/effect-page.json" 2> "$EVIDENCE/effect.stderr"
python "$SOURCE/tools/product_effect_probe.py" --verify "$EVIDENCE/effect-page.json"
# 从原件重新提取，不重复推理；应与原page逐字节一致。
python "$SOURCE/tools/product_effect_probe.py" --summarize "$EVIDENCE/effect-original.json" > "$EVIDENCE/effect-page-recheck.json"
cmp "$EVIDENCE/effect-page.json" "$EVIDENCE/effect-page-recheck.json"
sha256sum "$EVIDENCE/effect-original.json" "$EVIDENCE/effect-page.json" "$EVIDENCE/install-check.json"
wc -c "$EVIDENCE/effect-original.json" "$EVIDENCE/effect-page.json"
```

API top-5 logprob不是完整logits。完整hidden/logits金丝雀必须从Worker原件另证；缺少时留null，不推算。故障导致API缺失/非有限logprob时，工具会拒绝，原件和stderr全部保留回传失败，不修改工具强行生成通过页。token或文本改变不作为门禁，耗时只记录；changed<applied不是失败。

## 6. 特殊要求

- **Batch**：max_num_seqs=8；两个以上真实请求同decode批次才eligible。正常/故障/off各轮都保存实际批次大小、内部请求ID和唯一frame；逐元素scale精确、Q±1与真实QMM消费证据不可省略。金丝雀2–8条互异prompt必须逐条串行B1；非串行前置检查的拒绝不是注入成功。另做B1用户请求不注入负例，给出原因。enable失败要保留同次Worker traceback、CLI stdout/stderr/exit与permit/profile状态，不能套用39op的disable定性。
- **17**：无需--image，金丝雀为内置确定性64×64图；真实请求图来自sender。A3 TP2必须spawn。按真实视觉encoder/Conv3d调用统计，decode无视觉调用不补调用。不能用纯文本替代。
- **外部视觉Conv3d**：同样使用图片产品路径；证明实际Conv3d返回对象及后续消费者，输入/权重不变，改写仅在瞬态输出，字节恢复；不表示支持所有视觉模型。
- **20/26fia/41**：使用第3节精确Native库及CANN映射。20证明真实输入内容覆盖和独占消费窗口；26fia证明真实residual目的缓冲区、来源内容和消费者；41证明真实KV槽位/列/页面以及安全恢复点。后续正常写入不能被旧快照覆盖，资源为0，不能以重启代替恢复。
- **39op**：BF16 ND真实投影输出及消费者；新增Inf与上游已Inf分开。scope0完整detach/许可清理后scope1新owner；同一Worker保留。若OWNER_CHANGED，保存同次控制回包的group owner、provider owner、rank、generation和调用顺序；不同层级owner不应凭字符串相等判断。不得手工删wrapper后继续声称通过。
- **18三档/15**：三个变体各自独立双owner。15仅ND视图别名断开语义，不含私有格式触发条件；changed是调用数，changed_elements另报。

## 7. 负例清单

每槽/接线组合做一次通用负例；Batch、Native、视觉另做各自范围负例。拒绝必须CLI exit4加拒绝码、引擎存活、全部rank无残留，并有随后同引擎正对照。直接RPC HTTP200+结构化body可以接受，CLI exit5不算通过。

1. 缺验收许可；未知profile；未知fault99；均不得残留租约。capabilities阶段出现ACTUAL_CALL_NOT_YET_OBSERVED须记录，最终enable仍需明确缺许可拒绝，不能只凭前置未观测当缺许可负例。
2. 已enable再次enable：exit4且原会话不变；重复disable：幂等成功或结构化exit4，最终资源零。
3. 普通请求生成5token、超出已审查范围：用户请求正常处理但不注入并报告原因；不得为了负例改模型文件。
4. Batch非串行金丝雀拒绝后引擎活着，修正串行后同引擎正对照；B1用户请求不注入。
5. 非法Native路径/错误SHA、用户试图传native_library/rank/layer：在可信配置或解析边界拒绝，不加载未知库。错误外部factory SHA/持久区域v1同样拒绝。
6. 纯文本profile不能冒充图像；视觉请求必须匹配processor/图像范围。普通用户不能改变bootstrap鉴权回调/配置路径。
7. r8 A3 `PRODUCT_REGISTERED` 必须返回未登记拒绝；不能拿BLUE资格或临时许可声称免许可通过。

已有黄区effect adapter实现check_negative时，可执行：

```bash
python "$SOURCE/tools/product_effect_probe.py" --adapter "$ADAPTER" \
  --adapter-sha256 "$ADAPTER_SHA" --config "$EFFECT_CONFIG" \
  --negatives --output "$EVIDENCE/negative-original.json"
```

它覆盖missing_permit/unknown_name/already_enabled/repeat_disable；未知profile和各项特例仍用黄区现有驱动补齐原件，不假称该命令覆盖全部七条。

## 8. 一页回传、审查与r8.1

每项独立原样粘贴effect-page文本（≤4096字节，不截断），加 verify 输出。完整原件留黄区，只回文件名、字节数、SHA。不要使用固定r5审查器的extract_registration_record.py来审r8产品原件。

一页/关联文本必须包括：机制与命名变体、标签/源码/许可清单SHA；硬件原名和rank/逻辑设备，vLLM/vllm-ascend/torch/torch_npu/CANN实际版本，config/量化描述/全部权重索引SHA，TP/DP/PP/EP和请求/调度参数；两个owner与原件对照；每rank observed/eligible/attempted/applied/changed/consumed/restored/failed及changed_elements；真实消费者、区域恢复、资源清零、精确金丝雀、baseline/fault/off的top-5变化；负例拒绝码/exit/存活和正对照；耗时；原件/工具/提取文本SHA。image_digest未观测留null，非登记前提；其余必需项不能补默认值。实际映射库及Native身份另附只读身份文本；未观察到的字段保留缺失。

effect-page的seq只是宿主控制顺序，不是wire RPC调用ID。各rank owner/请求归属与Worker身份要交叉核对。工具页不含的标签、许可SHA、负例和耗时可另贴短关联段，以原件SHA/owner关联，不手改page。常规product_http原件可另提取：

```bash
python "$SOURCE/tools/review_product_mode.py" "$PRODUCT_RECEIPT" --tp 2 --output "$EVIDENCE/product-review.json"
# Native三项在上条命令额外加 --a3-native。
python "$SOURCE/tools/extract_product_record.py" "$PRODUCT_RECEIPT" \
  --runtime-identity "$RUNTIME_IDENTITY" > "$EVIDENCE/product-pages.jsonl"
```

这组命令只接受对应product原件schema，不接受effect-original替代。原件SHA、各rank计数、消费者/恢复必须一致；failed=0，eligible=attempted=applied=consumed=restored，changed≤applied。缺字段/矛盾/未通过不登记。

回传到达当天逐份审查，给通过/不通过/需退补结论。通过的精确A3产品组合进入从本r8发行提交开出的r8.1维护分支，**只改登记表**，不夹带33/V4或机制修复；附免许可冒烟单。外部三个示例仍为受控许可路径，不能借公共免许可登记。最终主项目CLI合入、黄区自有适配器修改和台账由黄区完成；公共运行时问题交codex维护，不在黄区自行修改。
