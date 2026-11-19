# demoncatpfi-yellow-rc1-20261119：openchamber 接入说明

这是**黄区内源集成测试候选包**，不是全部故障完成或黄区验收通过。
黄区接入负责人：**openchamber**；蓝区不能直接连接黄区。
源码冻结提交、压缩包 SHA256、三包安装回执和逐文件清单见仓库
`releases/demoncatpfi-yellow-rc1-20261119/`。源码提交与交付物提交分别记录，
不把 Native 构建、源码版本和部署资格合并成一个版本号。
包发布后不覆盖；修复和新增能力进入 RC1.1/RC2。
RC1范围/公共Host及三条受影响路线的105项CPU合同通过；导出器30项中29通过、
1项因Windows符号链接权限跳过。实际源码包的aarch64无设备安装结果单独记录，
CPU检查不替代黄区模型请求。

## 首批可测试范围

共同蓝区实测环境：124 Ascend910B4/A2、aarch64、CANN 8.5.1、
torch/torch_npu 2.9 系列、vLLM/vLLM-Ascend 0.18 系列，固定镜像
`sha256:35f9e569dbe3dc8db90cb361886d7889489efd8fa065860c8de319e13ab025d0`。
精确 Python/依赖版本见交付安装回执；黄区须检查实际版本、源码/ABI与当前目标。
RC1 不授予所有8.x/9.x、A3、graph、TP2或其他模型兼容。

| 能力 | 已验证部署和成本 | 蓝区实际效果与恢复 | RC1分类 |
|---|---|---|---|
|39.output_proxy|Qwen3-0.6B BF16，TP1 eager，独立单目标；normal/F/OFF/health四请求|1元素有限值→Inf，实际SwiGLU消费；OFF/health完整hidden/logits恢复|首批部署适配测试；须匹配postwrite库|
|quant.batch_shared_scale|Qwen3-0.6B动态W8A8，TP1 B8纯首decode；normal/reference/lossless/calibration_off/F/off六batch|Q 5565/8192元素、scale 7/8变化，真实QMM消费；OFF精确恢复|首批部署适配测试；须匹配量化Native bundle|
|18.api_half|Qwen3-0.6B BF16，TP1 eager；normal/F/OFF/health四请求|实际RMS返回14336元素减半，hidden/logits变化，文本不变；OFF/health精确恢复|首批部署适配测试；无新增自有Native库|
|8、17、20、22、26两变体、33、39.matmul_range、41|各自原限定部署与接口保留|原报告保留，不授新普通入口资格|后续接入；普通任务拆分/宿主适配按各项状态进行|
|28 denominator|只继续真实TND路线；正常输入关联已核|当前故障准入/完整F/OFF尚未完成|实现中，默认不可开启|
|28 FA/FIA legal|旧限定范围|历史原件保留|历史保留，不进入RC1默认测试清单|
|15 alias_detach|当前部署无合格写读传播链|不构造辅助目标|不适用|
|25 reduce_scatter|无替代机制|创建入口明确拒绝|不支持|

参考：[39实测](output39_source116_single_task_review.md)、
[Batch实测](review_quant_ordinary_source118.md)、[18独审](review_rms18_ordinary_source126.md)、
[逐项矩阵](product_delivery_20261119.md)。以上“首批”只表示可以进入黄区适配，
不是当前 Worker 可直接开启。能力目录、实际部署匹配和本次动作准入分别展示。
没有“必须先完成39”关卡；由黄区当前部署匹配结果选择测试项。

## 安装与同名包边界

三个发行单元保持不变：`demoncatpfi==0.2.0`、
`demoncat-worker-extension==0.1.0`、`demoncat-fault39-core==0.1.0`。
本RC标识不修改故障 `implementation_version='1'`。
源码包包含三者的源码及配方，不包含权重、CANN/vendor库或完整张量。

先核随包 `SHA256SUMS`，再运行原导出器的 `verify`。
解压后 `sources/` 是构建根；Native 构建管理器也指向这棵树。
以下在**既有推理环境的副本/隔离候选目录**执行，所需setuptools和wheel预先准备，
不要在线升级推理栈：

```sh
# 位于解压后的 demoncatpfi-integration/sources；路径由部署工具设置。
python3 -m pip wheel --no-index --no-deps --no-build-isolation \
  --wheel-dir "$RC_WHEELS" . ./src ./fault39
python3 -m pip install --no-index --no-deps --target "$RC_SITE" "$RC_WHEELS"/*.whl
PYTHONPATH="$RC_SITE" python3 -B tools/check_rc1_install.py \
  --expected-root "$RC_SITE" --output "$RC_INSTALL_RECEIPT"
```

`RC_SITE`、wheel目录及回执路径使用新目录/文件。检查器只核导入来源、版本、
公共配置、默认未授权状态和25拒绝，不创建Worker、不加载故障库或执行模型。
该检查在实际yellow进程导入组合中仍需重做；隔离目录通过不能证明同名包无冲突。

**不要直接覆盖黄区现役 `src/demoncat`。** 本包的 `demoncat`、`demoncat.runtime`
是namespace；不提供它们的顶层`__init__.py`。黄区若是普通package或已有
`demoncat.runtime.vllm_ascend.controller`，简单加PYTHONPATH可能被遮蔽或替换现役Controller。
openchamber须先记录双方 `__file__`/`__path__`，核wheel文件重叠，再在现役仓库审核合入
`worker_extension` 和RPC委托；保留原Controller行为。已有vLLM extension须显式组合，
发现同名RPC/属性冲突即拒绝，不能无检查地用新class替换。
三包版本号相同不代表字节相同，回退使用保存的原包、原SHA及启动配置。

Native获取/构建与加载详见[RC1 Native交接](rc1_native_dependencies.md)。
缺库、缺匹配源码/ABI或构建未审时显示 `待配置/部署不匹配`，不显示可开启。

## 唯一产品接口与五处接线

只推荐 `FaultConfig + WorkerFaultClient(HostAdapter(existing_executor))`。
下面标为“现役”的调度/部署代码由openchamber接入真实项目，不是新增公共API。

```python
from demoncat.runtime.vllm_ascend.worker_extension.config import FaultConfig
from demoncat.runtime.vllm_ascend.worker_extension.host import HostAdapter, WorkerFaultClient
from fault_injection.rc1_profile import prepare_binding

config = FaultConfig(fault_id=18, variant="api_half", implementation_version="1")
host = HostAdapter(existing_executor,
    prepare_binding=prepare_binding(trusted_deployment_callback))
# executor已由现役vLLM创建并初始化；callback只准备匹配的inert binding。
worker = host.route(config)           # 多Worker需可信select_worker策略，不默认rank0
capabilities = host.capabilities(worker)
client = WorkerFaultClient(host, config)
```

`existing_executor.collective_rpc(method, timeout=..., args=...)` 是实际路由边界。
无需让用户填层名、rank、地址、kernel或库路径。库/模型manifest由部署管理层取得。

| 接线位置 | 包内接口 | openchamber需要提供的实际代码位置 | 当前状态 |
|---|---|---|---|
|Catalog/能力查询|`host.route`、`host.capabilities`；静态`worker_core_manifest()`|现役目录与部署匹配/拒因展示|黄区待接线|
|Controller|`WorkerFaultClient`动作与回执|现役任务创建、状态查询、恢复入口|黄区待接线|
|已加载Worker/executor|`DemoncatWorkerExtension`及可信inert binding|Worker初始化后的绑定时机、现有extension组合|黄区待接线|
|现有请求调度|`advance`或`advance_batch/begin_decode_capture/collect_replay`|真实enqueue、内部ID映射、prefill/decode边界|黄区待接线|
|结果出口|动作返回`response.provider_state`和`query()`|现役接口/UI数值效果、历史效果、恢复展示|黄区待接线|

RPC class完整路径为
`demoncat.runtime.vllm_ascend.controller.DemoncatWorkerExtension`。
只在现役部署流程启动/加载Worker时配置或合并；extension自己不创建、重启、reload模型。
控制RPC不调用模型forward。未加载扩展的既有进程不能靠客户端import补出RPC。
RC1 callback和能力查询的`current_delivery.rc1_test_candidate`使用同一允许集合。
未入首批的普通RC1任务在调用部署callback前拒绝；原历史工厂为既有显式合同保留，
不能因它存在而绕过RC1 profile。候选部署不匹配时callback仍须拒绝，不能用静态名单
代替实际Native、对象及消费者门禁。若绑定拒绝发生在attach之后，用原client进行detach收尾。

部署层在**实际Worker线程、model已加载**后安装匹配binding：

| 能力 | 真实函数及必要参数 |
|---|---|
|18|`serving_tensor_binding.install_rms18_task_binding(worker, prompt_token_ids=..., weight_manifest=..., workload_sha256=..., environment_sha256=...)`|
|39|`serving39_output.install_fault39_output_serving_binding(worker, transport=verified_native_postwrite, prompt_token_ids=..., runtime_only=True)`|
|Batch|`serving_quant_projection.install_quant_projection_binding(worker, root=reviewed_native_bundle, build_sha256=actual_build_json_sha)`|

模块均位于 `demoncat.runtime.vllm_ascend.worker_extension`。
prompt来自这次真实任务的固定已tokenize输入；manifest和SHA来自实际权重文件、工作负载与环境，
不是填充常量。自动选点仍发生在当前调用中。binding只准备部署，不表示已注册、已启用或已合格。
没有匹配binding时必须保留provider的实际拒因。不要启用 `_simulation` 或使用CPU transport。
可选 `HostAdapter(..., prepare_binding=trusted_callback)`只允许准备inert绑定并返回prepare/register设置；
callback接收真实身份与config，不能执行推理或获取故障lease。

## 正常任务、效果与恢复

所有动作都返回完整Host回执；先检查 `receipt['status']`，再取
`receipt['response'].get('provider_state')`。动作应为`ACTION_COMPLETED`，query为`QUERY_OK`；
`PROVIDER_UNAVAILABLE`、`NATIVE_PRELOAD_NOT_ATTACHED`、`REJECTED`或不确定回复均停止正常推进。
不要只看register/enable成功。

39与18：`prepare → register → check → enable`；随后现役调度器以相同输入、fresh cache
分别enqueue normal/F/OFF/health。每轮在**实际enqueue取得内部request ID后、执行前**调用
`advance(actual_internal_id)`，调度完成后 `query()`。RPC不能代替enqueue或推理。
成功后 `finish → disable → unregister → detach`，再由部署层移除该binding。
18用 `remove_tensor_serving_binding(worker, '18.api_half')`，39用
`remove_fault39_output_serving_binding(worker)`。下一任务创建新client/provider/owner及正常作用域。

Batch：同样prepare/register/check/enable；按返回的`serving.next_phase/next_action`完成六batch。
每批8个真实fresh ID：enqueue → `advance_batch(ids)` → 真实prefill →
`begin_decode_capture()` → 真实首decode → `collect_replay()`。
六批输入保持一致，不能重复使用旧请求ID或已有KV。完成真实OFF后finish/disable/unregister/detach，
再用现有 `serving_quant.remove_quant_binding(worker)`移除binding。

| 能力 | 当前效果位置（provider_state内） | 恢复与历史 |
|---|---|---|
|18|`current_request_effect`|`last_fault_effect`明确历史；`serving.recovered`及实际OFF/health哈希|
|39|`evidence.current_request_effect`|`evidence.last_fault_effect`；`matched_off_and_health_confirmed`与Native释放分别核对|
|Batch|`serving.current_request_effect`|`serving.fault_effect`为历史F；完整Q/scale和OFF/hidden/logits核验|

以上字段无足够证据时保持未验证/null；pending OFF不能显示成当前F。
显示变化元素数、有限最大差、Inf/NaN数量或Q/scale关联，不要求文本改变。
本包不默认采dump/msprobe，既有统计历史不代表本次请求。

故障或响应不确定时保留原owner/ticket、首错和资源状态，先`query`，再按现有
`recover()`返回的恢复阶段，通过真实调度完成所要求的OFF/health。drain/restore未知时停止
这项能力后续注入并交给黄区运维；不得换Worker身份、清attempted或强制释放后宣称恢复。
`disable`本身不保证完整健康证明；lease为零与数值恢复分别展示。
回退先完成可执行的恢复，再按现役部署流程切回保存的旧包和启动配置；重启不算数值OFF证据。

## openchamber 首轮反馈

一次收集环境，不对每个故障重复取证：

1. 实际Python、vLLM、vllm-ascend、torch、torch_npu、CANN、硬件；模型/权重身份、量化、TP/PP/DP、eager/graph与缓存/调度模式。
2. 现役`demoncat`、`demoncat.runtime`、Controller、extension导入来源，以及三包文件冲突处理方式。
3. 上表五个真实接线位置（仓库、提交、文件和函数）；蓝区不能代填。
4. 本部署各RC1候选的匹配结果/精确拒因；先默认关闭从现役入口正常推理。
5. 匹配能力的实际故障现象、OFF/health结果、资源清理和下一独立任务结果。

问题分别归类：安装冲突、Controller/调度接线、部署版本不匹配、无合格目标、效果不符、
恢复失败/未知。当前黄区没有已完成的调用结果，五处接线均待openchamber回传。
蓝区继续开发不修改本包；未完成28、普通33、更多模型及Site整理不阻塞RC1交接。
