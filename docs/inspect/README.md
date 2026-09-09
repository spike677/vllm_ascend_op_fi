# 列层—看参数—采一次—选目标—导出—验证

当前状态：**本地接口与历史回放完成；真实双卡验证延后**。不改变正在运行的评测服务。Inspect 是开发者只读入口，不是自动注入器，不会因发现一个参数就 arm。

## 1. 新会话的单一 extension

仅在旧任务结束、故障移除、原服务恢复且资源锁交接后配置。保留原 vLLM/CANN/torch_npu 版本、模型参数及启动方式，不切 eager、不关 fusion 来伪造观察命中。

只读入口：

```bash
export PYTHONPATH=/path/to/interactive-inspect-v1/src:$PYTHONPATH
export VLLM_SERVER_DEV_MODE=1
# 在已经验证的启动器中，仅选择这一个 extension 参数：
--worker-extension-cls ascend_fi.inspect.extension.InspectWorkerExtension
```

需要接已有 R1 IDENTITY 时，改用一个组合入口（不能同时给两个 extension 参数）：

```bash
export PYTHONPATH=/path/to/interactive-inspect-v1/src:/path/to/pinned/reference:$PYTHONPATH
export ASCEND_FI_REFERENCE_SHA256=56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7
--worker-extension-cls ascend_fi.inspect.combined.InteractiveWorkerExtension
```

组合类只继承现有已安装研究扩展，不复制算法；校验失败拒绝启动。Inspect 使用独立 lazy state，不依赖自身 `__init__`，不替换 `execute_model`/`load_model`。RPC 方法只有 `demoncat_inspect`，action 是白名单。

vLLM 参数用途见[官方 engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)。模块观察采用 `register_forward_pre_hook(..., with_kwargs=True)`，返回 None，参见 [PyTorch Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)。发布文档不能代替当前版本实机验证。

## 2. 开始使用

以下 `$URL` 是用户已授权的**私有开发者 endpoint**。可通过已有 SSH 隧道访问 localhost；不新增公开 RPC 服务。CLI 复用当前 HTTP transport；有额外鉴权需求时通过客户端 transport 注入既有配置，不关闭鉴权。

```bash
python tools/interactive_inspect.py status --base-url "$URL" --output status.json
python tools/interactive_inspect.py list --base-url "$URL" --output modules-page1.json
python tools/interactive_inspect.py describe --base-url "$URL" --args describe.json --output target.json
```

`describe.json` 只含实际 list 得到的名称：

```json
{"module_name":"model.layers.1.mlp.down_proj"}
```

list 使用 `{"filters":{"layer_index":1},"offset":0,"limit":50}`；精确数字层匹配，1 不会匹配 10。`next_offset` 非空时继续翻页。每页最多 200 条及 256 KiB，模块总数/参数总数有上限。别名、未知 meta/lazy/layout 会明确显示；对象别名不声称覆盖所有 storage view alias。

## 3. 有限观察窗口

先从 status 取得各 worker 的 `service_epoch`，不要把 results 数组下标当成 rank。`observe.json` 示例：

```json
{
  "target_service_epochs":["从status复制的worker-epoch"],
  "module_names":["从list复制的完整模块名"],
  "request_window":"my-single-request-01",
  "exclusive":true,
  "ttl_seconds":60,
  "max_events":256,
  "max_bytes":1048576,
  "samples_per_module":2
}
```

```bash
python tools/interactive_inspect.py observe --base-url "$URL" --args observe.json --output window.json
# 用户通过自己的客户端发一条正常请求；不是固定 suite，也不自动 arm。
python tools/interactive_inspect.py poll --base-url "$URL" --args poll.json --output observed.json
python tools/interactive_inspect.py stop --base-url "$URL" --args poll.json --output stopped.json
```

`poll.json`：`{"target_service_epochs":["epoch"],"session_by_epoch":{"epoch":"observe返回的session"}}`。多个 worker 各自有 epoch/session，使用映射；epoch 不匹配的 worker 返回 skipped。

无 scheduler request ID 时，只能由用户确认独占单请求窗口，**不能检测或证明并发请求归属**。默认最多 60 秒，硬上限 300 秒，最多 2048 events/2 MiB，每模块最多 8 个样本。TTL 到时自动移除本任务 hook，异常也清理；客户端断开不造成无限观察。显式 stop 后核实 `installed_handles=0`，清理失败不可继续下一窗口。

记录 args/kwargs 嵌套 tensor 路径、dtype/shape/stride、实际 invocation、rank 来源及 epoch；不保存 activation 本体、不做 CPU 拷贝。没有可信阶段 metadata 时 `phase=UNKNOWN`；安装未命中为 `HOOK_INSTALLED_BUT_NOT_OBSERVED`，不是观察通过。当前只观察模块入口，模块输出与融合内部参数不在此处冒充已捕获。

## 4. 选候选、导出 DRAFT

```bash
python tools/interactive_inspect.py candidates --base-url "$URL" --args filter.json --output candidates.json
python tools/interactive_inspect.py export --base-url "$URL" --args export.json --output plan.json
```

过滤示例：`{"filters":{"logical_role":"MLP_DOWN","dtype":"torch.bfloat16"},"limit":50}`。支持 module_name/layer_index/module_class（list）、capability_id/backend/origin 等；不以字符串前缀扩大层范围。

`export.json` 需要实际候选 ID、目标 worker epoch、runtime fingerprint，以及**从已有 manifest 读取的完整 `frozen_arm`**，不能手填更强参数：

```text
target_service_epochs: [目标worker epoch]
candidate_ids: [选择的实际候选ID]
runtime_fingerprint: status中的指纹
frozen_arm: 原manifest的r1_arm_config对象
scale_source: reference
```

导出记录实际 module 集合、安全转义且首尾锚定的 regex、明确参数 selector、rank/layer、依赖与待验证项。默认 `DRAFT / enabled=false / arm_allowed=false`。

- `args[0]` 可转换为已有 `tensor_index=0`；`kwargs.hidden_states` 可转换为 `kwarg:hidden_states`。
- 嵌套 tensor 可以观察，但现有 backend 不能准确选择时拒绝。
- 共享模块存在多个调用别名时不能证明目标集合等价，拒绝精确注入计划。
- `tensor_absmax` 是 **REFERENCE_SCALE_POLICY**，不是模型原 activation scale。
- 用户要求 model scale 而没有匹配 provider 时拒绝，不静默回退。
- 找到属性名 scale 仅是线索；没有类型/版本/源码证据合同，字段角色为 unknown。

证据进度分别显示 FOUND、OBSERVED、ROLE_RESOLVED、BACKEND_BOUND、LOSSLESS_VALIDATED。目前仅观察/导出不会自动将后两项设为 true。

## 5. 显式无损验证，再委托已有注入/Evaluation

```bash
python tools/interactive_inspect.py validate --args validate.json \
  --adapter your_existing_leased_adapter:factory --confirm-lossless --output validation.json
```

factory 是**本地可信现有 lifecycle 接入点**，不是 RPC 执行代码；返回 `ExistingOwnerIdentityValidator`，并负责已有资源锁、配置、单请求 probe 与实时 epoch 查询。`validate.json` 含 plan、service_epoch、runtime_fingerprint。不要把虚构的 factory 路径当作已经安装的命令。

该桥只调用已有 owner 的 preflight、baseline、disabled install、`fault_request(identity=True)`、finally disarm/remove、recovery；不实现新 arm/controller。固定 owner 的实际模块集合和 selector 必须与 plan 等价，不支持的子集明确拒绝。IDENTITY 返回原 tensor，不能用有损 REFERENCE_REQUANT 冒充无损。

验证输出仍为 `ValidationResult`。后续有损请求与单次效果分析继续交给已有 lifecycle/EvaluationResult；Inspect CLI 不直接 arm。本轮并未运行真实 IDENTITY，只有桥接 fixture 测试与历史 native 验证回放。

## QMM：独立算子视图

```bash
python tools/interactive_inspect.py recorded --args recorded.json --output operator.json
```

`recorded.json`：`{"kind":"candidate","input":"artifacts/evaluation/qwen_2card_fault_effect_v1/v4/candidate.json"}`。

支持 kind：candidate、typed_qmm、qmm_run、native_parameter。输入需为一个 JSON 对象，最大 8 MiB；混合 JSONL 先由已有 trace 工具选择具体记录，不能猜测绑定。

输出区分 module_args / torch_operator_args / native_formal_args / runtime_slot_proven；旧 GWS 地址与热 Run 地址不必相等，仍复用已有 semantic slot proof。weight_scale 不能当 activation_scale；可选字段未记录不等于 None。当前 native runtime 不可用时显示 `NATIVE_BACKEND_UNAVAILABLE`，保留 recorded 视图。

仅有算子证据时 `module_name=null`，不能根据同 shape 推断层。当前没有 module→native 精确关联，也没有新 QMM live adapter，因此不得从旧证据直接 arm 新 epoch。

## 验收状态

- 本地 138 项测试通过（包括原有 91 项）；CLI help 与历史回放通过。
- 历史数据与 fixture 分开；大体积/现场数据留 `artifacts/inspect/private/`，不提交。
- 实机模块/参数观察、现场具体量化实现合同、真实 IDENTITY：**未完成**。
- 原双卡长任务仍运行，本任务状态：`LIVE_DEFERRED_ACTIVE_EXPERIMENT`。
- 详见[复用地图](implementation_reuse_map.md)与[实机交接清单](live_handoff.md)。
