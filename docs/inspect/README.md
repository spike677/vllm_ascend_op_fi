# 列层—看参数—采一次—选目标—导出—验证

单卡新用户实测入口：[Qwen3-0.6B 识别 → 注入 → 评估使用说明](qwen06b_single_npu_user_walkthrough_zh.md)。本分支新增显式 private deployment 配置，已在 2123 的单卡模型上完成 112 个目标的真实绑定、IDENTITY、三题有损 B/F/R；不依赖旧双卡 E1 目录。下方旧双卡描述保留其历史适用范围。

当前推荐：[FakeRequant 工具 V1 功能说明](fake_requant_tool_v1_usage_zh.md)与[完整指标解释](../evaluation/fake_requant_tool_v1_metrics_zh.md)。本页保留通用 Inspect 与早期入口细节；新任务默认走 TargetPlan runner。

新增入口：[Qwen／DeepSeek FakeRequant 自动目标适配 V1](model_aware_fake_requant_v1_zh.md)。新 epoch 的 Qwen 双卡已完成 256 目标自动绑定、IDENTITY 与一题真实有损 B/F/R：262144 hit，明显重复输出，精确恢复。**原生 DeepSeek 和独立执行步完整窗口仍未验收**；总体 PARTIAL，不与下述旧入口的历史通过结论混用。

当前状态：**双卡 Inspect / IDENTITY 已通过，基于 MT-Bench 官方八题首轮子集的真实有损集成也已通过**。8/8 输出分叉并精确恢复；唯一标点规则告警来自表格分隔线，不能直接认定真实刷屏。详见[有损实机报告](mtbench_lossy_live_validation_20260910_zh.md)与[数据集接入说明](dataset_backed_lossy_integration_zh.md)。早期只读/无损证据仍保留在[原实机记录](live_validation_20260910_zh.md)。Inspect 自身仍是只读入口，有损执行委托已有 R1 owner，不会因发现参数就自动 arm。

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
  --adapter ascend_fi.inspect.qwen_r1_adapter:factory --confirm-lossless --output validation.json
```

这是实际存在并已通过双卡 CLI 验证的 factory，返回 `ExistingOwnerIdentityValidator`，委托项目已有 `VerifiedOwner`，不是第二个 controller。仅适配当前已部署的 R1 环境：需要原 helper、冻结 manifest 和已经取得的真实资源锁；不支持仅给任意模型 URL 就直接执行。禁止手写 owner 记录伪造资源交接。

`validate.json` 使用本次实机导出的文件，字段包括 `session_dir`、`suite.request_protocol`、`probe`、`fault_configuration`、`plan`、`service_epoch` 和 `runtime_fingerprint`。可设置唯一 `validation_run`，结果保存于该 session 的 `inspect_identity/<validation_run>/`；已存在则拒绝覆盖。factory 会确认所有 worker 都没有 Inspect hook，再检查目标 worker epoch。

该桥只调用已有 owner 的 preflight、baseline、disabled install、`fault_request(identity=True)`、finally disarm/remove、recovery；不实现新 arm/controller。固定 owner 的实际模块集合和 selector 必须与 plan 等价，不支持的子集明确拒绝。IDENTITY 返回原 tensor，不能用有损 REFERENCE_REQUANT 冒充无损。

验证输出仍为 `ValidationResult`。实机 baseline/disabled/IDENTITY/recovery 均回答 `84`，IDENTITY rank0 命中 256 次、changed=0、rank1=0。后续有损请求与效果分析继续交给已有 lifecycle/EvaluationResult；Inspect CLI 不直接 arm。八题数据集有损集成已在独立报告中完成；新的完整请求窗口配置探索仍按[预运行契约](config_exploration_v1_zh.md)单独记录，不能与冻结 R1 混写。

已有受控会话的小规模验收入口：

```bash
python tools/run_interactive_inspect_live.py --session "$SESSION" --stage inspect-identity
# 仅前一阶段通过、同 epoch、用户提供本次输入后才执行：
python tools/run_interactive_inspect_live.py --session "$SESSION" --stage smoke --prompt "用户自己的正常输入"
```

脚本不获取锁、不启动服务，不替换原参数或补齐 4096 次命中。`smoke` 路径已提供但本轮未实际运行，不能标为实机通过。旧 session 恢复原服务以后，旧 epoch/plan 必须重新观察与导出，不能直接重放。

## 当前现场量化合同

- `down_proj` 实机为 `AscendUnquantizedLinearMethod`；不捏造模型 activation scale。
- 代表性的 `qkv_proj / o_proj / gate_up_proj` 为 `AscendLinearMethod → AscendW8A8LinearMethod`（**静态** W8A8）。
- `input_scale` 是静态 per-tensor activation scale；重复派生的 `aclnn_input_scale`/reciprocal 给 quantize 使用。
- `weight_scale` 是 per-output-channel 权重尺度；`deq_scale` 是实际 QMM dequant 参数，不能与 input_scale 混用。
- `tensor_absmax` 仍为参考 FakeRequant 规则。未实现模型尺度 provider，不自动把静态 scale 传入原 R1。
- 合同要求完整类名与源码 SHA256 相符；未知或不同构建回退 UNKNOWN，不按字段名猜测。动态 W8A8 合同仅有源码审计，没有当前模型命中证据。

## QMM：独立算子视图

```bash
python tools/interactive_inspect.py recorded --args recorded.json --output operator.json
```

`recorded.json`：`{"kind":"candidate","input":"artifacts/evaluation/qwen_2card_fault_effect_v1/v4/candidate.json"}`。

支持 kind：candidate、typed_qmm、qmm_run、native_parameter。输入需为一个 JSON 对象，最大 8 MiB；混合 JSONL 先由已有 trace 工具选择具体记录，不能猜测绑定。

输出区分 module_args / torch_operator_args / native_formal_args / runtime_slot_proven；旧 GWS 地址与热 Run 地址不必相等，仍复用已有 semantic slot proof。weight_scale 不能当 activation_scale；可选字段未记录不等于 None。当前 native runtime 不可用时显示 `NATIVE_BACKEND_UNAVAILABLE`，保留 recorded 视图。

仅有算子证据时 `module_name=null`，不能根据同 shape 推断层。当前没有 module→native 精确关联，也没有新 QMM live adapter，因此不得从旧证据直接 arm 新 epoch。

## 验收状态

- 本地最新 214 项测试通过；早期 factory CLI 与新 TargetPlan runner 的实机证据分别保留。
- 历史数据与 fixture 分开；大体积/现场数据留 `artifacts/inspect/private/`，不提交。
- 实机 2 worker 各枚举 967 个模块、64 层；早/中/晚层代表观察与全 R1 目标观察通过。
- 数据集有损集成已完成；QMM 新 live adapter、module→native 精确关联仍为**未验收/不支持**，分开报告。
- 旧长任务已结束并清理；本轮使用独立会话，不改旧证据、不 push。
- 详见[复用地图](implementation_reuse_map.md)与[实机交接清单](live_handoff.md)。
