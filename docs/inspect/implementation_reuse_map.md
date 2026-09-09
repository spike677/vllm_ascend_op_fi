# Interactive Inspect V1 复用地图

基线：`b4a1077d3d4d9602948f96250d3010993be1d745`。分支 `feature/interactive-inspect-v1`。本轮不修改既有 native、R1、Evaluation、scorer 或 collector。

| 原入口 | 本轮使用 | 边界 |
| --- | --- | --- |
| `src/ascend_fi/contracts/candidate.py` | Inspect 候选直接构造 `Candidate`，兼容 extra 保存路径/epoch/证据进度 | 不增建 registry；OBSERVED 不冒充 LOSSLESS_VALIDATED |
| `src/ascend_fi/contracts/validation.py` | `ValidationResult` 用于 IDENTITY 委托和已有 native 无损回放 | 保留 MODULE_IDENTITY 与 native slot proof 的区别 |
| `src/ascend_fi/discovery/adapters.py:adapt_qmm_run_trace` | `inspect/operators.py:native_run` | 保留 v17 storage-order proof；去掉旧 device→rank 回退，不输出裸地址 |
| `src/ascend_fi/discovery/resolver.py:resolve_candidate` | native 候选与现有 manifest 匹配 | RECORDED 候选仍不能 arm 当前进程 |
| `src/ascend_fi/validation/pipeline.py:validate_target` | 原样回放已有 QMM fixture 无损证据 | 不把 fixture PASS 当当前服务验证 |
| `fault20_fast_validation/source/l4_metadata_lifecycle_proxy/libopapi_metadata_proxy.cpp:capture_qmm_metadata` | 读取正式角色定义，显示 x1/x2/scale/offset/bias/out | 未取得的字段显式 UNKNOWN；无 C++ 修改 |
| `fault20_fast_validation/source/l4_buffer_substitute_allhit_v1/buffer_substitute_preload.cpp` | 适配 `qmm_param_collection`、`qmm_run_trace`、`qmm_weight_observation`、`qmm_x2_scale_observation` | formal ordinal 不当成 Run slot；不同事件不靠 shape/时间合并 |
| `fault20_fast_validation/source/fault26_readonly_trace_v1/fault26_worker_extension.py:_QmmCallableProxy` | 读取 `typed_qmm_call_exit` 的既有 schema | 旧 probe 只记部分参数，不声称省略字段为 None；本轮不安装该全局 wrapper |
| Research `1d715725f78a64ffb56bf8d1d75ec37275aef51f` 的 `experiments/fake_requant_reference/source/fake_requant_reference_extension.py` | 启动时可选 `combined.InteractiveWorkerExtension` 组合；IDENTITY 委托给已有 lifecycle | 外部依赖，不复制算法；SHA256 `56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7` 与原任务启动审计相同 |
| 上述 extension 的 `_select_tensor` | 导出 `tensor_index` 或 `kwarg:<name>` | 嵌套路径可观察，但现有 backend 不支持时拒绝导出；first_float 仅在实测唯一时等价 |
| `tools/run_normal_diagnostic_live.py:VerifiedOwner` | `ExistingOwnerIdentityValidator` 消费其既有 preflight/request/install/fault_request/cleanup 接口 | 不动正在运行的 owner；不同目标集合不强行让固定 owner 执行 |
| `src/ascend_fi/evaluation/live.py:OpenAICompletionClient` | 复用 HTTP transport 调用现有 `/collective_rpc` | 不另建 Web 服务；Inspect RPC 不提供 arm/任意代码/地址读写 |
| `profiles/discovery/qwen_2card_injection_surface_v1.json` | 保留旧四类投影的已提交历史参考 | 本轮运行枚举不硬编码 64 层、不直接导入旧地址或历史 hookable 标签 |

## 未提交源码边界

原评估 worktree 的 `src/ascend_fi/discovery/module_surface.py` 仍为 untracked，未复制、未移动、未当作本分支依赖。本轮仅实现 bounded registry 枚举与有限 Qwen 名称提示；未来合并已提交的分类器时可替换该提示函数。没有复制它的大型 DeepSeek 角色映射，也没有声称 DSV4 live 支持。

## 三个 adapter

- `surface.ModuleSurfaceAdapter`：读取模块注册表、参数/buffer metadata、对象级共享别名；不调用全量 state_dict，不读取 tensor 值。
- `quantization.QuantizationResolver`：量化方法类型与 runtime fingerprint 匹配可信部署合同；没有已审计合同则只列字段线索，role/granularity/encoding UNKNOWN。
- `operators.OperatorEvidenceAdapter`：Torch typed probe、native 正式角色、native Run slot 分视图；只接入既有 recorded schema，不添加 live interceptor。

当前没有随包宣称已验证的现场量化合同。尝试把远端量化文件全文传回本地被安全审核拒绝，因此没有获取该载荷，也没有通过其他路径重试。后续需要明确授权源码只读核对，或由用户提供可审计的文件及哈希，才能将具体现场实现从 UNKNOWN 升级为已解析。

## 审计时保护的 Git 状态

| 分支 | 原 HEAD | 保留内容 |
| --- | --- | --- |
| main | `86af009e3d2aa6640570b07217eac24259ad5784` | 两个现有 Fault26 修改 |
| feature/evaluation-discovery-v1 | `1fd19955d8f0c397c2f1ceb459533d362e51dffa` | 两个 __init__ 修改与未提交实现/工件 |
| research/fake-requant-reference | `1d715725f78a64ffb56bf8d1d75ec37275aef51f` | 未提交 token_layout_probe |
| research/fault-capabilities | `165774bb`（短 hash，见 Git 实际分支） | 原分支不合并 |
| feature/distortion-evaluation-v1 | `557d6aa08e0564b64999a5b47e7e0ec76e01a27e` | 冻结运行版本不改 |
| codex/distortion-scoring-audit-v2 | `b4a1077d3d4d9602948f96250d3010993be1d745` | 评分修订不改 |
| chore/docs-consolidation-v1 | `a7c70727fa64a12bf3720276ed223e9acc19570f` | 文档分支不改 |
