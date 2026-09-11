# 产物输入输出合同 V1

本页是现有文件合同的说明，不增加schema。相对链接供源码浏览；**CLI文件路径相对调用进程cwd，不默认相对JSON所在目录**。私有部署应使用显式Linux绝对路径，host/container同路径挂载；不要把Windows路径写入Linux计划。

## 总表：配置、绑定、证据、派生物

| 产物 | 生产者 → 消费者 | 核心字段／必要条件 | 有效期／缺失处理 |
| --- | --- | --- | --- |
| deployment.json | private_demo_service.start → PrivateServiceOwner | base_url字符串(127.0.0.1 HTTP)、tp_size整数、device_ids数组、served_model_name、model_path/source_root/reference_path、lock_owner_path/owner/lease_id、container_name、reference_arm对象 | 当前服务输入；核对模型/lease/hash/鉴权。历史记录永存，但换服务不可直接改epoch当新部署 |
| access.token | starter → middleware/client/owner | 私有随机token文件；owner要求长度≥32 | 秘密，仅会话本地；缺失拒绝；不进入manifest或报告 |
| service_ready.json | starter health检查 → 人工/审计 | healthy、container | 当时ready证据，不代替执行前preflight |
| source_suite.json／公共profile | 历史固定公共suite → prepare_private_demo_suite | version/hash/probes/request_protocol/thresholds、dataset revision与原题hash | 可复用原题和阈值，不能当已按新tokenizer计数的suite |
| suite.json | prepare工具 → observe/runner/replay | probes、generation_config、demo_scope.prompt_token_counts/max_prompt_tokens/source_suite_sha256、content hash | 换模型/template重新计算另存；来源hash失配拒绝；父目录须存在 |
| 01_inspect/model_target_reports.json | inspect_model_targets → 人工筛选/报告 | descriptor、targets、blockers、report_sha256 | 结构候选，不是可执行计划；旧结构可审计但必须fresh观察 |
| 02_bound_plan/model_target_reports.json | run_model_target_plan_live.observe_and_build → builder | 同epoch实参、observed/bound、selector、runtime/model指纹 | 换服务/实现必须重新观察；未bound不继续 |
| target_plan.json | plan_builder → TargetPlanExecutor | schema、state=DRAFT、enabled=false、workers、selected_modules、prospective_arm、request_window、plan_sha256 | 校验内容hash/当前身份/真实输入；不准手改worker或目标借旧许可 |
| identity.json | executor.validate_identity → 审计/runner | normal/disabled/IDENTITY/recovery、目标覆盖、changed=0、精确相等 | 历史无损证据；本次执行器还需有效内存receipt，不是可跨进程/epoch导入的许可证 |
| *_pair.json | executor/runner → dataset_effect/summarize | pair.baseline/fault/recovery、measurement_query、recovery_exact、plan关联、全响应 | 长期原始证据，不因服务停止失效；缺失请求/机制不能补造 |
| raw/ | owner → pair/audit/replay | 原HTTP/RPC请求、响应、状态、fault_transaction | 路径取决于session/owner；实际以文件记录为准，不承诺都在03_evaluation/raw；禁止公开敏感原文/认证 |
| replay_input.json | dataset_effect → distortion replay | suite、cycle配对记录、配置与机制证据 | 与原pair/评分器hash关联；新重放另存，不改历史输入 |
| numerical_effect.json | summarize_private_demo → 报告 | 每probe全调用汇总与sampled_*诊断 | 派生；采样分母单独保留，不从32条history推总hit |
| evaluation_result.json | 既有评价器 → Dashboard/程序 | EvaluationResult及distortion/model_targeting等extra | 统一合同，不新建第二套；UNKNOWN/PARTIAL/null不能视PASS/0 |
| 中文报告／effect_dashboard.html | summarize/report → 人工 | 完整原文、指标和来源记录 | 派生展示，绝不是注入控制入口 |
| replay_provenance.json／live_source_audit.json | report/audit → 版本复核 | 输入、源码、参考文件、运行环境hash | 哈希对应原字节，不能换行/移动重写JSON后沿用 |
| final_state.json／service_stopped.json | runner／starter.stop → 收尾审计 | disabled/clean/health或container_stopped/lease_released/time | 证明对应时刻；不是未来资源空闲承诺 |

生产源码：[starter](../../tools/private_demo_service.py)、[suite准备](../../tools/prepare_private_demo_suite.py)、[runner](../../tools/run_model_target_plan_live.py)、[owner](../../tools/private_service_owner.py)、[plan](../../src/ascend_fi/inspect/plan_builder.py)、[executor](../../src/ascend_fi/inspect/plan_executor.py)、[汇总](../../tools/summarize_private_demo.py)。

## Hash与人工编辑

- TargetPlan/report使用model_descriptor.digest的规范JSON哈希；suite使用distortion.contracts.content_hash（排除自身hash字段，排序/紧凑JSON、禁止NaN）。文件SHA是**原始字节hash**，与内容hash不同。
- source_suite可以作为只读配置来源。改变意图（角色/倍率/题目/预算）应创建新配置并经builder/prepare重新生成，不手工修改旧计划后沿用IDENTITY。
- deployment由新session启动产生；里面的reference_arm含历史安全模板max_hits=4096，**最终实际执行预算来自TargetPlan.prospective_arm**，不能用模板推断剂量。
- 实时输出中的worker状态是查询快照。字段缺失与null/false/0不同，未知环境不使用镜像tag或数组索引补足。

## 文件覆盖与失效策略

runner的out目录、summarize的out目录要求不存在；其save采用exclusive create。prepare的out文件exclusive create。starter的关键JSON/token也exclusive create，session/reference可以预先建。其他旧Inspect CLI不统一保证拒绝覆盖，因此始终使用独立新路径。

原始raw/pair、旧报告、旧manifest不做本次批量修正。新统计版本通过新目录＋来源hash连接；路径映射另存。换模型/服务后只复用配置意图和公共数据，重新生成实际deployment、绑定、预算和identity。

`request_completion_unproven.json`是停止后续请求的保护记录，不是删除即可继续的缓存。移除失败/留锁等边界见[算法生命周期](../architecture/algorithms_zh.md)。

## 典型数据流

```text
source + pinned reference + model
  → new session/deployment + ready
  → public source suite → model-specific suite
  → structural Inspect → clean observation → disabled TargetPlan
  → same-epoch IDENTITY
  → B/F/R pair + raw + mechanism
  → EvaluationResult + numerical summary + Dashboard
  → final_state + stop/lease record
```

阅读当前流程用[单卡手册](../inspect/qwen06b_single_npu_user_walkthrough_zh.md)；字段解释用[指标字典](../evaluation/fake_requant_tool_v1_metrics_zh.md)，两者不能从历史报告推断当前服务有效性。
