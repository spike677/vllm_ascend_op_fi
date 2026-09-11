# FakeRequant 工具 V1：功能与使用说明

本页是历史双卡工具V1手册，冻结分支为 `codex/fake-requant-tool-v1`，原[版本记录](../releases/fake_requant_tool_v1/README.md)保留。**当前新用户默认使用[Qwen0.6B单卡手册](qwen06b_single_npu_user_walkthrough_zh.md)及[source-baseline-v1](../releases/source_baseline_v1/README.md)**，不依赖下面的旧E1/R1路径。本项目仍是开发者模式受限源码工具，不是任意Qwen/DeepSeek通用发行包。

已验收：指定 Qwen3-32B 双卡环境的自动目标发现、真实输入绑定、IDENTITY、有损执行、精确恢复和离线报告。原生 DeepSeek 接入、独立执行步覆盖、新组合的 GSM8K／PPL／三周期矩阵是独立后续任务，不阻塞这个最小功能版本。

## 1. 唯一推荐流程与安装边界

```text
Inspect 当前模型 → 选择角色 → 真实请求观察 → TargetPlan
                                         ↓
                         CLEAN / disabled / IDENTITY / recovery
                                         ↓
                              baseline / FAULT / recovery
                                         ↓
                            EvaluationResult → Dashboard
```

运行入口为 [run_model_target_plan_live.py](../../tools/run_model_target_plan_live.py)，不再推荐新建 Qwen C3 或 DPSK 专用 driver。底层 Inspect 仍独立、只读；旧 R1、C3、16-probe 脚本保留供历史复现，不是并行的默认版本。

安装方式是完整仓库源码布局，不是 `pip install ascend_fi`。Python 3.10+ 语法；本次离线测试的解释器版本记录在发布审计。CPU 离线功能使用标准库；worker 实际注入需要原验证环境的 torch／torch_npu／vLLM-Ascend。不要通过此说明升级运行时依赖。

```powershell
# Windows：在本发布分支根目录
$env:PYTHONPATH = "$PWD\src;$PWD\tests"
python -m unittest discover -s tests -q
python tools/inspect_model_targets.py --help
python tools/run_model_target_plan_live.py --help
```

```bash
# Linux：保留 src、tools、profiles 的相对位置
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python -c 'from ascend_fi.inspect.plan_executor import TargetPlanExecutor'
```

仓库当前没有 wheel/pip 安装契约。发布审计会复制独立源码布局并执行导入、帮助和测试；不声称完成了任意离线镜像安装。

### 实机依赖必须另行具备

- 已正常推理的私有 developer-mode 服务，实验锁和单请求独占 owner。
- pinned `fake_requant_reference_extension.py`，SHA256 为 `56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7`。`combined.py` 缺库／哈希不符时拒绝导入，不下载替代版本。
- 现有 `VerifiedOwner` 依赖已验证 E1 client、manifest 和 NPU_LOCK；具体路径在 `tools/run_normal_diagnostic_live.py` 的 `ROOT/OLD/HELPER`，client SHA 存在实测 `preflight.json`。这些不是本轮新实现的第二套生命周期。
- 当前 CLI 部署适配固定为 localhost:18031 的双卡实验环境。通用的是 TargetPlan 和执行器，**不是任意 URL 的通用启动器**。
- 只在新 epoch 启动时使用 `ascend_fi.inspect.experiment_extension.ExplorationWorkerExtension`；不热改正在使用的 runtime。锁、启动和服务恢复仍由既有 supervisor 管理。

无这些依赖时可以离线导入／重放，不能声称实机安装已通过；不要伪造 owner 记录。

## 2. 模型身份与通用 Inspect

使用实际模型 config／text_config、根类、父模块类、目标类及运行时源码指纹共同判断架构。服务名不决定适配器。Distill-Qwen 走 Qwen 规则，不能用它替代原生 DeepSeek 认证。

```bash
python tools/interactive_inspect.py status --base-url "$PRIVATE_URL" --output status.json
python tools/interactive_inspect.py list --base-url "$PRIVATE_URL" --output modules.json
python tools/inspect_model_targets.py --base-url "$PRIVATE_URL" --out "$NEW_OUT/discovery"
```

`inspect_model_targets` 默认 URL 为 `http://127.0.0.1:18031`；`--out` 必填且必须不存在。不带 `--roles` 时只生成报告，绝不 install／arm。实际报告包含 `descriptor`、适配器版本、`targets`、绑定状态和拒绝原因。

底层 RPC `demoncat_inspect` 的 `describe_model/auto_targets` 都是只读 action。CLI 自动每页取 100 条并校验分页期间报告哈希；底层每页最高 200 条／256 KiB。整体枚举截断或报错不能被“分页结束”掩盖。

手工观察使用 `interactive_inspect.py observe --args observe.json`；参数和真实例子见 [Inspect 说明](README.md)。默认 TTL=60 秒、256 events、1 MiB、每模块2样本；硬上限300秒、2048 events、2 MiB、每模块8样本、512目标。自动计划流程请求300秒观察窗，采两次／目标；不拷贝完整 activation。

状态分别保留 `discovered/observed/role_resolved/bound/identity_pass/lossy_mechanism_pass/visible_effect_pass`。结构存在但未调用，不能自动变成可注入。

## 3. TargetPlan：输入、参数和有效期

推荐从当前 epoch 自动构建；也可以只读导出：

```bash
python tools/inspect_model_targets.py --base-url "$PRIVATE_URL" --out "$NEW_OUT/plan" \
  --roles ATTN_QKV ATTN_OUT MLP_GATE_UP MLP_DOWN --rank 0 \
  --reference-arm frozen_reference_arm.json --multiplier 20 \
  --max-tokens 1024 --max-prompt-tokens 69
```

这里 69 是已冻结 Qwen 套件的 prompt 上界，不是 DeepSeek 默认值。`--reference-arm` 读取已有实际配置，不根据示例重造原 R1 参数。

| 参数／字段 | 默认与约束 | 输入／输出含义 |
| --- | --- | --- |
| `--roles` | 不填则只读；执行时显式选择 | 使用适配器的逻辑角色，不要求手写全部路径 |
| `--rank` | 只读导出默认 `[0]` | 当前 live CLI 自动方案固定 rank0；不声称可随意改 rank 执行 |
| `--multiplier` | 默认20；`0 < m <= 20` | 参考量化步长倍率，不是严重度分数 |
| `--max-tokens` | 导出默认1024 | 必须等于实际冻结请求预算 |
| `--max-prompt-tokens` | 无默认、导出必需 | 为有限硬上限提供经过记录的保守依据 |
| `selected_modules` | 精确集合，最多512 | 融合逻辑角色只绑定一次，父子重叠拒绝 |
| `workers[].targets[].binding` | 必须真实观察 | semantic argument、args/kwargs、dtype、layout、局部特征维 |
| `plan_sha256` | 内容哈希 | 计划改动需重新验收；不是签名或安全认证 |
| `state/enabled` | `DRAFT/false` | JSON 不携带可继承的 IDENTITY 通行证 |

首版只支持可证明的二维连续 FP16/BF16/FP32 单一浮点 activation；最终 selector 为明确的 `tensor_index` 或 `kwarg:name`，不退回 `first_float`。residual/router/cache、别名不清、多浮点辅助输入、嵌套 selector、特征维不符均拒绝。

模型重载、module 对象替换、源码指纹、epoch、拓扑变化后必须重新发现、绑定和 IDENTITY。未适配 PP／EP／DP、图模式、不同 worker 需要异构集合／selector 时不自动执行；不静默切 eager。

原生 DeepSeek V2/V3/V4、Qwen MoE 当前为只读候选规则。`VARIANT_READONLY_UNTIL_LIVE_ADAPTED` 是明确边界，不应删除门禁来让迁移“通过”。

## 4. 执行 FakeRequant 与三种预算

在已有 supervisor 已持锁的新私有 session 中：

```bash
python tools/run_model_target_plan_live.py \
  --session "$SESSION" --suite "$FROZEN_SUITE" --out "$NEW_OUT/acceptance" \
  --roles ATTN_QKV ATTN_OUT MLP_GATE_UP MLP_DOWN \
  --multiplier 20 --max-prompt-tokens 69 \
  --identity-probe mtbench_82_turn0 --probe mtbench_149_turn0
```

也可用 `--plan current_epoch_plan.json` 替代自动 `--roles` 构建。`--suite/--session/--out/--identity-probe/--probe` 必填。输出目录必须新建。即使给定计划，runner 仍执行 IDENTITY，不相信手填的通过标志。

该入口会调用已有 owner 的 preflight、install、fault_request、disarm/remove；不是独立裸 arm CLI。对需要直接组合 API 的受信任开发者，`TargetPlanExecutor.validate_identity(probe)` 后才可 `run_pair(probe)`；中间实际 RPC 为 reference 的 install/arm/query/disarm/remove。观察与绑定 RPC 不隐含注入。

模式固定 `REFERENCE_REQUANT + tensor_absmax`：原参考算法计算 scale，再以 `m` 放大步长，量化并反量化到原浮点输入类型，仅替换选中参数，不改权重或原生量化 scale。`m<1` 倾向裁剪；`m>1` 倾向细节归零；`m=1` 仍有损，不是 IDENTITY。

| 预算 | 当前合同 | 耗尽含义 |
| --- | --- | --- |
| 生成 | 来自冻结 suite，本例1024 token | 模型 `length` 终止；不自动代表病态 |
| 注入安全上限 | `目标数 × (max_tokens + prompt_upper_bound + 16) × 2`，不超过 `2^21` | 触顶则窗口不完整；不自动清零续打 |
| 日志 | 单事件保留首16／尾16；其余用有界聚合 | 不会因为只保留32条事件而停止注入 |
| 附加数值采样 | 每模块第1、65、129…次合格调用 | 与累计 changed/hit 的全事件分母不同 |
| 单请求时限 | 当前计划固定3600秒 | 异常后清理，不追加 padding 请求 |

当前是隔离服务的单在途请求事务，不是任意线上并发的 request-local 注入。

## 5. 评价、恢复与异常处理

完整输出写到 raw 中；`EvaluationResult` 和 Dashboard 沿用现有格式。V2 默认拆开任务答案、生成终止、输出病态；指标详细口径见 [指标说明](../evaluation/fake_requant_tool_v1_metrics_zh.md)。

当前 Qwen 最小验收：第82题 IDENTITY 全文／token pieces 精确相同；第149题 rank0=262144 hit、rank1=0；重复与词汇坍缩告警，恢复951 token与 baseline 精确一致。N=1，仅单样本功能证据。新组合的 Accuracy、PPL、三周期稳定性均没有新增测量。

`finally` 调用 disarm/remove；之后 query 必须同时确认全部 worker `enabled=false/installed=false`、无输入保护句柄、服务健康。恢复生成还必须与同题 baseline 全文及 token pieces 精确一致，不能只看 health。失败会使本计划的 IDENTITY receipt 失效；不继续后续故障请求。

超时或 driver 异常时由原 supervisor 执行最后清理／恢复服务／释放锁。若无法证明清理，停止下一次请求，保留失败证据后按已有生命周期处理；不要用 kill worker 或删除锁来假定安全。CLI 本身不负责释放 supervisor 的项目锁。

## 6. 证据、离线重放与错误解释

```text
model_target_acceptance/
  structural_reports.json / model_target_reports.json
  target_plan.json / identity.json / pair.json / final_state.json
  raw/cycles/cycle_00/              # 观察与无损
  raw/cycles/cycle_01/{baseline,fault,recovery}/
  effect/                         # 当时评分和展示版本
source_manifest.json / source_verification.json
final_environment.json             # 当时的环境状态，非今天实时健康证明
```

当前推荐离线验收入口：

```bash
python tools/verify_fake_requant_release.py \
  --evidence /path/to/downloaded/live_result \
  --suite /path/to/frozen/suite.json \
  --manifest docs/releases/fake_requant_tool_v1/release_manifest.json \
  --output /path/to/NEW/release_audit
```

无网络调用、无 arm、无 NPU_LOCK。它核对逐文件源码、计划读写、独立 raw 与文本哈希、当前机制门禁、精确恢复，并用原 suite 重新运行 V2 compare；在新目录生成原 `EvaluationResult` 的报告。报告清单绑定原 raw、suite 和当前 report 源码哈希。

一般历史16-probe／旧完整周期用 `tools/run_distortion_evaluation.py --replay ... --output ...`；默认 `--mode=distortion`，`analysis/both` 为辅助展示。不能把 MT-Bench 最小一题强塞进16-probe三周期门禁。旧 `replay_fault_effect_profile.py` 保留概率／任务解释，不作为本版病态默认判定器。

| 失败 | 正确处理 |
| --- | --- |
| `LIVE_REVALIDATION_REQUIRED` | 运行源文件漂移，补一题真实链路；不自动继承旧 PASS |
| `RAW_RESPONSE_OR_TEXT_HASH_MISMATCH` | 保留数据，定位取错文件／内容改动；不覆盖原文 |
| `SCORER_REPLAY_DIFFERENT` | 新旧评分不同应独立版本审计，不当成新推理结果 |
| `STALE_MODEL_RUNTIME_OR_WORKER_EPOCH` | 重新绑定与 IDENTITY |
| `OWNER_NOT_CLEAN/EXCLUSIVE_OWNER_REQUIRED` | 不注入，先完成资源与恢复交接 |
| `UNPROVEN` full window | 只认证观察到的目标调用；本轮不开发独立 scheduler observer |

对 DeepSeek 的下一步只做具体实例身份、有限目标绑定、CLEAN／IDENTITY／FAULT／RECOVERY；共用本执行器和结果合同。接入通过与可见异常通过分开，不能因一题不退化而无限搜索。
