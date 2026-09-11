# FakeRequant：手选／显式全选与一次有损推理

本文是操作手册，字段精确定义见[目标选择合同](../reference/target_selection_contract_v1_zh.md)。新入口复用原 FakeRequant 算法、owner 和 `TargetPlanExecutor`，不做敏感位置搜索，不自动找“最容易打坏”的模块。

本轮新增路径以离线选择和模拟 worker 编排验收为界；本轮未占用 NPU、没有新的 Ascend 多 rank 实测，且本地测试环境没有可用 PyTorch。历史 Qwen 单 rank 结果不能作为新增全 rank 模式的实机证明。

## 1. 两个入口，由用户明确选择

| 模式 | 用户意图 | 工具行为 |
| --- | --- | --- |
| `manual`（默认） | 指定模块族、层、精确模块名、global rank | 只保留过滤条件交集；目标未绑定即报错，不自动换点或扩张 |
| `all_bindable`（显式） | 在所选范围内同时选择全部可绑定目标 | 将合格目标展开为明确计划，记录 blocked 目标和结构缺口；不是逐点扫描 |
| `--ranks all`（显式） | 对本次纯 TP 请求的全部参与 worker 生效 | 从 descriptor 解析完整参与组，不硬编码双卡，不等于整机所有 NPU |

多个 `--roles` 内部是“任一角色匹配”；roles、layers、modules 三类条件之间是 AND。`--modules` 必须从目标报告复制完整 canonical 模块名，不填字符串猜测、通配符或父模块来代替叶子目标。

当前参考 owner 要求所有参与 worker 具有相同的可绑定模块名集合，即使某个 rank 不注入也需要该集合的完整报告。异构集合、缺失 worker、无法证明的 global/TP rank 映射都会拒绝；不会偷偷缩成共同子集。

## 2. 环境与输入准备

Windows 用于源码编辑、离线报告和计划预览；实际服务仍在 Linux／Ascend。沿用[单卡启动与 suite 准备](../inspect/qwen06b_single_npu_user_walkthrough_zh.md)，使用新 session 和已独占的项目 lease，不抢占其他进程。本文不启动服务，也不复用历史 epoch 的绑定许可。

新宿主一次有损工具使用项目推荐 Python 3.10+。旧启动器曾能运行的 Python 3.7 不代表本入口兼容；本轮未安装或升级远端解释器。

以下 Bash 变量必须对应当前部署，不照抄旧实验目录：

```bash
export DEMO_CODE=/path/to/current/repository
export DEMO_SESSION=/path/to/current/session
export DEMO_BASE_URL=http://127.0.0.1:18131
export DEMO_MAX_PROMPT_TOKENS=58
```

`58` 仅是历史 Qwen0.6B 三题 suite 的示例；实际应由当前模型 tokenizer、模板和冻结 suite 重新确定。改变生成预算也要重新生成计划，不修改已经验收的 JSON。

本 session 应有 `deployment.json`、`access.token`、`reference_arm.json` 和 `suite.json`。密钥不粘贴进终端命令参数值、报告或 Git；CLI 只接受密钥文件路径。每个 `--out` 使用尚不存在的新目录。

## 3. 先查看目标，不 arm

在能访问服务的同一宿主网络命名空间执行：

```bash
python "$DEMO_CODE/tools/inspect_model_targets.py" \
  --base-url "$DEMO_BASE_URL" \
  --api-key-file "$DEMO_SESSION/access.token" \
  --out "$DEMO_SESSION/01_target_view"
```

先打开 `target_preview.md`，需要细节时看同目录 `target_preview.json`：位置、rank、层、角色、selector、dtype、layout、输入特征、观察来源、状态、blocker 和下一动作集中显示。结构存在但 `bound=false` 的条目不能直接注入。

如果报告提示未观察，先用原有正常请求做真实输入观察并停止 observer。如下是四个已支持角色的显式全选观察示例；它只生成关闭计划，不执行有损变换：

```bash
python "$DEMO_CODE/tools/run_model_target_plan_live.py" \
  --session "$DEMO_SESSION" --deployment "$DEMO_SESSION/deployment.json" \
  --suite "$DEMO_SESSION/suite.json" \
  --selection-mode all_bindable --ranks all \
  --roles ATTN_QKV ATTN_OUT MLP_GATE_UP MLP_DOWN \
  --multiplier 20 --max-prompt-tokens "$DEMO_MAX_PROMPT_TOKENS" \
  --identity-probe mtbench_82_turn0 --probe mtbench_149_turn0 \
  --discover-only --out "$DEMO_SESSION/02_observed"
```

probe ID 必须存在于当前冻结 suite。`--discover-only` 运行的是正常观察请求；名字为 `identity-probe` 的参数此时选观察输入，并不表示已经执行或通过 IDENTITY。真实 IDENTITY 在正式执行前完成。

## 4. 手选：先看清范围，再生成计划

从同 epoch 的观察报告生成手选示例：MLP_DOWN、层 0 和 1、rank0。它不会把其他层或模块族加入计划。

```bash
python "$DEMO_CODE/tools/inspect_model_targets.py" \
  --reports "$DEMO_SESSION/02_observed/model_target_reports.json" \
  --selection-mode manual --roles MLP_DOWN --layers 0 1 --ranks 0 \
  --reference-arm "$DEMO_SESSION/reference_arm.json" \
  --multiplier 20 --max-tokens 512 \
  --max-prompt-tokens "$DEMO_MAX_PROMPT_TOKENS" \
  --out "$DEMO_SESSION/03_manual_plan"
```

精确模块模式可用 `--modules` 代替 `--roles`，参数值从刚才报告逐项复制；如果同时填写 roles/layers，精确名字仍须符合这些条件，否则报过滤冲突。

`--reports` 完全离线，不访问服务；预览标记 `OFFLINE_REPLAY_CURRENT_EPOCH_UNVERIFIED`。即使文件来自 LIVE 采集，也不证明服务仍是原实例。文件完整性校验与实际执行前 rebind 是两件不同的事。

## 5. 显式全选：全部合格目标与全部参与 rank

```bash
python "$DEMO_CODE/tools/inspect_model_targets.py" \
  --reports "$DEMO_SESSION/02_observed/model_target_reports.json" \
  --selection-mode all_bindable --ranks all \
  --reference-arm "$DEMO_SESSION/reference_arm.json" \
  --multiplier 20 --max-tokens 512 \
  --max-prompt-tokens "$DEMO_MAX_PROMPT_TOKENS" \
  --out "$DEMO_SESSION/03_all_plan"
```

这里省略 roles/layers/modules 表示在报告提供的全部已支持目标中选可绑定者，不表示任意浮点张量或融合 kernel 内部都受支持。也可以显式加入过滤条件缩小范围。

执行前核对 `target_preview.md` 和 `target_plan.json`：

- 实际注入 global ranks 与参与 ranks；物理 device ID 不是 rank。
- 每 worker 的精确模块名、每层/角色覆盖、排除原因与结构缺口。
- selector、输入合同、倍率/provider、plan/source/runtime/model 指纹。
- 每 worker 硬上限，以及各注入 worker 上限之和；不是共享全局预算。
- 计划仍为 `DRAFT`、`enabled=false`、`arm_allowed=false`；预览没有执行 hook 安装、arm 或故障请求。

相同名字在 rank0 和 rank1 是两个 worker 目标，不能跨 rank 合并。单个 worker 的逐字重复报告行可去重；真实多别名/父子冲突等未支持情况会阻塞，不偷偷安装两次。

## 6. 一次有损后完整停止模型实例

推荐新的“结束实例”路径在 Linux **宿主机**执行，宿主需有 Docker CLI 和本 session 所有权。`host` 不是子命令。示例选择全选计划；改成手选时仅替换计划文件路径。

```bash
python3 "$DEMO_CODE/tools/run_private_demo_once.py" \
  --session "$DEMO_SESSION" --suite "$DEMO_SESSION/suite.json" \
  --plan "$DEMO_SESSION/03_all_plan/target_plan.json" \
  --identity-probe mtbench_82_turn0 --probe mtbench_149_turn0 \
  --out "$DEMO_SESSION/04_once" --confirm-stop-instance
```

此确认只授权停止该 session 自有模型容器，不授权终止其他服务。执行器按当前 epoch 验证计划并运行既有 IDENTITY；通过后运行一个正式有损请求。清理仍尝试 disarm/remove，随后复用原停止流程结束完整模型实例。

IDENTITY 内部有正常/disabled/identity/恢复校验，不属于第二个正式有损请求。正式有损之后不再运行同进程 recovery 请求：记录应是 **实例已停止；文本恢复未测**，不能写成热卸载恢复 PASS。

旧 `run_model_target_plan_live.py` 的同进程 B/F/R 路径保留，供明确需要 baseline/fault/recovery 的实验使用。两种结束策略不能混淆；若完整重启后再推理，称为“重启后正常对照”。新实例必须重新绑定和 IDENTITY，不复用旧许可。

## 7. 从统一产物入口看结果

先打开本次 `--out` 下的 `README.md`；机器可读入口是 `artifact_index.json`，逐项记录路径、是否存在和文件 hash。它关联结构／绑定输入、关闭计划、IDENTITY、raw/pair、既有 `EvaluationResult` / Dashboard 和实例停止记录。未采集条目明确标记缺失，不造记录。

| 文件 | 阅读用途 |
| --- | --- |
| `execution_preview.json` | IDENTITY 前的精确目标数、worker epoch、倍率、预算和执行策略；`arm=false` 是当时预览状态 |
| `identity.json` | 当前计划的 IDENTITY 验收，不复用历史许可 |
| `single_run.json` | 本次 baseline/fault 原文、窗口证据和清理尝试；`recovery=null` |
| `effect/evaluation_result.json` | 既有结果合同，保留未测恢复与完整周期 N=0 |
| `effect/effect_dashboard.html` | 只读显示输出与数值，不是 arm 控制台 |
| `final_state.json` | 容器内 driver 的清理／query／health 尝试，不能代替宿主停止确认 |
| `final_instance_state.json` | 宿主最终停止结果及 lease 释放状态；停止失败时保留错误与不再请求要求 |

`EvaluationResult` 在宿主停止前生成，所以其 `stop_confirmed=false/STOP_PENDING` 不是最终停止失败结论；最终停止状态应以索引关联的 `final_instance_state.json` 为准。两份记录属于先后阶段，不覆盖早先证据。

先按以下顺序判断：

1. **机制**：所有目标 worker 的 arm/disarm、每模块 candidate/hit/reject 是否符合计划；缺失 rank 不按“没有效果”计分。
2. **数值**：changed elements 与采样误差能量确实变化；采样 NMSE/SQNR 不当成额外独立故障证据。
3. **输出**：全文、token pieces、首分叉、长度、finish_reason；没有整数 IDs 就保持 null。
4. **表征**：规则告警和具体异常片段；普通改写、合法表格或单纯预算耗尽不是病态证明。
5. **结束**：清理尝试、旧 worker 停止、容器与 lease 状态；未测恢复明确记未测。

仅增加命中范围不保证故障单调加重；开放题没有独立答案标签不能报告 Accuracy 下降。没有独立 scheduler 执行覆盖证据时，`full_request_window` 仍为 `UNPROVEN`，不由 candidate=hit 自动升级。

## 8. 常见阻塞与正确动作

| 阻塞 | 正确动作 |
| --- | --- |
| 未观察／observer 未停止 | 补正常观察，停止并确认无残留观察 handle |
| 旧 epoch／指纹变化 | 在新实例重新采集、生成计划和 IDENTITY |
| 某 worker 缺失或集合不同 | 保存具体缺口，停止，不悄悄降低 rank 范围 |
| selector／别名／layout 不受支持 | 明确适配缺口，不回退 first_float |
| IDENTITY 失败 | 不执行正式有损请求，保存结果并结束实例 |
| 请求完成或清理无法确认 | 不自动重试有损请求；按自有实例停止策略记录结束结果 |
| 只停止未再推理 | 文本恢复未测，不制造 recovery 数值 |

机制与清理的源码边界见[hook 核对](../architecture/hook_mechanism_compatibility_v1_zh.md)。本轮不把异常热卸载完备性作为继续扩大实验的理由。
