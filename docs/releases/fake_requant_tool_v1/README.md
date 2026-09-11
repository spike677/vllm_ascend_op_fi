# FakeRequant Tool V1：主体冻结与迁移交接

发布标识：`FAKE_REQUANT_TOOL_V1`。推荐主线 `codex/fake-requant-tool-v1`；下一适配线 `codex/deepseek-target-adaptation-v1`。本轮只本地提交，不 push、不修改 main、不删除历史分支、worktree、dirty文件或evidence。

## 发布结论

**工具主体最小发布条件通过，Qwen证据可复用；原生DeepSeek待具体实例。**

该结论不等于旧 `FAKE_REQUANT_MODEL_AWARE_AUTO_TARGETING_V1_LIVE_COMPLETE` 或 `ROBUST_VISIBLE_DISTORTION_PASS`。原始Qwen `EvaluationResult.status=UNKNOWN`、完整执行步 `UNPROVEN` 仍保持；这里另行说明本轮更窄的“工具功能可交付”范围，不改历史评分。

| 项目 | 结果 |
| --- | --- |
| 主体整合 | Inspect → model adapters → 实测binding → TargetPlan → 现有owner → EvaluationResult/V2/Dashboard |
| 源码与实测一致性 | 实测捕获114文件，112完全一致；仅报告展示和报告测试不同，运行链路不变 |
| Qwen最小验收 | 当前代码离线重新检查真实raw、IDENTITY、目标绑定、有损机制、精确恢复，PASS |
| 离线回归 | 独立源码布局安装／导入、5个CLI帮助、计划校验、raw评分重放、221项测试、diff check通过 |
| 新增NPU请求 | 0；未获取锁、未碰当前服务 |
| 参考算法、权重、native | 不修改；保留独立pinned reference依赖及已有owner |
| 新×20配置任务/PPL/三周期 | 未测，后置研究，不作为此次最小版本门禁 |
| 原生DeepSeek | 版本化只读候选存在，实机接入资格尚无；不以Distill-Qwen替代 |

使用：[功能说明](../../inspect/fake_requant_tool_v1_usage_zh.md)、[指标说明](../../evaluation/fake_requant_tool_v1_metrics_zh.md)。

## 分支依据：已经包含的提交不重复合并

基准工作区为 `D:\project\demoncatpfi-worktrees\interactive-inspect-v1`。本地发布分支在该worktree创建，保留原 `feature/interactive-inspect-v1` 在 `a9b67095c1f0104744ba79734886effbc6c6651b`。原dirty工作均逐项检查后随发布内容保存，不通过reset/stash丢弃。

| 分支／盘点HEAD | 处理 |
| --- | --- |
| `main` / `86af009e` | 不变；该worktree另有Fault26修改，本轮不触碰 |
| `feature/evaluation-discovery-v1` / `1fd19955` | 已在当前提交祖先；其未提交扩展保留原worktree，不静默吸收 |
| `feature/distortion-evaluation-v1` / `557d6aa0` | 已在祖先，无需再次merge |
| `codex/distortion-scoring-audit-v2` / `b4a1077d` | 已在祖先，V2默认解释已经纳入 |
| `codex/fault-effect-profile-v1` / `6fe7a833` | 非直接祖先，但对应profile已移植且增加严格机制门禁；不merge退回旧宽松条件 |
| `research/fake-requant-reference` / `1d715725` | 保留独立reference边界；未提交PD/布局探查不纳入 |
| `research/fault-capabilities` / `165774b` | 历史能力引用，不借工具冻结升级Fault41认证 |
| `chore/docs-consolidation-v1` / `a7c70727` | 已有文档布局继续使用，不开展第二次大迁移 |
| `research/fault18-determinism` / `10400a11` | 独立任务及dirty文件全部保留 |

完整实际分支、HEAD与dirty清单见 [分支盘点](branch_inventory.json)。这次收束的是推荐开发入口，不是让所有历史分支强制合流或删除。发布后的DPSK分支从同一冻结提交开始；当前只是工作线准备，不表示已实现新适配。

## Qwen最小实测引用与变更分类

原实测：`/home/haoran/inspect_20260910_modelaware_v1/`；本地完整副本：`artifacts/inspect/private/model_aware_v1_20260910/live_result/`。

- Qwen3-32B-w8a8-nopdmix；910B4/A2，CANN8.5.1，TP2、eager；指定部署版本不变。
- 自动绑定4族256目标；真实参数`input_`，本次`args[0]`。
- Plan SHA256 `62e15c5e29caf7d5ee4c939c2b3f58dbeb93fb87c6b61dc19536925307aa20f0`。
- IDENTITY rank0 34560 hit，changed=0；各无损状态全文/pieces一致。
- 正式149题rank0 262144 hit、rank1 0、changed=1850723432；×20、tensor_absmax、cap567808；具体配置以清单和原plan为准。
- 正常951token／fault1024／恢复951；首个piece分叉，明显`diri`重复；恢复全文/pieces精确一致。
- 当时清理、恢复服务、释放锁已证明。**9月11日没有进行服务实时探测，不能把9月10日健康记录当作今天的实时状态。**

发布变化仅是增加离线审计入口与测试、功能/指标/发布文档和默认入口索引。昨晚live后已有报告展示修正也纳入：从实际配置显示4族、m20和cap567808，不再写旧R1默认值；不改变检测器/阈值或raw。原错误展示版本仍保留在原evidence。

由于绑定、输入替换、生命周期、请求编排以及原依赖均未变化，本轮使用原始证据重放，不新增Qwen模型请求。今后若这些运行文件变化，`verify_fake_requant_release.py` 将输出 `LIVE_REVALIDATION_REQUIRED`，届时只补最小真实链路，不自动继承本版PASS。

## 发布清单与可复查性

[release_manifest.json](release_manifest.json) 关联：源码逐文件哈希、原实测源码、模型／adapter／reference指纹、实际plan和arm、冻结请求及数据集、采集和评分版本、阈值、raw哈希、离线验证、已知缺口。清单是manifest而非新的EvaluationResult schema。

发布身份以本清单的内容hash和携带它的Git提交共同确定，避免在清单内部循环填写自身提交hash。当前提交可用 `git log -1 -- docs/releases/fake_requant_tool_v1/release_manifest.json` 查询；不把冻结前HEAD当成发布HEAD。

原始响应、私有session、access token、模型文件、.so、dump和源码安装临时副本不提交；只提交代码、测试、文档和清单。完整验收包留在私有artifact目录并由清单逐文件校验。独立安装测试所需的历史manifest引用也只读复制，不改其内容。

离线完整命令见功能说明。发布验收结果位于 `artifacts/inspect/private/release_20260911/`；首轮安装测试暴露缺少fixture引用，已补齐安装副本里的原fixture，未更改测试断言；失败audit_01保留，后续使用新目录。

## 原生DPSK迁移最小合同

只接受明确的原生模型路径／服务、硬件、runtime、TP/PP/EP与执行模式。准备后在适配工作分支：

1. 同一Inspect读取真实身份，非品牌匹配。
2. 只绑定实际观察过的一组浮点入口，导出该实例自己的TargetPlan。
3. 同一owner接口完成CLEAN／IDENTITY／FAULT／RECOVERY；必要绑定适配留在Model Adapter，不复制执行器。
4. 记录数值变化与严格恢复；同一EvaluationResult/报告交付。
5. 分别给 `DPSK_ACCESS_ACCEPTANCE` 和 `DPSK_VISIBLE_EFFECT_EXAMPLE` 结论；无可见异常不无限加倍率。

不默认256目标、64层、m20。融合expert内部、V4函数式调用没有观测时明确缺口；未知版本不借用Qwen执行资格。如果必须改变共享绑定/运行流程，执行相关离线回归并补Qwen一题，而不是直接宣称全兼容。
