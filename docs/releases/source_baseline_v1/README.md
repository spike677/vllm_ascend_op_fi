# 项目源码基线与分层文档收束 V1

本次是源码与文档交付，不是新模型实测。canonical：`D:\project\demoncatpfi`；分支 `codex/source-baseline-v1`；发布tag `source-baseline-v1-20260911`。具体提交可用 `git rev-parse source-baseline-v1-20260911^{commit}` 查询，避免把包含自身的manifest写成循环hash。

## 1. 来源、整合范围与回滚

| 来源 | 开始时HEAD | 处理 |
| --- | --- | --- |
| canonical main | 86af009e3d2aa6640570b07217eac24259ad5784 | main引用不改；2个未提交Fault26文件单独保存后切到整合分支 |
| interactive/Qwen0.6B | b8b167d709271296b8114b42a627d3a7827d03b1＋工作区修改 | 15个演示实现/测试/手册文件冻结为f67b4186b7ed5652d49ac7c40aa9cc762adfeaf1；基线从此继续 |
| research/fake-requant-reference | 1d715725f78a64ffb56bf8d1d75ec37275aef51f | 仅按固定Git提交纳入必要运行文件和原数学测试 |
| evaluation-discovery | 1fd19955开头，以inventory完整HEAD为准 | 不合入、不改工作区 |
| 其他research/docs/evaluation worktrees | [盘点](worktree_inventory.json) | 原分支/独有修改保留，不因日期新而整支merge |

完整[工作树盘点](worktree_inventory.json)记录开始时10个worktree的分支、HEAD、Markdown数、待提交文件SHA。canonical原两处Fault26修改保存到：

```text
D:\project\demoncatpfi-worktrees\preserved-main-before-source-baseline-v1
branch: codex/preserve-main-before-source-baseline-v1
additional retained stash: 4f3f1698720a1e2fb788767c0ea5d959cd40881d
```

两文件仍保持未提交状态，与开始时SHA完全相同。原main、Qwen演示分支、其他开发线与C盘legacy均保留。回到旧main应先检查canonical是否有新修改，再通过Git切换；不要reset或手工搬.git。无需把保存工作树合入发布。

## 2. 源码与依赖

[source_manifest.json](source_manifest.json)覆盖Python/CLI/测试/配置等交付文件；文本SHA按Git LF规范计算，固定reference另外严格核对原始字节。source manifest不包含自身hash，发布commit/tag固定整体树。

参考文件来源1d715725，SHA `56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7`，已纳入[原目录边界](../../../experiments/fake_requant_reference/README.md)。组合扩展不再依赖其他研究worktree。公共MT-Bench源suite来自已有固定证据中的相同字节，不改题目/阈值；运行suite由本模型tokenizer重新生成。

本次f67b418之后**未改任何已有运行Python/native、测试语义或评分合同**。新增代码只有离线文档／源码审计helper；其余为必要原依赖、固定公共配置和文档。

## 3. 与实际Qwen源码逐项核对

从服务器只读取回当时 `client_live_v1.tar.gz`，未启动服务。归档SHA：`e89c8c59fffd355fecd0ee52372d6b4a553e754dfd5175dd349cb1b17db5eba9`。

[证据索引](evidence_index.json)包含33个原始文件与当时audit逐字节一致的检查，以及85个实测Python文件的对比：**83个LF源码hash一致，2个已有差异**。

1. `dataset_effect.py`：原先从最多32条保留event汇总，已有修正改用全部接受调用聚合。只影响派生数值报告；原始raw及原评分不改。
2. `private_demo_service.py`：已有调整将设备/源码路径检查提前到lease之前，设置session下的TRITON_CACHE_DIR。**该启动器前置检查／缓存调整未重新做实机验收**，完整diff已保存，不能说与当时启动器逐字相同。

本次没有继续修改这两项。注入数学、输入selector、owner生命周期、评分器与既有实测对应；不能将上述差异隐去后宣称整份release已重新实机认证。其他新report/helper不反向改变当时模型行为。

## 4. 已有实测绑定（不是本轮新实验）

Qwen3-0.6B、BF16权重、28层；单910B4物理卡5→rank0、TP=1。CANN安装记录8.5.1，torch2.9.0+cpu、torch_npu2.9.0，vLLM0.18.0+empty、vLLM-Ascend0.17.0rc2.dev143+gf83cb0e6d。镜像tag不代替实际版本。

配置：4个角色×28层=112目标，REFERENCE_REQUANT/tensor_absmax/m20/rank0/interval1；512token、temperature0、seed42、thinking off；每请求cap131264。模型/计划身份与证据见索引和[单卡手册](../../inspect/qwen06b_single_npu_user_walkthrough_zh.md)，旧epoch不可执行复用。

| MT-Bench首轮原题 | B/F/R token数 | hit（candidate相等） | changed / total元素事件 | 精确恢复 |
| --- | --- | ---: | --- | --- |
| 82 | 139/512/139 | 57344 | 113904247 / 114200576 | PASS |
| 86 | 106/244/106 | 27328 | 57097327 / 57200640 | PASS |
| 149 | 512/512/512 | 57344 | 111147413 / 111390720 | PASS |

IDENTITY全部112目标、changed=0、normal/disabled/identity/recovery全文和pieces相等。正式三题首pieces分叉0、clip_rate=0、3/3恢复；N=1，不是3个完整周期。零/L2采样分母见[指标](../../evaluation/fake_requant_tool_v1_metrics_zh.md)。token_ids为null，不把pieces冒充ID。

第82题故障输出数字/标点/多语碎片；86/149有重复/词汇坍缩规则告警。149 baseline本身length，不把预算终止单独判错。完整原文留私有包，不能用本表代替原文。

结论边界：机制与目标调用窗口PASS；scheduler全请求窗口UNPROVEN，原EvaluationResult UNKNOWN／model_targeting PARTIAL保留。无该配置Accuracy/PPL、无官方MT-Bench分数，无N≥3稳定性或DeepSeek认证。

## 5. 检查与交付

[离线测试](offline_tests.json)：226项主体测试＋5项原参考测试；[源码/链接/CLI检查](verification.json)：固定文件SHA、参考SHA、单根workspace、推荐8个CLI help、Markdown本地目标检查。测试环境为已有Windows Python，**不是Windows运行移植**。

Linux特有运行检查本轮NOT_RUN；只读归档检查不等于服务启动。无NPU使用、无fault arm、无pull/rebase/push。原始33项证据hash保持，历史大文件/秘密不进入新提交。新增文档链接检查不检查外网可用性或Markdown锚点语义。

正式目录切换后再次执行231项离线测试；发现旧MT-Bench报告的6个链接依赖旧工作树私有文件，已改为证据包路径说明，没有搬数据或改结果。最终verification来自canonical目录复核，而不只是在实验工作树内通过。

已知待办：[算法限制](../../architecture/algorithms_zh.md)记录CPU/tensor极小scale eps差异、remove失败的物理hook确认缺口、启动早期留锁、请求timeout完成证明；本轮不修。通用概率事务、全窗口observer、DeepSeek和一键安装单独推进。

## 6. 阅读顺序

[项目入口](../../../README.md) → [用户手册](../../inspect/qwen06b_single_npu_user_walkthrough_zh.md) → [产物合同](../../reference/artifact_io_contracts_zh.md) → [算法](../../architecture/algorithms_zh.md) → [指标](../../evaluation/fake_requant_tool_v1_metrics_zh.md) → [源码](../../../src/README.md)。

本次完成状态表示 **SOURCE_BASELINE_AND_LAYERED_DOCS_V1_READY**，不表示新模型或全环境认证。
