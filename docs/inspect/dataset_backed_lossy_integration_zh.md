# 从官方数据样本验证 Inspect 有损集成

本入口解决的是“当前 Inspect 发现的真实模块能否交给既有 R1 执行器，并留下配对输出和恢复证据”。
它不开发新 fault，不修改 R1，不执行原生 QMM 注错，不把模块输入绑定升级为 native slot 证明。

## 数据与边界

读取发布方 `lm-sys/FastChat` 的 `fastchat/llm_judge/data/mt_bench/question.jsonl`。
获取时先固定 40 位 revision，再下载该 revision 的原始问题、仓库 LICENSE 和评测 README；
保存 HTTP receipt、SHA256、获取时间和许可范围说明。下载不关闭 TLS 校验。

固定问题 ID 为 `82,83,86,88,141,143,146,149`，只用 `turns[0]` 的英文原文。
不翻译、不添加答案 marker、不提供参考答案或 judge rubric。编号缺失或 hash 不符时直接拒绝。
该范围名为 `MT_BENCH_FIRST_TURN_TARGETED_SUBSET`：它是定向 smoke 子集，不是完整多轮 MT-Bench；
没有 judge，也不输出官方 benchmark 分数。

公开下载入口：

```powershell
python tools/acquire_inspect_mtbench.py --output <全新数据目录> --revision <完整提交SHA>
```

如确有网络需要，可附加用户授权的 `--proxy http://host:port`。代理仅在该公开文件下载进程内生效；
不能用于 SSH、私有 RPC、模型输入输出或全局系统代理。

## 执行路径

```text
既有独占资源锁及受控服务会话
  → 数据原文与当前 tokenizer 对齐
  → 全部 8 题 Fault OFF 长度校准
  → 冻结同一生成协议
  → 当前 epoch 下观察 64 层 MLP_DOWN
  → 导出当前 plan，停止全部 observer
  → 用 question 149 通过 disabled / IDENTITY / recovery
  → question 149 真实 R1 fault / recovery / 报告
  → 其余 7 题逐题 baseline / fault / recovery
  → 清理、恢复原无扩展服务、释放锁
```

仍由 `qwen_r1_adapter` 创建项目已有 `VerifiedOwner`，由它安装、arm、查询、disarm 和 remove。
新增的是数据加载和编排，不是第二套注错控制器。

运行已获得锁的实验会话：

```bash
python3 tools/run_interactive_inspect_live.py \
  --stage dataset --session /home/haoran/inspect_<new_session> \
  --dataset-file <dataset>/question.jsonl \
  --dataset-manifest <dataset>/dataset_manifest.json
```

此命令不会抢锁或自行启动模型。先按已经验证的 launcher 和恢复流程建立新会话，不能使用旧 epoch 的 plan。
测试输出目录必须不存在；不覆盖历史结果。每个 fault request 使用 `try/finally` 清理，失败保留 raw。

## 冻结内容

- 模型及执行方式不变：双卡 TP2、eager、已有 chat template，`enable_thinking=false`。
- R1：rank0、64 层 `mlp.down_proj`，`REFERENCE_REQUANT`、`tensor_absmax`、倍率 0.25、interval 1、max_hits 4096。
- 此模型的 down_proj 使用未量化 linear；注入作用于其浮点输入 activation，不修改模型原 W8A8 scale。
- 逐请求重新 arm，4096 是每请求上限，不补齐请求，也不声称整段长回答都在注入。
- temperature 0、seed 42，采样、penalty、stop、chat template 和输出预算显式进入 raw payload。
- 初始预算 1024；只允许在任何有损请求之前按 1536、2048 全集校准并重新冻结；
  每次均验证模板化 prompt tokens + max_tokens ≤ 4096。仍未完成则停止，不制造 fault 数据补位。

## 如何解释结果

使用原 `EvaluationResult` 和修正后的
`DISTORTION_SCORER_V2_TASK_TERMINATION_PATHOLOGY`；记录四个 scorer 源文件的 hash，冻结原阈值。
数据集适配只增加输入溯源和结果范围说明，不改变检测规则。

| 维度 | 证据和解释 |
| --- | --- |
| 机制 | 本请求实际 rank / hit / changed elements / layers / reject；无变化不能通过 |
| 任务 | 开放式题没有可靠 validator，`NOT_EVALUATED`；不伪造准确率 |
| 终止 | 单独记录 stop / length / error；length 不自动等于推理错乱 |
| 病态 | 通用重复检测按实际输出长度触发；保留重复片段、位置、seq-rep-4 等 |
| 输出 | 完整 B/F/R 文本、首个 token-piece 差异；没有整数 token IDs 就记录 null |
| 语义 | `SEMANTIC_REVIEW_REQUIRED`；没触发规则不等于语义正确 |
| 恢复 | runtime clean、服务健康、表型恢复和文本精确一致分别记录 |

两项独立交付：

- `DATASET_BACKED_LOSSY_INTEGRATION`：溯源、当前绑定、实际数值变化、配对请求和精确恢复是否成立。
- `VISIBLE_DISTORTION_OBSERVATION`：实际观察到的规则异常；没有则如实写未观察到。

本轮每题一个 B/F/R，不是跨轮 repeatability。旧 16-probe / 3-cycle robust gate 标为
`NOT_APPLICABLE`，不能拿它判本轮执行失败，也不能伪称原 robust gate 已通过。

## 产物

`dataset_acceptance/` 保存数据 manifest、原文选择、校准 attempts、冻结 protocol、当前 plan、
ValidationResult、逐题完整 raw / mechanism / clean 状态、scorer hash 和完整 EvaluationResult。
`effect_1/` 是 149 首次闭环，`effect_8/` 是最终八题报告。Dashboard 默认并排显示完整回答，
并明确 hit 上限、恢复结果、语义审查边界，不用 600 字符摘要替代原输出。

原始 response、输入、运行时日志留在隔离 artifact 目录；本命令不上传 Git，不调用外部 judge。
