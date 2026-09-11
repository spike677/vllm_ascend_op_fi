# MT-Bench 首轮子集：Inspect → FakeRequant 真实有损验证

## 结论与直接入口

**`DATASET_BACKED_LOSSY_INTEGRATION=PASS`。**

8 个官方原始输入均通过新 epoch 的 Inspect 目标绑定、既有 R1 有损执行及恢复检查。
rank0 每请求命中 4096 次、覆盖 64 层，rank1 为 0 hit；累计 **32,768 hits**、
**716,447,861 changed elements**。这是各次命中的累计 activation 元素数，不是不同权重的数量。
8/8 故障回答分叉，8/8 recovery 的完整文本和 token pieces 精确恢复；24 个正式 B/F/R 请求均 HTTP 200。

可见效果必须分两层表述：

- 原冻结规则：`VISIBLE_DISTORTION_OBSERVATION=OBSERVED`，1/8 触发 `PUNCTUATION_SPAM`。
- 上下文核对：唯一命中位于第 146 题的 Markdown 表格分隔线，不足以证明真实刷屏或胡言乱语。
  原分数和阈值全部保留，只增加格式上下文注释，没有事后调参。

本轮证明了“新入口能执行有损变换、改变生成轨迹并精确恢复”，**没有证明模型出现明显病态崩溃**。
以下是本工作区私有产物，不是普通 Git 中已经提交的文件：

- [完整输入及 B/F/R 并排 Dashboard](../../artifacts/inspect/private/mtbench_live_20260910_v1/effect_final/effect_dashboard.html)
- [全部原始回答的中文 Markdown 索引](../../artifacts/inspect/private/mtbench_live_20260910_v1/effect_final/responses_full_zh.md)
- [完整 EvaluationResult](../../artifacts/inspect/private/mtbench_live_20260910_v1/effect_final/evaluation_result.json)
- [规则上下文核对](../../artifacts/inspect/private/mtbench_live_20260910_v1/effect_final/rule_context_review.json)
- [在线 / 离线一致性](../../artifacts/inspect/private/mtbench_live_20260910_v1/effect_final/offline_parity.json)
- [最终环境状态](../../artifacts/inspect/private/mtbench_live_20260910_v1/final_snapshot/final_environment.json)

## 数据与协议

使用发布方 [lm-sys/FastChat 固定 revision 原始问题文件](https://github.com/lm-sys/FastChat/blob/587d5cfa1609a43d192cedb8441cac3c17db105d/fastchat/llm_judge/data/mt_bench/question.jsonl)。
保留原文件、revision receipt、仓库 Apache-2.0 LICENSE、上游评测 README 和获取记录。
正常 HTTPS 直连成功，TLS 校验开启，本次未使用代理。

```text
revision = 587d5cfa1609a43d192cedb8441cac3c17db105d
question.jsonl SHA256 = 119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7
question_ids = [82,83,86,88,141,143,146,149]，turn_index=0
scope = MT_BENCH_FIRST_TURN_TARGETED_SUBSET
```

英文原文未翻译、未改写、未追加 marker 或 detector 信息。149 先完成首次闭环，
再按 82、83、86、88、141、143、146 顺序运行；不为凑数量重复 149。
8/8 Fault OFF 校准均自然停止，最长 951 tokens，所以直接冻结初始 1024 输出预算。
模板化输入为 38–69 tokens，输入加预算均不超过 4096。

```text
/v1/chat/completions；单条 user message = 原始 turns[0]
enable_thinking=false
temperature=0，seed=42，max_tokens=1024
top_p=1，top_k=-1，min_p=0
presence_penalty=0，frequency_penalty=0，repetition_penalty=1
stop=[]，stop_token_ids=[]，ignore_eos=false，min_tokens=0
n=1，add_generation_prompt=true，logprobs=true，top_logprobs=1
```

仅完成一个含八题的 B/F/R cycle，不宣称跨周期稳定性或统计显著性。
**不是完整多轮 MT-Bench，不计算官方分数**；旧 16-probe / 3-cycle robust gate 为 `NOT_APPLICABLE`。

## 环境、新 epoch 与冻结 R1

| 项目 | 实际记录 |
| --- | --- |
| 工作区 / 分支 | `D:\project\demoncatpfi-worktrees\interactive-inspect-v1` / `feature/interactive-inspect-v1` |
| 起始 HEAD | `a9b67095c1f0104744ba79734886effbc6c6651b`；新代码以 source manifest / 包 hash 绑定 |
| 设备 | A2 / 910B4，物理 2、5，逻辑 0、1，TP2 |
| 模型 | `/home/haoran/models/Qwen3-32B-w8a8-nopdmix` |
| 服务名 / 端口 | `qwen3-32b-w8a8-eval` / 18031，eager |
| CANN | 8.5.1，实际安装文件复核 |
| vLLM / vLLM-Ascend | `0.18.0+empty` / `0.18.0rc1` |
| torch / torch_npu | `2.9.0+cpu` / `2.9.0.post1+gitee7ba04`；实际计算由 NPU worker 证明 |
| 实验 worker | rank0 PID 1853370；rank1 PID 1853371 |

复用既有 NPU_LOCK，串行执行。新 epoch 每 rank 采到 128 条真实输入事件，覆盖全部 64 个 MLP_DOWN。
例：`model.layers.0.mlp.down_proj.args[0]`，BF16，`[44,12800]`，stride `[12800,1]`。
无可信 scheduler 阶段时仍记录 `phase=UNKNOWN`。导出计划和 owner 实装模块集合精确相等。
全部 Inspect observer 在执行控制/故障前已停止：`active=false / installed_handles=0`。

所检查 down_proj 使用 `AscendUnquantizedLinearMethod`：本轮修改其浮点 activation，
不是修改模型原 W8A8 scale。第 149 题通过当前 epoch 的 disabled / IDENTITY / recovery：
四次完整回答均为 951 tokens 且精确一致；IDENTITY rank0 4096 hits、64 层、changed=0，rank1 hit=0。
该 IDENTITY 计数不计入有损总数；保存了新 plan 和 `ValidationResult.json`，没有复用旧 epoch。

```text
rank_filter=[0]；module_regex=(?:^|\.)mlp\.down_proj$
layer_filter=None，实际集合=全部64层
mode=REFERENCE_REQUANT；scale_provider=tensor_absmax；scale_multiplier=0.25
max_hits=4096；interval=1；eps=1e-12
fixed_scale=None；runtime_scale_attr=None
tensor_selector=first_float（本次证明与唯一 args[0] 等价）
```

逐题重新 arm，测量后 `finally` disarm/remove。**4096 是上限，不是整个长回答持续注入。**
八题均耗尽上限并有后续 `reject:max_hits`。例如 149：rank0 后续拒绝 61,440 次，
rank1 因 `rank_filter` 拒绝 65,536 次。

## 逐题结果

| ID | Baseline tokens | Fault tokens | Recovery tokens | 首个差异 piece（从0计） | changed elements |
| --- | ---: | ---: | ---: | ---: | ---: |
| 82 | 134 | 117 | 134 | 9 | 96,282,964 |
| 83 | 269 | 258 | 269 | 7 | 105,192,800 |
| 86 | 229 | 217 | 229 | 5 | 83,395,571 |
| 88 | 133 | 135 | 133 | 2 | 83,340,012 |
| 141 | 906 | 826 | 906 | 22 | 80,223,113 |
| 143 | 615 | 547 | 615 | 0 | 93,107,162 |
| 146 | 724 | 777 | 724 | 2 | 89,896,175 |
| 149 | 951 | 1024 | 951 | 0 | 85,010,064 |

每行 rank0 为 4096 hit、64 层；rank1 为 0 hit，拒绝原因 `rank_filter`。
比较基础为真实 API `token_pieces`；没有整数 token IDs，明确记录 `token_ids=null`。
149 fault 的 `finish_reason=length`；其他正式请求均为 `stop`。

149 正常回答开头为 `Machine learning (ML) is a subset of artificial intelligence (AI)`；
故障回答从 `**Machine Learning** is a subset of **artificial Intelligence (AI)**` 开始，
后续章节、例子和排版也不同。Fault 在 `Each type of learning has its own strengths and is suited to different kinds`
处耗尽预算。这是生成分叉和预算内未完成，不能单凭 length 认定胡言乱语；完整原文见上方入口。

### 第 146 题：保留告警，但不扩大结论

冻结 punctuation detector：baseline 最长标点串 13，fault 最长串 21。
fault 命中区间 `[2687,2708)`，片段是 21 个 `-`，所在完整行：

```text
|------|-------------|---------|---------------------|---------|
```

它是 Markdown 汇总表分隔行。最终保留原 `PUNCTUATION_SPAM / DEGENERATION` 记录，
另加 `MARKDOWN_TABLE_SEPARATOR` 上下文注释；没有修改阈值或重算出更“好看”的分数。
**不能把 1/8 规则告警泛化成“12.5% 的回答真实刷屏”。**

没有独立任务 validator / judge，任务状态 `NOT_EVALUATED`，全部仍为 `SEMANTIC_REVIEW_REQUIRED`。
普通改写、语义错误、病态崩溃不混为一类。

## 溯源、验证与结束状态

```text
运行源码包 SHA256:
3e54d4c78f2e3bcfb9d4d963731333b378ea1412e06620eba3c92f413275d868
R1 runtime SHA256:
56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7
R1 arm config SHA256:
331bd956268c3f8d08a696ed1a8b75b9ff80e7601dda3c9eb4611960835698c9
Scorer = DISTORTION_SCORER_V2_TASK_TERMINATION_PATHOLOGY
Scorer composite SHA256:
69a2ae6346809dcb38bab813bdad3b0c4215617dfd3c7d50ffabae321cc3ff4e
证据归档 SHA256:
9463f39cffb80d0f46385c2b70d4a21cef12106ecd622a882c185f938a94275c
```

远端：`/home/haoran/inspect_20260910_mtbench_v1`。
Windows 快照：`artifacts/inspect/private/mtbench_live_20260910_v1/final_snapshot/`。
归档约 9.9 MB，170 个文件通过 SHA256 校验。历史 evidence 没有移动、删除或覆盖。

Windows 最终展示增加原始输入、机制标签、窄屏布局和格式上下文注释。
在线与离线评分、门禁完全一致，原在线 `effect_8/` 保留。新增的是非评分注释，不是改写 live 结果。

- 本地测试 154/154；原 Research FakeRequant 数学测试 5/5。
- Dashboard 1365 / 390 像素检查通过，8 行，JS error=0；已核对渲染截图。
- 实验服务日志未观察到 ERROR、OOM、HCCL failure 行；原日志保留。
- 2026-09-10 04:42:07 UTC（北京时间 12:42:07）：原无 Inspect/FakeRequant worker-extension 服务恢复，health PASS。
- Fault20 `enabled=0`；FakeRequant 方法已不存在；NPU_LOCK 已释放；恢复无 error / restore_error。
- 未 push；未修改 R1 算法、native Fault20 算法、模型权重或冻结强度。

运行方式见[数据集有损集成说明](dataset_backed_lossy_integration_zh.md)。
