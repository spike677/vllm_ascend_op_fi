# Qwen／DeepSeek FakeRequant 自动目标适配 V1

## 当前结论

状态：**Qwen 新入口的自动绑定、IDENTITY 和真实有损 B/F/R 已通过；跨模型／独立执行步完整窗口验收仍为 PARTIAL**。

本轮实现的是“模型身份 → 结构候选 → 实测输入绑定 → 明确目标计划 → 既有 owner 执行”，不是再增加一套 Qwen C3 或 DeepSeek 专用注错脚本。FakeRequant 数学实现、原始 R1 配置、Fault20 native、模型权重和检测阈值均不改。

当前可核实的结果：

- 新入口不再要求固定 64 层。离线测试覆盖 1、4、28、64 层配置。
- 本地完整测试为 **214 项通过**，其中本轮新增 46 项目标适配测试、2 项报告配置展示回归；含模拟 owner 的完整生命周期测试，但不将模拟测试算作模型 E2E。
- 对历史 C3 两个 worker 的真实模块清单进行只读结构回放，均自动得到与原清单完全一致的 256 个目标，缺失／额外目标均为 0。
- 回放发现并适配了现场事实：Qwen3 使用 `Qwen2MLP`，投影为 Ascend linear 类；其实际 `forward` 参数名为 `input_`，不是臆定的 `hidden_states`。
- 随后在新 epoch 真实观察并自动绑定了两个 worker 各 256 个目标；IDENTITY 命中 34560 次且 changed=0；正式 MT-Bench 149 故障请求命中 262144 次，出现明显重复输出，恢复全文与 token pieces 精确一致。不是借用旧 C3 的通过结论。
- 2026-09-10 初次检查时 NPU 2／5 被其他服务使用，未抢占。后来确认资源释放，按项目锁和既有 supervisor 冷启动隔离环境；完成后恢复无扩展服务，20:59:37（UTC+8）健康并释放锁。初次只读审计记录保留，不覆盖为本轮实测结论。
- 未分配原生 DeepSeek 的现场验证 endpoint。DeepSeek／Qwen MoE 分支只提供版本化只读候选规则，**不自动继承 Qwen dense 的执行资格**。

## 实现边界

| 组件 | 文件 | 责任 |
| --- | --- | --- |
| 模型描述 | [model_descriptor.py](../../src/ascend_fi/inspect/model_descriptor.py) | 配置／文本子配置、实际类、源码 SHA256、backend SHA256、epoch、TP／PP／DP／EP 和执行模式 |
| 模型规则 | [model_adapters.py](../../src/ascend_fi/inspect/model_adapters.py) | 按架构、父模块、线性类、路径解释角色；生成预期覆盖要求 |
| 输入解析 | [target_resolver.py](../../src/ascend_fi/inspect/target_resolver.py) | 真实调用的语义参数、args／kwargs 位置、dtype、layout、局部分片特征维度 |
| 计划构建 | [plan_builder.py](../../src/ascend_fi/inspect/plan_builder.py) | 明确目标集合、融合去重、父子冲突、请求安全预算、完整性校验 |
| 通用执行 | [plan_executor.py](../../src/ascend_fi/inspect/plan_executor.py) | 委托现有 owner 做 IDENTITY、单请求 B/F/R、finally 清理、证据分层 |
| 输入保护 | [input_contract.py](../../src/ascend_fi/inspect/input_contract.py) | 每次实际调用前只读复核输入合同；不改 tensor，不实现另一套 QDQ |

`InspectWorkerExtension` 新增两个只读 action：`describe_model`、`auto_targets`。都保留分页，不会自动 install／arm。观察日志增加参数语义、对象别名、连续性和调用计数；预算／TTL 终止原因不会被随后显式 stop 覆盖。

`ExplorationWorkerExtension` 按需增加输入保护 mixin。仅显式计划执行时安装保护 hook，且在真实 reference hook 之前运行。异常输入直接中止该受控请求；最终 owner 同时清除 reference hook 与输入保护 hook。旧 C3 入口未使用该功能时不会安装额外 hook。

## 支持矩阵：识别不等于执行已认证

| 架构 | 自动规则 | 当前执行边界 |
| --- | --- | --- |
| Qwen2／Qwen3 dense | Attention QKV／Out、MLP Gate-Up／Down 等明确投影 | 当前 Qwen3-32B 双卡实测通过；其他实例仍须本 epoch 的绑定和 IDENTITY |
| DeepSeek-Distill-Qwen | 根据实际 `qwen2/qwen3` 配置进入 Qwen 规则 | 不算原生 DeepSeek 验收 |
| Qwen2／Qwen3 MoE | 可见投影、shared MLP、明确外层 MoE 候选 | 只读；融合 expert 内部不冒充 down projection |
| DeepSeek V2／V3 | 独立 MLA 投影和 MoE 结构候选 | 只读，尚无本轮现场资格 |
| DeepSeek V4 | 独立 `fused_wqa_wkv/wq_b/wo_a/wo_b` 等候选 | 只读；其他 Ascend 实现类、compressor/indexer 需对应合同，不按名字猜 |

V4 合并入口可携带多个逻辑角色，但仍只有一个物理目标。模块存在但没有调用记录时，保留结构信息并写明 `MODULE_ENTRY_NOT_OBSERVED`，不声称已经证明函数式绕行；只有现场调用证据才能区分“该请求未执行”和“确实绕过 module hook”。

注册表不是按服务别名匹配。配置身份与实际根模型类不一致时拒绝；未知版本不借用最相似规则。HF 的 Distill-Qwen 配置与原生 V4 代码可作为结构参考，但不能替代现场版本证据。[Distill-Qwen 配置](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/blob/main/config.json)、[vLLM V4 attention 实现](https://github.com/vllm-project/vllm/blob/main/vllm/models/deepseek_v4/attention.py)。

## 输入绑定与首版限制

正式计划冻结：实际完整模块名、逻辑角色、层、worker epoch、模型／实现指纹、语义参数、`args[index]` 或 `kwargs[name]`、FP16／BF16／FP32、二维连续 strided layout、动态正 token 维、固定局部分片特征维。

首版明确阻断：

- 多个浮点辅助参数／别名合同不清；residual、router logits、cache 等不能因为是浮点就被选中。
- 嵌套 selector、布局未知、输入维度无法与模块分片属性对齐。
- 同时选择父模块和子投影；同名模块共享别名未解决。
- 同一 reference owner 需要混合 selector，或 TP worker 上需要不同安装集合。
- 未适配 PP／EP／DP 拓扑，非明确 eager 配置，模型／worker／源码指纹变化。
- 模块枚举／观察预算截断。发现页全部读完不等于整体枚举完整。

当前超过初始模块／参数枚举硬上限时，沿用旧 Inspect 的明确 RPC 错误；尚未增加超限树的流式部分返回。观察阶段则保留预算终止原因与已获得样本。两种情况都不得认证完整覆盖。

`first_float` 仍可用于旧探索入口，但新计划不会退回它。对于普通线性入口，`input_` 是经签名确认的 activation；不是仅凭“第一个浮点 Tensor”判定。运行时保护只复核选中的参数，不对所有同地址参数做自动替换。

请求期间改变模块树后，旧 Inspect surface 会被拒绝，需重建 epoch。模型指纹包含实际 forward 源码和 backend 源码信息；它是绑定完整性校验，不是签名认证或远程安全凭证。

## 执行与三个不同预算

流程为：

```text
结构候选 → 官方固定请求观察 → 自动绑定计划
                         ↓
clean baseline → disabled → IDENTITY → exact recovery
                         ↓
baseline → 当前计划 FakeRequant 请求 → finally disarm/remove → recovery
```

同一个 `TargetPlanExecutor` 消费不同模型的明确计划，不解释模型名、不展开通配角色。部署／HTTP／NPU_LOCK 生命周期仍交给已有 owner。

当前提供的 CLI owner 适配是已验证的双卡 18031 实验环境，不是通用模型启动器。换 DeepSeek 不仅需要规则，还需要实际服务、现场绑定资格和对应部署 owner。不会为了让实验跑起来修改其他服务的启动参数。

区分：

1. 注入窗口：隔离实例中单个在途请求，逐请求 arm，正常结束即关闭；不是任意并发服务的 request-local 隔离。
2. 安全上限：`模块数 × (max_tokens + prompt_token_upper_bound + 16) × 2`，上限不可超过现有 reference 保护范围 `2^21`；超出则拒绝计划，不静默截小，也不触顶归零。
3. 日志预算：单事件只保留有界首尾 32 条，数值附加统计每模块首个及每 64 次采样一次。日志采样不控制注入次数。

以当前 C3 目标集合及原预算为例，自动算出的 cap 是 567808，4096 不再是新入口的实际剂量上限；不补 padding 请求。

倍率是量化步长倍率：小于 1 主要收窄动态范围、增加裁剪；大于 1 主要使步长更粗、细节归零。`multiplier=1` 仍是 QDQ，绝不能代替独立 IDENTITY。首版计划只允许 `REFERENCE_REQUANT + tensor_absmax`，不将未知原生 scale 静默替换成参考 scale。

## 证据与状态

逐目标保留：`discovered / role_resolved / observed / bound / identity_pass / lossy_mechanism_pass / visible_effect_pass`，不压成 `supported=true`。结构报告中的后三项默认 false；后续验收证据引用同一 `plan_sha256`，不会反写历史报告。

运行门禁分别给出：

- `mechanism`：arm/disarm、真实参数合同、计划参数一致、rank 策略、changed elements、逐模块候选和命中。
- `target_call_window`：观察到的目标调用是否全覆盖，是否碰到安全上限。
- `full_request_window`：还需要来自独立 scheduler execution-step 的请求／epoch／逐目标计数证明。

**当前没有新增 scheduler observer，故新工具不会仅靠 hook 比例宣称 FULL_REQUEST_WINDOW_PASS。** 没有独立执行步证据时，此项为 `UNPROVEN`。不能从 `shape[0]=1` 猜 decode，也不能把某个 hook 一次命中当作所有 decode 步都覆盖。

输出效果沿用已有 `EvaluationResult` 与 Dashboard。原枚举没有 `PARTIAL`，所以综合未验收时保留 `status=UNKNOWN`，在 `model_targeting.status` 中记录 `PARTIAL`，不新增第二套评分 schema。短样本 B/F/R 只给该样本结果；不宣称 Accuracy 或跨模型稳定退化。

## 使用入口

仅在原实验完成、资源实际空闲并通过项目锁获取独占后，用现有 supervisor 启动新 epoch；不能热替换旧进程的源码。该 supervisor 负责结束时恢复服务和释放锁。

新 epoch 使用既有组合扩展入口：

```bash
export VLLM_SERVER_DEV_MODE=1
export ASCEND_FI_REFERENCE_SHA256=56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7
# 在当前环境已验证的启动参数中选择一个 extension；不是完整启动命令。
--worker-extension-cls ascend_fi.inspect.experiment_extension.ExplorationWorkerExtension
```

单独只读查看结构／已存在的观察结果：

```bash
python tools/inspect_model_targets.py --base-url http://127.0.0.1:18031 --out "$NEW_OUT/read_only"
```

完整自动目标入口：模块名、层数与输入位置无需填写；`SESSION` 必须是现有 supervisor 已持锁的独立目录，`SUITE` 必须是已有官方 MT-Bench 固定套件，不是临时提示词。

```bash
python tools/run_model_target_plan_live.py \
  --session "$SESSION" --suite "$SUITE" --out "$NEW_OUT" \
  --roles ATTN_QKV ATTN_OUT MLP_GATE_UP MLP_DOWN \
  --multiplier 20 --max-prompt-tokens 69 \
  --identity-probe mtbench_82_turn0 --probe mtbench_149_turn0
```

上例的 `69` 仅来自当前已冻结套件的 prompt 上界，不是通用模型常量。更换模型／模板时须重新测量并记录，不能直接复用。工具还可通过 `--plan` 消费当前 epoch 导出的计划，但仍重新执行 IDENTITY，不信任 JSON 中手填的通过标志。

## 本轮交付与下一步

分支 `feature/interactive-inspect-v1`，HEAD `a9b67095c1f0104744ba79734886effbc6c6651b`。保留原有未提交改动，无 pull／rebase／push。原始 reference 文件 SHA256 再次只读核验仍为上述固定值；当前 `_requant_tensor` 与 C3 打包源码 AST 相同。

本地证据位于 gitignored 的 `artifacts/inspect/private/model_aware_v1_20260910/`：历史输入副本、每次独立 audit 结果与本地源码 manifest。`audit_01` 的 MLP 缺口也保留，没有覆盖失败记录；后续回放修正规则后目标精确一致。实测源文件哈希以 `live_result/source_manifest.json` 为准，不以 HEAD 代替未提交源码快照。

### 2026-09-10 新入口真实验收

环境：Qwen3-32B-w8a8-nopdmix，Ascend 910B4／A2，CANN 8.5.1；TP=2，物理 2／5 对应逻辑 rank0／1；已验证的 eager 配置，localhost:18031。torch `2.9.0+cpu`，torch_npu `2.9.0.post1+gitee7ba04`，vLLM `0.18.0+empty`，vLLM-Ascend `0.18.0rc1`。

新入口自动读取当前 64 层、4 类角色各 64 个模块，生成明确的 256 个目标，而不是由 runner 硬填层数和完整路径。输入为当前模块签名中的 `input_`，本次实际调用绑定 `args[0]`，使用显式 `tensor_index=0`，不是 `first_float`。

- Plan SHA256：`62e15c5e29caf7d5ee4c939c2b3f58dbeb93fb87c6b61dc19536925307aa20f0`。
- 策略：`REFERENCE_REQUANT / tensor_absmax / multiplier=20 / rank_filter=[0] / interval=1`。
- 角色：`ATTN_QKV / ATTN_OUT / MLP_GATE_UP / MLP_DOWN`；完整名称和输入合同保存在计划中。
- 安全 cap：`256 × (1024 + 69 + 16) × 2 = 567808`，未触顶。
- 官方套件：FastChat revision `587d5cfa1609a43d192cedb8441cac3c17db105d`；第 82 题做无损门禁，第 149 题做正式有损对照；原 prompt 和生成合同不改。
- epoch：rank0 `67ac9c3605b24fac8fc796bee8523067`；rank1 `05d6a731c8f24a4298bfde88850c5e87`。这些计划不可直接重放到已恢复服务。

| 验收项 | 本次真实结果 |
| --- | --- |
| 当前 epoch 绑定 | 两个 worker 各 256 个目标，无缺失和额外目标 |
| IDENTITY | rank0 hit=34560、changed=0；rank1 hit=0；baseline／disabled／identity／recovery 均 134 token 且全文／pieces 精确一致 |
| 有损机制 | rank0 candidate=hit=262144；rank1 hit=0、rank_filter 拒绝=262144；输入合同违例均为 0 |
| 逐模块族覆盖 | 四个族分别 hit=65536；全部 256 个目标实际命中 |
| 数值效果 | changed elements=1850723432（累计 activation 元素事件，不是不同权重数） |
| 观察到的目标调用窗口 | `target_call_window=PASS`，未触顶，无未解释拒绝 |
| 独立执行步完整窗口 | `full_request_window=UNPROVEN`；没有 scheduler 侧独立覆盖核对 |
| 正常／故障／恢复 | 951／1024／951 token；HTTP 均 200；stop／length／stop |
| 首次输出分叉 | token-piece 下标 0，即第一个 piece；未提供整数 token IDs，保存为 null |
| 规则告警 | repetition、lexical_collapse；其他覆盖规则未告警 |
| 恢复 | 全文和 token pieces 精确一致；故障与 hook 移除；服务健康、锁释放 |

真实数值统计仍为既有有界聚合：每模块首个及每 64 次合格调用做附加采样。四族采样新增归零比例依次约为 QKV 97.20%、Out 93.00%、Gate-Up 88.71%、Down 98.24%；相对 L2 误差约为 0.590、0.412、0.525、0.318。不能将这些采样比例冒充全量测量。clip_rate 本次为 0，与高倍率步长变粗的机制一致。

第 149 题正常输出开头：

```text
Machine learning (ML) is a subset of artificial intelligence (AI) that enables systems to automatically learn and improve from experience without being explicitly programmed.
```

同题故障输出开头（原文摘录）：

```text
ediren

�

 to  -,edirin,diri
diri()
diri
edir,diri diri
edir,dir,dir,diri diri diri diri diri diri diri diri
```

后续大量重复 `diri`。这里的可见异常是相对于正常解释的循环／词汇坍缩，**不是仅凭 `finish_reason=length` 判断**。本次是倍率与覆盖同时改变的组合效果，不归因于某个单独模块，也不报告开放题 Accuracy 下降。正式有损周期 N=1，不声称稳定性或跨模型可迁移的有效剂量。

证据位置：

```text
远端：/home/haoran/inspect_20260910_modelaware_v1/
本地：artifacts/inspect/private/model_aware_v1_20260910/
  live_result/
    source_manifest.json / source_verification.json / remote_tests.json
    final_environment.json
    model_target_acceptance/
      structural_reports.json / model_target_reports.json
      target_plan.json / identity.json / pair.json
      raw/                         # 全部请求、响应、token pieces、命中与数值记录
      effect/                      # 原始导出保留；其 MD/HTML 曾误用旧 R1 描述
  live_summary.json
  effect_corrected/                # 当前应阅读的 Dashboard / Effect Card
  post_live_verification.json
```

最终审计发现旧 dataset 报告模板写死了单模块族、倍率 0.25、cap=4096。已修成读取实际 `fault_configuration`，并显式显示 `full_request_window=UNPROVEN`。只离线重新渲染到 `effect_corrected/`，原始报告、请求和评分不覆盖；新旧 `EvaluationResult` 语义完全一致，检测器与阈值不改。报告源码变更发生在 live 结束后，单独记录哈希。

待补的真实验收：

1. 补独立执行步覆盖证明；没有该证据前不提升完整窗口结论。
2. 原生 DeepSeek 实例现场识别与覆盖审计，等待具体模型／服务；Distill-Qwen 不能代替。
3. 对已通过的实际变体最小补适配合同，再完成无损和有限有损验收。
4. 不同层数目前只有离线测试，仍需另一真实模型验证。
5. 最后才在相同正式数据集上做倍率／覆盖搜索，并用未参与选配置的样本确认效果。

本轮不声称 `FAKE_REQUANT_MODEL_AWARE_AUTO_TARGETING_V1_LIVE_COMPLETE`。
