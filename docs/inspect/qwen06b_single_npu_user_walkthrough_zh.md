# Qwen3-0.6B 单卡：新用户如何识别、注入和评估

这是一条完整的工具使用路径，不是手写一组模块名后直接装 hook，也不是只测一个最小 single-hit。先用真实服务识别模型与输入，生成关闭状态的 TargetPlan，再通过同一计划的 IDENTITY 和有损对照，最后用已有 EvaluationResult / Dashboard 展示效果。

本轮环境：SSH `root@123.60.231.86:2123`；模型 `/home/gyl/models/Qwen3-0.6B`；物理 NPU 5 → rank0，TP=1；28 层，BF16 权重。镜像为服务器已有 `d519927a2b4d`，未下载新镜像、未修改其他容器或模型文件。容器内 CANN toolkit/ops 的安装记录均为 8.5.1。

## 1. 安装和启动：明确依赖，不用旧双卡实验目录

需要完整源码布局 `src/ tools/ profiles/`，以及已经冻结的参考实现：

```text
fake_requant_reference_extension.py
SHA256=56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7
```

参考实现按研究提交固定纳入本仓库的 `experiments/fake_requant_reference/source/`，不是本次重写的算法，不再依赖其他工作树当前内容。见[依赖与执行环境](../development/workspace_and_dependencies_zh.md)。新部署应将整个发布源码放到明确的Linux路径；下面的历史服务器路径只是本次已验证示例。

先将本次发布源码部署到明确的Linux目录。以下是新部署示例，不依赖旧演示client或旧实验结果；实际路径可以自行指定：

```bash
export DEMO_ROOT=/home/gyl/demoncat_demo
export DEMO_CODE=/home/gyl/demoncat_source_baseline_v1
```

再次演示应创建新 session，不能覆盖本轮 `run_02`：

```bash
export DEMO_SESSION=$DEMO_ROOT/run_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$DEMO_SESSION/reference"
cp "$DEMO_CODE/experiments/fake_requant_reference/source/fake_requant_reference_extension.py" "$DEMO_SESSION/reference/"
sha256sum "$DEMO_SESSION/reference/fake_requant_reference_extension.py"

python3 "$DEMO_CODE/tools/private_demo_service.py" start \
  --session "$DEMO_SESSION" --source-root "$DEMO_CODE" \
  --model /home/gyl/models/Qwen3-0.6B --device 5 --port 18131 \
  --image d519927a2b4d --lock-root /home/gyl/demoncat_coordination
```

仅在 NPU 空闲、项目 NPU_LOCK 可独占获取时执行。启动器不停止其他服务；只使用本地镜像。服务只监听 `127.0.0.1`，生成、Inspect、collective RPC 均需 session API key。密钥不打印、不放进报告。

启动配置为 TP=1、eager、开发者模式、禁用 prefix cache / async scheduling、max_num_seqs=1。它装载 `ExplorationWorkerExtension`，但故障默认关闭。没有 Fault20 native preload，没有把 BF16 模型改成 W8A8。

启动后 `deployment.json` 记录真实容器名、模型、卡映射、引用源码哈希和资源 lease。记录中的 `reference_arm` 是构建新计划的配置模板，不是声称复用旧模型的实际 hit 数。

进入打印出的容器；以下命令在该容器内运行，`DEMO_CODE` / `DEMO_SESSION` 设置为上述宿主机相同绝对路径。模型和代码按只读方式挂载，结果目录可写。

## 2. 只读 Inspect：先知道模型里有什么

```bash
python "$DEMO_CODE/tools/interactive_inspect.py" status \
  --base-url http://127.0.0.1:18131 \
  --api-key-file "$DEMO_SESSION/access.token" \
  --output "$DEMO_SESSION/inspect_status.json"

python "$DEMO_CODE/tools/inspect_model_targets.py" \
  --base-url http://127.0.0.1:18131 \
  --api-key-file "$DEMO_SESSION/access.token" \
  --out "$DEMO_SESSION/01_inspect"
```

不带 `--roles` 的 Inspect 不 install/arm，也不自动生成有损请求。主要查看 `model_target_reports.json`：

- descriptor：实际模型类、config、运行时版本、epoch、rank、TP/PP/DP、eager。
- targets：角色、真实模块路径、模块类、是否 observed/bound，以及 blockers。
- 不把“模块存在”当作“可以注入”；未观察真实输入之前通常仍是未绑定状态。

本轮实测枚举 427 个模块，四类投影各 28 个，共 112 个：

| 角色 | 实际后缀 | 本轮典型输入特征维数 |
| --- | --- | ---: |
| ATTN_QKV | self_attn.qkv_proj | 1024 |
| ATTN_OUT | self_attn.o_proj | 2048 |
| MLP_GATE_UP | mlp.gate_up_proj | 1024 |
| MLP_DOWN | mlp.down_proj | 3072 |

角色由实际模型/父类/模块类共同判定。融合 qkv 和 gate_up 各只绑定一次，不同时挂它们的逻辑子投影；不挂父模块导致重复注入。

## 3. 真实输入观察 → 导出关闭的计划

本次使用已有 MT-Bench 首轮原题 82、86、149，预算改为本演示独立固定的 512 token，temperature=0、seed=42、关闭 thinking。没有修改题目、检测阈值，也没有用 LLM judge。

首次先准备 suite：`--source-suite` 指向发布中固定的公开原题快照，不要求旧实验目录。输出是新的模型专属suite，不覆盖来源。

```bash
python "$DEMO_CODE/tools/prepare_private_demo_suite.py" \
  --source-suite "$DEMO_CODE/profiles/evaluation/qwen06b_demo_source_suite.json" \
  --model /home/gyl/models/Qwen3-0.6B \
  --max-tokens 512 --out "$DEMO_SESSION/suite.json"
```

本模型实际模板 token 数为 58、42、44；取最大值 58 作为保守预算输入。更换模型/模板/数据时重新计算，不沿用此数值。

```bash
python "$DEMO_CODE/tools/run_model_target_plan_live.py" \
  --deployment "$DEMO_SESSION/deployment.json" --session "$DEMO_SESSION" \
  --suite "$DEMO_SESSION/suite.json" \
  --roles ATTN_QKV ATTN_OUT MLP_GATE_UP MLP_DOWN \
  --multiplier 20 --max-prompt-tokens 58 \
  --identity-probe mtbench_82_turn0 \
  --probe mtbench_82_turn0 mtbench_86_turn0 mtbench_149_turn0 \
  --discover-only --out "$DEMO_SESSION/02_bound_plan"
```

这一步只运行正常请求采样，不做数值修改。`target_plan.json` 应显示 `DRAFT / enabled=false`，112 个目标均 observed/bound、无 blockers。本次输入为二维连续 BF16，语义参数 `input_`，selector 为 `tensor_index=0`；不是 first_float 猜测。

新用户可以在此查看/审查完整目标列表和参数。改变角色或倍率，应重新生成独立计划，并重新做该计划的 IDENTITY；不要直接修改 JSON 后冒充已验收计划。

## 4. 用户选择的故障配置是什么意思

本次选择四模块族、全部实际层、rank0、倍率 20，不使用 4096 作为剂量上限。

```text
base_scale = absmax(input) / 127
fault_scale = max(base_scale, eps) × 20
q = clamp(round(input / fault_scale), -128, 127)
faulty_input = (q × fault_scale).to(original_dtype)
```

输入最终仍是 BF16，真实线性层继续执行；权重和原生量化参数不变。这个参考 INT8 Q/DQ 的粗步长模拟的是激活精度损失，不是 packed INT4 kernel，也不是训练/压缩模型。上式仅帮助读配置，精确dtype、舍入、eps及CPU辅助实现差异以[算法说明](../architecture/algorithms_zh.md)为准。

每请求有限硬上限：`112 × (512 + 58 + 16) × 2 = 131264`。覆盖的是观察到的目标调用；触顶记为不完整，不自动清零。倍率 1 仍有损，不可替代 IDENTITY。

## 5. 执行：同一计划的门禁与三题对照

```bash
python "$DEMO_CODE/tools/run_model_target_plan_live.py" \
  --deployment "$DEMO_SESSION/deployment.json" --session "$DEMO_SESSION" \
  --suite "$DEMO_SESSION/suite.json" \
  --plan "$DEMO_SESSION/02_bound_plan/target_plan.json" \
  --identity-probe mtbench_82_turn0 \
  --probe mtbench_82_turn0 mtbench_86_turn0 mtbench_149_turn0 \
  --out "$DEMO_SESSION/03_evaluation"
```

先执行 normal → installed-disabled → IDENTITY → removed-recovery，要求全文/token pieces 完全一致、changed_elements=0、全部目标实际命中。随后每题执行独立 B/F/R：baseline → install/arm → fault → finally disarm/remove → recovery。

新计划或新服务 epoch 不继承旧 IDENTITY 通行证。门禁失败即停，不静默删除无法绑定的目标，也不继续下一题。调用者不能用单纯手动 arm 绕过此入口。

## 6. 怎么评估：不是只看输出“像不像坏了”

| 层次 | 本次检查 | 不能混淆的边界 |
| --- | --- | --- |
| Mechanism | 实际目标、rank、candidate/hit/reject、每模块覆盖 | 装 hook 不等于生效 |
| 数值 | changed_elements、clip_rate、scale 范围、采样新增归零与相对 L2 | 采样统计不冒充全量 |
| Output | 连续原文、首个 token-piece 分叉、长度、finish_reason | 分叉不自动等于严重病态 |
| Phenotype | 原 V2 规则中的 repetition、lexical collapse、spam 等 | length 截断、合法结构不自动判异常 |
| Recovery | fault disabled、hook removed、同题原文及 token pieces 精确恢复 | health 200 不足以单独证明恢复 |

Dashboard 沿用现有结果合同，不建立另一套指标。没有整数 token IDs 时保存 null，比较 token pieces。这里只有三道开放题一个周期，Accuracy/官方 MT-Bench 分数、PPL、统计稳定性不应被捏造。

离线生成中文连续文本报告，并重放同一评分器：

```bash
python "$DEMO_CODE/tools/summarize_private_demo.py" \
  --evaluation "$DEMO_SESSION/03_evaluation" \
  --out "$DEMO_SESSION/04_user_report"
```

查看 `04_user_report/demo_walkthrough_zh.md`、`numerical_effect.json` 和 `effect/effect_dashboard.html`。HTML 可下载到 Windows 后本地打开，不需要暴露开发 RPC。

`full_request_window` 仍保留 UNPROVEN：逐目标 candidate=hit 可以证明目标调用覆盖，但当前未新增独立 scheduler token/执行步 observer。

## 7. 结束与再次演示

runner 正常结束会清理注入并验证服务健康，但服务资源 lease 属于启动器。不要在 driver 仍运行时 stop。

全部完成后退出容器，在宿主机执行：

```bash
python3 "$DEMO_CODE/tools/private_demo_service.py" stop --session "$DEMO_SESSION"
```

只停止本 session 所有的命名容器；保留结果/日志、停止后的容器和代码，确认停止后释放自己的 NPU_LOCK。其他容器不动。再次实验使用新 session、重新 Inspect/绑定/IDENTITY，不复用旧地址或 epoch。

当前交付为源码布局工具，不声称是任意环境下一条 pip 命令即可完成的发行版。首次接入如遇 CANN/vLLM/模块签名变化，保留失败证据并修部署适配，不改 FakeRequant 数学或放松绑定门禁。

## 8. 本轮实际结果与交付位置

本次使用 `run_02`，服务 epoch 为 `cfe39c6d5d964321856be512025960cd`；已停止的容器为 `demoncat-qwen06b-d46720b758`。计划 SHA256 为 `ec4e332a54ddd5c82caac8f5a08c26378d30a7e7699d2450b2f9e56486c3d287`。它们只是本轮证据，不能当作新服务的绑定配置直接复用。

| 样本 | 正常/故障/恢复 token 数 | 实际 hit | 改变元素比例 | 首个分叉（0 起始） | 精确恢复 |
| --- | --- | ---: | ---: | ---: | --- |
| 82：反馈邮件 | 139 / 512 / 139 | 57344 | 99.74% | 0 | PASS |
| 86：市场描写 | 106 / 244 / 106 | 27328 | 99.82% | 0 | PASS |
| 149：解释题 | 512 / 512 / 512 | 57344 | 99.78% | 0 | PASS |

每题 112 个目标均实际命中，candidate=hit、reject 为空，均未触及 131264 硬上限。裁剪率都是 0；粗步长造成的新增归零与相对误差仍然明显。归零比例和相对 L2 使用每模块第 1、65、129…次调用的采样，详细分母见 `numerical_effect.json`。

第 82 题从正常邮件变成数字、标点、多语碎片混杂；第 86/149 题出现重复/词汇坍缩/标点规则告警。第 149 题 baseline 自身达到 512-token 预算，该终止状态没有被单独当作推理故障。本轮规则、阈值未调整；不将三题效果外推成通用模型准确率损失。

服务器路径：

```text
/home/gyl/demoncat_demo_qwen06b_20260911/
  client/                            # 可复用工具与本文
  client_live_v1.tar.gz               # 实测时源码快照
  reference/                         # 哈希固定的原 FakeRequant
  run_02/
    01_inspect/                      # 只读识别结果
    02_bound_plan/                   # 正常请求观察 + 关闭状态计划
    03_evaluation/                   # IDENTITY、原始 B/F/R、EvaluationResult
    live_source_audit.json           # 源码/runtime/模型文件/原始证据哈希
    service_stopped.json             # 安全停止与锁释放记录
    service.log
  user_report_v1/
    demo_walkthrough_zh.md           # 三题完整连续原文
    numerical_effect.json
    effect/effect_dashboard.html
```

初次启动曾因宿主机旧 Python 缺少 `shlex.join` 失败，尚未创建容器/运行模型；失败目录与锁归还记录已保留，修复为兼容拼接后使用独立 `run_02` 重试。当时收尾没有遗留 armed fault，NPU 进程为空，原 `gyl_test` 保持运行；这是当时记录，不是本次实时服务器检查。实测来源是 `b8b167d709271296b8114b42a627d3a7827d03b1` 加工作区修改，随后冻结为 `f67b4186b7ed5652d49ac7c40aa9cc762adfeaf1`。当前正式入口见[源码基线发布](../releases/source_baseline_v1/README.md)，不再依赖演示工作树；实测时226项离线测试通过。启动早期留锁／remove失败的现有边界见算法说明，未在本次文档收束中修复。
