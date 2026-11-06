# 运行交接（2026-11-06）

**a4 固定布局算子 B/F 已通过机理与独立 msprobe 验收。真实 Qwen3 适配模型的四份独立作业已通过 `PASS_ADAPTED_MODEL_AND_STATISTICS`。**

- SSH：`npu-dual`，123.60.231.86:2125；仅使用物理卡 5。
- 远端任务根：`/home/fault33-paged-tiling-20261105`。控制器持有 `/var/tmp/fault22_npu_2_5.lock`，每次检查设备门禁。
- 18 次 stock 发现、2 次 stock statistics/mix 重复已完成，公开摘要见 `artifacts/attention_tiling/fault33_paged_statistics_20261106/stock_discovery`。
- `outputs/bridge_a1` 已完成 4 次 stock/private OFF 原生调用，容器清理成功。正式验收 `outputs/bridge_a1_acceptance.json` 拒绝：`native_scope=false`。该次作业在 F 前结束。
- 原因：普通非 TND、非 rope/NZ 的 PA 分支只设置 `inputLayout_`；`inputKvLayout_` 保留 `BSH_BSND` 默认值。a3 守卫误要求它为 BNSD。a4 已按固定源码修正该内部枚举断言，保留公开 BNSD 属性、实际 K/V 形状和其他条件。
- a3 的失败、成功续跑、源码冻结、私有 stage、原始采集全部保留。a4 使用独立 `source_policy_a4`、`source_a4`、`vendor_stage_a4`；不得覆盖 a3。
- 当前运行源码已指向 a4。正式 `setup/replay_a4` 和 `setup/model_a4` 已在桥接前同时封存。
- a4 构建证据位于远端 `build/evidence_a4`；源工具位于 `source_policy_a4`，a3 的冻结路径未修改。
- a4 build/stage 均已通过，manifest SHA `f48a82e89fdf7bbfafba5ac3e05334dbde01b422c55dc726f786152db5874cfe`。
- `bridge_a4_a1` + `fault_restore_a4_a3` 已完成 10 次调用，`outputs/replay_a4_acceptance_a2.json` 为 `PASS_CONTROLLED_REPLAY_AND_STATISTICS`。正式算子公开包为 artifact 下 `operator_a4`，原报告在同级 [reports/replay_a4_acceptance.json](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/reports/replay_a4_acceptance.json)。
- `fault_restore_a4_a1` 被外部卡 5 占用挡住；a2 创建超时且未执行任何 case。迟到的本任务 Created 容器已逐项核 token/name/image/task 后移除；原始失败回执不覆盖。另按用户授权停止了当时卡 5 的 ATK 容器。
### 正式模型的四个独立成功作业

原多用例作业 `outputs/model_a4_a1` 的 B 曾返回；F 遇到外部卡 5 的 ATK 任务后，资源门禁中止了整份作业。该作业状态为 FAIL，只保留诊断材料，**其中的 B 和已产生的 F 片段均排除在正式模型验收之外**。使用修订后的控制器和同一冻结源码，从 B 重新开始，每组使用独立进程和独立作业目录。

| 顺序 | 用例 | 通过验收的远端作业目录 |
| ---: | --- | --- |
| 1 | B，OFF | `outputs/model_baseline_a4_a3` |
| 2 | F0，inner1024 | `outputs/model_fault_r0_a4_a1` |
| 3 | F1，inner1024 | `outputs/model_fault_r1_a4_a1` |
| 4 | R，OFF | `outputs/model_restore_a4_a1` |

四次控制器均记录成功、各自容器已移除、锁已释放，资源样本通过门禁。四份原始源码清单 SHA256 均为 `90c2dd280ec119793e44738738821ae0577663452d986d97e2e2ac71da14500d`，实际时间顺序互不重叠。单次启动前的资源操作记录在 `outputs/manual_launch/<attempt>.json`。依用户授权，B 前停止了 1 个经核验的卡 5 专用 ATK，F0、F1、R 启动时无停止操作。该动作发生在控制器持锁区间之前。[四份启动前回执](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/resource_prelaunch) · [控制器恢复与验证边界](analysis/CONTROLLER_RECOVERY.md)

验收器从四份完整作业各取一个成功用例，分别校验 28 个原生 IFA 窗口和全部 8 个请求。两轮 F 的 **56 / 56 次**符合条件的 decode IFA 均应用策略，两轮 OFF 共 56 次走原生选择。msprobe 两个 B/F 对照首差均为第 0 层 IFA 输出，8 个请求的 logits 数值均发生变化，8 个 top-1 均相同。模型报告 SHA256 为 `e4c3af499ea9081dba7a44e8dac3f5da0bec2de1335157d6e614325e58af8aaf`。[正式模型报告](../../artifacts/attention_tiling/fault33_paged_statistics_20261106/reports/model_a4_acceptance.json)

### 离线复核

从 124 的任务根运行下列命令。完整验收会重新读取原有完整日志、私有输出和原始统计，报告写到一个尚不存在的新文件，无需再次调用 NPU。

```sh
python3 -B analysis/accept_model.py \
  --attempt outputs/model_baseline_a4_a3 \
  --attempt outputs/model_fault_r0_a4_a1 \
  --attempt outputs/model_fault_r1_a4_a1 \
  --attempt outputs/model_restore_a4_a1 \
  --vendor-manifest vendor_stage_a4/vendor_manifest.json \
  --abi-schema build/evidence_a4/ifa_abi_schema.json \
  --output outputs/model_a4_acceptance_recheck.json
```

公开包分别保留四组原生统计、四份作业审计及一套经逐份验证的共同源码。以下命令从 Git 仓库根运行，只核对公开内容和库存哈希。重新计算逐元素 logits 差或整份原生日志覆盖需要 124 上的私有输入文件。

```sh
python3 -B experiments/fault33_paged_tiling_20261105/analysis/package_evidence.py \
  --verify artifacts/attention_tiling/fault33_paged_statistics_20261106/model_a4
python3 -B experiments/fault33_paged_tiling_20261105/analysis/package_evidence.py \
  --verify artifacts/attention_tiling/fault33_paged_statistics_20261106/operator_a4
```

### 预检及交付边界

- `outputs/model_stock_pilot_a1` 已完成无注入的真实 HF 模型预检：28 次原始 prefill SDPA、28 次 paged IFA decode、8 请求/224 层请求映射、step0 statistics/mix，控制器清理成功。日志 1,339,155,480 字节，stock logits SHA `c3fdb5deb61baf55845ec0cfd77660687c7540fbc691a2de3431f7d6e274fcf2`。这是预检，不是 B/F 验收。
- 顺序：新 stock/OFF 桥接 4 次并独立通过 → fault+同进程 OFF/ON/ON/OFF 共 6 次 → 模型 B/F0/F1/OFF 四进程、112 次 decode IFA。正式四份完成记录分别归档。
- 模型为真实 HF Qwen3-0.6B forward 加明确的分页 IFA decode 适配器，不是自然 vLLM/MindIE 调度。固定 8 个请求、16,383-token prefill、一个已知 teacher token、28 层。
- 分别验完整输出、重复稳定、关闭恢复、公共 serializer ABI 与实际 NNOP launch；`kernel_argument_raw_observed=false`。独立 msprobe 分析只读原生 dump/stack/construct/config，不用策略日志指定首差。
- 公开交付只包含审查后的代码、原生 statistics/mix 和运行元数据。私有完整张量、权重、SDK/kernel 库及完整 DEBUG 日志保留远端，不提交。

发布分支为 `codex/fault33-paged-attention-20261105`，起点为已发布的 `9a16ccc3dc2bd116a5541baf1c362d274e6f811f`。本交接以两份正式验收报告及其源清单为准。
