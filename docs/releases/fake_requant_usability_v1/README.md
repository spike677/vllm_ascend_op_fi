# FakeRequant 可解释选择与使用体验 V1：离线候选收束

日期：2026-09-11。canonical：`D:\project\demoncatpfi`；分支：`codex/fake-requant-usability-v1`。
起点实际 HEAD：`0dadd3b12404920f6abc7345e0a7884b6b52ad4b`，开始时工作树干净。未 reset、pull、rebase、push；未修改其他 worktree 或旧证据。

本轮状态：**IMPLEMENTED_OFFLINE / LIVE_UNVERIFIED**。这是工具开发候选，不是新增 Ascend 多 rank 实测认证。提交与文件 SHA 见 [source_manifest.json](source_manifest.json)，实际候选 HEAD 用 `git rev-parse HEAD` 查询。

## 1. 本轮交付

| 工作包 | 已实现与检查 | 不作的声明 |
| --- | --- | --- |
| C：目标解释与手选 | 同一清单展示 rank/layer/role/module、实际输入、selector、依据、状态、blocker/下一步；手选过滤交集、精确名称不扩展 | 无敏感度排序或最坏位置搜索 |
| C：全量与参与 rank | 显式 `all_bindable`、`--ranks all`；动态纯 TP 组；同名跨 worker 保留；同 worker 重复报告去重；安装与 arm 完整确认后才发请求 | 不代表 DP/PP/EP、任意 rank 映射或整机所有 NPU 已支持 |
| A：数值 | 同采样集合原始 E/S 派生 NMSE、SQNR、相对 L2；先加能量后求比；缺失/零/无穷/溢出合法 JSON | 不改变 Q/DQ、统计采样或病态阈值，不是三份独立证据 |
| B：检测验证 | 完整原文、原 token pieces、分组与协议分层、标签来源、区间并集；确定性回归和独立复核分母分开 | 尚无独立金标准，不作泛化认证 |
| 使用体验 | 执行前预览、既有 IDENTITY、一条有损、finally 清理尝试、宿主完整停止自有容器、统一索引和既有 Dashboard | 不把停止实例记成热卸载恢复，不构建新控制台 |
| D：可选概率 | 单窗口固定 reference token IDs/mask/身份；严格 echo logprob 对齐；NLL/PPL/ratio；写既有 EvaluationResult 的离线扩展 | 未实现 TargetPlan 概率请求事务、多窗口合并、Top-1 或现场概率注入 |

源码入口仍是同一参考实现与 `TargetPlanExecutor`。没有新增注入器、Python dispatcher patch、native fault 或模型权重修改。固定 reference 原始 SHA 仍是 `56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7`。

## 2. 真实限制与错误动作

- 当前 reference owner 在各纯 TP worker 安装同一模块名集合。所有参与 worker（包括非目标 rank）必须提供完整、同质绑定；异构或缺 worker 明确阻塞，不取交集降级。
- 当前纯 TP 支持 `global rank == module TP rank == reference resolver rank` 且组为 `0..TP-1`。本轮对其他映射显式拒绝，不把 device ID、LOCAL_RANK 或 RPC 数组顺序猜成 global rank。
- 所有模块需通过既有浮点/二维/连续布局/唯一参数/无别名输入合同。真实多别名和父子边界冲突仍阻塞；全选只包含已支持目标。
- 新计划保留每 worker arm-lifetime 有限硬上限，另记录目标 worker 上限之和；不是共享全局配额。预算耗尽不清零。
- candidate=hit 仅支持目标调用覆盖。没有独立 scheduler 执行步证据，`full_request_window=UNPROVEN`。
- 新实例必须重新观察、绑定与 IDENTITY。持久 session reservation 只是工具串行使用约束，不是对任意裸 RPC 调用的安全权限边界。
- 新宿主工具需要项目推荐 Python 3.10+、已就绪且归属明确的独占容器；不自行启动、抢锁或重试。

## 3. 恢复命名与新使用路径

```text
正常观察 → 手选／全选 → 预览 → 同 epoch IDENTITY
→ 正常对照 → 一次有损 → 保存完整原文及数值 → 尝试 disarm/remove
→ 宿主完整停止自有 worker 容器 → 释放自己的 lease
```

旧 `run_pair` / 默认 `same-process-bfr` 保留。新路径不做故障后的同进程 recovery 请求。

| 实际证据 | 命名 |
| --- | --- |
| 同 worker 移除后再推理并通过对比 | 同进程卸载后恢复 |
| 完整重启并重新正常推理 | 重启后正常对照 |
| 只证明自有容器停止 | `INSTANCE_STOPPED_TEXT_RECOVERY_NOT_MEASURED` |

EvaluationResult 在停止前生成，保留 stop pending；不事后改写。最终以 `final_instance_state.json` 为准，统一入口是 `README.md` / `artifact_index.json`。若 stop 无法确认，状态为 `STOP_UNCONFIRMED_KEEP_LEASE_NO_FURTHER_REQUESTS`，不继续请求、不假报清理成功。

异常热 remove 的物理确认缺口仍后置。源码风险与已复现故障不同，见 [新旧 hook 核对](../../architecture/hook_mechanism_compatibility_v1_zh.md)。

## 4. 离线检查与历史回放

检查命令和真实结果见 [offline_validation.json](offline_validation.json)。主体测试当轮 293 项：291 PASS、2 项因本机无 PyTorch SKIP；原参考测试 5/5 PASS。新增 C/UX 26 项中 25 PASS、1 项 PyTorch SKIP，包含两模拟 worker 的 IDENTITY→全 rank 有损编排。它们不是 Ascend 实测。

收束检查：141 项源码/配置 SHA 匹配，固定 reference 原始字节一致；9 个推荐 CLI `--help` 成功；391 个本地文档链接无失效，`git diff --check` 通过。外网链接与真实启动不由该检查认证。

原始 Qwen0.6B 报告离线全选预览得到 112 模块、1 个参与 rank、每 worker cap=131264；明确标记 `OFFLINE_REPLAY_CURRENT_EPOCH_UNVERIFIED`，不是可复用的现场 IDENTITY 许可。

历史三题在新数值报告中离线回放：

| 原题 | 采样元素事件数 | NMSE | SQNR dB |
| --- | ---: | ---: | ---: |
| 82 | 13,045,760 | 0.2980401448 | 5.2572523432 |
| 86 | 9,031,680 | 0.2160637544 | 6.6541808180 |
| 149 | 10,235,904 | 0.3546571610 | 4.5019126679 |

以上为**此前单 rank 实测的重新派生**，不是本轮新有损请求。分母是采样 activation 元素事件；公式由未舍入能量计算，表格仅显示舍入值。原 distortion score 保持完全一致，原 B/F/R、阈值、原始文件未改。

同时纠正 `summarize_private_demo.py` 新生成说明中的 tensor eps 顺序：`max(base_scale, eps) × multiplier`；只修报告文字，不改参考算法，不覆盖旧生成报告。

检测审计：12 条合成完整原文，23/23 回归断言通过；历史 0001/0002/0004 的 9 条周期记录，使用原始 token pieces 时 9/9 通过。仅使用完整文本导出但缺原 pieces 时 6/9，0004 尾循环漏报保留为已知兼容缺口，未调阈值。全部标为 regression expectation，独立质量标签仍 `INDEPENDENT_LABELS_PENDING`。

## 5. 文件与操作入口

- [手选／全选／单次结束实例](../../usage/fake_requant_manual_all_targets_v1_zh.md)：真实 CLI 和产物入口。
- [选择与 rank 合同](../../reference/target_selection_contract_v1_zh.md)：输入范围和拒绝条件。
- [数值与检测器验证](../../evaluation/usability_numerical_detector_audit_zh.md)：派生定义、独立标签边界。
- [可选 D](../../evaluation/reference_probability_offline_v1_zh.md)：严格 reference 概率解析和未实现部分。
- [历史源码基线](../source_baseline_v1/README.md)：此前 Qwen 单卡实测，不升级为本轮全 rank 结果。

新派生产物只写本地 `artifacts/inspect/private/usability_v1/`，不提交完整 raw 或私有部署配置：

```text
detector_regression_v1.json
legacy_detector_audit_v1.json
qwen06b_all_preview/            # 历史报告离线预览，不代表当前服务
qwen06b_offline_effect/         # 新 NMSE/SQNR + 原评分与连续原文
```

## 6. 本轮没有做

没有远端访问／启动／NPU_LOCK 获取／arm，现有远端状态未读取也未修改。没有 Windows NPU 适配、依赖升级、模型下载、倍率扫描、敏感搜索、阈值改阳性、故障算法变更、历史证据重写或 push。

现场下一步仅限在获授权的新实例：当前选择计划→同 epoch IDENTITY→一条有损→停止；多 rank 必须单独实测。D 与异常热卸载完备性不阻塞本轮离线候选，但不能把其未验证项写成 PASS。
