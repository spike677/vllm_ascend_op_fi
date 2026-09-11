# FakeRequant 新旧 hook 路线与清理证据边界

核对日期：2026-09-11。核对基线：`0dadd3b12404920f6abc7345e0a7884b6b52ad4b`。
本说明是只读源码审计，不是一次异常卸载实机验证，也不新增生命周期实现。

## 1. 结论：当前仍是模块输入 hook

当前 FakeRequant 使用 `register_forward_pre_hook(..., with_kwargs=True)`，对指定 activation 做 Q/DQ，再返回替换后的 `args/kwargs`。原模块继续执行。它没有改成 LD_PRELOAD，不修改模型权重，也不替换整个原生量化 kernel。

新增部分主要是目标发现、输入合同、计划与实例身份验证，以及调用既有参考实现的编排。实际数值逻辑仍来自[固定 reference](../../experiments/fake_requant_reference/source/fake_requant_reference_extension.py)，SHA256：

```text
56097835861bb441232663380eed2953acd97f3aa03c49262368dce6defeaca7
```

PyTorch 的 pre-hook 在模块 forward 前执行；启用 `with_kwargs` 时，修改输入需要返回新的 positional/keyword 参数对。注册返回可移除 handle。[PyTorch 2.9 Module 文档](https://docs.pytorch.org/docs/2.9/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook)

## 2. 旧内源路径：区分历史描述与已核对源码

用户提供的历史描述是：

```text
activation / fake_requant
→ RuntimeBinding 选择 hook
→ FaultHandler 安装与管理
→ activation Q/DQ
→ 原模块
→ clean / unregister
```

历史描述还提到 `catalog.py`、`bindings.py` 与持久 `fault_state`。本次在 canonical 仓库及 C 盘 legacy clone 的可见 Python/Markdown 中，未找到这些旧内源实现的完整源码；legacy clone 没有对应的 `src/` 和 `experiments/` 源树。因此：

- 上述旧流程标记为 **USER_HISTORY_NOT_SOURCE_VERIFIED**，不能据此断言旧实现所有异常分支均已闭合。
- 当前仓库[合入边界文档](../overview/demoncat_merge_contract.md)说明这些产品职责由隔离 DemonCAT 项目持有，不等于这里已经包含它们的实现。
- 无法给出“旧版成功、新版失败”的对照结论；本次没有复现卸载失败。
- 机制同为 hook 不代表数值策略相同。当前验证路线是 `REFERENCE_REQUANT + tensor_absmax`；历史 runtime scale 描述不构成两者数值等价证明。

## 3. 当前源码对应关系

| 位置与符号 | 实际职责 |
| --- | --- |
| [session.py](../../src/ascend_fi/inspect/session.py) `InspectSession.observe_start/_observe/stop` | 只读观察 hook 记录真实参数 metadata；回调返回 `None`，不替换输入。保存 handles；移除失败的 handle 留在 pending 列表。 |
| [input_contract.py](../../src/ascend_fi/inspect/input_contract.py) `PlanInputContractMixin.fake_requant_bind_target_plan` | 单独的前置合同 guard，`prepend=True`；核对本次输入并返回 `None`。不是数值注入。 |
| [reference](../../experiments/fake_requant_reference/source/fake_requant_reference_extension.py) `FakeRequantReferenceExtension._state` | 将 `ExperimentState` 持久挂在 worker/extension 的 `_fake_requant_reference_state`，不依赖单次 RPC 临时对象。 |
| 同文件 `fake_requant_install` | 遍历匹配模块，保存包含模块与 handle 的 `HookBinding` 到 `state.bindings`；安装后默认 disabled。 |
| 同文件 `fake_requant_arm/_apply_pre_hook` | arm 配置并重置计数；运行时按 rank、模块、层、selector、interval、有限 hit budget 筛选。 |
| 同文件 `_select_tensor/_requant_tensor/_replace_tensor` | 找到明确输入，计算新 activation，将它放回复制后的参数容器；不是原 tensor storage 上的直接修改。 |
| 同文件 `IDENTITY` 分支 | 记录零变化统计后返回 `None`，原参数原样继续；不是 `multiplier=1` 的近似替代。 |
| [combined.py](../../src/ascend_fi/inspect/combined.py) | 启动时核对 reference 文件 hash，再继承该实现；不做运行时动态换库。 |
| [experiment_extension.py](../../src/ascend_fi/inspect/experiment_extension.py) `ExplorationWorkerExtension` | 复用 reference 的有损输出，仅补有界诊断统计。 |
| [plan_executor.py](../../src/ascend_fi/inspect/plan_executor.py) `TargetPlanExecutor` | 当前实例核对、IDENTITY 许可与运行编排；具体安装、arm、清理由 owner 执行。 |

## 4. disarm、remove 与 owner 的确认缺口

以下事实针对上面的核对基线，属于工程风险分析，不是故障复现记录。

`fake_requant_disarm()` 只将 `enabled=False`、`config=None`，保留已安装 hook。随后 hook 仍可能被调用并计数，但不执行 Q/DQ。

`fake_requant_remove()` 先复制 bindings 并清空状态，再逐 handle 调用 `remove()`。异常收集到 `remove_failures`，以 `ok=False` 返回。由此不能仅用 `query.installed=False` 推出物理 hook 已全部移除；该 query 读取的是已经清空的状态列表。由于 enabled 已关闭，残留 handle 也不自动等于“仍在有损注入”。

安装中的异常回滚会逐个移除已注册 handle，但该回滚循环没有为每个 `remove()` 独立保底。若回滚本身再次异常，完整清理也不能只凭安装失败消息推断。

[VerifiedOwner.disarm_and_remove](../../tools/run_normal_diagnostic_live.py) 在核对基线中捕获 RPC 异常，但没有逐 worker 将 reference 返回的 `ok=False/remove_failures` 提升为失败；输入合同 guard 的清理则检查 `ok` 和 `handles=0`。[PrivateServiceOwner](../../tools/private_service_owner.py) 继承这一 owner，RPC transport 确认 HTTP 和 results 结构，并不等于每个 worker 的 remove 成功。

`fault_request()` 的 `finally` 尝试 disarm 并保留结果；`TargetPlanExecutor.run_pair()` 的 `finally` 委托 owner 清理，然后检查 clean state。它们构成已有正常流程保护，但不能宣传为“任何异常之后物理 hook 都已确认移除”。

PyTorch `RemovableHandle.remove()` 从 hook 注册表和关联字典删除对应 ID；它不会回滚已经产生的 activation、KV state 或输出轨迹。因此，移除动作与模型行为恢复也应分别验证。[PyTorch 2.9 RemovableHandle 源码](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/utils/hooks.py)

本轮保持现有 disarm/remove 尝试与日志；没有据此重建生命周期系统。发生异常或无法确认清理时，不在同一实例继续尝试后续有损请求，应按已有部署流程结束该实例。后续若增强返回值检查，需单列代码与验证记录，而不是回写历史实测结论。

## 5. 一次有损推理后结束实例：三种结果不能混用

| 实际动作与证据 | 正确报告名称 | 不能推导 |
| --- | --- | --- |
| 同一 worker 内移除 hook，随后同协议推理与 baseline 对比 | 同进程卸载后恢复；按实际相等性记录 PASS/FAIL | 所有异常热卸载都可靠 |
| 确认旧模型 workers 完整停止，启动新 workers，重新推理正常 | 重启后正常对照 | 热卸载恢复 PASS |
| 只确认模型 workers 已停止，没有再启动与推理 | 实例已停止；文本恢复未测 | 恢复输出精确一致 |

只重启 HTTP 前端，或同 Python worker 内重新加载权重，都不能替代完整 worker 结束确认。停止记录应关联旧实例身份；重启后的健康、epoch 和文本对照属于新的记录。

新 epoch 必须重新 Inspect/绑定并完成该实例的 IDENTITY。`TargetPlanExecutor.rebind()` 核对服务 epoch、runtime/model fingerprint 与 worker 集合；IDENTITY receipt 由当前 executor 建立，不能把旧 JSON 中的 `PASS` 当作新实例许可。手选/显式全选改变的是计划目标范围，不应绕过这一约束。

推荐单位仍为：当前实例计划预览 → 同 epoch IDENTITY → 一次有损请求 → 保存原文和证据 → 尝试 disarm/remove 并记录 → 完整停止该模型实例。异常热卸载完备性后置，不改变 FakeRequant 算法，也不构建第二个注入器。

相关定义见[算法说明](algorithms_zh.md)与[单卡用户手册](../inspect/qwen06b_single_npu_user_walkthrough_zh.md)。
