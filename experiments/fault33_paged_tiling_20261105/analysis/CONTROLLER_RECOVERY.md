# 不确定容器创建结果的恢复补丁

**正式模型作业已使用恢复补丁，四组均正常创建、执行和清理。** `controller.py` 的 SHA256 为 `54625a941d004eff891988a0f4d98c658253426e3fb2f83054b5e15a381991d0`，与评审过的补丁候选相同。恢复逻辑的 14 项 CPU 测试在模拟时钟和 Docker 后端上执行。

原 `controller_recovery_proposal.json` 和 `controller_recovery_cpu_report.json` 保留提出补丁时的 `PROPOSED_NOT_APPLIED` / `patch_applied=false` 历史状态，其base SHA不是当前controller SHA。旧operator attempts的冻结controller snapshot、原completion和迟到创建的人工核验/清理记录均保留；不追改旧attempt，也不将旧snapshot解释为已含新逻辑。

旧控制器在 `docker run -d` 的 40 秒调用超时后，只按随机 token 和任务标签查询一次 `docker ps -a`。一次空结果触发提前返回，清理记录可显示查询已完成，容器实际是否已经创建仍未知。作业 `fault_restore_a4_a2` 曾在创建超时后出现迟到的 Created 容器；该作业未执行 case，随后逐项核验容器归属并移除，原失败回执保留。

`controller_recovery.patch` 只修改三处：新增恢复窗口常量、替换uncertain恢复函数、保证该函数异常时仍保存uncertain receipt。正常docker-run成功后已有ID的路径保持原样。

- 恢复总窗口60秒，poll为0.5秒；每次ps/inspect调用上限为5秒与总剩余时间的较小值。
- 空ps仅记录 `NO_MATCH_OBSERVED` 并重试，不证明不存在。
- 仅接受唯一完整64字符ID，再通过既有 `verify_owned` 核对完整ID、固定image、exact task/token/name。多候选、无效ID、inspect歧义、身份不符均拒绝，不到达stop/rm。
- daemon/inspect瞬时错误保留逐次记录并在窗口内重试。窗口耗尽或迟到响应超出deadline仍为 `UNRESOLVED_CREATE_OUTCOME`；owned_container_removed明确为false，并通过既有cleanup error收敛为 `FAIL_CLEANUP`。
- `uncertain_create_recovery.json` 在成功与异常两条分支均保存。恢复receipt仅证明ownership恢复，`cleanup_claimed=false`；后续真正的stop/rm和最终completion独立记录。未解决状态保留token/task/name/expected image、候选及逐次观察，供下一次严格身份复核。

CPU测试从reviewed candidate中提取恢复函数和实际main-finally AST执行，时间与Docker backend均为mock。包含多个空快照后迟到Created、迟到Running、无匹配耗尽、查询timeout共用总deadline、inspect/daemon瞬时错误、晚于deadline的响应、无效/多ID、inspect歧义，以及ID/image/name/task/token逐项身份不符。另有旧逻辑反例和正常成功路径无recovery/no-sleep回归测试。

```sh
python3 -B -m unittest discover -s experiments/fault33_paged_tiling_20261105/analysis -p test_controller_recovery.py
```

原提案轮14项CPU mock测试通过时，controller原字节SHA为 `e244d133b35e76a7c0a620622c765f4aaae963cda8d6005998e4a0a505377392`。patch基准及candidate SHA见 `controller_recovery_proposal.json`，当时测试结果见 `controller_recovery_cpu_report.json`。采用unittest入口可复核当前精确candidate而不覆盖这份历史report；旧逻辑反例在base已被替换时按测试设计skip。

正式模型作业 `model_baseline_a4_a3`、`model_fault_r0_a4_a1`、`model_fault_r1_a4_a1` 和 `model_restore_a4_a1` 均通过独立资源及源码校验。四份冻结源码清单逐字节相同，逐组核实自有容器、运行时间、设备锁、选定物理卡上的资源样本和最终清理回执。模型验收结果为 `PASS_ADAPTED_MODEL_AND_STATISTICS`。这四次实机执行覆盖控制器的正常创建及清理路径；迟到创建及超时恢复各条分支的覆盖来自上面的 CPU 模拟测试。[四份原始审计](../../../artifacts/attention_tiling/fault33_paged_statistics_20261106/model_a4/audit/model/attempts) · [正式验收及作业哈希](../../../artifacts/attention_tiling/fault33_paged_statistics_20261106/reports/model_a4_acceptance.json) · [完整离线复核命令](../RESUME.md)

若达到恢复时限仍无法确认创建结果，控制器记录 `FAIL_CLEANUP` 和已知身份线索，供后续复核。

`launch_model_case.py` 是controller之前的有界ATK清理辅助程序，其card5限定validator的CPU安全测试见 `test_launch_model_case.py`。测试仅用inspect fixture和mock backend，拒绝分支不得执行stop或controller；辅助程序原字节保持不变。这些测试不证明外部ATK队列不会在清理后重新创建工作，controller仍执行自己的锁和反复资源门禁。

```sh
python3 -B experiments/fault33_paged_tiling_20261105/analysis/test_launch_model_case.py
```

本地13项CPU测试全部通过，覆盖exact ID、ATK name/job label、必须含card5、其余物理卡或设备、privileged/device requests/cgroup rules、root或dev bind，以及main中不合格inspect不得到达stop/controller的顺序。结果见 `launch_model_case_cpu_report.json`；helper SHA仍为 `068342cb25638230ac6fe79fdfe1a0b6508824fa05c3abe566d7f7531855b193`。
