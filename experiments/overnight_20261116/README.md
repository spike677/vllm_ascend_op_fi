# 17 条故障工程整理 · 2026-11-16

本分支从核实的`bfdd6abc1adb5d08b034db7a96ec97ffbec0b9bf`建立。
**17条已审计；445项CPU测试通过；15/17/18新增NPU机制级B/F/OFF通过。
新完整预训练模型故障闭环、共享Worker故障闭环及严格统计全局首差通过均为0。**

| 交付 | 位置 |
|---|---|
| 总结、17项表、范围与失败原因 | [最终报告](../../docs/fault_closure_report.md) |
| 每项实际状态及历史范围 | [fault_matrix.json](fault_matrix.json) |
| 每项源码/模型/算子/实施方案 | [17张能力卡](../../profiles/capabilities/) |
| 实际探索与未执行部分 | [exploration_ledger.json](exploration_ledger.json) |
| 模型及环境事实 | [model_matrix.json](model_matrix.json)、[environment_matrix.json](environment_matrix.json) |
| Native、OFF、独立统计 | [native_receipts](native_receipts/index.json)、[recovery_receipts](recovery_receipts/index.json)、[independent_reports](independent_reports/index.json) |
| Worker实际发现与缺口 | [worker_receipts-a2.json](worker_receipts-a2.json)、[worker_injection_gap.json](worker_injection_gap.json) |
| CPU最终回归 | [cpu_final_receipt.json](cpu_final_receipt.json) |
| 覆盖资格 | [coverage_matrix.json](coverage_matrix.json) |
| 阻塞和下轮工作 | [blockers.md](blockers.md) |
| 发布校验 | [checksums.json](checksums.json) |

## 新 B/F 原件

根目录是124的`/home/overnight-17-20261116`，每组`capture/`保存原生统计及配置。

| 机制 | B | F | 真实step |
|---|---|---|---|
| 15 | `native15-a1/fault15_baseline` | `native15-a1/fault15_fault` | 0、1、2 |
| 17 | `native17-a2/fault17_baseline` | `native17-a2/fault17_fault` | 0、1、2 |
| 18 | `native18-a1/fault18_baseline` | `native18-fault-a3/fault18_fault` | 0、1、2 |

这三组都是构造推理模块，不是Qwen模型token步数。
固定`rtol=1e-5, atol=1e-8`，独立严格结果全部`INCOMPLETE_COMPARISON`。
局部可见变化及null说明见[诊断](STATISTICS_ALIGNMENT_DIAGNOSIS.md)。
新原始`dump/stack/construct.json`、完整数组、运行时地址及vendor二进制不上传。

## 可复核执行

`run_cpu_validation.py`执行五组本地测试；`worker_native_cpu.py`是124无设备C++/ctypes验证。
`run_cases_124.py`是实验控制器，使用既有锁和每case空闲门禁，最多8case，清理仅限本轮容器。
所有实际执行参数和冻结源码hash留在`cases_*.json`、`worker_cases_*.json`和`source_manifest-a*.json`。

`npu_mechanism_probe.py`当前文件对应source-a3：修正采集器先于故障wrapper安装的顺序。
15/18使用source-a1，17最终B/F使用source-a3；provider算法一致，保留各次失败原件。
Worker使用source-a2；三个wheel的独立安装证明与实际Worker源码加载证明分开记录。

`compare_capture.py`执行旧严格比较器；`diagnose_statistics_alignment.py`只读同一原始统计，
报告可配对字段及结构差异，不覆盖严格失败。`summarize_remote.py`只提取运行/恢复回执，
其结果不参与首差选择。`finalize_results.py`将执行范围覆盖到基线审计卡，保留历史单独栏目。

首次安装顺序错误、设备繁忙和RPC配置失败均保留，不把重试成功改写成初次成功。
本轮没有对外部任务执行停止，没有改Docker入口或启动等待自动化。
