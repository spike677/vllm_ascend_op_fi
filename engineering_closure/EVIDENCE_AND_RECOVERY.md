# 证据与恢复

本交付固定核对 `8f73f8293f47049f0c4dc4eb0b797a99b9278f74` 的历史检查点。新增工程改动通过工作树文件和本地测试记录追踪。执行新的实机任务时，重新生成 source manifest，使用其实际 SHA 绑定构建及任务。

`mechanism_matrix.csv` 对账全部 17 个机制。`historical_evidence/` 保存历史模型、CANN、芯片、TP、实现边界与证据出处。历史报告保留原范围。每个新作用域仍按自己的执行链、模型、版本与恢复结果验收。

`python -m engineering_closure audit` 验证原发布清单的 205 项文件，列出本次工程修改导致的哈希变化，检查 17 个机制的索引是否完整。它同时返回历史矩阵中的限定 C2/C3 状态和各机制当前工程、NPU 缺口。远端原始张量及统计在本机不可访问的情况会明确保留。

统一报告采用 `demoncat.engineering.evidence_bundle.v1`。`scope` 包含 `mechanism`、`model`、`cann`、`soc`、`tp`、`source_commit`、`implementation_sha256`、`environment_sha256`、`workload_sha256`。原始运行回执应继续记录实际 owner、generation、rank、Worker、request、executor 和 Storage lifetime。

`stages` 的十个字段依次为 `source`、`build`、`native_inventory`、`target_binding`、`baseline`、`fault`、`numerical_effect`、`off`、`recovery`、`independent_audit`。每个字段是原件引用数组，每条引用包含证据根目录内的 `path` 和实际 `sha256`。缺少的阶段保留 `MISSING`。文件哈希通过后，普通引用的 `semantic_review` 仍为 `PENDING`，由对应机制的审查确认语义。

外层 model、CANN、SoC、TP 和 source 标签由报告作者提供，输出明确标记其环境与模型语义核验待完成。coverage 会核对 mechanism 与三个执行范围哈希，模型和环境标签还需要与真实模型清单、构建记录和 Worker 回执对应。

可选 `coverage` 和 `c2` 也使用 `path`、`sha256`，并支持 JSON Pointer 字段 `pointer`。`coverage` 必须是现有 `demoncat.coverage_session.v1` 格式。审计复用 `fault_injection.coverage_audit.audit_saved_coverage`，重新计算目标、独立调用、hit、lease、全组 drain 和同进程健康恢复。作用域哈希需要与外层报告一致。每次匹配的实机启动仍须持有当前进程中的 live C2 proof。

可选 `statistics` 包含 `baseline` 与 `fault`，各自提供 `dump.json`、`stack.json`、`construct.json`、`config`、`collector` 的原件引用。比较复用 `fault_injection.statistics_audit.compare_raw_statistics`。固定公式为 `abs(F-B) > 1e-8 + 1e-5*abs(B)`。完整输出保留 null、NaN、Inf、错位和不可见字段，以及首个可观察差异之前的缺口。

```powershell
python -m engineering_closure evidence .\run_bundle.json --artifact-root D:\reviewed-evidence --out .\audit-result.json
```

`engineering_tests/cpu/test_evidence_closure.py` 验证字节变更、缺少原件、作用域错配和历史状态继承。已有 `tests/test_coverage_audit.py` 与 `tests/test_statistics_audit.py` 继续覆盖分母、C2/C3 关联、固定容差和异常统计。

恢复沿用 `CoverageSession` 的先 stop、再全目标 drain、再 recover/release、最后同 Worker 健康复跑的顺序。任一目标 drain 失败时保留租约，后续重试使用同一个 owner/generation。每个 Operator Adapter 负责真实设备资源和机制状态的恢复。容器退出码和容器清理结果仅作为控制器记录。

Engineering Coverage 记录发现、绑定、调用观测、注入入口、恢复及测试范围。Mechanism Coverage 记录指定模型、CANN、硬件和 workload 的真实 C2/C3。新 CPU 测试结果单独存档。本交付的历史矩阵仍为限定标准 Worker C2 3/17、C3 3/17。
