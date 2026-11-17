# 工程闭环交付

本目录提供六项工程目标的实现与审计，并对账 17 个机制的历史范围、当前源码、工程缺口和 NPU 缺口。历史审查基线固定为 `8f73f8293f47049f0c4dc4eb0b797a99b9278f74`。

发布时已接入远端截至 `f352dc9f296650f817fd0a44c167f8602f2292d4` 的 31 个新增提交，保留其 Conv3d、FA/FIA、Native26、量化及 live C2 proof 更新。下述机制矩阵、继承验收与差异队列仍是上述历史基线的审查快照；其中 FA 等工程缺口应结合后续源码和审查记录重新对账。本次集成验证覆盖合并后的 Python 源码，未新增 NPU 验收结论。

从仓库根目录运行，需要 Python 3.11 及 PyTorch 2.5 或兼容的现有部署版本。纯证据和队列审计使用标准库及已有项目模块。Worker 入口需使用目标 vLLM/torch_npu 环境。完整测试应使用含 RMSNorm 的 PyTorch 与 Python 3.11 的 unittest。

```powershell
$env:PYTHONPATH='.;src;fault39/src;tests;tests/worker_extension'
python -m engineering_closure audit
python -m engineering_closure queue
python -m engineering_closure.compatibility --cann 8.5.1 --mechanism 20.buffer_substitute --operator aclnnQuantMatmulWeightNz
python engineering_closure/engineering_tests/run_validation.py --out engineering_closure/engineering_tests/validation-new.json
```

兼容检查在没有 CANN 的 Windows 环境会返回实际缺失条件和退出码 2。NPU 队列先显示待满足的进入条件。`prepare` 子命令可生成 Worker18、Output39 和 KV41 的原控制器 case JSON，其他已有任务脚本在各批次的 `preparation_entrypoint` 中列出。详细执行方法见 `npu_execution/dependency_graph.md`。

| 文档 | 内容 |
| --- | --- |
| [ENGINEERING_ARCHITECTURE.md](ENGINEERING_ARCHITECTURE.md) | 公共 Runtime 与四种执行入口的集成 |
| [MODEL_PATH_DISCOVERY.md](MODEL_PATH_DISCOVERY.md) | 实际调用图、观察状态和 Worker 接入 |
| [NATIVE_RUNTIME.md](NATIVE_RUNTIME.md) | 全量调用、库绑定、身份与生命周期 |
| [OPERATOR_ADAPTERS.md](OPERATOR_ADAPTERS.md) | 17 项机制的统一接口和实际 Provider |
| [TILING_KERNEL_ADAPTERS.md](TILING_KERNEL_ADAPTERS.md) | FA/FIA、Host/Device 与版本构建边界 |
| [CANN_COMPATIBILITY.md](CANN_COMPATIBILITY.md) | 多 CANN、模型和硬件的精确兼容检查 |
| [EVIDENCE_AND_RECOVERY.md](EVIDENCE_AND_RECOVERY.md) | 原件引用、独立统计和恢复审计 |

`historical_evidence/inherited_pass.csv` 保留模型、CANN、芯片、TP、故障实现、出处和验收边界。`incomplete_evidence.csv` 记录每项缺口。`historical_scope_limits.csv` 记录继承范围。Fault15 的 alias-detach 已找到旧 Fault27/NZ 构造用例与 11/16 NPU B/F/OFF 记录，仍待自然模型路径。Fault22 的公开记录缺少精确 CANN 和芯片型号，该信息保留待核实。

`mechanism_matrix.csv` 的 C2/C3 继承自已发布的限定标准 Worker 审查，分别为 3/17。新工程测试记录独立于实机覆盖。历史 Qwen Fault08、17、22、旧 Fault26 代理实验和已完成通信校准按原范围继承。后续任务只针对各自明确缺口。

2026-11-18 合并复核补充：上述 CSV 是历史快照，不是当前资格。最新实际构建审查确认 Native20/Native39 的 SDK 元数据数组未释放，两条全资源 C2/C3 已撤回为待修复重验；当前严格标准 Worker 资格为 1/17，仅 Fault18。原命中、数值和 OFF 证据保留。当前判定见[主控矩阵](../experiments/c2c3_closure_20261117/C2_C3_MATRIX.md)和[实际构建所有权审查](../experiments/c2c3_closure_20261117/coverage_frozen_metadata_ownership_review.md)。本目录新增的 Postwrite 多租约实现仍需新构建和 NPU 验收，不继承旧单租约实机资格。

源码打包继续使用 `experiments/c2c3_closure_20261117/package_source.py`，已纳入本目录的代码、文档、CSV 和队列。新打包中的 source manifest 反映真实当前字节，后续构建和运行必须引用该身份。

尚待补齐的工程能力包括通用第 N 次调用触发、任意模型调用到 Host/Kernel 的逐调用关联、FA 新 SDK 可移植构建和 FIA replay 自然模型执行入口。跨版本实机资格、NPU 多目标效果及黄区宿主接入继续按实际证据推进。本目录各测试通过只覆盖其记录的验证环境。

本轮验证运行 222 项 Python 测试，其中 220 项通过，2 项 Linux 专属测试在 Windows 跳过。Ubuntu WSL 单独运行的 3 项真实 ELF/LD_PRELOAD 测试全部通过。生产 postwrite.cpp 使用 MSVC 严格警告配置与 CPU ACL mock 编译执行通过。回执分别位于 `engineering_tests/validation.json`、`engineering_tests/linux_loader/loader_receipt.json` 和 `engineering_tests/native_mock/native_lease_receipt.json`，包含实际源文件 SHA-256。

本机验证使用项目内 `.codex-tmp/engineering-venv` 的 Python 3.11 / PyTorch 2.5.1 CPU。Ubuntu WSL 为加载测试安装了 GCC 和 libc 开发包及其依赖更新。测试回执中的 NPU 调用数为零。
