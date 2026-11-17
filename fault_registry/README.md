# Unified Interface Closure

基线为 `9dfdb0c6e0e900f8a67d04dcafc53544bc7fc477`，工程分支为 `integration/demoncat-unified-api-v1`。

本轮交付统一机制注册、配置和调用契约。17/17 项都有唯一身份、版本化参数 Schema、Adapter、结果模型、状态机和标准故障卡。统一 API 当前实际接通 **17.spatial_padding、18.api_half 两个 CPU 模块代理**；其余 15 项执行入口返回 `NOT_IMPLEMENTED`，原因是旧 Provider 的参数、触发和恢复语义尚未接入新协议。原有独立执行入口继续保留。

`registry.yaml` 使用 JSON 兼容的 YAML 1.2 表达，标准库即可读取。它保存机制身份、语义、后端、当前统一桥接状态和原有目录描述。参数 Schema 是其引用的版本化定义。`fault_injection/catalog.py`、`capabilities.py` 与 `engineering_closure/operator_adapters.py` 从注册表加载视图。Fault39 的两个机制出现在公共目录中，仍由独立包执行。

原 `engineering_closure/mechanism_matrix.csv`、五批实验队列和历史测试回执是冻结的审查记录。它们不会被新接口测试覆盖。新的生成矩阵、卡片和队列位于 `generated/`；队列只列出接入条件，不直接启动 NPU。注册表内 `historical` 字段保留原审查范围。

从项目根目录执行：

```powershell
python -m pip install -e .
python -m demoncat example 18.api_half
python -m demoncat validate fault_registry/examples/f018.api_half.json
python -m demoncat inspect F039.MATMUL_RANGE
python -m demoncat generate --check
```

安装不会安装 PyTorch；CPU 执行测试需已有 Python 3.11 / PyTorch 2.5.1 环境。目录和 Schema 查询不导入 PyTorch，不初始化设备。

文档索引：

- [Python、CLI、Worker API](docs/api.md)
- [参数与触发语义](docs/parameters.md)
- [命名规范](docs/naming_convention.md)
- [机制定义与证据](docs/mechanism_definition.md)
- [兼容性与工程边界](docs/compatibility.md)
- [分支与 PR 规范](docs/branch_policy.md)
- [17 项生成目录](generated/mechanism_catalog.md)
- [当前统一执行矩阵](generated/mechanism_matrix.csv)

`tests/unified/run_validation.py` 独立统计 Adapter Conformance Suite 和旧接口回归，输出实际源码 SHA、环境、测试失败和跳过原因。原有 222 项回归另行运行并保存本轮回执。真实 Kernel 关联、FIA Native 执行、跨版本硬件资格和新增 NPU 验收均未由这些 CPU 测试授予。
