# D74：0.30/0.32 黄区接手参考资料

用途：源码咨询和迁移设计参考，**没有在真实 V4 0.30/0.32 环境验证过**。不是运行时升级、不是可执行 D13/公共 8 包，无 wheel/Native 库/许可/登记。

## 取件身份

- 分支：`codex/d74-reference-handoff-20261127`。
- 内容提交：`67f03bea`（完整号见 manifest 的 source_commit）。
- [参考 ZIP](d74-reference-20261127/d74-v030-reference-20261127.zip)：169964 字节。
- ZIP SHA256：`683891acf0acd7a604a2f68c76aa8e316ad6433c288b641d1219d6c6d5515972`。
- [逐文件 manifest](d74-reference-20261127/manifest.json)、[包内文件校验清单](d74-reference-20261127/SHA256SUMS)、[发行目录校验清单](d74-reference-20261127/DELIVERY-SHA256SUMS)、[离线验证记录](d74-reference-20261127/validation.json)。

源码资产逐项指定提交，不以本交接分支的完整运行时作为安装来源：

|资产|固定来源|限制|
|---|---|---|
|长请求产品准入与原六项 CPU 测试|27cadf797014a6a34bbee23e5eaeae1977f5488f|关闭路径；enable 始终拒绝；没有真实 V4 环境验证|
|D13/公共 8 门槛、r3 执行单|065931d986e44a970af6c8ed3af1423b24f1258a|历史责任安排被 D74 覆盖；观测通过不等于注入资格|
|审查工具、未接线 Python 观测核心|743611c4a8de9a7c0f762e6a3a38f7a55c350304|18 项 CPU 是核心测试；没有实际 V4 Worker/tensor 接线|
|0.26 检查位置原始源码快照|e38c4113b421ac979699071a60e80d3babd05e07|选定文件供阅读，不是完整安装树；不混入 r8.2 开发修复|

## 校验和阅读

把 ZIP 下载到独立目录，先核对上述 ZIP SHA，再解压。Python ≥3.10、仅标准库：

```sh
sha256sum d74-v030-reference-20261127.zip
unzip d74-v030-reference-20261127.zip -d d74-reference
cd d74-reference
python -I verify_reference.py
python -I verify_reference.py --tests
```

预期：50 个 payload 文件逐字节匹配 manifest；原六项 CPU 契约全部通过。验证器为测试创建隔离 namespace，仅加载两个参考模块，不执行完整生产 initializer；这不是整包安装验收。不要将 `reference/` 加入生产 PYTHONPATH，不要将交接分支整体安装到现役 SITE。

蓝区实际完成：50 文件校验、46 个 Git 源文件与固定 blob 字节一致、原六项 CPU 测试 6/6、独立解压运行、篡改拒绝、两次构建 ZIP 逐字节一致。全部为离线验证，NPU 用时 0；没有修改 0.26 运行时或任何冻结标签。

## 黄区接手顺序

1. S1/M1 继续收尾，将 0.26 留作已验证参考；S0-UPGRADE-001 转咨询，不再等蓝区源码审查/探针包。
2. 阅读 [D74 决定](../docs/decision-log/20261127.md) 与 [版本绑定检查清单](../docs/worker_core_v1/runtime_026_version_bound_checklist_20261127.md)。包内 runtime-source-locations.json 提供冻结源码 SHA 和函数行号。
3. 黄区在 r8 基础上另开 0.30 迁移分支，R0 逐函数审签名、语义、调度、请求映射、异步/stream、owner 恢复，运行时代码交独立审查会话。不得只扩充 SHA 白名单。
4. R1：18.api_half、39mr 的 TP2 两 owner 正例、真实效果、一页总结、负例。R2：D13 与公共 8 独立开发验收。R3：其余适用项重验重登记；S2 行在哪版正式收尾由黄区记录 Y-D。
5. LEDGER-FINAL 分列 0.26 与 0.30；旧登记和实现身份不得跨版本借用。0.32 另按同样流程迁移。

此后蓝区只按明确请求解释设计或源码复核，不做 0.30/0.32 环境运行和出包；不作为黄区合入/冻结的常规前置。黄区无需回传静态源码以解除蓝区阻塞，只有需要咨询时再提供相关代码即可。
