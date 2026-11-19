# 黄区测试部署 RC1.1

冻结标识：`demoncatpfi-yellow-rc1-1-20261119`。源码提交：`e39d48c1a86f629241e4edc3817048544f1bf722`。发行提交由同名标签指向，与源码提交分开。原 RC1 标签、包和 SHA 保持不变。

- [源码包](demoncatpfi-yellow-rc1-1-20261119.tar.gz)、[逐文件清单](source-review.json)、[发行清单](release-manifest.json)、[安装结果](install-summary.json)、[校验清单](SHA256SUMS)
- [调用、前提和清理接口](../../docs/worker_core_v1/yellow_rc1.md)、[Native 依赖](../../docs/worker_core_v1/rc1_native_dependencies.md)
- 三个同源 wheel：[公共能力](wheels/demoncatpfi-0.2.0-py3-none-any.whl)、[Worker Extension](wheels/demoncat_worker_extension-0.1.0-py3-none-any.whl)、[Fault39](wheels/demoncat_fault39_core-0.1.0-py3-none-any.whl)。版本号沿用，必须按本发行 SHA 辨认。

源码包 SHA256：`2a2364677353af302fe74d114bdc8e96f41b0771ae9b19047fa175f3fdf75909`。

## 修复与范围

黄区六请求问题的主因是宿主混用了历史 C1 协议。普通 39 仍按到达顺序处理 normal/fault/off/health 四请求；请求名称不选阶段，OFF/health 状态机未改。第五个 advance 现在返回 `OUTPUT39_TASK_PHASES_EXHAUSTED`；真正未关闭请求仍返回 `PREVIOUS_REQUEST_NOT_CLOSED`。公共工厂拒绝 `runtime_only=False`，返回 `OUTPUT39_ORDINARY_TASK_BINDING_REQUIRED`，不再回落 C1。安装函数签名不变。

结果从 `response.provider_state.evidence` 读取。正常结束按 finish、disable、unregister、detach/remove；失败（包括用完阶段后错误追加请求）按 recover/abort、clean、unregister、detach/remove，保留 Host 的失败隔离。文档补齐串行 eager 全部参数、add_request 内部 ID、自定义算子库和 GLIBCXX_3.4.29 前提。

声明候选仍为 18、39 与普通动态 W8A8 Batch。Batch 需要匹配权重和 bundle、TP1/B8 首次纯 decode，沿用六批协议。15 不适用，25 不支持；28 未完成，默认不可开启；其他历史验收入口不构成本包普通任务授权。

## 验证与限制

CPU 回归 162 项：161 通过、1 跳过、0 失败/错误；原 RC1 同一扩展选择为 158 通过、1 跳过，新增三项 39 回归。旧发行的 134/1 是更小的选择，不能直接混为同一测试集。

实际导出的 773 个源码文件来自冻结 Git 对象。124 上 aarch64/Python3.11 无设备、无网络容器离线构建安装三个 wheel，公共导入/default-closed 检查通过；270 个已安装 Python 源文件逐字节匹配。没有 NPU 调用、没有新推理验收、没有新原始统计数据或私有完整张量。沿用未改机制的蓝区既有结果。

用户转述的 openchamber 结果表明，原 RC1 的 18/39 在黄区 Qwen3-0.6B BF16 TP1 串行 eager 测试部署各通过两条独立任务；quant 待部署 W8A8。请 openchamber 按本包回归 18/39。此包不代表黄区现役 TP4/8/16 部署已支持，`product_closed=false`。
