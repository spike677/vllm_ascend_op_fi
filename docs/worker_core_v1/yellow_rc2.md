# 黄区测试部署 RC2

冻结目标 `demoncatpfi-yellow-rc2-20261119`；实际发行提交、源码提交和 SHA 以 `releases/` 下该发行清单为准。本页不提前声称冻结成功。

普通任务子集：18 api_half、39 output_proxy、8 restore_offset、17 spatial_padding、22 wrong_row、quant batch_shared_scale。只覆盖已登记的部署组合，查询实现存在不表示可执行。部署来源与硬件分别核对，黄区 RC2 组合当前尚未登记。

15：当前部署无私有视图写入/原 base 消费链；25：不支持；28 legal：历史保留，不扩展；28 denominator：真实 TND F/OFF 未完成；20、26 两变体、33 Source105、41：历史机制保留，普通任务未完成。这些项不从公共入口回退到研究协议。39 matmul_range 也不是本包普通任务候选。

## 安装与身份

三个同源包升级为 `demoncatpfi==0.3.0`、`demoncat-worker-extension==0.2.0`、`demoncat-fault39-core==0.2.0`。按发行 `SHA256SUMS` 核验后，在独立测试部署安装三个 wheel：

```sh
python -m pip install --no-deps --no-index wheels/*.whl
python tools/check_rc2_install.py --expected-root ACTUAL_INSTALL_ROOT --output install-check.json
```

安装检查必须在无模型栈的新进程运行；检查导入来源、发行版本和默认关闭，不创建 Worker 或 NPU调用。已有内源普通 `demoncat` 包需审核 namespace/合并接线，不能因能导入就认为黄区主项目已接通。正式安装记录还需核对所有已安装 Python 文件与源包相同字节，三包来自同一源码提交。

Native 不包含在 wheel：39 使用 [原 postwrite 身份与构建要求](rc1_native_dependencies.md)；Batch 使用已验证动态量化 bundle、源码/SDK/DSO/ABI/启动 vendor 检查。更换环境需重新构建和验证；不能以运行版本字符串冒充构建证明。8/17/18/22 的实际源码/调度/消费者检查继续保留。

## 接入与恢复

沿用已有 `DemoncatWorkerExtension` 和真实 vLLM Worker，不在 Extension 内创建/重启 Worker。可信 Host 可用 `fault_injection.rc2_profile.prepare_binding(deployment_callback)` 限定本包子集；它只调用部署侧原 binding，不能绕过运行时准入。

黄区需要 [部署侧许可与登记](deployment_acceptance_rc2.md)，主项目不要把蓝区实验 probe 当成产品入口。无许可的新组合默认关闭；授权许可只在指定 worker/model/pid/thread/机制/attempt/source SHA 使用，记录到每次结果，检查失败时不能降级。

宿主统一 `advance(actual_internal_id) → 等待真实完成 → query`；Batch 使用 `advance_batch(actual_ids)`，仍保留原生六批校准/故障/OFF协议。query只读。普通四请求机制为 normal/fault/off/health；Batch 真实六批为 normal/reference/lossless/calibration_off/fault/off，另有前后常规健康请求。效果及资源从实际 provider_state 读取，不能以 enable flag 或日志代替故障命中。

正常结束 finish→disable（适用时）→unregister→detach→移除 binding，最后撤销验收许可。失败 preserve first error，按 recover/abort/clean 处理；无法确认恢复则保留 owner，不继续注入。Native、hooks、method/API patches、readback 与输入强引用按机制实际计数，不把容器退出视为恢复。

两个独立作用域回传 [有界总结](deployment_acceptance_rc2.md)。evidence 上限512 KiB、provider_state 2 MiB；超限是 incomplete，不能静默压缩成通过。Host 历史16 MiB上限独立，长任务可用已有cursor/stream；登记总结最多64 KiB。

## 声明范围

[蓝区逐项验收](ordinary_tasks_rc2_20261119.md) 与 [黄区验证清单](openchamber_rc2_preparation.md)分别记录。黄区仅验证18/39/22，8/17本轮蓝区验收，Batch黄区待配置。RC1.1 已合入且18/39两作用域通过为用户回传事实，不自动成为RC2组合登记。

0.18/CANN8.5.1/A2结果不能推广到0.26/0.32、新CANN、A3、TP8/DP2/EP、异步或图模式。新 DeepSeek 测试优先 V4.1 W4A8，W8A8只作现役参考，各量化链分别验收。现役inspect与RC3范围确认仍待黄区资料；`product_closed` 以openchamber实际主项目验收为准。
