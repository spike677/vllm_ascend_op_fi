# 黄区测试部署 RC2

发行标签 `demoncatpfi-yellow-rc2-20261119`；源码提交 `d07c72156e0fe4e3608dcee58cdb422b49834c66`。发行提交由该标签指向，与源码提交分开。RC1/RC1.1保持不变。

- [源码包](demoncatpfi-yellow-rc2-20261119.tar.gz)、[逐文件清单](source-review.json)、[发行清单](release-manifest.json)、[安装校验](install-summary.json)、[SHA256SUMS](SHA256SUMS)
- [公共组件](wheels/demoncatpfi-0.3.0-py3-none-any.whl)、[Worker Extension](wheels/demoncat_worker_extension-0.2.0-py3-none-any.whl)、[Fault39 core](wheels/demoncat_fault39_core-0.2.0-py3-none-any.whl)
- [安装/调用/恢复](../../docs/worker_core_v1/yellow_rc2.md)、[部署验收许可及登记](../../docs/worker_core_v1/deployment_acceptance_rc2.md)、[黄区步骤](../../docs/worker_core_v1/openchamber_rc2_preparation.md)

源包SHA256：`1a59219c0ec5e37204f208c12a2a9e6f42edcb491eee82353ccba6a0fae050d8`。

范围为18、39 output_proxy、8、17、22、Batch六项已实现普通任务；只对精确已登记组合默认允许。蓝区A2组合与黄区测试部署/专用实例分开。黄区RC2记录当前为空，18/39升级后默认关闭；用可信部署操作员许可跑两作用域并回传总结，由下一RC2.x审查登记。RC1.1黄区两作用域通过不自动继承RC2组合。

8/17/22已有各两作用域；Batch本轮a3补齐第二独立Worker/owner，各作用域内OFF/健康精确恢复。a3真实Q改变5565/8192、scale改变7/8，decode logits改变、逻辑请求7的decode Top-1改变；资源归零。新许可审计18条state有效。其他项不因历史机制证据获准普通任务，原因见发行清单。

源包818个文件来自冻结Git对象。124 aarch64无设备、无网络容器离线构建安装三个wheel；285个已安装Python源文件逐字节匹配。安装过程零NPU调用。CPU选择138项=137通过+1跳过，发行选择61项=60通过+1跳过，最后边界32项通过；有重叠不相加。不提交新原始statistics、完整张量、权重或Native二进制。

不授予新CANN、0.26/0.32、A3、现役Qwen3.8/DeepSeekV4.1 W4A8或TP8/DP2/EP/图模式资格。实际黄区主项目闭环和product_closed由openchamber验收。4.7准备完成；4.8迁移需负责人确认。
