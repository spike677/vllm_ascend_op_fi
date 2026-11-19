# 黄区内源测试 RC1

固定包：`demoncatpfi-yellow-rc1-20261119`。黄区负责人：**openchamber**。

- [下载源码包](demoncatpfi-yellow-rc1-20261119.tar.gz) · [逐文件源码清单](source-review.json)
- [版本、能力及Native身份清单](release-manifest.json) · [三包安装检查](install-summary.json) · [SHA256SUMS](SHA256SUMS)
- [统一接线/安装/恢复说明](../../docs/worker_core_v1/yellow_rc1.md) · [Native构建取得方式](../../docs/worker_core_v1/rc1_native_dependencies.md)

源码冻结提交：`a315d89c962c93eacd61546bb9188a3fdbe424a2`。该提交是包内容的来源，交付目录所在提交另由冻结tag标识。
源码包SHA256：`694f9007f79018242744b1ffad96ff38d63f40006b0cb31dc448ef4e4e74a702`。

首批可进入黄区部署适配测试：**39 output_proxy、Batch普通动态W8A8、18普通单任务**。
参考实测均为Qwen3-0.6B、TP1/eager、CANN8.5.1/A2；Batch另限B8首decode。
28未完成，默认不可开启；33等保留历史限定范围，不据此授新普通入口。15不适用，25不支持。

实际导出包已在124固定aarch64/Python3.11环境离线构建安装三个wheel，
270个已安装Python源码逐字节匹配。公共导入及默认未授权检查通过，未导入模型栈、未挂NPU。
相关CPU检查134通过、1项Windows权限跳过。Native与NPU原结果按未改机制复用，未重新运行黄区请求。

## 同源已安装检查的wheel

以下wheel仅含Python和自有源码，不含CANN/vendor库；版本号沿用原发行单元，必须按本包SHA辨别。

- [demoncat_fault39_core-0.1.0-py3-none-any.whl](wheels/demoncat_fault39_core-0.1.0-py3-none-any.whl)
- [demoncat_worker_extension-0.1.0-py3-none-any.whl](wheels/demoncat_worker_extension-0.1.0-py3-none-any.whl)
- [demoncatpfi-0.2.0-py3-none-any.whl](wheels/demoncatpfi-0.2.0-py3-none-any.whl)

黄区现役`demoncat`可能与namespace/Controller冲突；先在隔离候选目录检查导入来源，
由openchamber审核组合，不能直接覆盖现役Controller。五处真实接线和黄区正常/F/OFF请求均**待测**。

发布后不覆盖本包。后续修改使用RC1.1/RC2；蓝区开发线继续，未完成工作不混入此冻结快照。
