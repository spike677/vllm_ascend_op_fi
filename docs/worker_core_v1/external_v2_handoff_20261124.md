# 外部 provider v2 接入说明（r7.4 候选）

由负责人转交。v2仅新增可信 `persistent_weight` / `persistent_kv` 路径，v1瞬态site和原有公共机制不改。发行包包含示例源码；示例不是默认注册、不是黄区四项能力已通过的声明。

## 安装与可信接线顺序

1. 在独立环境安装本发行的三个wheel，核对交付清单。Python≥3.10；真实模型示例基于Python3.12、0.26、CANN9.1。A3不得套用蓝区硬件身份。
2. 由原有可信bootstrap提供 `authority`、`operator` 和鉴权回调。在首个attachment之前、各rank分别调用 `install_persistent_runtime(worker, authority=authority, operator=operator)`。已有tenant/lease时拒绝，不能强换Runtime。
3. 审查部署方adapter源码，计算 `external_provider.factory_sha256(factory)`；登记 `PersistentSite(mechanism, region_kind, factory, source_sha256, expected_deployment)`，调用 `register_persistent_site(worker, spec, authority=authority, operator=operator)`。expected_deployment必须是该部署经审查的实际身份，不能沿用示例的BLUE配置或为通过而删字段。
4. 验收阶段仍安装原有部署验收许可，机制名为已审查的 `external.<named_variant>`，许可source_manifest_sha256使用发行 `source-review.json` 的SHA。没有许可必须exit4；注册site本身不授予资格。
5. 宿主使用 `PersistentProductClient(executor, PersistentFaultConfig('external', variant), operator=..., ingress=..., run_canary=..., canary_request=..., control_boundary=...)`。宿主控制保持 attach/enable/status/disable/detach。所有入口必须经过同一 `PersistentIngress`；pause/drain不能仅翻Python flag，须实际排空HTTP提交、调度和异步设备读者。
6. enable先完成稳定金丝雀，再各rank prepare/register/check/enable；全rank成功才开放入口。disable先关闭并排空入口，各rank恢复、unregister/detach，金丝雀通过后才释放入口所有权。失败时保留所有权及关闭入口，禁止“清掉错误状态”后继续服务。

用户CLI不接收地址、层名或数值参数；由黄区将命名变体映射到可信会话，不复制蓝区示例的无鉴权调试RPC到生产入口。

## 示例和证明边界

|文件|作用与边界|
|---|---|
|external_weight_scale_v2_example.py|真实dense QMM读取BF16 weight_scale；发现并暂时失效已审查派生副本，恢复原对象。未知NZ/预打包副本拒绝|
|external_persistent_product_probe.py|专用network-none、唯一HTTP提交者的weight两owner示例；不是通用公开HTTP管理端|
|external_kv_v2_example.py|真实16层full attention，写V cell，实际FIA消费者，合法更新保护；仅B1、4tokens、无共享页|
|external_kv_engine_plugin.py / external_kv_product_probe.py|实际EngineCore/KVManager页归属与释放前屏障；私有general-plugin元数据，不改第三方文件；源码SHA不符拒绝|
|review_persistent_example.py / review_persistent_kv_example.py|独立原始回执审查。KV冻结验收使用`--require-negatives`，核对真实abort及恢复，而不是只看标志|

示例位于发行目录 `experiments/worker_core_v11_20261118/`，运行前需将其及源码根加入PYTHONPATH。所有示例固定 `/model`、`/source`、`/evidence` 挂载用于隔离验收；不会替部署方创建生产Worker。

## 黄区四项的适配责任

|黄区能力|v2可复用部分|仍须黄区证明|
|---|---|---|
|weight/faulty_requant|持久权重独占、同代际字节备份/恢复|int8真实权重与NZ/预打包副本的全部消费者；蓝区scale例不代表int8验收|
|weight/param_perturb|weight_scale示例与派生副本协议|本机实际scheme、dtype及派生副本；开启期间所有请求读取修改值|
|kv_update_misdirect|请求/页归属、真实释放前恢复|实际写错槽位属于哪个请求、不得跨共享/已复用页；本例不声称原kernel自身写错地址|
|kv_rope_quant_loss|同stream写入、后续token消费、请求结束恢复|真实K/V变体、cache布局和消费者；GDN、MLA未由本示例验收|

每项两个owner，同组Worker；八项持久写入计数与reader_observations分开，真实消费者、字节恢复、资源清零、金丝雀、缺许可和超范围负例。HTTP取消须有调度器结束状态及页释放证明，断开socket本身不是完成证明。prefix cache/connector/异步调度/未知派生缓存不支持时明确拒绝。

蓝区结果详见r74_v2_acceptance_20261124.md。A3源码及布局核对通过后才运行示例或黄区adapter；不通过先报部署不匹配。所有未审查组合保持关闭。

## 33

本包只提供 `paged33_launch_capture_v2.cpp` 和完整捕获审查工具。控制重算及A3适配属于r7.5候选分支。完整捕获、公开候选身份、字段边界、三形状、每组≥99%控制一致率都是独立门禁；本包不放行33 A3。
