# r8共享路径影响审查（冻结前）

基线为r7.4.1发行提交 `59f6aea5`。当前实机候选 `3d4c9e39`；后续 `dd625aa3` 只增加文档、审查记录和发行准备，未改变该候选运行时Python。最终发行时必须再次核对，不将本页当作未来提交的自动批准。

[字节清单](validation/shared-byte-comparison.json)覆盖基线481个原有运行时文件，其中474个Git字节相同、以下7个共享文件变化。机制算法、实际site、observer和Native原语没有变化。当前NPU使用工作树快照，wheel使用Git字节，不能仅据提交号宣称两者相同。

[实测快照到规范源码的逐文件核对](validation/npu-to-canonical-comparison.json)另外检查449个运行时、实际HTTP探针和审查器文件：448个逐字节相同，唯一差异为`src/pyproject.toml`的wheel版本号；没有执行代码或换行差异。两个源码清单的文件范围不同，清单SHA不相等，分别保留。后续只读登记JSON更新仍须单独核对，并完成最终安装字节的免许可正例。没有确诊的Y1公共机制修复被混入本轮。

## 七个既有文件

路径均相对 `src/demoncat/runtime/vllm_ascend/worker_extension/`。

|文件|变化及触发边界|影响范围与检查|
|---|---|---|
|deployment_acceptance.py|仅在已绑定product且mode为PRODUCT_REGISTERED时调用精确登记检查；用审查记录决定can_prepare。验收许可路径保留|所有公共产品项的prepare准入；每个组合不能借其它变体资格，A2代表项需免许可正例与未登记负例|
|deployment_authorization.py|已登记模式在控制动作时重新读取当前binding/profile/登记SHA，不把一次prepare的结论永久缓存|所有已登记项的控制路径；身份/登记改变后拒绝，不能继续使用旧资格|
|extension.py|增加受信bootstrap RPC；契约拒绝结构化exit4，未知执行异常exit5并标恢复待证|新bootstrap动作；不把鉴权回调、路径、许可创造权限暴露给用户RPC。既有产品RPC入口保留|
|product_binding.py|允许公共profile选择PRODUCT_REGISTERED；capabilities明确资格/拒绝原因；attach锁定登记SHA。external仍显式拒绝免许可模式|所有公共产品attach；各rank一致性；Native/15/视觉原门禁继续执行；外部三个示例需回归既有许可路径|
|product_config.py|配置支持两种mode；已登记配置不生成required_permits，真实身份、图像/Batch金丝雀、Native路径和库身份仍核对|所有从部署文件创建的profile；不能仅靠改mode获得资格；Batch仍是串行金丝雀和B8调度配置|
|product_host.py|ProductClient核对全rank相同登记SHA；在状态中报告product_qualified和registration_sha256|全部宿主生命周期；每项仍走同一attach/enable/status/disable；登记不替代恢复证据|
|providers/product.py|保留mechanism供每次准入复核；未建立ticket且prepare之前拒绝时，允许受owner约束的惰性清理|全部产品失败回滚；不能清理另一owner或外来wrapper；拒绝后引擎存活、资源清零和正对照必须验证|

## 新增文件与隔离边界

- `product_registration.py`及只读JSON登记表：严格对比每rank身份、模型制品、运行库、参数范围、实现SHA、模式和变体；只读发行记录，不提供在线写登记接口。EP登记未开放；A3记录不能由BLUE结果生成。
- `product_bootstrap.py`：部署方本地初始化鉴权回调/配置路径/金丝雀token后才开放受信RPC。RPC不能替换上述对象。模型对象变化、未知principal和未知字段拒绝。
- `product_cli.py`、`product_cli_entry.py`：受信session参考分发和 `dcatpfi-product` transport helper。黄区现有入口仍是 `python -m demoncat.cli fault ...`；helper不创建daemon、Worker或许可，CPU transport/鉴权测试不等于黄区CLI验收。
- `fault_injection/dsv4_product_v030.py`：此前准备的D路线CPU身份/范围检查；没有接入本轮Qwen产品路由，V4/EP仍关闭。但该文件属于安装实现指纹，加入后旧实现SHA不能沿用，必须按最终安装内容重新核对。

## 按能力的实际影响

18的三个命名变体、22、39op、39mr、28、17、Batch、15、41、26fia、20都经过上述共享准入/宿主路径，因此本次用当前候选各自做A2 TP2真实HTTP双owner回归。不能据机制字节相同跳过A2回归，也不能用代表18替代Batch/Native/图片路径。

三个外部示例通过共享宿主与旧external site路由，在验收许可模式独立回归；不授予免许可资格。15保留ND别名断开限制。41/26fia/20保留各自已审查Native库、同stream消费者和安全恢复点，不改成Torch效果代理。33 A3、V4和公共8不因新登记机制而放行。

A3迁移按 [增量单](r8_incremental_acceptance_20261126.md)：所有已审查登记项做免许可冒烟；39op、41、26fia两个owner重做；Batch若有公共修复也两个owner重做。其它项只有正式r7.4.1结果审查、机制字节和本共享路径影响审查都通过后，才沿用原双owner结果。A3未登记项不得借用A2资格。

当前39op/Batch的黄区失败尚缺同次运行内层证据；不把蓝区通过写成黄区已修复。26fia的补配置成功声明与旧configure失败按不同运行分开记录。实际确诊修复若出现，必须在本页追加文件、触发条件、影响能力和重测范围，最终冻结前重新生成字节清单。
