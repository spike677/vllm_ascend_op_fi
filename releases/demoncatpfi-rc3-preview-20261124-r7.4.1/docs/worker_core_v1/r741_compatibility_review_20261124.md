# r7.4.1兼容性与复测范围

基线为r7.3.1 d23cfc9f；r7.4的v2新增路径继续保留。原18.api_half、22、39op/mr、28、17、15、Batch与Native 41/26fia/20机制算法文件不改。不能因此声称全部共享文件逐字节不变。

|既有共享文件|本轮变化边界|复测|
|---|---|---|
|catalog、delivery_scope、capabilities、integration|增加18.variance_over4/variance_under4命名变体及其说明，不改变旧ID算法|目录、配置、身份、未知变体拒绝CPU回归|
|product_contract|新18变体进入相同请求范围限制|范围正负例、旧机制请求计划CPU回归|
|continuous_binding、registry|新18工厂及显式external vision路由|新18与vision A2；旧18和17 A2回归；所有旧公共工厂CPU回归|
|external_provider|明确允许已审核的vision stage，持久区域仍不能进v1|v1/v2准入CPU；图片实际消费者A2；无授权/无源码身份拒绝|
|product_binding|vision capability中的真实image_grid|查询不授予资格；图片范围CPU与A2|
|product_request|新18效果与external vision证据；旧恢复分支仍保留|新18/vision与旧18/17 A2；产品请求、恢复和所有公共机制CPU回归|

本轮B2的module/dispatch示例只增加部署方factory，不增加共享产品状态机。B5的DiagnosticProductClient继承原控制逻辑，失败只采集实际已返回数据，不重试、不修正cursor。15只读诊断器仅用于验收，不替换Alias15Site；原机制保持不变。

r7.4中weight_scale和KV两个owner的实机证据继续引用：本轮未改其独立v2路径、persistent ingress或恢复实现；不把它们写为本轮重新跑过。A2结果不自动成为A3资格。外部适配器源码SHA变化后必须按新身份验收。

冻结时以source-delta.json列出每个新增、变化、删除文件，以shared-byte-comparison.json列出原运行时文件的SHA及差异。本文的受影响清单不能替代最终机器可读对照；任何未列出的共享运行时改动都应阻止冻结。
