# product_closed剩余工作与排期依赖

截至本轮，r5验收模式不等于黄区产品闭环。以下往返数为依赖齐全且一次通过时的估计，不作为验收结果。

|工作|归属|前置条件|预计往返|完成证据|
|---|---|---|---|---|
|r5 14组合审查、r5.1登记|Codex已完成|方案B，原文本null不改|0|已发布0e6ded80，14个验收组合；不是产品资格|
|18/39op/22/28/17黄区产品CLI接线|openchamber运行；Codex契约/复核|r5产品RPC与可信会话、身份、17内置金丝雀/spawn|1–2|真实用户HTTP请求自动发现、两owner范围、关闭后新canary恢复|
|Batch产品入口|Codex已完成A2；openchamber A3验收|并发decode真实批次映射、max_num_seqs8、稳定基线|1–2|B1不注入、B>=2全真实行共享scale、真实QMM消费、两作用域恢复|
|39mr产品迁移|Codex已完成A2；openchamber A3验收|复用持续site、受支持投影/范围与证据预算|1–2|自动HTTP请求、持续命中、控制与消费、canary恢复|
|33 A3控制与产品迁移|Codex工具；openchamber取证|实际launch、固定QKV99%、A3分组配置|2–3|控制检查点后两作用域；不继承A2分组|
|41/26fia/20产品迁移|openchamber提交A3构建；Codex审查适配|CANN9.1当前Native库完整构建身份及审查|2–3|同stream对象区间写/消费/恢复，真实HTTP关闭闭环|
|DeepSeek第二批及8|openchamber只观测/标定；Codex适配|V4-0731 W8A8 0.32/CANN9.2.1、源码指纹、噪声标定|2–4|实际site/EP映射、故障超噪声、恢复与资源零；不是V2资格继承|
|r6发行|Codex|候选源码/范围明确、CPU门禁、包/安装一致|0；A3另算|三wheel+源码+SHA+无设备安装+交付步骤|

产品最终关闭必须黄区真实CLI→控制通道→Worker→用户请求→关闭恢复贯通，由openchamber实测回传；Codex不能单凭A2或CPU声明product_closed。未具备条件的能力默认关闭。r5.1仅登记；r6已冻结，新代码或工具使用独立补充件或后继版本。耗时记录，不调整门禁或优化。

本轮补充：Batch、39mr、33已完成A2/0.26/Qwen3.8社区W8A8/TP2公共HTTP产品入口两个作用域，见[实测范围与独立复核](product_r6_20261123/README.md)。以上三项的下一步是A3适配与验收，不再列为尚未开始的A2开发。33仅覆盖已测长prompt与分组，A3仍关闭。黄区本地运行提取工具，回传提取文本，完整summary和原始回执留在黄区，只回传SHA；不等待summary全文。


2026-11-23调整：product_closed 要求12项全部在适用环境的产品模式下完成，不以验收模式代替。DeepSeek适用项同样需产品模式。12项为18、22、39op、39mr、Batch、41、17、8、28、20、26fia、33。17只限视觉路径；8只限MoE；不适用须据实声明，不能虚报通过。r6可冻结已通过子集，r6冻结不代表product_closed。

11-23补充：r5.1已按文本证明发布；A组产品执行不等待B组。统一工具ZIP已补齐DeepSeek capture/observe，Native三项仍按构建身份清单等待黄区对现有产物提取，不重建。

## r7 N 批蓝区更新

41/26fia/20 的 A2/0.26/CANN9.1/Qwen3.8 W8A8/TP2 产品接线与两个 owner 真实 HTTP 验收已通过独立审查，Native C++ 未改。A3 构建身份审查、A3 产品执行及登记仍由黄区回传后继续；八项共享回归正在进行。D/P 继续独立等待其运行证据，不阻塞 N 在材料齐全后单独冻结。
