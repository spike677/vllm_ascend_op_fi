# product_closed剩余工作与排期依赖

截至本轮，r5验收模式不等于黄区产品闭环。以下往返数为依赖齐全且一次通过时的估计，不作为验收结果。

|工作|归属|前置条件|预计往返|完成证据|
|r5 14组合审查、r5.1登记|负责人转交提取文本；Codex审查发行|规范提取文本、完整身份、原件/summary SHA|1；缺字段另加1|14条审查与YELLOW_DEDICATED_INSTANCE，冻结新发行|
|18/39op/22/28/17黄区产品CLI接线|openchamber运行；Codex契约/复核|r5产品RPC与可信会话、身份、17内置金丝雀/spawn|1–2|真实用户HTTP请求自动发现、两owner范围、关闭后新canary恢复|
|Batch产品入口|Codex开发A2实测；openchamber A3验收|并发decode真实批次映射、max_num_seqs8、稳定基线|开发后1–2|B1不注入、B>=2全真实行共享scale、真实QMM消费、两作用域恢复|
|39mr产品迁移|Codex开发A2实测；openchamber A3验收|复用持续site、受支持投影/范围与证据预算|1–2|自动HTTP请求、持续命中、控制与消费、canary恢复|
|33 A3控制与产品迁移|Codex工具；openchamber取证|实际launch、固定QKV99%、A3分组配置|2–3|控制检查点后两作用域；不继承A2分组|
|41/26fia/20产品迁移|openchamber提交A3构建；Codex审查适配|CANN9.1当前Native库完整构建身份及审查|2–3|同stream对象区间写/消费/恢复，真实HTTP关闭闭环|
|DeepSeek第二批及8|openchamber只观测/标定；Codex适配|V4-0731 W8A8 0.32/CANN9.2.1、源码指纹、噪声标定|2–4|实际site/EP映射、故障超噪声、恢复与资源零；不是V2资格继承|
|r6发行|Codex|候选源码/范围明确、CPU门禁、包/安装一致|0；A3另算|三wheel+源码+SHA+无设备安装+交付步骤|

产品最终关闭必须黄区真实CLI→控制通道→Worker→用户请求→关闭恢复贯通，由openchamber实测回传；Codex不能单凭A2或CPU声明product_closed。未具备条件的能力默认关闭。r5.1仅登记；新代码只进入后继r6。耗时记录，不调整门禁或优化。


2026-11-23调整：product_closed 要求12项全部在适用环境的产品模式下完成，不以验收模式代替。DeepSeek适用项同样需产品模式。12项为18、22、39op、39mr、Batch、41、17、8、28、20、26fia、33。17只限视觉路径；8只限MoE；不适用须据实声明，不能虚报通过。r6可冻结已通过子集，r6冻结不代表product_closed。
