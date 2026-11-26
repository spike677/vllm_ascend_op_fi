# 33 A3 TP2快速取证判定（r7/r7.1不开放注入）

今晚只做Qwen3.8 W8A8、Ascend910_9382、CANN9.1.0、v0.26.0rc1-a3、TP2。执行[取证命令](rc3_r6_paged33_a3_acceptance.md)第1、2节，但源码/wheel统一换成r7或r7.1，跳过其中所有TP1安排。两个rank各自保留真实Q/K/V与原生输出，不回传完整张量。

每rank回传以下小段文本，每段≤4KiB，未知项为null，不能用A2预期值填入：

- 部署：芯片原名、TP/rank/逻辑设备、实际软件版本及运行时CANN来源、模型config/量化描述/全部权重索引SHA、发行清单SHA。
- 调用：实际算子名、prefill/decode、Q/K/V shape/dtype、布局、scale、实际KV长度、cache block大小、sparse模式。
- launch：实测tiling key、核数、scheMode、实际S2区间；每个key恰好一次；launch原件字节/SHA、对应真实QKV文件SHA、serializer源码SHA与字段偏移、实际host/kernel源码身份。key相同不能单独证明布局相同。
- 完整性：捕获API、是否截断/达上限、未覆盖的launch和字段。不把null当相同。
- 控制：`control-rankN.json`全文，包括两轮逐位相等元素数/总数、有限性、重复精确一致、设备身份、实际分块。各轮≥99%，重复100%；不修改阈值。

蓝区参考仅作对照：decode key=5100000000010200203、核数10、scheMode1；Q=[1,12,256]，K/V=[L,2,256]，BF16，scale=0.0625，block=128，实际L=2305～2307。控制按512分块及实际尾块，故障候选256；算术BF16_MM_FP32_OUT_PBF16_LSE_FP32。prefill key=5000000000000200203、核数20，不能拿prefill记录当decode证据。

判定：

1. 两rank实际布局/分组/算术与蓝区相符、身份和源码来源完整、全部控制通过：Codex可准备r7.2的A3 TP2受信候选，补CPU/安装门禁和两作用域执行单；不修改r7/r7.1，不直接登记合格。能否当晚发布取决于证据和门禁完成，不能预先承诺通过。
2. key、分组或算术不同：记录“A3分组待适配”及逐字段差异；不沿用A2常量、不强放行。
3. 缺字段、控制不足99%、不稳定或非有限：停止注入，记录待补证据或控制失败。未提供A3数据时状态为待取证。

候选验收仍须同组Worker两个owner、全rank持续触发、真实消费者、OFF/health精确恢复、资源清零、三项负例；产品回执独立审查通过后才讨论登记。耗时只记录。其他能力继续，不等待33。
