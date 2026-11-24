# r8免许可验收草案（尚未发行）

组合分别为A3 Qwen3.8 TP2/CANN9.1.0，以及仅在D0/D1和8注入通过后建立的A3 DeepSeek V4 TP8+EP/CANN9.2.1。本页不产生登记，也不授权使用未审查组合。

前置：最终r7.x和D包验收完成；逐组合一页总结、原件SHA、两个owner和每rank关联经过审查；登记键包含硬件原名、设备、TP/EP、实际软件栈、模型config/量化/全部索引SHA、范围、变体、源码指纹和映射运行库身份。缺失不补默认值。按最终代码指纹建立记录，旧r8的A2正例不能自动迁移到新指纹。

负责人转交后，黄区可信宿主安装正式bootstrap和鉴权回调；用户入口仍为黄区CLI。部署配置选择PRODUCT_REGISTERED；不得同时安装临时验收许可。所有bootstrap、capabilities和状态回包按rank、owner、调用关联核对。

每个登记组合选择该组合实际通过的一个代表机制：

1. 确认没有临时许可，capabilities声明登记来源与登记SHA；先运行正常金丝雀。
2. 第一个owner执行attach/enable，发送范围内真实用户请求，status收集每rank八项计数、消费者和效果；disable完成排空、区域恢复、资源清零，关闭后金丝雀通过，detach。
3. 同组Worker第二个新owner重复；不能以同一owner的第二次查询替代。
4. 受信测试配置仅更改一个身份字段（如TP或config SHA），要求未登记组合结构化拒绝、CLI exit4且保留拒绝码，原引擎存活、无残留；恢复配置后正对照通过。不得伪造运行时观测字段去获取资格。
5. 另核对缺字段、跨TP借用和身份漂移；都不能退回临时许可或用历史BLUE记录放行。验收模式仍须许可。

Qwen金丝雀精确恢复；DeepSeek只有在噪声协议审查通过后才用demoncat.dsv4_0731.noise.v1，token一致、hidden/logits/logprob不超标定上界。区域字节和资源恢复仍是精确条件。

回传用统一一页工具：两个owner、每rank、资格记录SHA、确认无许可、效果、三项恢复、拒绝码/引擎存活、原件SHA，单页不超过4KB。此处不填假设的CLI参数或未发行包SHA；最终r8冻结时替换成实际入口与包清单，再交负责人转交。

当前状态：r8仅开发。历史A2登记实现指纹为3dd010c686cafcabc1d7b5e8703cb12e100b0ead0090a52b27fad93dd6734739；维护同步后指纹改变，CPU覆盖IDENTITY_MISMATCH拒绝，未重做当前代码的A2免许可正例。没有新增YELLOW登记。
