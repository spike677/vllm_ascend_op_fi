# r7.1 总单 A2 核对说明

这份核对只针对冻结r7.1源码/公开ProductClient，不冒充黄区CLI运行。蓝区没有黄区demoncat.cli和session daemon，CLI语法依据负责人提供的帮助文本；今晚必须使用python -m demoncat.cli。新增r8参考console不是今晚的替换包。

已核对：r7.1冻结包原安装检查；333个实际Native运行时文件与冻结源字节相同；三个Native profile通过真实Worker configure并返回capabilities（rank0/1）；同包审查器复核NPU回执。完整原始运行数据留蓝区私有目录，仅提交审查结果/哈希/隔离记录。

总单v2-r1的三处更正：bootstrap属于黄区r6 A组既有可信包装；Native路径和SHA从受信JSON进入，不是用户RPC参数；capabilities已经census attach，同一client不能重复attach。旧capabilities.py只描述研究路线，不能据其not_implemented判产品41不存在。

六份文档ZIP含完整原始capabilities输出。PRODUCT_CANDIDATE不是产品资格：产品mode仍PRODUCT_ACCEPTANCE_ONLY，product_qualified=false，实际enable仍须许可和现场准入。A2通过也不能将A3登记为通过。

仍须在黄区确认：真实daemon的鉴权/CLI映射、owner生命周期、三项CLI负例及正对照。蓝区原回归负例记录缺许可、未知profile、用户Native路径；未知机制99有公开配置/新helper的CPU负例，不能写成黄区CLI实测完成。
