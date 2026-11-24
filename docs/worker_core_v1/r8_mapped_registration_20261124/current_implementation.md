# r8 当前实现免许可门禁（2026-11-24）

开发线整合r7.3.1采样接线及r7.4持久区域后，先重新做验收模式，再审查BLUE记录，然后独立重启免许可复核。未以更新旧哈希代替验收。

组合仅为A2/Ascend910B4、vLLM0.26.0/vllm-ascend0.26.0rc1、CANN9.1.0、社区Qwen3.8 W8A8、TP2，代表机制18.api_half。源码实现指纹`d2dcac1ac38e8ebbaff1d9a90e456c9f3c7ac3d100b679cad93c1127e8e77ccc`；登记SHA`d3c5082dfb6342270c639ff4e4b2921f62d4120ab47cd60b13aad1fef8e0b7fb`。登记不包含A3、TP1、V4或其他能力。

验收模式：r8int1124a1，286.878秒；两个owner、每owner四个真实HTTP请求；实际缺许可/未知profile负例及正对照、消费者、字节恢复、资源清零、金丝雀均独立审查通过。对应回执和关联见r8int1124a1_review.json及blue_registration_review.json。

免许可：r8intreg1124a1，292.431秒；两个新owner，capabilities声明PRODUCT_REGISTERED并绑定同一登记SHA；全过程未安装临时许可。CLI handler未知机制和未授权操作者exit4；两个owner恢复和NO_PERMIT_USED清理通过。不是黄区CLI或socket传输实测；A3仍要执行自己的代表组合正例与未登记组合负例。

CPU整合门禁1357项：1355通过、2项既有研究入口跳过；打包5项、当前登记身份17项补充回归通过。首次整合发现漏同步v1采样和控制文件，已补入原发行内容后重跑通过。旧记录拒绝新指纹、错误启动参数、队列ID过长均保留失败记录，未伪造为NPU通过。两个队列ID拒绝发生在设备窗口之前。

窗口绝对UTC时间、原件SHA与解除隔离见每份review及intermediate_windows.json。无Docker入口/驱动/CANN更改，无外部数据删除。当前实测完成后隔离已解除。原始回执私有保存，Git仅审查摘要。

r8仍未冻结：黄区逐组合产品证据和负例、33的A3三形状控制、D0/D1及D包均未闭合。未新增YELLOW登记；负责人确定最终收口范围并审查真实材料后再发行。
