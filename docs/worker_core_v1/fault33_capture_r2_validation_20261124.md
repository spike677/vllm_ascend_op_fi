# 33采集器r2：验证与交付边界

N1独立实现按名加载登记；N2黄区完整报告仍需退补；N3冻结r7.4.1和追加统一正式测试说明已完成；N4 main已依次快进到r7.4和r7.4.1。

- Python相关CPU测试36/36通过；124无设备容器模拟ACL 9/9通过，含9000句柄/launch、容量/长度/未ARMED、仅按名字加载、后到entry、先到entry、查询失败和key过滤。
- 模拟库验证原参数及返回值的转发，不调用ACL设备初始化，不等同A3 ABI验收。
- 真实CANN SDK无设备编译通过；compiler/opp/runtime/asc-devkit元数据均9.1.0，编译器g++11.4.0，架构aarch64。
- 源码：8366字节，SHA `2d472c2e3268080c52bfa3ba497e6239a22f77e062849bbed379e5177578a835`。
- 库：171536字节，SHA `c87db1e157bca28e395e1c8280b0958fdea2d057b29affa21576c727996e023e`。构建回执含源码提交、头文件逐项SHA、版本文件、编译参数、ELF build-id；见同名目录。
- 本轮无NPU任务、无卡抢占、无调度隔离；两个无设备容器均正常退出。此前已解除的隔离没有重新开启。

r2维持原二进制header和原launch参数，只增加句柄来源与每条旁证。按名字加载的header key=0只是临时占位，review输出key=null；不能将unknown=0当作已取得真实tiling key。后到的entry只影响后续记录，不追补旧记录。key过滤模式明确排除name-only，窗口一必须ALL=1。

Python审查器要求新版每个bin都有合法binding sidecar，并核对pid/serial/entry；解码CLI再次核对真实entry。库缺失sidecar、写入失败等均不能判完整。9000条测试同时验证此链，不只是静态查找hook名称。

N2接收仅保留文本证明，可复用9库身份和金丝雀；自改采集器和单形状结果不作为最终窗口一。下一步由负责人转交r2，黄区采集三形状×两rank及目标FIA的唯一API/launch映射。未收到前不进入控制重算，不发r7.5注入资格。

r2是独立工具补充件，r7.4.1产品wheel及所有冻结标签不变。黄区Qwen统一正式测试采用r7.4.1，执行范围见`releases/RC3-PREVIEW-R7.4.1-QWEN-FINAL-ADDENDUM-20261124.md`；本轮不重做已通过的A2双owner任务。
