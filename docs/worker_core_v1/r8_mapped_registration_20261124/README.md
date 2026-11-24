# r8 运行库映射身份与免许可正例

组合仅为蓝区 A2 Ascend910B4、0.26、CANN 9.1.0、社区 Qwen3.8 W8A8、TP2。不是 A3 资格；本目录不发布 r8，不登记任何黄区组合。

## 运行库身份

原容器 overlay 下 `/proc/self/maps` 的设备号与路径打开后的 fstat 不同，原门禁正确返回 `MAPPED_FILE_IDENTITY_CHANGED`。没有放宽门禁，也没有拿路径 SHA 代替实际映射身份。

从固定镜像 `sha256:c9c9432b69767cf1855efad44f20eaa8ab02520ae05b75c936e67ea4e7eb2702` 复制原始两个库（共 1,265,968 字节）到本项目目录，设只读，再只读挂载到容器内原解析路径。原镜像、宿主驱动/CANN、Docker 配置和其他任务不变。无设备检查确认 maps 与 fstat 的设备/文件身份相符、库字节与固定镜像一致；随后实机两个 rank 均通过同一严格检查。

|库|字节数|SHA256|
|---|---:|---|
|libascendcl.so|329544|123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480|
|libruntime.so|936424|7bc2b610dff552df2f114c95701156b4aba43ffcba24e06e3c249e1cf57bbc21|

CANN 安装元数据 SHA：`a9b17ef608ebeb70ade09cde1c12a545eb5e8f3e5312f6806ca0d368b0464848`。这是需要部署方明确配置的映射方案，不是任意 overlay 的身份回退。无法证明映射身份的组合继续拒绝。

## 首轮失败与修正

`r8reg1124a1` 在身份、capabilities 和 attach 已通过后，prepare 拒绝；没有运行故障请求。原因是 ProductProvider 构造时未保存 mechanism，注册路径的逐操作复核读取该字段失败。第二个问题是此时尚未执行 prepare/bind，恢复触发 owner 不匹配。

修正只在未发行的 r8：保存实际机制标识；新建且尚未绑定的 ProductProvider 允许恢复动作绑定当前 Runtime owner，仍核对 binding、owner/generation 与资源。已经绑定的实例不能借用别的 owner。注册身份仍逐次复核，漂移继续拒绝。CPU 用真实 ProductProvider 验证注册复核，实际 Runtime 的 world_size=1/2 测试覆盖 prepare 之前拒绝与各阶段部分 rank 失败后的资源释放。

旧源码的带许可正例不能替代修复后证据。`r8map1124a3` 重新验收后重新生成 BLUE 记录，再启动免许可正例；最终结果见同目录独立复核 JSON 和总结。旧开发记录的 SHA 不作为新版本资格。

## 最终复核通过

`r8map1124a3`：277.745秒，修复后两个owner、真实缺许可enable拒绝/回滚、同客户端后续阳性、未知profile引擎仍健康，均通过。`r8reg1124a2`：276.873秒，两个owner各4个真实HTTP请求（3个范围内正例、1个超范围负例），所有rank的PRODUCT_REGISTERED身份一致，未安装临时许可；可信CLI helper的list/enable/status/disable通过，未知机制99和未授权principal均exit4。

独立审查确认全rank覆盖、消费者、区域恢复、Runtime资源零、金丝雀精确恢复，同一组Worker两个不同group owner；完整回执SHA见r8reg1124a2_review.json。登记SHA为`938f2e2ca9cb778de3f09f15abdd44afe9c7a861b27a3947a31b3c675062a27a`，仅对应这一个BLUE组合。

CPU最终门禁1248项：1246通过、2项带理由跳过，0失败/错误，见上级r8_development_cpu_20261124.json。早期一次完整CPU调用因未向子进程继承测试PYTHONPATH而失败，保留原报告；修正调用环境后全量重跑通过。没有把CPU或旧源码验收算作本次实机正例。

## 证据边界

真实 HTTP 推理、同组 Worker 两个 owner、全 rank 覆盖与真实消费者、超范围请求不注入、字节恢复/资源零/金丝雀精确恢复均由独立审查器读取原回执检查。group owner 与各 rank 的 provider owner 分别保存并关联，不要求它们字面相同。

注册正例使用可信 `TrustedProductSession` 的 list/enable/status/disable 分派器，含未知机制和未授权 principal 负例；这是 CLI helper 的服务端路径，不是黄区 socket/CLI 已验收。黄区 CLI 和 master 冒烟仍需黄区回传。

仅新增精确身份的 `18.api_half` BLUE_COMBINATION。A3、TP1、其他机制、其他实现 SHA 不得借用。无需临时许可的正例与缺许可验收模式负例分别记录。运行时间只记录，不作门禁。

## 隔离与恢复

每窗口使用共享资源锁和最长 2100 秒的卡2/5有界清理器；仅 docker stop 占用这两卡的非本项目容器，不改 Docker 入口/配置、不删外部容器/数据。控制器结束时停止并 join 清理器，派发能力恢复。每窗口原始 controller/isolation 回执留存私有目录，公开 SHA 与时间见 windows.json。

本轮7个实际窗口共1934.735秒（约32.25分钟，包含模型启动和失败窗口，不是纯kernel时间），全部确认隔离解除。首次过长case ID在设备运行前被拒绝，不计NPU窗口。停止外部容器的事件数逐窗口记录，不把旧停机授权扩展到其他卡。

退出时 Python resource_tracker 的 semaphore/shared_memory 警告保留；故障 Runtime 资源零不等于全部宿主 IPC 为零。首轮 registered 失败不计为产品通过，隔离解除与引擎退出也不等于该次租约正常恢复。
