# Yellow delivery index 2026-11-25 r4.2 — V4 0.30只读观测r3

只增量交付观测工具，原r4.1继续有效。r3不能注入、不授予D13/公共8资格。由负责人转交，不直接联系黄区。

## 交付物与身份

- 标签：yellow-delivery-20261125-r4.2。
- 源码/验证提交：f5dc3768（观测实现提交eafb43fa；之后只有证据、文档及打包器修改）。
- [ZIP](v4-v030-tools-20261125-r3/v030-tools-r3.zip)：994448字节。
- ZIP SHA256：`4bc2af4cf8b715a3c6f4ed55a2e88ea0f46da254cc1b97cf1e7881e7d269f8fc`。
- 源码清单SHA256：`1518830652bce91d38e3c775a8ead6d4e488972af11d24b25c49d12e8efd4062`。
- [逐文件manifest](v4-v030-tools-20261125-r3/manifest.json)、[SHA256SUMS](v4-v030-tools-20261125-r3/SHA256SUMS)、[字节对照](v4-v030-tools-20261125-r3/shared-byte-comparison.json)。
- [CPU](v4-v030-tools-20261125-r3/cpu.json)、[独立解压CPU](v4-v030-tools-20261125-r3/cpu-isolated.json)、[A2审查](v4-v030-tools-20261125-r3/a2-review.json)、[隔离记录](v4-v030-tools-20261125-r3/isolation.json)。

r2既有194个Python文件中193个逐字节相同；唯一变化为dsv4_observe_probe.py，新增4个Python文件。capture、noise、probe_support、probe_extension、旧提取/噪声审查工具均相同；全部fault_injection从r2包原样继承。没有修改r7.4.1或r8机制文件。包自带所需项目依赖，Python3.12，采集使用部署自带torch/torch_npu；只读文本提取使用标准库。

## 黄区完整顺序

1. 原P2/P3不停止。r2做已有D0/D1时不能通过options补出浮点或消费者事件。
2. 若G1尚未启动，独立解压r3，把r3目录作为该次启动PYTHONPATH第一项，扩展类不变。按原严格绑定/插件规则核对。
3. 先完成全部G1噪声采集：observe必须关闭；formal/canary按既定prompt-manifest、顺序和样本数。
4. capture全部stop后，在同组Worker按r3执行单做定点观测和卸载恢复。1/2 token取证不是正式4 token注入验收。
5. 用新extract_dsv4_targeted_record.py生成并verify文本；D13真实router消费先审，公共8单独审combine行归属。
6. 若G1已用r2启动，不热替换Worker模块。保留G1，r3定点采集需要单独安排重启；不能以文件相同为理由伪称活跃进程已升级。

四份完整文档均在ZIP内：

- [r3执行单与实际options](../docs/worker_core_v1/v4_v030_probe_r3_handoff_20261125.md)
- [定点补采单](../docs/worker_core_v1/v4_d1_targeted_capture_20261125.md)
- [公共8审查方案](../docs/worker_core_v1/v4_public8_d1_review_plan_20261125.md)
- [库行13独立放行方案](../docs/worker_core_v1/v4_d13_independent_release_20261125.md)

## 验证结论与边界

CPU55/55、独立解压55/55。最终A2（Ascend910B4、0.26/0.26rc1、CANN9.1、社区Qwen3.8 W8A8、TP2）烟测通过：每rank20条定点算子、16条消费者候选、20个完成事件、12316个浮点样本值，无错误/截断。观测前/中/后输出一致，原方法身份恢复；r2不变capture四步及采样关联通过。消费者语义仍需源码审查，A2无EP不代表V4资格。

实机提取548段，最大1634字节，两次逐字节相同，verify通过。全部原件留蓝区，交付审查页及SHA。源码清单与两份CPU、最终A2和ZIP一致。旧首轮诊断已作废，详情在隔离记录；不引用其结果放行。

最终验证22:13:16–22:20:55（北京时间），458.882秒；原30分钟授权截止22:37:06，监控已停止，无长期占卡。

D产品路由准备另在r8开发线推进，当前不能执行：0.30、TP8+EP、2500–4092 token的基础准入CPU已准备，不在本只读ZIP中开启产品模式。39op/26fia同次证据定性、r8全项A2回归和A3登记仍按主线推进。

