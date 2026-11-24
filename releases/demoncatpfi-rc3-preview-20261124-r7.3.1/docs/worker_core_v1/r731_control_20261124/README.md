# r7.3.1 A2控制拒绝与统一效果工具

组合：Ascend910B4/A2，vLLM0.26、vllm-ascend0.26.0rc1、torch2.10、torch_npu2.10.post4、CANN9.1，Qwen3.8社区W8A8，TP2。
只登记蓝区验证结论，不授予A3或黄区CLI资格。原始大回执保留私有本地/124目录，本目录提供原件SHA与独立审查摘要。

* `cpu.json`：最终1216项，1214通过，2项废弃研究入口带理由跳过。版本断言第一次仍指向post3导致1项失败，已随post4发行修正并完整重跑。
* `npu_18_final.json`：最终拒绝码、缺许可/未知profile/未知名称、重复开启、重复关闭均exit4，保留rank原拒绝；同引擎完成两个owner正例，资源零。`npu_18.json`为首轮维护证据，最终版以final为准。
* `npu_batch.json`：非串行金丝雀返回PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1，HTTP health200、真实推理继续；随后同client正确串行金丝雀和两个owner正例通过。
* `effect_tool_review.json`：18和Batch各两个新owner，同服务baseline/fault/off，最终提取器重复运行逐字节一致。
* `effect_18_page.json`为2172字节，两个owner均token/text不变、top5变化。
* `effect_batch_page.json`为2234字节，两个owner均token/text/top5变化；按frame去重，各rank eligible/applied/consumed/restored=912，rank1 changed=911有效。
* 两个机制OFF的API字段精确恢复，原件中的区域字节/资源/完整hidden-logits金丝雀恢复通过。完整fault logits没有额外采集，一页明确为null。

五个有界卡2/5窗口约1930秒（包括启动、观察和清理，不是纯kernel时间）；每窗均isolation_removed=true、dispatch_ability_restored=true。停止过的容器、镜像、绝对时间在对应isolation_log。未改Docker入口、驱动或CANN。vLLM退出时提示semaphore/shared_memory resource_tracker警告，单独保留，不把它等同于fault租约泄漏；租约与owner清理由实际Worker审查。

冻结前各次远端源码快照与最终发行源码分开标识；最后的稳定拒绝码由final窗口验证。效果工具最后增加的提取完整性校验在原件上重新运行，未重写采集原件。CPU验证与真实模型验证分别记录，不互相替代。

统一工具的HCCL和黄区CLI适配仍需黄区提供真实入口与原件映射；本目录不宣称已验证HCCL。发布不新增YELLOW_DEDICATED_INSTANCE。
