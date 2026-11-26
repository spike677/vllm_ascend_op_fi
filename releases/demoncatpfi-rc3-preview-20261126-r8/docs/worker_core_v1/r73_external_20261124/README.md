# r7.3 A2 复测与门禁

组合：Ascend910B4（A2）、vLLM 0.26 / vllm-ascend 0.26.0rc1、CANN9.1、Qwen3.8 社区 W8A8、TP2，卡2/5。
两个示例均通过真实 HTTP 请求、公共 ProductClient、原 Worker Runtime；同组 Worker 两个独立 owner。
不是黄区外部能力的替代验收，也没有新增 A3/DeepSeek/EP 或免许可资格。

| 示例 | 每个 owner / 每个 rank 的 observed/eligible/attempted/applied | changed | consumed/restored | failed |
|---|---:|---:|---:|---:|
| external.gain_three_quarters | 1932/1932/1932/1932 | 1930 | 1932/1932 | 0 |
| external.exponential_perturb | 12/12/12/12 | 12 | 12/12 | 0 |

每 owner 三条故障 HTTP 请求，每条4生成 token（prefill + decode），覆盖全 rank 的所有匹配调用。
模型示例另有5-token超范围负例；采样示例另有该负例及temperature=0贪心负例。
变化调用数小于应用数不是失败；changed_elements 单独记录在独立审查回执。

- 模型示例实际消费者为 aten::index、npu::npu_dynamic_quant、vllm::unified_attention_with_output。
- 采样示例实际消费者为 aten.div_.Tensor；原采样函数生成噪声后，真实除法读取修改的 q；设备读取完成后恢复 q。
- 每个 owner 均通过区域字节恢复、故障资源清零、关闭后新金丝雀 hidden/logits 精确恢复及 detach。
- q 的恢复不会撤销已经选出的 token；不要求当前故障请求回到无故障轨迹。
- 同一现有 tools/review_product_mode.py 独立复核为 PRODUCT_BLUE_REVIEW_PASSED。
- CPU 完整门禁1195项：1193通过、2项既有废弃研究入口有理由跳过。

## 窗口与取证

| 窗口 | 耗时（秒，含加载与退出） | 结果 |
|---|---:|---|
| r7.2 RPC 诊断 | 194.781 | 正常/延迟/结构化拒绝48调用全部对齐；rank0抛异常后复现rank1落后一拍 |
| r7.3 模型示例 | 276.309 | 两作用域通过 |
| r7.3 采样示例 | 267.442 | 两作用域、贪心/范围负例通过 |

原始回执留在本地/124，只提交审查摘要、原件 SHA、源码 manifest 与隔离摘要。
模型/采样快照的代码与最终接口一致；后补 external_control helper 只经过CPU控制测试，实机结构化回包模式由独立RPC诊断验证。
打包版本、文档与后补helper不混写成原实测manifest；最终发行另有 source-review 身份和安装检查。

三个验证窗口均结束并解除隔离；另有两个120秒有界预清理窗口，已自动结束。只对独占映射卡2/5的外部容器执行docker stop，
不修改Docker入口、调度配置、驱动、CANN，不对其他卡执行操作；日志保存停止对象、镜像和绝对时间。
预清理第一次启动发生日志字符串语法错误，未执行任何stop，修正后才运行；记录不将这个CPU脚本错误当NPU失败。
vLLM退出时Python resource_tracker提示与故障Runtime零资源分开记录，不宣称检查了全部宿主IPC对象。

[模型示例独立审查](model_a1_review.json) · [采样示例独立审查](sampling_a1_review.json) · [RPC诊断](rpc_review.json) · [CPU门禁](cpu.json) · [隔离记录](isolation_summary.json)
