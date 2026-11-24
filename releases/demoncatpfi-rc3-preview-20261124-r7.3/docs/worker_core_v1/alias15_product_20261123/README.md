# Fault15 B档：ND视图别名断开产品验收

**A2 TP2产品入口已通过；不含私有格式触发条件，不代表历史NC1HWC0缺陷复刻。** 独立分支 `codex/alias15-product-20261123`，不进入冻结r7。A3入口仍拒绝，等待现役普查和部署适配审查。

组合：Ascend910B4、vLLM0.26 / vllm-ascend0.26.0rc1、CANN9.1.0、社区Qwen3.8 W8A8、TP2、eager、单请求单chunk、4生成token。真实HTTP请求自动发现，无宿主逐请求通知。

## 实际实现和调用链

`Attention.forward` 自然分配ND输出base并建立视图 → `AscendAttentionBackendImpl.forward(output=view)` → 原生实现写入output → 外层返回原base的视图 → 真实 `attn_output * sigmoid(gate)` → o_proj。

命中时仅将该次impl的output参数改为独立存储、初值完全相同的临时视图。原生impl执行原有真实写入，外层base没有重新绑定，仍保留分配时旧字节。真实gate读取旧base，o_proj读取gate结果。没有在模型中增加copy_、没有把调度索引当激活、没有修改stride或事后消费者输出。CPU测试的确定性allocator替身只用于单元测试，未进入NPU路径。

实现：`fault_injection/sites/alias15.py`；接线：`ContinuousRequest.install_site → ProductRequest`；生命周期、许可、request_plan和多rank协调沿用现有框架。每次调用持有原base、临时视图及RegionLeases的对象/存储身份；同stream事件确认消费完成，`release_unmodified`验证base字节和版本仍未改动后释放，**不把旧快照写回**。后续合法原地更新导致验证失败时保留失败租约，拒绝伪装恢复。持久KV/SSM状态不在此选点范围。

## 结果

同组Worker两个独立owner；每owner三条不同用户文本及一个5-token范围负例；每条合格请求16个full-attention层 × prefill及3步decode = 64次/rank。GDN分支无该自然output视图写入边界，未人为制造调用。

|作用域|rank|observed/eligible/attempted/applied/changed/consumed/restored|failed|changed_elements|
|---|---:|---:|---:|---:|
|0|0|各192|0|1,424,607|
|0|1|各192|0|1,424,622|
|1|0|各192|0|1,424,607|
|1|1|各192|0|1,424,622|

合计768次，逐次证明临时存储与原base独立且初值相同，原base写后SHA仍为旧值；临时视图的实际应写值与base不同，真实gate读取旧base，o_proj读取gate结果，设备消费完成。`changed`是变化调用数，`changed_elements`是每次base与应写值逐元素字节差异的数量之和，二者不混用。

三项恢复通过：原base区域字节未被注入器改写、每请求租约/hook归零、关闭后新请求完整hidden/logits金丝雀精确恢复。污染中的请求不要求回到正常轨迹。两个owner的Worker身份保持不变。窗口261.745秒，卡2/5有界隔离已解除；详见 [窗口](window.json) 和 [独立机理审查](review.json)。耗时倍数未测量，不从窗口时间推断性能。

## 独立复核与限制

`tools/review_alias15_product.py` 先执行通用产品审查，再独立检查实际请求关联、初值、独立存储、base未更新、消费者、changed/changed_elements一致性及恢复，不读取“命中成功”文本替代证据。45项相关CPU测试通过，包含后续合法写入时拒绝恢复、消费者缺失、非独立存储、计数矛盾负例。

base来自真实torch.empty，旧值没有初始化语义；没有人为清零，保留初值是否有限。当前范围仅ND BF16连续完整区间、已审核源码、自然输出边界及gate消费者。不能据此宣称私有格式、其他形状/模型、所有Attention实现或A3已通过。未绑定、未获许可及A3当前部署仍默认拒绝。
