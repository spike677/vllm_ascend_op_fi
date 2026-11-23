# r7.1 Fault15 A2 TP2新回归

已在r7基线上重放15，并按新候选源码完成真实产品回归。组合：Ascend910B4/A2、vLLM0.26、vllm-ascend0.26.0rc1、CANN9.1.0、社区Qwen3.8 W8A8、TP2。

- 同组Worker两个owner，每owner三条不同4-token请求及5-token负例；每rank每owner192次真实匹配，合计768次；八计数相等（failed=0），changed=applied。
- 每owner rank0 changed_elements=1424607、rank1=1424622。视图初值正确且脱离存储，真实impl写临时输出，原base保持旧值；gate读旧base，o_proj读gate的真实结果。独立审查768条证据。
- OFF/health新请求完整hidden/logits精确恢复，活动租约/钩子/inflight清零；不把故障请求历史改回正常结果。
- 窗口267.825740秒；只停止卡2上一个ATK容器，完整名称/镜像/绝对时间在window.json。有界清理器已退出，派发能力恢复，Docker入口/配置未改，未删除外部容器或数据。
- 服务器退出仍报第三方resource_tracker信号量/共享内存告警；与已核验的注入资源清零分别记录，不宣称修复第三方退出问题。

原始回执留私有目录，公开独立review及其原件SHA。性能倍数未测不推算，耗时不作门禁。语义为ND别名断开，不含私有格式触发条件。A3受信候选不是A3已通过；见[执行单](../alias15_a3_execution_20261123.md)。r7的其他机制结果只在文件核对确认未改的范围内沿用。
