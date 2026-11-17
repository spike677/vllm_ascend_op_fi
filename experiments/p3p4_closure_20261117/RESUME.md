# 继续位置

P3 已提交0c6ec2e，原严格失败仍保留。26 FIA 已在 SmolLM/GLM 的真实公共Worker完成29/39次writer→consumer→恢复，见FIA26_REPORT。GLM无损/B自比NaN及两模型OFF整池统计差异保留；完整新请求hidden/logits恢复不等于全部缓存字节恢复。FIA阶段581项CPU测试通过。
26 residual只读分配调查在两个模型完成，但新运行无NaN，不能推断旧rank1来源；见ALLOCATION_REPORT。
下一项为batch逐行scale/Q及33连续生成。quantization.py、batch_scale_v2_only.cpp和build_quant_v2.py为本地未完成准备，尚未获得NPU资格，不能混入FIA阶段通过范围。

工作目录：本实验；124：`/home/p3p4-closure-20261117`。一次只用空闲物理卡2或5，
使用 `/var/tmp/fault22_npu_2_5.lock` 和已有 `idle_gate`。不得停止外部任务或恢复过期Docker入口隔断。
先检查旧 Provider 的核心方法替换，不能继承为新公共Worker通过。保留所有失败与未观测项。
发布仅实现、总结及哈希，不发布新的原始统计、完整张量、权重或vendor二进制。
