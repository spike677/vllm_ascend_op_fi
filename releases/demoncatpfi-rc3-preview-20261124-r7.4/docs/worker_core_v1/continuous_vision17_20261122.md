# 17：Qwen3.8 视觉公共入口持续验收

2026-11-22，结论：**A2 / Ascend910B4 / CANN9.1 / vLLM0.26.0 + Ascend0.26.0rc1 / 社区 Qwen3.8 W8A8 / TP2 公共入口两作用域通过**。这不是 A3 或 TP1 结果。新增开发线登记，不修改已经冻结的 r3；加入下一次预览候选。

## 实际路径与机制

`LLM.enqueue(image + 新 multimodal_uuid)` → 当前 scheduler 的 encoder feature → `_execute_mm_encoder` → `model.visual.patch_embed.proj` → `AscendConv3dLayer._forward_conv` → 实际 `aten::conv3d` → 输出 reshape → `aten::add` 位置编码 → 视觉 transformer → 本请求 encoder_cache → `embed_input_ids` → 原文本模型 → hidden/logits。

保持原 kernel/stride=(2,16,16)、原图像、权重、bias、dtype；只在 fault 阶段把实际 Conv3d 参数 padding 从 (0,0,0) 改为 (0,7,7)。每个真实 patch 输入为 (2,16,16)，输出空间仍为1×1×1。没有改输出值或预处理像素来制造差异。该 patch 投影的权重在两 rank 复制且哈希相同；后续视觉 attention 存在 TP 分片，不能说整个视觉塔复制。

适配器按实际 scheduler 请求和 feature 对象建立 encoder→cache→embedding→模型的关联。四阶段每次使用新 UUID、关闭处理缓存；每次 prefill 必须实际运行 encoder。decode 的 `_execute_mm_encoder` 空入口仍按原实现返回 []，不算一次视觉调用，也不人为重跑 encoder。两作用域使用同一组 Worker；每阶段4生成token。

## 结果

|每 rank、每作用域|observed|eligible|attempted|applied|changed|consumed|restored|failed|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|fault 实际 patch 调用|1|1|1|1|1|1|1|0|

两个 rank、两个作用域均满足以上记录。消费者为实际 `aten::add`，以共享存储及值检查证明读入，再记录同stream事件并等待完成；后续视觉缓存和 embedding 消费也有独立对象关联检查。两 rank 的 hidden 和 logits 都变化；OFF/health 的全部4步完整指纹精确回到 normal。所有阶段全有限；不以生成文本质量或 token 改变作为验收。

**恢复类型为 CALL_ARGUMENT_ONLY_NO_STORAGE_OVERWRITE。** 注入不写张量存储；验证原输入/权重/bias字节未变，正常调用参数保持零padding、wrapper释放。`bytes_restored` 在此表示无被污染的原始存储，不声称把故障 Conv3d 输出恢复成正常输出。租约/钩子/在途计数清零，无许可创建与额外 advance 拒绝均通过。

故障/正常请求耗时：作用域0 **1.0550369465×**，作用域1 **1.0037788918×**。固定64×64测试图、单图、max_num_seqs=1、max长度1024、单chunk；不外推视频、多图、其他形状或 encoder DP。

## 证据与代码

- 实机源码清单：`1496957b6fd55255969c81ebe9cc396c0bb1ee50f22c1919ce15fa224c1485d5`。
- 独立回执复核：`experiments/worker_core_v11_20261118/continuous_public_vision17_review_20261122.json`。
- 源码审查：同目录 `continuous_vision17_source_review_20261122.json`；原回执仅本地/124保留，不提交完整张量。
- 实现：`fault_injection/sites/vision17.py`、Worker `continuous_vision17.py`；复用 `ContinuousProvider`、MultiRankTaskClient 和原四阶段生命周期。
- CPU：986项、0失败/错误、2项已有废弃研究skip；另行新增的真实V4.1 metadata切片测试通过，未混入这986项统计。

## 窗口与失败保留

|窗口|UTC开始—结束|用时秒|结论|
|---|---|---:|---|
|vision17-a1|06:39:37—06:44:49|311.974459|无故障视觉预检，两次完整输出精确稳定|
|vision17-a2|06:52:32—06:57:34|301.406569|观察器错误拒绝 decode 的空encoder入口；未进入故障阶段|
|vision17-a3|06:59:07—07:05:59|411.617631|修正空入口契约后公共两作用域通过|

全部窗口仅卡2/5，采用有界精确设备容器停止器；实际外部停止清单均为空。每个窗口结束均停止清理器并确认派发能力恢复；无Docker入口/宿主驱动/CANN变更。a3 shutdown 有 vLLM process-manager/shared_memory 警告，保留为宿主退出限制；不冒充原进程没有任何资源警告。任务内部租约、钩子、许可已在退出前独立确认清零；窗口自有容器已清理。
