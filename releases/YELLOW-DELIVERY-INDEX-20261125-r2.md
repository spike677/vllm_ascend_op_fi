# 黄区固定交付索引：20261125-r2（T2控制拒绝修复）

本索引是新增补充，不修改 `yellow-delivery-20261125` 的索引、汇总ZIP、r7.4.1或旧参考适配器包。负责人从GitHub下载后转交；不需要黄区服务器直接连接GitHub。

原完整材料仍从 [20261125固定索引](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125/releases/YELLOW-DELIVERY-INDEX-20261125.md) 取得。

|文件|字节数|SHA256|用途/任务|
|---|---:|---|---|
|[reference-control.zip](https://github.com/spike677/vllm_ascend_op_fi/raw/refs/tags/yellow-delivery-20261125-r2/releases/yellow-reference-control-20261125-r1/reference-control.zip)|22837|`c79a41de6fe08fb0b72c57eddcf4d1130f654c51f56c08a09ba194cc1678f0d1`|T2：可信Worker控制边界预检、结构化拒绝；包内manifest逐文件SHA、接线和A2独立审查记录|

本次仅增加参考适配器控制拒绝保护。A2/0.26/CANN9.1/社区Qwen3.8 W8A8/TP2：四项负例全部exit4，同client正常请求继续完成、两个rank完整hidden/logits四步SHA精确一致、资源为零。原回执SHA `22b27d00502efc8573216c89196a8d35fd31062dbac7702212b45db85751cb56`；源提交16f25855，源清单SHA `6a874345f2e00d3951ff7d1c91788be69ac8870a945a8da95126826dbc97ad68`。窗口331.238秒。

包内 `dcat_reference_control_guard.py` 可与旧参考包并存；黄区可信Worker包装类按HANDOFF加入mixin。不会自动改变公共机制、验收许可或组合登记。实际forward/kernel异常不被吞掉。

第12行v2权重行、第32行类FP8、第34行尾数截断的新实现仍待各自实机审查，本索引不授予这些新变体资格。后续完成后再追加r3索引，不改本文件。
