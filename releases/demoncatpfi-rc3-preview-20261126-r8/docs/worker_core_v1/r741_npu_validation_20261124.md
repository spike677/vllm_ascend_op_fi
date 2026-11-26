# r7.4.1 定向 NPU 验收：18全输出变体与外部图片路由

2026-11-24，**四项定向产品测试通过**：新增18均方代理、外部Conv3d输出示例，以及原18/17回归。首次图片尝试失败已保留，修复后重新验收通过。
本报告不是r7.4.1发行冻结或A3资格登记；既有标签和包未改动。

## 组合、源码与执行

- 124物理卡2/5，Ascend910B4 A2，TP2，标准mp Worker；每项两个作用域使用同组Worker，实际HTTP用户请求，无宿主逐请求注入钩子。
- vLLM 0.26.0、vllm-ascend 0.26.0rc1、torch 2.10.0+cpu、torch_npu 2.10.0.post4、CANN 9.1.0；社区Qwen3.8 W8A8，eager，prefix cache关闭，max_num_seqs=1，max_model_len/max_num_batched_tokens=1024。
- 原18/17及新18：源码 `bfb8bc9e`，清单SHA `18c73e8013ea3b8c61fd3e2e2204a473bdc637672fa10ad56e19eda869ceeff5`。
- 图片修复复测：源码 `acd82e03`，清单SHA `ccf7db983fed1845a4244f7b15d1510524b24443c331bff658755f4cdd510d8b`。变化仅为外部示例、显式vision证据字段及其审查/测试，不修改18/17算法。
- 模型config、量化描述、索引SHA、每份回执的字节数/SHA、owner及计数见[结构化报告](r741_npu_validation_20261124.json)。原件归档在本地与124，不提交新的原始回执或张量数据。

## 结果

每项每个owner包含3个有效用户请求和1个超范围负例。有效请求含prefill及后续decode，共4次模型步；视觉卷积只在真正的encoder调用触发，不制造decode视觉调用。

|机制|同组Worker作用域数|每个有效请求、每rank applied|容器占卡墙钟时间（含启动）|结果|
|---|---|---|---|---|
|`18.api_half`|2|644|278.00s|通过|
|`17.spatial_padding`|2|1|323.15s|通过|
|`18.variance_over4`|2|644|285.24s|通过|
|`external.conv3d_output_gain`|2|1|332.43s|通过|

所有有效调用 eligible=attempted=applied=consumed=restored，failed=0；changed按发生数值变化的调用计，changed_elements另报。超范围请求不注入。

- **18.variance_over4**：每个请求、每rank 644次；两个owner合计全rank 7,728次。每次改写完整连续BF16输出，固定gain=0.5。每请求的消费者分别为512次 `npu_dynamic_quant`、128次 `unified_attention_with_output`、4次 `aten::index`。输入区间覆盖检查不是逐元素读取证明：最终index只选择末token行。
- **外部图片**：每请求、每rank 1次真实 `aten::conv3d`，临时输出进入 `aten::add`；每次18,432元素发生变化。每owner每rank 3次，合计全rank 12次。输入、权重不变，encoder cache及embedding读取映射通过；使用内置64×64金丝雀与两张确定性用户图。
- 两owner均完成区域字节精确恢复、租约/钩子/在途资源清零、关闭后新请求hidden/logits金丝雀精确恢复。故障请求不要求恢复为未注入结果。
- 缺许可、未知profile、用户Native路径等负例均未获准；原18另跑完整控制负例，缺许可/未知profile/非法99均为CLI等价exit 4，拒绝后正对照继续运行。
- 独立审查先运行 `tools.review_product_mode`；两个新路由再运行 `tools.review_r741_routes`。本地从原始回执重算，四份均通过。

## 首次图片失败与修复

首次失败为 `EXTERNAL_ACTUAL_PATCH_CONV3D_REQUIRED`，发生在改写前。实际模型用vLLM `Conv3dLayer`，而示例误要求 `nn.Conv3d` 子类。不是设备冲突，不把失败计作通过，也不声称该失败进程完成了正常OFF恢复。
修复后5维权重只用于发现候选；实际dispatch必须证明Conv3d、权重/bias归属、返回对象一致。未执行卷积的伪模块拒绝。新增producer证据进入回执及独立审查。修复及相关32项CPU测试通过；产品专项91项通过，两组存在覆盖重叠，不相加称独立总数。

## 抢占与恢复记录

负责人追加授权“做抢占，完成npu验证”后启动。只绕过保留的Kubernetes分配记录，活跃/未知占用仍作门禁。有界清理器每5秒检查，只对设备映射严格限于卡2/5的外部容器执行docker stop；不改Docker入口、驱动、CANN或派发配置。以下均为北京时间（UTC+8），日期2026-11-24。

|窗口|起止时间|持续时间|恢复|
|---|---|---|---|
|window-r741routes1124a1|21:05:15—21:15:17|602.69s|已停止清理器，未改派发配置|
|window-r741reg1124a1|21:15:17—21:25:24|606.13s|已停止清理器，未改派发配置|
|window-r741routes1124a2|21:25:24—21:31:00|336.03s|已停止清理器，未改派发配置|

回归窗口停止两个新派发实例，镜像均为 `cann:9.0.0-910b-torch2.10.0_npu-atk26.4.30`：

- 21:17:12：`atk_q_ops-math-v900-aclnnAbs-e5f531c1de3e4f0f_dev2_slot0_96b06c7e`（仅卡2）。
- 21:18:04：`atk_q_ops-math-v900-aclnnAbs-e5f531c1de3e4f0f_dev5_slot0_10521b8e`（仅卡5）。

未删除上述容器、镜像或数据；无设备映射的bridge/e2e容器保留。三个清理器均结束，恢复外部派发能力。最终检查卡2/5无计算进程，无本轮测试容器或隔离控制器。原外部容器不由本任务擅自重启。

## 限制

18是RMS均方误差的增益效果代理，不是减均值方差或epsilon精确公式；图片示例是输出增益，不代表黄区的低精度卷积或ConvTranspose故障。用户请求未采相同输入的完整模型B/F对照；报告不推导token变化或性能倍数，只记录绝对HTTP耗时。NPU墙钟含模型启动与预热，不是kernel运行时间。
本轮不替代其余公共机制的完整发行回归、无设备安装与发行源码一致性门禁，也不扩大A3、其他模型或未登记组合的资格。
