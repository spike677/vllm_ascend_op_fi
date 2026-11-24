# 外部 provider v1：挂载点、恢复和库行接入

本页按能力而非故障库行数授权。示例不是黄区原语实现，不授予其产品资格。
共享 ProductRequest、ProductProvider、生命周期与准入保持现有路径；新增模块输入和 dispatch 输入示例均使用 stage=model。图片走已批准的显式 stage=vision，采样走 stage=sampler。

|挂载点|示例/边界|真实消费者与恢复|适用限制|
|---|---|---|---|
|模块输入|external_input_examples.ExternalModuleInputSite；自动发现具备 quant_method、input_size、output_size 和二维真实 weight 的投影（排除Embedding）|前置 hook 改输入末行前两列；真实非视图运算覆盖区域后，同stream完成并恢复；模块返回核对整个输入恢复|BF16连续二维输入；消费者原地写或无法证明覆盖时拒绝；不改持久weight|
|模块输出及只读输入/权重|external_vision_example.ExternalVisionGainSite|真实Conv3d返回值被add消费；核验输入/权重未变、输出区域恢复；encoder cache到embedding另有请求证据|仅已审查图片配置；不是低精度重算机制本身|
|dispatch调用参数|external_quant_argument_example.ExternalQuantArgumentSite|匹配实测npu::npu_dynamic_quant；仅在operand0副本上改两元素，原DynamicQuant读取后恢复副本；原输入不变，原Q/scale返回值交给后续QMM|BF16连续二维输入；其余参数原样转发，校验真实int8量化返回；不修改Q/scale结果、不改持久区；不是任意op通用资格|
|采样|external_sampling.ExponentialNoiseSite|真实随机噪声到采样除法的消费者；同stream读取后恢复|已有采样范围；贪心调用不制造随机采样|
|持久区|external_persistent独立v2路径|按权重/请求页代际和读取者协议恢复|v1拒绝：EXTERNAL_V1_PERSISTENT_WEIGHT_OR_KV_RESTORE_UNPROVEN|

输入示例固定两列只是部署命名变体，不接收用户layer/rank/value。其余列和行必须逐字节保持。changed表示数值变化的调用，changed_elements独立统计；0变化也计applied。每个rank按当次真实调用计数，不用基线清单补造调用。

## 黄区逐行接入指南

|库行|推荐site/阶段/区域|必须补充的机制与消费者证据|
|---|---|---|
|2|v2持久weight或weight_scale|派生NZ/预打包副本、全部读取者和恢复；不可按临时输入注册|
|3|model模块输出或SwiGLU真实dispatch输入|真实SwiGLU分支；固定命名变体；MLP消费者|
|9、10、11、31、37|model投影输入或真实量化dispatch边界|Q与scale、clipping/舍入规则和实际QMM读取；库37保留配置说明，不全局替换round|
|14、40|v2真实KV写入区域|请求/页/slot代际、后续token读取及结束重建；禁止恢复覆盖后续合法更新|
|16、32|vision Conv3d输出＋只读输入/权重，或已证明的Conv3d参数边界|真实5维调用、dtype/布局、低精度重算或cast的具体语义；图片encoder到embedding链；TP2 spawn|
|19|model投影输出|有限扰动与改变累加顺序分开描述；真实消费者；保留CLOSE_MATMUL_K_SHIFT说明|
|21|按寿命选model输入/输出或v2持久参数|先确定param属于临时激活还是持久权重/scale，不能仅按字段名判断|
|24|sampler|真实采样而非argmax；噪声到除法/采样的消费链|
|25、38|黄区独立HCCL通道|不在v1模块接口中，按黄区已有通道验收；不假装本接口覆盖|
|35|公共28已有命名变体|仅错误输出代理，不声称合法归约重排同机制|
|7、8|model MLP输出的行变换|明确真实请求/token行；置零/替换padding只是现象代理|
|12|待原始描述|不因lm_head未量化判不适用，也不提前选择错列机制|
|6、17、23、30、34|公共22、17、39op、18.variance_over4、39mr或黄区输出变体|按41行台账的现象等价边界；不以库行号自动选公共ID|
|13|V4 router model输出，D包|动态token→router→专家归属；V2仅预演，不授予V4资格|

## 注册与运行

可信部署调用 register_external_site(worker, ExternalSite(mechanism, factory, factory_sha256(factory), observed_deployment, stage=...), authority=authority, operator=...)，在每个rank分别注册同一审核源码。另配置产品profile、许可及CLI显式映射。注册不等于准入；能力以 ProductClient.capabilities() 和 product_contract 为准。

输入两个示例的验收命令：

```bash
export DCAT_EXTERNAL_SITE_EVIDENCE=/evidence
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism 18 --external-input module --status-diagnostics
# 另一个独立evidence目录：
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism 18 --external-input dispatch --status-diagnostics
python -m tools.review_external_inputs /evidence/product_http_18.json --output /evidence/independent-review.json
```

原始逐调用sidecar与产品回执按rank和真实request ID交叉核对；孤立记录、缺失调用、计数矛盾均拒绝。使用同组Worker两个owner；第四个请求超范围不注入；卸载后无许可、未知profile、未知字段继续拒绝。CPU结果不是A2结果，A2不是A3资格，实机结论见本轮验收报告。

首轮aten.add.Tensor参数示例在该Qwen3.8路径未观察到合格调用，返回PRODUCT_NO_ACTUAL_SITE_CALLS；保留为CPU语义示例，不标为该模型A2通过。实际验收改用已观察到的DynamicQuant边界。
