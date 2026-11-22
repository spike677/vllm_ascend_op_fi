# 历史记录：宿主逐请求钩子的蓝区验收

本页固定记录9ec051ea的旧接口（含next3），不再是当前产品契约。新接口及真实HTTP结果见[product_http_acceptance_20261122.md](product_http_acceptance_20261122.md)。旧原件不改写为新接口通过。

## 组合和边界

A2（Ascend910B4）、CANN9.1.0、torch2.10.0+cpu、torch_npu2.10.0.post4、vLLM0.26.0、vllm-ascend0.26.0rc1、Qwen3.8社区W8A8、TP2。模型config SHA为 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`。容器镜像 `sha256:c9c9432b69767cf1855efad44f20eaa8ab02520ae05b75c936e67ea4e7eb2702`。

每机制同组Worker两个独立owner作用域：all、next3；每作用域三个不同用户prompt，均真实prefill加三步decode。第四个请求max_tokens=5超出范围，正常推理完成、零注入并返回原因。最终复测另外在next3作用域提交一个范围合格的第五请求，必须仅因名额耗尽不注入。17使用部署金丝雀图像之外的两张不同64×64用户图片，不核对用户图片与金丝雀SHA相同。

产品请求仅通过真实内部request ID接入现有executor，没有注册用户prompt。共享注入site，未改变故障公式。17的变化元素数来自同一算子输入的原padding局部对照，不伪造用户模型normal结果。所有请求的模型级baseline比较均明确为未执行。

## 结果

**五项最终均通过独立复核**。最终文本窗口 `product-config-text-a2`、图像窗口 `product-config17-a2` 使用同一份源清单 `81df89c2ea64ae5e9482a508644b3af2ec54568078a6da60e78086118836b475`。两轮均通过正式部署文件入口，不在请求间清理缓存，且完成范围合格请求的next-N名额耗尽对照。以下eligible均全部attempted/applied/consumed/restored，failed=0；每项3个合格用户请求×2作用域×2rank。

|能力|每rank每个合格用户请求eligible|实际消费者|恢复要求|
|---|---:|---|---|
|18 api_half|644|真实norm下游值消费者|区域字节、资源、金丝雀|
|39 output_proxy|1216|BF16 ND投影真实下游值消费者|区域字节、资源、金丝雀|
|22 wrong_row|64|实际融合rotary路径|区域字节、资源、金丝雀|
|28 denominator_replay|64|真实o_proj输入及消费者|区域字节、资源、金丝雀|
|17 spatial_padding|1|实际patch输出→视觉编码→embedding|原调用参数及输入字节无泄漏、资源、金丝雀|

开启期合格用户请求的实际执行耗时（不含模型加载与关闭金丝雀）分别为：18为3.132–3.267s，39为4.595–4.727s，22为1.494–1.551s，28为1.480–1.561s，17为1.242–5.310s。这些请求没有配对normal基线，不据此编造性能倍数；历史机制验收的同输入倍数仍独立保留。

17在prefill发生一次真实图像编码，decode没有视觉调用，不制造额外触发。changed与applied独立计数；非有限值针对注入区域，不声称是整张量计数。没有观察值时保留null。18/39/22/28通过真实流事件确认消费及字节恢复；不能用Python返回或容器退出代替。

独立总结见 [18](product_mode_20261122/review_18.json)、[39](product_mode_20261122/review_39.json)、[22](product_mode_20261122/review_22.json)、[28](product_mode_20261122/review_28.json)、[17](product_mode_20261122/review_17.json)。记录原回执SHA、实测源清单SHA和独立审查器SHA；原回执仅保存在蓝区及本地私有目录，不提交完整张量或新的statistics原始数据。

## 修正与未掩盖的失败

- 28首次产品矩阵漏记消费者算子名称。审查器以 `ACTUAL_CONSUMER_OR_BYTE_PROOF_MISSING` 拒绝；在真实消费者hook补记录后独立重跑通过，没有手工修补旧回执。
- 初轮脚本每个请求前执行了prefix-cache reset。正式产品库没有这个行为；后续专门移除脚本中的重置再测，避免把验收清理操作变成产品的隐含前提。
- next3最初的第四请求同时超出生成长度；新增第五个范围合格请求，使名额限制与范围拒绝分开验证。
- 1048项CPU回归最初仅旧RPC精确列表断言失败。新增正式RPC到清单，且增加无权限拒绝检查后，全矩阵复跑0失败/错误，保留2项旧研究skip。新增独立owner及next-N审查用例另外12项通过，见[CPU记录](product_mode_20261122/cpu_review.json)。

## 发布与主项目接线

[使用文档](rc3_product_usage.md)提供正式RPC、部署文件、ProductClient接口及五处接线契约。CLI示例是供黄区主项目实现的路由契约，不能当作已发布的独立命令。蓝区测试使用现有executor和正式客户端；测试脚本创建临时实例仅用于验收，不属于主项目产品入口。

r4九个SHA清单成员复核原字节未变；本次未冻结新包、未改变标签。待提交的18个运行时/site/产品探针源码与最终NPU清单逐字节一致，见[源码核对](product_mode_20261122/source_reconciliation.json)。五项作为下一版产品候选，不自动继承旧验收记录为产品准入。A3主项目的Catalog、Controller、调度、结果出口及关闭五处实际接线尚未验收，`product_closed=false`。

33长序列split8/split16两次均未满足rank0模型传播，按止损继续关闭；8新combine输出代理仍待适配，不把旧索引路径通过结果当作新代理通过。V4稳定版的版本与inspect未齐，不继承V4.1或V2-Lite资格。

## 抢占与隔离记录

仅卡2/5使用有界按设备docker-stop清理器，最大35分钟；不改Docker入口、调度器配置、驱动或CANN。9个产品窗口均已停止并join清理器，恢复回执完整；未发生外部容器停止事件。完整原件在Git外 `product-windows.json`，公开[起止摘要](product_mode_20261122/windows.json)。最终两个窗口如下（UTC；容器运行时间包含初始化/加载/预热，并非纯NPU kernel时间）：

|窗口|开始|结束|容器运行秒数|隔离恢复|
|---|---|---|---:|---|
|最终四项文本|2026-11-22 10:58:53|2026-11-22 11:05:15|378.150|通过|
|最终17图像|2026-11-22 11:06:45|2026-11-22 11:12:16|326.675|通过|
