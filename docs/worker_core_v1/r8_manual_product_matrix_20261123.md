# r8 手工测试与范围表（开发候选，今晚不下发）

蓝区组合：Ascend910B4/A2、0.26/CANN9.1、社区Qwen3.8 W8A8、TP2。A3或DeepSeek不能继承此结果。r7.1回归与r8新增免许可路径分开；产品免许可表初始为空，不能因候选被列出而跳过验收许可。

|机制|产品作用位置/范围|专门检查|当前限制|
|18.api_half|norm返回|实际消费者；新请求金丝雀恢复|r8免许可正例另验|
|22.wrong_row|实际rotary/QK行|行效果；prefill/decode均计数|不能套Qwen mRoPE到V4 partial rotary|
|39.output_proxy|BF16 ND投影输出|新增Inf与已有Inf分计；消费者|非BF16/ND拒绝|
|39.matmul_range|投影累加|全K控制、错误范围、非目标区|量化布局须已适配|
|28.fia.denominator_replay|attention分母重放|真实QKV/消费者；大误差代理|不声称历史kernel复刻|
|17.spatial_padding|视觉Conv3d|用户真实图；内置64x64金丝雀|spawn、固定视觉范围；不要求外部图片文件|
|quant.batch_shared_scale|decode DynamicQuant→QMM|实际B2/4/8并发；B1不注入|金丝雀串行B1；max_num_seqs=8|
|41.tail_column|Native KV尾列|真实FIA消费、字节恢复|实际KV布局和审查库；非paged阶段保留不适用|
|26.fia_destination|Native写活跃residual|FIA源→residual→真实消费→恢复|不是原kernel指针替换复刻|
|20.buffer_substitute|Native输入覆盖|动态量化/QMM消费后恢复|独占窗口、审查库|
|33.paged_decode|paged decode重分组|长prompt、控制检查点≥99%|A2实测范围；A3仍关闭|
|15.alias_detach|ND视图真实写入边界|view写对、base未更新、消费者读旧值|不含私有格式；当前候选TP2|
|8.restore_offset|MoE combine行映射|动态EP请求归属与恢复|Qwen不适用；V4等待只观测，默认关闭|

每项同组Worker两个owner，全rank持续触发；eligible=attempted=applied=consumed=restored且failed=0，changed≤applied；区域字节、资源清零、新请求金丝雀三项恢复。性能只记录，不作门禁，不要求文本或Top-1改变。

可信宿主按r8_trusted_product_integration_20261123.md建立bootstrap和部署自己的鉴权session。参考console只有下列动作，不创建daemon、不安装许可：

```sh
dcatpfi-product fault list --socket "$SOCKET"
dcatpfi-product fault enable --socket "$SOCKET" --fault-id "$MECHANISM"
dcatpfi-product fault status --socket "$SOCKET"
dcatpfi-product fault disable --socket "$SOCKET"
```

验收模式每owner由可信控制面装许可；正式登记模式必须命中精确组合且不存在临时许可。用户只发正常HTTP推理，不填rank/layer/value/内部request ID。第二owner不重启Worker。disable失败保留原session恢复，不能创建新owner掩盖。

三项负例分别保持其余条件有效：缺许可（验收模式）、未知profile、未知机制TEXT 99；每项需合法正对照和所有rank资源无残留。登记模式另测未登记、身份漂移、跨TP借用、缺必需字段、混入临时许可。CLI未知99需退出4，服务异常为5。scope中5生成token请求应不注入并写明原因。

保存公开ProductClient回执、review_product_mode审查和extract_product_record分页文本，owner—文件字节/SHA关联。蓝区不拥有黄区demoncat.cli源码或daemon，参考CLI测试不能替代其真实接线验证。
