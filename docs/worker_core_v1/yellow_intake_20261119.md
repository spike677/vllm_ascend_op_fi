# 黄区数据接收：2026-11-19

## 来源、保存和使用边界

负责人转交 openchamber 的 `RC2_DEPLOYMENT_PROBE.md` 摘要，声明记录提交 `4ea4d3f`。本次收到的是文本报告，未收到该提交源码、逐请求原始回执或 inspect JSON；提交真实性及执行细节尚未独立复算。原件在仓库外 `D:/DemonCATPFI/intake/yellow/20261119/`，文件 SHA256 为 `feff4270058a7cee7ef1bc81883fd7e6fe5ffee4b86c7ed91534cb2bdb689937`。本接收记录不收录机器地址、硬件编号、容器名，也不把原始文本提交 Git。

用途：现役摸底、部署不匹配定性；**不足以登记 RC2 黄区组合**。黄区结果不作为蓝区验证；原件缺项不从启动脚本、镜像标签、默认值或蓝区记录补齐。

## 两个已报告的软件栈

|项目|RC1.1 测试部署|Qwen 专用实例|
|---|---|---|
|来源/运行提交|现役仓库 `43c33ba`；记录仓库 `11f8b87`，不是同库串联提交|本次只读 inspect 摘要；实际安装源码 SHA 未提供|
|模型|Qwen3-0.6B / Qwen3ForCausalLM / BF16|Qwen3.8-27B-W8A8 / Qwen3_5ForConditionalGeneration / qwen3_5|
|模型 config SHA|660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd|191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab|
|栈|CANN8.5.0；torch2.9.0+cpu；torch_npu2.9.0.post1+gitee7ba04；vLLM/Ascend0.18.0；Python3.11.14；Transformers4.57.6|CANN9.1.0；torch2.10.0+cpu；torch_npu2.10.0.post4；vLLM0.26.0/Ascend0.26.0rc1；Python3.12.13；Transformers5.14.1|
|调度|TP/PP/DP=1/1/1；eager；max_seqs1、max_len2048|TP/PP/DP=1/1/1；eager串行；max_seqs1、max_len4096；MTP/KV connector/async/prefix关闭|
|硬件代际|未可靠确认|未可靠确认；报告用“蓝区只有A2”推断黄区代际，依据不成立；不能登记A3|
|已报告结果|18/39两作用域和负向检查通过；完整原始证明未随本批提供|短文本生成成功；18/39安装因pcp_size缺失拒绝；未进入任务、未注入|

Qwen 专用实例关闭 chunked prefill 后能启动并完成短请求，但有官方运行时警告，不能把短请求成功推成普遍语义正确。DeepSeek V4.1 W4A8 仅报告制品完整、专用实例未启动；不推断其调用路径或资格。

## 缺失与冲突清单（由负责人转交，不直接要求黄区操作）

1. **硬件**：权威硬件代际/产品型号的非唯一标识说明。机器、卡序列号等唯一编号不需要回传。当前“按蓝区说明推断A2”与现役A3目标相冲突。
2. **RC2 登记材料**：18/39/22各两作用域的 RC2 验收总结、部署授权/许可身份、source-review SHA、三个wheel SHA、安装源码校验、效果/OFF/health/资源清零、两项负向检查。现有RC1.1成功不能替代RC2。
3. **源码与身份**：实际运行 `43c33ba` 的源码组合清单、未提交CLI接线是否影响交付的确认；两个栈的已安装runner/量化/norm/rope实现指纹及完整inspect结构化数据。`11f8b87`只是记录提交，不作运行源码身份。
4. **0.26 调度事实**：完整 parallel/scheduler/model/compilation/KV配置、DCP实际group与runner来源、MTP/KV connector关闭的实际读回、调度CPU缓冲区及真实请求映射字段。不能补一个pcp_size=1绕过检查。
5. **调用关联**：模块→实际norm/rope/quant调用→消费者的对应，量化张量/scale形状、dtype、权重描述身份。现有全局算子计数不能证明208组DynamicQuant与QMM一一关联或scale粒度。
6. **decode计数限制**：报告用max_tokens=2减max_tokens=1得到第一次decode计数，属于差分推断；未收到两个运行的原始profile及同输入/暖机等条件，不作为单次decode实际链证明。
7. **现役参数**：报告中的对外服务TP8/DP2等来自未复查启动说明，与负责人当前Qwen每实例TP1目标不一致。按专用实例实测TP1分析，不回填现役。DeepSeek准确制品和资源窗口仍待负责人确认。

## 定性与后续

`NPUModelRunner.pcp_size`：**部署不匹配，转RC3调用语义适配**，不发RC1.x、不让接线方伪造字段。主开发线的 prerequisites 已有汇总缺失报告；RC2.2增加针对该缺项与其他同时不满足项的回归，不据此放开0.26。RC1.1 master保持；未创建RC2.1登记分支。

[Qwen第一批范围建议](rc3_qwen_scope_proposal_20261119.md)是待负责人确认的开发范围，不是验收结论。没有向openchamber发送任务。
