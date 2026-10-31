# 全模型 FA 策略覆盖 statistics/mix 证据子集

两组 B1/B8，各含 calibration、native、controlled、controlled、off 五轮，每轮 prefill+decode；共20统计step、560次SDPA。
原始 dump/stack/construct/config/receipt、模块/阶段/请求映射、调用JSON及native_window、公有host/POD均逐字节保留。
这是固定模型与已认证原生策略的覆盖证据，不声称复现历史故障。数值变化、逐请求logits变化和准确率损失必须区分。
使用已披露的任务本地FP32 high统计collector，不是未修改的官方collector。

本包保留collector来源与加载SHA；完整collector修正源码/依赖和运行环境仍需对应仓库版本及环境归档，并非本包自包含。

**整段DEBUG case.log、完整私有logits以及原prompt/token输入清单保留远端，没有放入本包。**
manifest列出这些远端文件的原路径、SHA与bytes。公开native_window不能替代整日志的漏调用检查；
本包可以校验自身原始统计/窗口及文件完整性，但不能单独重跑全日志census、原始全前缀校验或私有logits逐字节比较。
所附独立验收报告是在完整远端证据上产生的，不能把附带报告当成这些省略数据的本地重算。
