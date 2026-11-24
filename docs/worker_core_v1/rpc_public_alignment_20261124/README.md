# r7.2公共RPC回包与落盘核验（A2 TP2）

组合：A2 Ascend910B4 / vLLM0.26 mp executor / CANN9.1 / Qwen3.8社区W8A8 / TP2。
运行时使用冻结r7.2源码；另加只用于观测的Worker子类，原方法返回值不改变。每次调用前设置唯一token，逐rank写完整结果SHA、action、rank并fsync；宿主按本次返回重新算SHA，随后与两rank落盘全文逐项独立比对。
第一窗口rpcpub1124a1：37次binding调用、74份rank回包，全部精确对应。包括capabilities、attach、warmup/canary、status、失败prepare的rollback、reset/detach。
同组Worker两个18 owner真实HTTP通过独立产品复核；294.565秒，隔离已解除。review.json含回执SHA与完整独立审查摘要。
本窗口enable/disable由ProductClient实际执行，但最初观测器仅包住binding方法；不把它夸大成所有Runtime RPC均逐一关联。补测all探针结果如下。

上一轮r7.2隔离诊断证明：16正常+16rank1延迟+16结构化拒绝均对齐；rank0未捕获控制异常后，rank1连续返回前次结果。底层vLLM队列提前抛异常未排空后续rank是已复现原因。
本项目不通过多发capabilities或手动丢弃一次消息修复污染队列；已错位/超时后终止专用引擎，重新启动。
自定义控制RPC在Worker内捕获异常并结构化返回，宿主串行、每调用nonce+完整rank集校验。鉴权在捕获边界内，不吞forward/kernel异常。

黄区只有现象文本，没有其探测RPC源码及第一起异常日志。尚不能复刻其特有实现或直接归责于用法。
公共路径当前样本未见错位，不据此声称公共接口对任意第三方先前污染都免疫。无证据不发布虚构的维护修复；若all复测发现公共路径自行错位，再从对应冻结版本最小修复。
退出的Python resource_tracker提示独立记录，不把故障Runtime租约零表述为全部宿主IPC零。

## 全部公共RPC补测：已完成

rpcall1124a1：99次RPC、198份rank回包，调用序号/action/完整结果SHA与每rank fsync落盘全部一致。
包括真实ProductClient的capabilities/status，以及prepare/register/check/enable/disable/unregister/detach，真实缺许可回滚、同客户端后续阳性、两个owner。
原结果中的group ticket owner、各rank provider owner、请求ID均包含在完整结果摘要和落盘比对中，不将不同rank私有owner错误要求为相同。
独立产品复核通过；286.664秒，窗口隔离已解除。证据：rpcall1124a1_review.json。

结论：在本A2/0.26/Qwen3.8 TP2样本中，公共r7.2路径没有自行制造错位；当前没有依据发布公共接口维护修复。
已证明的是底层mp executor在未捕获控制异常后会污染后续回包序列；不等于已证明黄区第一次错位的触发点。
黄区需提供自建RPC源码及首次异常前后的调用序号、rank、owner日志；先用结构化拒绝+nonce/rank核对，发生传输错位后重建引擎，不继续使用污染队列取验收回执。
本次不改冻结r7.2/r7.3，也不把观测子类放进产品路径。新增离线审查器CPU5项覆盖旧rank回包、owner/落盘不一致、摘要篡改、缺rank和重复序号。
