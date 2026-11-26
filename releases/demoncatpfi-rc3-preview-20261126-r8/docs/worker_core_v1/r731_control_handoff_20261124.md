# r7.3.1：契约拒绝映射与引擎存活

从r7.3发行提交1b816dd6开维护分支，不覆盖旧标签/包。公共机制不改；只修改控制拒绝、Batch金丝雀前置拒绝及负责人新要求的效果测试工具。黄区master按最新决定直接从r7.3.1接入，正式测试使用该最终包，不再重复r7.2/r7.3矩阵。

## CLI与RPC口径

黄区入口保持 `python -m demoncat.cli fault ...`。蓝区没有黄区CLI源码，不能声称已经修改/执行过该CLI。宿主需采用以下明确映射，不能用异常字符串猜退出码：

```python
import json
from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_control_reply

body = product_control_reply(trusted_session_operation)
print(json.dumps(body, ensure_ascii=False))
raise SystemExit(body['exit_code'])
```

`ProductContractRefused`有类型和exit_code=4，保留每rank的原拒绝码。所有rank契约拒绝→4；rank部分成功/失败、未知内部异常→5，仍保留恢复责任。未知名称的配置校验→4，body保留原因。公开控制RPC会返回结构化body；直接RPC的HTTP200并不表示准入成功，也不要求必须HTTP4xx。

|场景|要求|
|---|---|
|缺许可|CLI exit4，原ORDINARY_ROUTE_NOT_ACCEPTED_FOR_DEPLOYMENT，资源零，同引擎后续正例通过|
|未知profile|CLI exit4，PRODUCT_PROFILE_UNKNOWN，真实推理和health继续可用|
|未知名称|CLI exit4，配置拒绝原因；不进入模型|
|已有故障再开启|结构化exit4；不得回滚原来已启用的owner|
|重复关闭|幂等成功或结构化exit4；不产生新租约，不影响服务|
|Batch非串行金丝雀|PRODUCT_BATCH_CANARY_REQUIRES_SERIAL_B1；模型继续正常执行，控制RPC拒绝，清理后同client可重试正确串行金丝雀|

Batch在execute_model进入真实计算前检查可判定的金丝雀契约。发现错误只拆除本故障的非注入观察器，锁存原因，原模型执行一次。它不吞掉模型/kernel错误，也不通过重试或替换输出冒充引擎存活。控制RPC错误统一结构化，不让预期拒绝流出为MultiprocExecutor FAILURE。

## A3正式测试

至少覆盖18（Torch hook）、41（Native）、Batch（并发）、15，各两个owner；其他公共项按既定正式唯一一次矩阵执行。新的统一工具见 [同服务效果测试](product_effect_test_20261124.md)，包括baseline/fault/off、两个owner、各rank八项计数和changed_elements，原件在黄区。

部署级负例只做一次：缺许可、未知名称、重复enable、重复disable；同时核对未知profile和非串行Batch拒绝后的引擎存活。不要故意让未捕获Python异常进入mp控制队列。真正传输失败/回包错位时停止并重建专用引擎，不能靠补发status冲刷。

r7.3.1不含v2持久区域实现，不使33 A3或8 V4自动可执行。v2/33由后续r7.4增量，8由D包；共享路径若变化必须给出受影响复测清单。r8在登记证据完整后整合，不能先填虚假组合。

## A2验证边界

A2 Ascend910B4 / vLLM0.26 / vllm-ascend0.26.0rc1 / CANN9.1 / Qwen3.8社区W8A8 / TP2。原件与独立审查摘要见 `r731_control_20261124/`。宿主映射函数按真实Worker拒绝验证；黄区CLI仍需使用上面的映射。性能只记录，不作为门禁。每次有界隔离均在窗口结束后恢复，不改Docker入口、驱动或CANN。
