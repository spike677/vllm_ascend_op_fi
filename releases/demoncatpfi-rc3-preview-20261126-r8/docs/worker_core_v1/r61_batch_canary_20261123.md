# r6.1：Batch串行金丝雀维护

从冻结r6发行提交 `ea42a246` 开独立维护分支。只修正Batch恢复观测协议及其宿主示例/审查器；不改变共享scale、Q校验、真实多请求decode注入规则，不改r6标签和包。

旧r6使用并发金丝雀并要求观察decode batch；正常服务调度可能改变批次，因此不能将并发金丝雀不同直接判为故障残留，也不能据此豁免恢复。新协议把金丝雀固定为同一prompt列表、同一顺序，每次一个请求完整完成后再发送下一个（B1），每个请求4步。Worker核对真实调度的prompt身份、prefill长度、computed数及每个decode步，不以宿主声称串行代替证据。

注入前最多6轮，必须连续3轮全有限且完整hidden/logits和调度逐字节一致；关闭并释放故障owner后跑2轮，均须与最后一轮正常参考精确相同。少一轮、并发混入、顺序漂移、任一值变化都不能通过。用户故障请求仍为真实并发多请求decode，B1用户请求不注入，原因保留。

## A2结果

124，Ascend910B4、0.26/CANN9.1、社区Qwen3.8 W8A8、TP2。两个作用域沿用同组Worker，各16个用户请求覆盖B1负例、并发B2/B4/B8及范围负例。两rank均3轮正常/2轮恢复通过；消费者、资源清零和精确金丝雀通过独立审查。详见 [验证记录](r61_batch_validation_20261123.json)。窗口434.813秒，结束后隔离解除；耗时不是性能门禁。蓝区结果不代表黄区TP1或TP2已验收。

## 黄区接线

在独立环境安装r6.1三个wheel，不混装r6与r6.1源码。保留r6原失败回执。宿主`ProductClient`接口不变；Batch的`run_canary(spec)`必须按 `spec['prompt']` 顺序逐条发送并等待真实完成，禁止把整个列表一次发送或使用线程池。只对金丝雀串行，不能把故障用户请求改成串行来绕过Batch触发。

```python
def run_batch_canary(spec):
    for prompt in spec['prompt']:
        completion = inference(prompt, max_tokens=spec['max_tokens'])
        wait_until_completed(completion)
```

参考可执行接线见 `experiments/worker_core_v11_20261118/product_http_probe.py`。使用r6.1同包 `tools/review_product_mode.py` 复核并保留其SHA；旧并发回执不满足新协议，不能重写旧字节作为新通过记录。恢复失败仍保留清理责任，不强制把状态改成成功。本维护版不新增任何登记组合。
