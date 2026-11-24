# 产品关闭与双rank状态诊断

本页适用于可信宿主，不改变 ProductClient 的状态机或 Worker 调用顺序。

## 关闭前的排空契约

宿主必须先暂停所有接收新请求的入口，然后等待已经提交的真实HTTP请求、调度中的请求及设备消费者完成，再由同一 ProductClient 执行 disable、detach。`drain` 回调必须实现这套过程，不能用空回调或 sleep 代替。验收脚本的串行请求已在关闭前完成；并发用例则显式等待每个已提交的 Future 返回。

判定依据是同一owner、同一请求集合、每个rank的完成记录和设备读取证据，不能只比较两个总数。关闭完成后同时检查字节恢复、leases/hooks/inflight为0、新请求金丝雀精确恢复。排空失败时保留错误，不伪造请求完成，不移动cursor来消除差异。范围内的外部HTTP提交者也必须经过同一个宿主入口；不能让未追踪流量穿过关闭边界。

## 保存失败的原始状态

可信宿主可使用 `demoncat.runtime.vllm_ascend.worker_extension.product_diagnostics.DiagnosticProductClient`。它继承原控制逻辑，只在 `_collect_status` 失败时保存本次真实RPC已经返回的内容：

- host cursor、已收集请求数、owner和预期rank集合；
- 各rank的rank号、cursor、active_request_id、active、more、failure；
- 每条已返回请求的真实ID和八项coverage计数。

异常继续原样抛出，附 `status_diagnostic`；宿主也可调用 `diagnostics()` 取出副本。没有回包时rank字段为null；缺少active字段时保持null。不会发第二次RPC补齐，不重试，不压掉原异常，不猜测rank1是哪一次调用的结果。

这是新增宿主类，旧的ProductClient不会自动获得诊断。需要诊断的黄区宿主显式改用下面的构造方式；已有executor、可信配置、身份、金丝雀和真实排空回调保持原样：

```python
from demoncat.runtime.vllm_ascend.worker_extension.product_diagnostics import DiagnosticProductClient

client = DiagnosticProductClient.from_deployment(
    executor, deployment_path, operator=trusted_operator,
    run_canary=send_deployment_canary, drain=pause_ingress_and_drain_real_requests,
)
```

上面的名字是宿主已有对象和回调，不是新增可由用户指定的CLI参数。日志层应在原异常处理路径中保存`error.status_diagnostic`或`client.diagnostics()`，随后仍按原失败恢复流程处理；不能记录后把错误当成成功。

使用前应核对产品进程及适配器源码身份；诊断器本身不授予新的组合资格。对应CPU用例包含完成数错位、传输失败、缺字段、畸形回包和正常路径。

## 复核命令

```bash
python experiments/worker_core_v11_20261118/product_http_probe.py \
  --mechanism 18 --drain-stress --status-diagnostics
python -m tools.review_disable_drain /evidence/product_http_18.json \
  --output /evidence/drain-review.json
```

压力用例在每个owner的关闭前额外提交两个真实HTTP请求，在至少一个尚未返回时调用disable，通过真实drain回调等待完成。两个owner仍共用一组Worker。该用例能核对排空契约，不能重建缺少原始调度轨迹的黄区历史事件。

本轮实测结论、源清单和回执SHA见 `r741_b1_b6_closure_20261124.json`。如果未重现黄区差一请求事件，台账可按任务书记录“旧驱动混入外部请求造成”，但其证据等级必须同时写为负责人指定的历史定性，而非本轮已复现的根因。不得以此排除其他未经复现的原因。
