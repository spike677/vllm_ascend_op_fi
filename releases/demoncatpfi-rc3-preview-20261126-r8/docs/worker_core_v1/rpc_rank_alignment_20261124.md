# rank1 回包落后一拍：r7.2 A2 复现与使用约束

组合：A2 / vLLM 0.26 / vllm-ascend 0.26.0rc1 / CANN 9.1 / Qwen3.8 W8A8 / TP2。
公共包使用 r7.2 `7ffd9a60`；增加的探测方法只在独立验收扩展中，未修改第三方 executor。
模型推理没有用于制造异常。错误发生后仅探测相关编号，不继续推理；最后关闭该临时引擎。

| 对照 | 调用数 | 结果 |
|---|---:|---|
| 同步正常返回 | 16 | 两 rank 的 token 均对应当前调用 |
| rank1 人工延迟50ms | 16 | 全部对齐，延迟本身不是错位原因 |
| rank0 结构化拒绝、rank1 延迟 | 16 | 全部对齐 |
| rank0 抛出探测异常、rank1 延迟250ms | 1 | HTTP500；随后 rank1 队列落后一轮 |

异常后的实际对照：

| 发送 token | rank0 返回 | rank1 返回 |
|---|---|---|
| after-0 | after-0 | error |
| after-1 | after-1 | after-0 |
| after-2 | after-2 | after-1 |

## 原因与归属

实际安装的 `vllm/v1/executor/multiproc_executor.py`：`MultiprocExecutor.collective_rpc` 的 get_response 按 rank 顺序收包，
遇到非 SUCCESS 立即 raise，后续 rank 的队列没有排空。Future 已结束，下一次收包仍读到旧的 rank1 响应。
超时也可能留下未消费响应，不能在超时后猜测队列已恢复。
这是 **vLLM 多 rank 控制传输的异常路径缺陷/使用限制**，不是 NPU 计算精度故障，也不是仅由 rank1 慢造成。
r7.2 已将公共 binding 的契约拒绝和内部控制错误转成数据，因此该边界没有重现错位；黄区自有探测 RPC 若继续直接抛异常，仍会触发底层问题。
黄区此前具体哪次调用先出错，当前材料未证明，需要对应首次异常/超时日志；不能只凭本复现断定其最初触发点。

原件 SHA、executor 源码 SHA、窗口时间与结果见 [独立诊断摘要](r73_external_20261124/rpc_review.json)。
本次复现窗口194.781秒，结束后解除卡2/5隔离。没有安装或修改宿主 vLLM、驱动、CANN。

## 黄区正确接线

可信探测方法的所有 rank 都返回结构化数据；不在探测 RPC 内执行可能破坏服务的推理或 kernel 改写。
r7.3 附 `external_control.control_reply / validate_control_replies`：

```python
# Worker 的可信扩展。probe_impl 是部署方已经审核的只读函数。
def yellow_readonly_probe(self, request_token, kind):
    from demoncat.runtime.vllm_ascend.worker_extension.external_control import control_reply
    return control_reply(self, request_token, lambda: probe_impl(self, kind))

# 宿主：每次新的 token；同一 executor 的控制请求串行等待完成。
import uuid
from demoncat.runtime.vllm_ascend.worker_extension.external_control import validate_control_replies
token = uuid.uuid4().hex
rows = executor.collective_rpc("yellow_readonly_probe", timeout=120, args=(token, kind))
rows = validate_control_replies(rows, request_token=token, world_size=ACTUAL_WORLD_SIZE)
# 先验证所有 rank，再读取各 rank 的 result/status；不能只取 rank0 或按到达顺序猜 rank。
```

鉴权检查也应位于 control_reply 的 operation 内，使鉴权拒绝同样返回结构化数据。
helper 复用纯控制数据，不加载用户代码，不绕过部署鉴权；包装器只放可信只读/控制边界，**不能吞掉 forward/kernel 异常**。
helper 的异常、序号、缺 rank、重复 rank、单 rank 路径有 CPU 测试；上表结构化拒绝模式已在实机验证。

出现 transport异常/timeout、缺 rank 或 token 不匹配时：停止继续控制/推理，保留日志，重建专用验收引擎。
不要用 capabilities 冲正；发送新消息通常只会继续保留一条积压。不要丢弃无法关联的响应后继续修改状态。
本包不自动重启 Worker；由原服务生命周期管理方执行重建。未知 profile 等公共预期拒绝应由 r7.2/r7.3 结构化返回，不应触发这条重建路径。

若需要修改 vLLM 本身，必须作为独立部署变更审查：所有响应排空/超时隔离、每次 RPC 关联 ID、并发及 future 生命周期均需覆盖。
不能在当前实例临时猴子补丁 dequeue 来宣称已修复。本次通过应用控制边界避免触发，并在无法证明关联时拒绝使用结果。
