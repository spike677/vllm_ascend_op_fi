# DemonCATPFI API v1

```python
import torch
import demoncat

model = torch.nn.RMSNorm(4).eval()
config = demoncat.example_config("18.api_half")
session = demoncat.configure("18.api_half", config)
request = demoncat.Request(args=(torch.ones(1, 4),), request_id="req-1")

def checked(result):
    if not result.ok:
        raise RuntimeError(result.to_dict())
    return result

checked(session.discover(model, request))
checked(session.inspect())
checked(session.arm())              # 自动保存独立 baseline，再安装受限模块代理
try:
    checked(session.inject())
    output = session.output        # 实际返回 Tensor；不混进 JSON 控制回执
    verification = checked(session.verify())
finally:
    checked(session.disarm())
    checked(session.restore())     # 清理失败会保留同一 owner/generation 的租约
audit = checked(session.audit())
checked(session.cleanup())
```

所有会话操作返回 `demoncat.operation_result.v1`，统一字段为 `operation / ok / code / state / mechanism / owner / generation / data / message / qualification_granted`。成功的查询不代表实际注入。`effect_observed` 单独报告，未命中和输出不变不会被写成数值效果通过。

顶层 `configure()` 返回会话，非法配置抛出有 `to_dict()` 的 `ConfigError`；CLI 把它转成 JSON 和退出码 2。会话的 `configure()` 只允许在未绑定模型的 CONFIGURED 状态更新同一机制配置，返回统一结果并建立新 generation。

| 操作 | 输入 / 行为 |
| --- | --- |
| discover | 实际 model 与 Request；结构候选不等于真实调用已观察 |
| inspect | 注册表说明、边界和能力 |
| configure | 完整 v1 配置；未知字段不丢弃 |
| arm | 当前接通的 CPU 代理捕获 baseline，安装拥有的钩子 |
| observe | 原始模型调用；禁止选择故障；独立记录实际 dispatch |
| inject | 执行绑定的 Request，触发器只在合格模块调用处计数 |
| verify | 独立 CPU 返回值比较，不宣称 Native 首差分 |
| disarm | 停止后续故障准入，DRAIN 后进入 OFF |
| restore | 释放拥有的钩子/包装并执行健康回放，失败可重试 |
| audit | B/F/OFF 的只读观察记录、比较与范围 |
| status | 状态、owner、generation、触发计数和历史 |
| cleanup | 仅在没有待恢复资源时释放模型引用 |

正常状态：`CONFIGURED → DISCOVERED → ARMED → HIT → DRAIN → OFF → RESTORED → CLOSED`。未命中可继续 ARMED。失败进入 ERROR 或 RECOVERY_FAILED；不得跳过恢复直接 cleanup。控制操作限定同一 owner 线程，运行期间的重入返回 BUSY。TriggerEngine 可供多线程调用，但这不赋予模型 Session 并发执行资格。

状态错误返回 INVALID_STATE；身份错误返回 OWNER_MISMATCH；未桥接后端返回 NOT_IMPLEMENTED；后端异常返回 EXECUTION_ERROR；恢复失败保持 RECOVERY_FAILED 状态和原始错误。此同步协议没有虚构异步设备 drain。

CLI：`python -m demoncat catalog`、`example ID`、`validate FILE`、`inspect ID`、`generate [--check]`。CLI 不反序列化模型、不根据字符串 import 任意执行器。真实执行使用 Python SDK 或已加载模型的本地 WorkerControl。

Worker：`WorkerControl(model, mechanism, config).discover(Request(...))`，随后 `dispatch("arm")` 等操作返回同一 JSON 结果。Tensor 留在 Worker 进程；本轮 CPU Request 身份是本地调用者绑定，不冒充实际 vLLM Scheduler 证明。FIA 入口另复用既有严格 prefill 元数据检查。
