# 124 r8.2 测试停止来源（2026-11-26）

组合：蓝区 A2 / 0.26 / CANN9.1.0 / Qwen3.8 W8A8 / TP2，卡2/5。本报告仅调查测试窗口，不评价 under4 或 Batch 能力通过与否。

## 已确认的停止链

以下均为北京时间，括号内 UTC。来源为现有 dockerd journal、Docker State、项目 controller 原件；没有改动宿主审计或 Docker。

| 对象 | 起始操作 | 最终状态 |
|---|---|---|
| under4，候选 d18cfe44 | 16:22:17.330（08:22:17.330Z），POST `/v1.39/containers/<本次容器>/stop?t=20` | 16:22:37.337，signal15 等待20秒未退出，Docker 强制终止；exit137，OOMKilled=false |
| Batch，同一候选 | 16:23:44.398（08:23:44.398Z），POST `/v1.39/containers/<本次容器>/stop?t=5` | 16:23:49.405，signal15 等待5秒未退出，Docker 强制终止；exit137，OOMKilled=false |
| 本项目队列控制器 | 只读观察到状态 Tsl（暂停）；暂停发送者未记录 | 16:30:21 发送 SIGINT+SIGCONT，仅让本项目 finally 完成清理；16:30:22.724 记录 isolation_removed=true |

因此“直接收到 SIGKILL”是结果，完整起因是 Docker stop 超时后的强制终止。不是已证明的模型异常，也不是 OOM。测试预算1800秒未耗尽；本项目控制器没有 timeout 标记。控制器自身清理函数只会停止指定的本项目容器，使用20秒等待；Batch 的5秒停止不是这段清理函数发起。

负责人已确认存在其他人工操作。本机日志只能证明以上 Docker API 调用，不能据此确认具体操作者或将其归给某个任务：

- dockerd 当前记录操作路径/时间，没有请求用户、连接源或会话身份。
- `auditctl -s` 显示 enabled=1，但 audit daemon pid=0；`auditctl -l` 为 No rules。journald 有登录/认证审计，没有目标信号或命令的 syscall 证据。
- `/root/.bash_history` 未命中本次目标；其中 docker stop 记录的时间是11月9日，不能拿来解释11月26日。
- SSH root 会话时间只能作为相关信息，不能证明是谁执行 stop；不将登录地址扩散到仓库。

## 恢复与当前安排

两轮项目隔离均已解除；本项目 `r82effects1126*` 无运行容器，控制器进程不存在。未改调度器、Docker入口、驱动、CANN或宿主权限。原始 controller 状态中的 RUNNING 是被中断前的旧值；终止时间及 finally 的 isolation_removed 是清理证据，不能据 RUNNING 误判仍在占卡。

按负责人最新指示停止抢占、新实测等待 NPU 空闲。检查同时包括卡2/5计算进程与仍持有设备的容器，不能只凭利用率0判断空闲。这些中断不计入机制止损，也不作为 OFF/health 恢复证明。

后续窗口防冲突：沿用项目互斥锁，开始前复核任务归属，保留清晰的项目标签、窗口起止和只读 Docker 事件；如又出现人工 stop 或控制器暂停，停止推进队列并保存现场，不无限重启与人工操作相互抢占。协作窗口标识和事件记录只能降低误停概率，无法阻止持有 root/Docker 权限者的操作；不通过改宿主权限或 Docker 入口来实现所谓“防 kill”。

## 验证状态

- 相关CPU：43/43通过。
- 442 个公共机制/运行时文件与 r8 逐字节一致。
- under4：尚未拿到同次 API_TOP5_REQUIRED 与 logits 的完整实机诊断，不能定性。
- Batch：协议和CPU重试用例已完成，“实际组成不同→重发后组成相同且OFF精确”的A2用例尚未完成。
- 17：同一蓝区真实原件两路径均55154；黄区两原件的请求/图片关联仍待补。
- r8.1：11项候选，0项写入；负例和逐rank身份未齐。不发布r8.1，也不把未完成实测的r8.2冻结为通过版。
