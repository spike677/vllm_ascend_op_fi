# V4 router 与 dispatch/combine 观测接线设计

状态：开发设计，等待已安装源码核对；不是 r3 新 options，不可直接在现役服务安装。新观测核心不改变 r3 capture/noise/strict binding 规则。仅可信 Worker RPC 调用，鉴权由部署方保留。

| 边界 | 最少采集 | 必须证明的联系 |
|---|---|---|
| runner 本步 | 请求 ID、query 起止、computed/scheduled tokens、step/run/rank/PID | 真实调度输入，不把跨 rank 同序号当作同一次调用 |
| 目标层 | named_modules 路径、layer_idx、hash、draft、internal-router、实际类与源码 SHA | 同一步、同一层、同一真实对象；不能仅按函数名匹配共享对象 |
| router 生产 | 内部分支实际 `_apply_router_linear` 或外部分支 F.linear 的输入输出 | hidden 与浮点 scores 区分；score 的形状、dtype、storage/generation、全列受控行值 |
| selector 前后 | scores、bias/group/scoring/normalization；原始 top-k IDs/weights | 分数实际交给选专家函数，候选存储边不足以代替源码契约 |
| 路由修正 | log2phy、shared experts、force-EPLB/profile 开关、最终 IDs/weights | 捕获全部改写；不把逻辑专家 ID 当物理 ID |
| prepare/SP | 切片偏移、padding/mask、router/hidden 重排、对象区间 | 调度行到本地 router/dispatch 行；unknown 不补全 true |
| dispatch | 实际 MC2/AllGather/AllToAll/fused 分支、schema、IDs/weights、assist/counts/masks | 原始选择结果到实际路由，整数预算内完整保存，超预算缺证 |
| combine/finalize | 对应辅助张量、输入输出区间、有效行、后续消费者与设备完成事件 | 输出行到原请求/token；不能以 shared output 或 Python return 代替 |
| 运行时组 | TP/EP/MC2 成员有序列表、size/local rank、通信名称、数据来源 | 实际组；不得在条件命中点增加 collective |

实现路线：保留 r3 TorchDispatch 的算子/张量采样；增加显式 Python 方法边界，覆盖 custom-op 内部的 Python selector/通信边界。核心已实现 source-pinned MethodLease，但不自动寻找对象、不猜属性链、不自动放行代理。普通 F.linear 非方法边界仍用 r3 算子观测并与已审源码和实际 gate 对象关联。若融合实现不暴露分数/selector，输出 `ROUTER_SCORE_SELECTOR_BOUNDARY_UNOBSERVED`，不以事后 torch 重算冒充实际执行。

预算建议待实际源码锁定后写入正式 options：单层、一个固定请求分别生成 1/2 token，最多 256 个 Python 边界调用、4 MiB 序列化数据；r3 tensor budget 原上限保留，浮点只采受控 2 行且需覆盖全部专家列，整数需涵盖被审调用的真实辅助张量。超过预算保留 truncated/缺件，不改变推理、不无界扩大。当前核心要求 snapshotter 自身有张量取样上限；序列化上限不能代替读取前预算。

关联键：run + rank + PID + runner step + layer + invocation + parent_call_id。共享通信对象必须按当前真实调用层筛选；跨线程用独立调用栈并记录线程，未知线程或设备 stream 缺证。地址相同但对象生命期、写版本或 generation 不同不得连接。字段缺失时 null；不得从候选边创建“已证明”的行映射。

设备顺序由张量快照器记录当前 stream 上的事件；涉及内部异步 stream 的分支，需源码与实际 wait/event 的联系。Python 方法返回与当前 stream 完成均不自动证明异步消费者完成。原始 tensor 返回值不替换，原方法不多执行，不采权重数据。观测结束待在途请求完成，再恢复自己持有的包装；部分 rank 失败时全 rank 在必达控制边界 stop，不移除外部包装。

正式包前 CPU 夹具：内部/外部 router、hash 拒绝、shared-only 不达标、prepare/SP/padding、logical→physical ID、MC2/AllGather 辅助字段、无 EP 默认值、预算耗尽、部分安装、包装冲突、原异常/kwargs/返回对象保留、同一步不同层/线程不误关联。当前 18 项仅覆盖源码审查与 Python 观测核心（包含线程独立父调用关联），其余待现场接线实现后补齐，不能宣称已完成。

交付后黄区顺序：独立目录校验 → 已加载源码/rank 身份 → inspect_binding → 噪声（observer 关闭）→ 噪声 stop → 定点 start → 真请求 → 全 rank stop → 方法身份与推理恢复 → 原件 SHA/提取。若须替换已导入的工具，安排一次有计划重启；不在 G0/G1 中间热换工具。蓝区不代开 V4 服务。
