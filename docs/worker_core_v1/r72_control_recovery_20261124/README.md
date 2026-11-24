# r7.2 控制拒绝与 enable 失败恢复

基线是冻结 r7.1 发行提交 `ef53263bdf9b9139ce92c4c8eb27bbb5e98c124d`。
本维护分支不包含 r8 外部 provider、V4 或其他新能力。

## 修复

1. `demoncat_product_binding` 把控制契约拒绝序列化为 `PRODUCT_BINDING_REFUSED`、exit_code=4，附 rank、reason。
   未知 profile 和鉴权拒绝不再逃出 Worker RPC、终止整个 vLLM executor。
   非预期控制异常返回 `PRODUCT_BINDING_ERROR`、exit_code=5、recovery_required；没有拦截模型 forward/kernel 的异常。
2. `ProductClient.enable` 在任一 rank 的 attach/prepare/register/check/enable 失败后 drain，逐 rank 调用现有 Runtime tenant detach。
   prepare 尚未取得 lease 的 attachment 也清理；Runtime 负责 unregister 和实际资源读回。某 rank 清理失败仍继续尝试其他 rank，保留恢复责任。
3. binding rollback 仅允许同一个 group owner；已有其他 owner 的绑定不会被移除。warmup observer 也在原有 close 边界退出。
4. 已确认清理的失败会话允许 disable/detach 幂等收尾，同一个 ProductClient 可安装正确许可后重新 enable。
   清理成功单独记录 `enable_rollback`，不伪称完成产品金丝雀验收。
5. provider 的历史 failure 仍保留；完成 close 且实际资源清零后，不再把历史错误当作活跃 replay_failed，避免永远无法注销。

## 实机证据

组合：蓝区 A2 / Ascend910B4 / vLLM 0.26 / vllm-ascend 0.26.0rc1 / CANN 9.1 / Qwen3.8 社区 W8A8 / TP2。

| 窗口 | 内容 | 结果 | 窗口秒数 |
|---|---|---|---:|
| a1 | 未知 profile → 健康 HTTP；无许可 enable → 清理 → 健康 HTTP；两个许可阳性 owner | 独立审查通过 | 290.649 |
| a2 | 增加失败后的 disable/detach；同一 ProductClient 装许可重试；再完成第二个 owner | 独立审查通过 | 282.165 |

两个 rank 的缺许可拒绝后 `leases/owners/bindings/wrappers` 都是 0。
正例复用同一组 Worker，每 owner 三条 4-token 用户请求及一条 5-token 超范围负例；真实消费者、区域字节恢复、故障资源清零和部署金丝雀精确恢复通过。
详见 `a1_review.json`、`a2_review.json`，原始回执留在本地/124，只发布摘要与原件 SHA。
负例测试没有注入或停止实际生产服务；实例由独立临时验收 Worker 构成。

a2 后补充 Host 异常分类及清理重试的 CPU 检查：内部 KeyError/无契约码的工厂错误保持 exit 5，带明确 provider_code 的缺许可拒绝为 exit 4；清理暂时失败后可通过公共 disable 重试；数值与设备路径不变。实机源码 manifest 与本次发行源码身份分别记录，不混写。

CPU 覆盖 prepare/register/check/enable 任一阶段的部分 rank 失败，world_size=1/2 均走同一 Runtime 路径；资源未确认释放时不伪报成功。完整门禁见 `cpu.json`。

## 对已报十一项通过结果的影响

不改变任何 site、故障数值、消费者或恢复字节规则，历史阳性结果不作废。
但黄区旧负向清理失败的记录不能冒充有效负例；需在新包上补测拒绝后同进程存活、零资源和正确许可阳性。
黄区主 CLI 应把库的 ValueError 契约拒绝映射为退出码 4，内部/恢复失败为 5；蓝区未运行黄区自有 CLI，不声称其接线已经通过。
十一项组合仍须逐项提取文本、身份和负向记录审查，r7.2 本身不追加资格。

## 隔离与已知限制

仅卡2/5、有界 Docker stop 清理器；两个窗口均记录开始/结束、被停止对象和镜像，并确认清理线程退出。未修改 Docker 入口、宿主驱动/CANN，未删除外部数据或容器。
控制器和隔离日志原件 SHA 在各 review 中。窗口时间包含启动/模型加载和清理，不是 kernel 用时。
vLLM 进程退出有 Python resource_tracker 提示；故障 Runtime 的零租约/零 hook 与此分开记录，不声称检查了全部宿主 IPC 对象。
