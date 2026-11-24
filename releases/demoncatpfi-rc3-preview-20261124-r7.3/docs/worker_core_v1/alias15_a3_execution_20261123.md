# r7.1：Fault15 B档 A3 TP2产品验收执行单

交负责人转交，今晚只做TP2。语义是“别名断开语义（ND视图），不含私有格式触发条件”；不是历史NZ/NC1HWC0触发条件复刻。r7不含15，r7.1包含受信候选，未授予A3资格。

## 安装与接线

使用[TP2总单](tp2_overnight_runbook_20261123.md)，L5卡8/9、独立端口/目录、Qwen3.8现役W8A8、Ascend910_9382、0.26rc1-a3、实际CANN9.1、eager、单chunk、1024/1024/1、4生成token。安装r7.1三wheel及同发行源码，校验清单；许可必须使用r7.1 source-review SHA。

黄区CLI目前只有原12项。接线人仅增加`15.alias_detach`显式映射，使用FaultConfig(fault_id=15, variant='alias_detach', implementation_version='1')，同名可信profile、native_library/native_sha256均null；禁止改准入和预置资格。宿主仍只用公开ProductClient，不加人为copy_或注入后输出补丁。

```sh
SOCKET=/absolute/path/to/l5-alias15-session.sock
dcatpfi fault list --socket "$SOCKET"
dcatpfi fault enable --socket "$SOCKET" --fault-id 15.alias_detach
dcatpfi fault status --socket "$SOCKET"
# 按总单发送三条不同真实纯文本请求，每条4token，再发5token范围负例。
dcatpfi fault disable --socket "$SOCKET"
```

## 执行前的自然链检查

沿用只读普查工具做正常请求，对照观察关闭前后；源码包包含alias15_readonly_probe.py及alias15_output_probe.py供可信扩展接线。探针main为蓝区隔离脚本，不能替换黄区服务启动器。确认Attention.forward预分配base→ND视图→impl真实写入→gate读取base→o_proj读取gate结果。私有格式或持久cache不在本变体范围；没有gate或不可改向的内部写入不套用本adapter，返回不适用后停止。

产品代码仍逐调用核验实际源码SHA、BF16/ND、几何、活跃对象、初值相同且临时存储独立。A3 attach核验TP2、芯片原名、toolkit9.1以及Worker实际映射的libascendcl/libruntime文件SHA和元数据9.1归属。缺失或不匹配均拒绝，不能用A2证据填充A3字段。

## 两个owner与独立审查

同组Worker每owner完整走金丝雀→enable→三用户请求及范围负例→disable/detach；第二owner重新装许可，保持pid/thread/设备相同。全rank所有合格full-attention调用持续触发。预热最多6轮、连续2轮hidden/logits有限且精确一致。故障请求本身可非有限，不要求恢复其历史；OFF/health用新请求完整精确对照。

每调用证据：视图初值正确、真实impl把应写值写入临时存储、base字节保持旧值、gate实际读取旧base、o_proj消费gate结果、设备完成事件、无快照回写覆盖合法更新。changed为数值变化调用数且≤applied；changed_elements为base与应写值不同元素数。八计数完整、eligible失败0、区域字节/活动租约钩子/新金丝雀三项恢复齐全。

```sh
cd "$SOURCE"
PYTHONPATH="$PREVIEW_SITE:$SOURCE" python tools/review_product_mode.py "$RECEIPT" --tp 2 --output "$REVIEW"
PYTHONPATH="$PREVIEW_SITE:$SOURCE" python tools/review_alias15_product.py "$RECEIPT" --tp 2 --a3 --output "$ALIAS_REVIEW"
PYTHONPATH="$PREVIEW_SITE:$SOURCE" python tools/extract_product_record.py "$RECEIPT" --runtime-identity "$RUNTIME_IDENTITY" > "$TEXT"
```

回传提取文本每页≤4KiB、独立机理审查全文、原件文件名/字节/SHA、两个owner表、缺许可/未知profile/机制99及合法正对照。源码/TP/格式不符和5token负例也留存。A3尚无实测回传时状态为待验收，不能把候选包发布当作产品通过。
