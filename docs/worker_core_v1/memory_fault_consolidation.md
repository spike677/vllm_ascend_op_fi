# 41 / 26 FIA / 20：真实内存持续方法整理

2026-11-21 夜，阶段4准备。**尚未启动新 Native 内存实机验收。** 门槛为阶段1完成且 Qwen3.8 公共持续 39 output_proxy、18、22 全部通过；目前后者未齐。研究变体和旧证据保留，不替代新的公共验收。

## 历史实现与新边界

|能力|现有实现/依赖|新交付写入方式|不沿用的主张|
|---|---|---|---|
|41|`deployments/postwrite_kv.py`、Worker `postwrite41.py` / `serving41.py`；实际 KV span 与 Native postwrite，旧固定布局、单次协议|按当前 ScatterPaKvCache 或 MLA latent 布局选择真实 KV 对象内的列/槽位/页面|不声称相邻分配器块或原 kernel 越界；旧布局资格不扩展到新布局|
|26 fia_destination|`deployments/writer26_fia_typed.py`、`writer26_group_runtime.cpp`、`serving26_core.py`；固定 descriptor ABI、GWS/execute 前后证据、typed桥接|真实 FIA 后取故障数据，在同一请求的活跃 residual 对象内部写入|不再做参数级目的地址重定向、不借旧 ABI 证明新栈；residual_stomp 本轮排除|
|20|`deployments/buffer20.py` 与旧 v17 MatmulWeightNz 地址桥接；依赖 native callback、实际模块归属及有界输入区间|原算子执行前，用另一个已确认活跃对象的内容覆盖实际输入；完成消费后恢复|不声称替换了执行器指针，也不支持任意裸地址|

公共 Native 底座位于 `native_adapters/postwrite/postwrite.h/.cpp`。现有 ABI 有 read_span/open/apply/restore/release/live 和原始快照读取；apply 主要面向 BF16 特殊值/符号/清零，不足以直接表示“活跃源区域→活跃目标区域”的全部需求。不得把 C ABI 接收指针误解为用户可提交地址。

## 分层接口与版本适配

1. **模型 site 层**：从真实调用选择 producer、目的对象和 consumer；记录请求 ID、rank、步、实际 shape/stride/dtype、KV slot/block table。Qwen 和 MLA 使用各自布局解码器。依赖 vLLM / vllm-ascend 实际函数源码指纹；版本变化时重新审查语义，不能靠版本字符串放行。
2. **对象区间层**：可信 typed binding 持有 Tensor 与 Storage 强引用、分配身份、device、offset/字节长度、owner/generation。公共配置只选机制。Tensor改变存储、区域越界、别名重叠、无法确认生命周期均拒绝；不依赖真实物理邻接。
3. **Native transport 层**：新增有界读、从受信对象区间写入、恢复；仅接受由对象层颁发的 handle/token，C内部再次核对 device/owner/generation/区间。复用公开 ACL copy/stream/event 接口并检查全部返回码，不解析未经登记的 executor ABI。构建记录绑定 CANN SDK、实际链接符号、SoC、二进制 SHA。
4. **顺序层**：同 stream 的备份完成→写入完成→实际消费者完成事件→恢复完成。默认不跨 stream。事件来源、消费者输入与写入区间重叠必须有运行证据；Python返回或命中日志不能替代。异常时保留失败租约，禁止自动给新对象写回旧快照。
5. **Runtime 层**：复用持续公共 Runtime / request_plan / MultiRankTaskClient；全 rank 独立命中，条件点无 collective，在共同控制边界汇总。八项计数、资源和完整新请求 OFF/health 与 torch 类相同。

torch_npu 升级首先检查 storage/format/view 与当前 stream/event 的可用接口；CANN 升级核对 ACL 原型、返回码、事件和设备支持；vLLM 升级核对调度/请求、KV生命周期及真实消费位置。任何一层无法证明则关闭对应组合。没有“8.x/9.x 一律相同”的默认推断。

## 每次命中的恢复点

|能力|可接受的恢复窗口|必须拒绝的情况|
|---|---|---|
|20 输入覆盖|输入由当前算子独占读取；备份后写入，算子同stream完成事件之后立即恢复；其后再允许其他读写者|别的stream/算子并发读写、输入其实为可合法原地更新的output、消费者提前保存异步使用但没有完成事件|
|26 FIA→residual|FIA生产完成；在已识别的后续 add_rms_norm 读取之前改写 residual。若消费者写回该residual，须在正常更新发生前有独立读窗口，或提供严格干净请求重建边界|不能在 add_rms_norm 已合法更新后用旧快照覆盖。无法拆分读/写又没有可证明重建边界时，当前调用不适用，不改成输出扰动代替|
|41 KV区域|选择能证明读取和后续正常写入顺序的对象区间；当前消费者完成后、下一次合法KV写入前恢复。或请求结束时销毁本请求独占缓存并证明新请求完全重建|共享prefix、跨请求复用、其他消费者仍在途、正常decode已更新同区间后仍恢复旧字节；没有可靠生命周期证据的MLA cache布局|

“新请求重建”必须单独记录，不把它说成原污染对象已逐字节恢复。当前统一标准仍要求区域恢复、资源清零、完整 OFF/health 三项；若只证明销毁重建而不能满足前者，需要负责人另定口径，不能自行通过。

## CPU 与实机计划

CPU先覆盖：真实对象身份与地址复用、越界/重叠拒绝、同stream先后、异常保留租约、合法原地更新拒绝回写、多个连续命中、world_size=1/2/8同一路径、EP无本地调用不制造调用。Native单元测试用替身ACL验证错误传播，不能作为NPU通过。

门槛通过后顺序41→26 FIA→20；Qwen3.8 A2/0.26/TP2，再按适用布局处理V2-Lite。每项同组Worker两作用域、四token、最多六次预热、真实consumer事件、全部eligible零失败、完整新请求恢复、资源零、性能倍数。各窗口卡2/5有界隔离，结束恢复；不开展严格邻接探索或修改第三方kernel。

登记用语为“真实内存故障：KV对象区域改写 / FIA数据写入真实residual / 输入内容覆盖”。历史typed桥接和Native缓存ABI变体仅在原已登记部署可用，不让研究入口经公共配置开启。
