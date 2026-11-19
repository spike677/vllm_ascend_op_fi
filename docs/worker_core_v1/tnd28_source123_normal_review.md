# TND28 Source123：正常调用关联复核

124 卡5的 Source123 在标准 vLLM Qwen3-0.6B FP16 推理中完成一个
2048-token TND prefill，以及解除绑定后的同 Worker 健康请求。原件由
Source127 的离线只读 reader 重新审核，结果为
`SOURCE_BOUND_NORMAL_PREPARED_INPUT_JOIN_RECORDED_NO_F_AUTHORITY`，无审核问题。
本页是总结，不是原始采集文件。

实际运行代码仍是 Source123，reader 更新没有改变该次运行或其原件。
读取器核对了11条原始 RPC、Host 回执、当前请求和清理摘要，以及14条 typed
Native 事件。实际输入对象与 K/V 列表描述符按源码 ABI 对应；实际 Slice 和 FIA
launch、注册 ELF、选中函数及库文件身份均有原件关联。

| 项目 | 实际结果 |
|---|---|
| 当前 public 对象与 typed 调用 | 对应成立 |
| K/V 描述符 | 按 ABI 位置逐字节核对，不从任意内存搜索地址 |
| FIA 实际 launch key | `5000000000000200103` |
| FIA 注册 ELF SHA256 | `d99ce5e9435a3df59ff4432f53f997305638de40e4b306918f587faf8d96dd81` |
| controller | 退出0；仍要求独立资格审核 |
| F 授权 | **否** |
| stock/private 完整数值相等、M128 control、F/OFF | **本次未证明** |
| C1/C2/C3/E5、实际主项目接线 | **不授予** |

读取器保留了前几次拒绝：固定协议中的约18.8 MB单条 RPC 超过旧16 MiB上限；
将所有 storage format 当作 ND；未识别实际 CANN 安装链接；将 ACL 的平坦分配
shape 当作 K/V 描述符的逻辑 shape。后继修正分别采用固定11条/单条32 MiB/
总计128 MiB预算、实际 SDK 的 NCL=47/ND=2、两条已观察且文件身份一致的
安装链接、以及 `NnopbaseSaveTensor` 的逻辑 shape 规则。均继续核原字节、
地址、偏移、分配范围及输入身份，没有放宽成仅比较元素总数。

固定依赖源码的描述符行为与实际参数一致；生产 NNOPBASE 完整源码匹配仍未证明。
Slice POD 本身未解码，false-LSE 临时占位不被说成 prepare 返回后仍存活的对象。
这些限制没有因为总状态通过而消失。

| 可追溯对象 | SHA256 |
|---|---|
| Source123 manifest | `a72d5feffef860028f3d0e10a72799c80c20fafc90f827c66c4f6b4413107524` |
| Source122 Native build completion | `58dfd1021a8a77267ab94e2a3b17d0a913c22365d4f0e7d2d84ba49473485181` |
| Source127 reader源码 | `7c7eb433408b5a90a5807a49723b57d263e6db68fbe656910c67bccf55b50c6c` |
| 完整独立复核报告 | `e0684e5479b74ddadf47ea64d32a25ea1211429510f5aa718bbd46f40ec3d942` |
| controller 原件 | `47e61941e00556a8471799d10834be44110b8ff8b4c1a88c9e919d245268bf32` |

私有原件位于124的
`/home/17-fault-c2c3-20261117/product-tnd-source123-card5-normal-a1/`；
复核文件为同父目录 `product-tnd-source123-normal-a1-review-reader127.json`。
本次发布不包含原始 RPC、完整张量或 Native 库。

下一步是把当前调用的 public 对象、同步 Execute workspace 和实际 executor
取消清理连接起来，再进行新包的正常/control/F/OFF检查。当前 normal 关联通过
不能替代这项实现，分母故障仍未完成。
