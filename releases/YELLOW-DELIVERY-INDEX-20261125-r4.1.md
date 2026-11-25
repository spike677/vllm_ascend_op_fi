# 黄区交付索引 r4.1：V4 0.30 探针 r2

r4.1仅纠正r4提交时Git换行转换造成的发行目录JSON副本SHA不一致；加入该目录的-text规则，按原始审查字节保存。r4标签保留；源码、ZIP、包内manifest和源码清单完全不变，使用本索引的r4.1链接校验。

仅追加本次独立探针补件，r3索引与所有冻结包不变；Qwen继续其当前测试，不等待本补件，不重新部署公共机制。负责人转交，Codex不直接联系黄区。

源码提交：`d037495ab022bbc500f90bfe3213eb6d66126fb3`。本次发行标签：`yellow-delivery-20261125-r4.1`。包目录：`releases/v4-v030-tools-20261125-r2/`。旧工具r1不再用于新的V4探测。

|文件|字节数|SHA256|
|---|---:|---|
|[a2-review.json](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/a2-review.json)|5179|`5e9594d27a317fbeb7b8990a11c1fa830796e7a6c0caac5c6fe6349965ef2b99`|
|[cpu.json](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/cpu.json)|6525|`fa21d3a92c493716dc13ea595cadc56c49d453542d1e30b91ea3b792a3568504`|
|[install-check.json](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/install-check.json)|912|`f52234540c642627116656c4974597dd59dd4daa967dc7167f1af04799f3d5a5`|
|[isolation.json](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/isolation.json)|2196|`9537f51bc0c0cfd0f88d9273321ec3bb3aada81ec5d8575f2136a71330828d8b`|
|[manifest.json](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/manifest.json)|48191|`fb2bb5d6dd4293e7940a293202f75ac68bdee77a41274ff1fc8d545e079357ea`|
|[SHA256SUMS](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/SHA256SUMS)|577|`77e3159b366f3036b85e48bb62ce511d3dda947616198dc91056dab85bd73c90`|
|[source_manifest.json](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/source_manifest.json)|29399|`c877ca5edb784d163bbeae21a46d87d96a46f5a5b8a7a2311159ca91cb8923d3`|
|[v030-tools.zip](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/releases/v4-v030-tools-20261125-r2/v030-tools.zip)|996972|`dbcb516a07d71dd7ffb6ea9dc60e3a7a5de8708561226c0a79acc165b13ecce2`|

交接单：[v4_v030_probe_r2_handoff_20261125.md](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r4.1/docs/worker_core_v1/v4_v030_probe_r2_handoff_20261125.md)。其余任务和此前发行入口沿用[交付索引r3](https://github.com/spike677/vllm_ascend_op_fi/blob/yellow-delivery-20261125-r3/releases/YELLOW-DELIVERY-INDEX-20261125-r3.md)。

## 门禁与来源

- 最终CPU45/45通过（原42项加包装对象、目标路径和目标层检查）；独立解压后，仅PYTHONPATH指向本包，再跑45/45通过。自带fault_injection.sites.dsv4_observer，目标环境提供Python/torch/torch_npu/vLLM，不依赖tools-r1目录。
- A2 / Ascend910B4 / vLLM0.26.0 / vllm-ascend0.26.0rc1 / CANN9.1.0 / Qwen3.8社区W8A8 / TP2：最终烟测324.149秒。两rank各3012次观察，无截断/错误；capture各4步完整、采样关联通过；原方法身份及前后推理结果恢复。rank1拒绝、rank0成功后的全组清理只移除本工具，外来包装保持。
- 最终源清单SHA：`c877ca5edb784d163bbeae21a46d87d96a46f5a5b8a7a2311159ca91cb8923d3`。ZIP根source_manifest.json、发行目录副本、CPU记录、A2回执完全一致；186个源码文件逐文件SHA复核。源清单base_commit为创建时的r3基础，新增源码由上述最终源码提交及清单共同锁定。
- 包内282个清单成员校验通过；258个fault_injection文件与r1依赖包逐字节一致，公共运行时/机制未改。源码对r3的fault_injection差异为空。
- 最终窗口北京时间21:13:02–21:18:26；隔离已撤销，项目容器已清理，无外部容器删除，无Docker/调度器配置修改。原授权硬截止21:33:30不再使用。
- 21:09的旧源码烟测不作为本包门禁；更早exit137/OOMKilled=false的一轮未通过，终止来源未证明。完整原件保留蓝区，未提交推理数组。
- A2结果只验证工具可移植性，不授A3/V4/EP资格；IPC退出警告已保留，不宣称宿主IPC全面认证。

## 执行要点

1. 独立解压核对SHA；PYTHONPATH只放本工具目录与必要的CANN环境。专用Worker只加载Dsv4ProbeExtension，宿主鉴权继续保留。
2. 第一次重启先identity/read和identity/inspect_binding，逐rank审查类/MRO/内层对象/包源码来源。仅同对象的同名类bound method准入，来源指纹不能放行未知包装。
3. 仅当实际绑定指向profiler插件时，按交接单的VLLM_PLUGINS精确白名单回退并额外重启一次。然后D1两轮、G0，紧接第二个独立Worker组G1。
4. formal/canary原文尚未提供给蓝区；G0前必须按交接单绑定其真实文件/token SHA与长度，不用占位符开始标定。两组各每prompt≥5样本，固定交错顺序；噪声上界规则不改。
5. journal记录逐调用及候选行关联，可同进程调整预算/实际target_layers；不凭采集成功解除COMBINE_OUTPUT_ROW_JOIN_UNPROVEN。公共8须证明combine输出处的请求归属、有效行和消费者。
6. 回传identity、binding、observe、noise四类提取文本。黄区diff四段尚未取得，收到后另作审查登记，不阻塞本包。
