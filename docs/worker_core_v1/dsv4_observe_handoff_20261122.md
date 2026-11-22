# V4-0731只观测试跑（不注入）

目标A3 / V4-Flash-0731-W8A8 / v0.32.0rc1-a3 / CANN9.2.1 / TP8+EP / DP1，4096以内长prompt单chunk。蓝区不运行V4。不得与V4.1 W8A8的16卡OOM/1M上下文实验混合。

源码清单为[dsv4_0731_source_review_20261122.json](dsv4_0731_source_review_20261122.json)。请在镜像已加载的vllm_ascend包上逐个返回清单paths的SHA，不用Git候选的SHA覆盖实际文件。`dsv4_observe_probe.source_fingerprints(paths)`以实际import包根目录读取，只返回相对路径和SHA，缺失为null。逐项与候选比较，相同也只是源码身份，不是注入通过。

可信验收驱动使用`experiments/worker_core_v11_20261118/dsv4_observe_probe.py`的observe callable，经现有受信collective_rpc：`source`（options.paths为清单路径）、`start`、真实服务推理、`stop`。不创建Worker，不注入，不写入张量，不改变调度，不创建fault许可。必须先完成源码/硬件/软件/模型身份收集；只观测可接收不匹配源码以发现实际差异，绝不据此放行注入。

start仅在现有runner.execute_model边界装观察上下文，每次实际execute内部进入、退出TorchDispatchMode；stop在空闲控制边界移除自身包装。已有边界owner时拒绝，不能覆盖其他任务的wrapper。没有调用的rank记录0，不制造调用。观察器不会让原算子返回新张量。

每rank返回18、22、39op、39mr、Batch、41、8、28的observed/eligible、prefill/decode/unknown计数、实际operator、Python调用路径、输入输出shape/dtype、不适用原因和实际request ID。这里eligible只表示类型和请求映射具备候选条件；`injection_adapter_validated=false`，不等于可注入。28/41的压缩/稀疏布局没有运行join证据时eligible=0并明确原因。C++融合内部未暴露到TorchDispatch的调用报告不可观测，不能将0解释为模型没有算子。

每次推理≥2500 token，固定4生成token、≤4096单chunk。输出按rank分文件，文本回传时按rank/机制拆分，保持所有不适用和不可观测项。stop结果、源码SHA、运行时实际CANN身份和prefill/decode独立计数交负责人转回。这里只观测入口CPU测试通过；等待真实结果后调整site，再制作第二批验收包。
