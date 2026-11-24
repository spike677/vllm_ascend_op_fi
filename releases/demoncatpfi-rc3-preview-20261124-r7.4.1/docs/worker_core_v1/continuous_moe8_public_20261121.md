# V2-Lite 故障8：公共入口持续索引错位

组合为124 A2 Ascend910B4 / CANN9.1.0 / vLLM0.26.0 + vllm-ascend0.26.0rc1 / torch2.10+cpu + torch_npu2.10.post4 / DeepSeek-V2-Lite-W8A8 / TP2、EP关闭、eager、单真实请求。不外推V4.1、TP8、EP或多请求路由。

public-a20 两个作用域在同组Worker通过。每rank每作用域：26个实际AscendMoERunner，4次forward，observed/eligible/attempted/applied/consumed/restored均104，failed=0，changed=26。78个decode调用只有一个token组，合法错位退化为原值，仍真实应用、消费、恢复，不伪造新变化。338个实际路由行确认属于当前内部请求ID。

实际链路是当前调用的 token_dispatch → 专家MLP → token_combine → A2 DeviceOperator 的索引abs → 原生 `npu::npu_moe_token_unpermute`。每次保存当前metadata和真实专家输出对象，按对象及请求ID关联；不使用基线路由列表要求fault出现同一专家。对真实反排列索引和对应概率按下一token组（尾组保持）重排，由原combine消费一次。同stream完成后恢复两个输入对象。没有替换MoE输出，也没有追加一次模拟kernel调用。

两作用域所有阶段输出有限，fault hidden/logits变化，OFF/health完整输出精确恢复，租约/hook/在途资源为0；真实无许可创建拒绝和耗尽后advance拒绝通过。耗时1.585177×/1.498553×。独立报告：`experiments/worker_core_v11_20261118/continuous_public_moe8_review_20261121.json`，保留原回执SHA和审查器SHA。源码清单 `56de605fe3c6c58026231647b61f22235795bd203eb09d715284688dc935eccb`。

## 调试与隔离记录

|窗口|UTC开始→结束|结果|
|---|---|---|
|a18|13:19:18.391063→13:21:57.192016|旧forward_impl选点漏掉0.26实际AscendMoERunner；未注入|
|a19|13:24:46.598291→13:27:28.157257|真实A2消费者带abs派生索引，原对象检查拒绝；补实际数据流跟踪|
|a20|13:28:56.994447→13:33:15.015590|258.021秒，公共两作用域独立复核通过|

日期均2026-11-21。每个窗口的卡2/5有界隔离均解除，无外部容器停止事件；不修改Docker入口。失败窗口退出清理不冒充同进程恢复，只有a20用于验收。源码指纹固定dispatcher、MoE pipeline和AscendMoERunner，MC2/All2All/EP不支持时拒绝。r2发行物不变。
