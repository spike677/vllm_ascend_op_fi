# 全 rank 持续触发公共 Runtime：阶段1记录

依据2026-11-21晚最新任务书。阶段0已提交并推送至 `d71c9351`；原8个诊断提交也已推送。冻结发行物没有改动。以下公共入口工作仍在独立审查与实机复测中，尚未授予新组合或冻结RC3。

## 接线

- `continuous_binding.py`：可信部署安装 typed binding，固定实际模型、源码指纹、组成员、prompt及算术后端。安装不授予许可、不创建请求。源绑定复用0.26实际 execute/preprocess/sample 和并行组核查。
- `providers/continuous.py`：复用 WorkerFaultRuntime 的 owner/generation 与 MultiRankTaskClient 的持续模式；不是独立故障控制服务。normal/fault/off/health由真实enqueue内部ID推进；每请求观察prefill及3次decode。
- 预热最多6次，Worker内观察两次连续有限且完整hidden/logits SHA一致。故障前未满足预热门禁就拒绝prepare。
- 实际调度证明单chunk；请求集、CPU调度镜像、真实模型参数对象、sample返回ID必须对应。视觉调用拒绝。逐调用处不加入collective。
- 新torch绑定只适配Qwen3.8路径。没有typed绑定时，28 Native研究路径保持关闭；V2-Lite公共适配仍待阶段3。
- 结果分别报告注入区字节恢复、消费者事件、资源清零、OFF/health新请求完整输出恢复。故障失败后的recover不改写失败结论。

## 对象租约

`engines/region_lease.py`对每次真实改写持有tensor和storage强引用，记录设备、storage身份、数据地址、大小、offset、区间、shape/dtype，重叠拒绝。备份、写入、消费者完成、恢复按同stream事件排序。地址/存储变化时拒绝旧快照回写；消费者后有合法原地变化时也拒绝回写。inference tensor没有version计数时，用持有的预期字节核对，不能假定版本可用。

Batch Q/scale使用该租约。GQA替换返回行和原输出区间分别持有租约；实际消费者前验证原输出是原值或已经复制的已知替换值，消费者后才恢复。Native内存三项未在此阶段开放。

## Batch算术与抽样范围

保留CPU FP32全量参考路径，新增NPU FP32共享量化。共享scale必须逐行与CPU对设备全量absmax标量计算的FP32 `/127` 完全一致；CPU输入/Q参考为最多1024个确定性样本，包含极值位置，不声称全量CPU逐元素核验。

Q差异只能为±1且FP32除法结果距离半整数不超过2 ULP；其他差异拒绝。差异数量与比例均以抽样数为分母，不能外推全张量差异率。所有实际Q/scale仍交原QMM消费。

首次NPU运行a2因scale非精确被拒绝，没有放宽scale门槛。新增NPU补偿式 `s + ((max - 128*s) + s)/127`，利用127=128−1恢复近邻残差；CPU BF16范围枚举验证±1 ULP初始误差的纠正。有限性、scale精确比较和Q边界检查仍保留，不能把CPU检查当作NPU通过。

## 实机尝试

均为124 A2 Ascend910B4 / CANN9.1 / vLLM0.26、vllm-ascend0.26rc1 / Qwen3.8社区W8A8 / TP2。完整私有回执不提交Git。

|尝试|结果|窗口秒|隔离恢复|
|---|---|---:|---|
|public-a1，28|公共入口两作用域初轮完成；每rank每作用域64次applied/changed/consumed/restored，failed=0；OFF/health完整hidden/logits精确一致；1.2638×/1.1547×。尚需整合租约后的最终复核|318.452|已解除；未停止外部容器|
|public-a2，Batch NPU|scale精确检查失败，未通过；本项目失败容器主动停止后退出137，不是外部冲突|350.488|已解除；未停止外部容器|
|public-a3，Batch NPU补偿|公共入口两作用域完成且独立复核通过：每rank每作用域304投影、1216次applied/consumed/restored，failed=0；changed分别1214/1213；完整OFF/health恢复；6.2112×/6.0967×|333.361|已解除；未停止外部容器|
|public-a4，Batch CPU FP32对照|公共入口两作用域独立复核通过；6.0880×/6.4494×；许可/租约/tenant清零|329.595|已解除；未停止外部容器|
|public-a5，28最终逐次租约|公共入口两个作用域严格复核通过；每rank64次；1.4265×/1.3416×；真实无许可创建拒绝和最终清理通过|322.398|已解除；未停止外部容器|
|public-a6，Batch最终交付脚本|严格独立复核通过：真实创建负向、通用部署许可、每rank1216次；6.3440×/6.2005×；全部恢复和清理通过|337.352|已解除；未停止外部容器|

a3每rank每作用域分别抽样1,245,170和1,245,168个Q元素，样本差异均为0，样本差异率0；不能外推为全张量零差异。scale检查通过。独立总结见 `experiments/worker_core_v11_20261118/continuous_public_batch_npu_review_20261121.json`。

窗口沿用资源锁和每次门禁、只针对卡2/5的有界清理器；不修改Docker入口、驱动或CANN，不删外部容器或数据。

## 独立审查及交付缺口

`tools/review_continuous_public.py`只读公共回执，重查实际rank身份、两个owner、真实请求映射、4次完整输出、预热、覆盖、OFF/health和资源；通过也不自动写登记表。缺字段或不一致即拒绝。

只读 `demoncat_inspect_readonly` 不经过TP1注入前提；按128模块分页，多rank汇总保留缺失和失败rank。summary保留18元素级、22行级效果及覆盖计数；原始recovered为null时另行披露，以明确恢复证据派生展示值，有矛盾时报告失败。

阶段1已完成：28和Batch精确BLUE组合登记、公共验收、RC3预览构建及124无设备安装源码校验。交付必须使用 `demoncatpfi-rc3-preview-20261121-r2`，见 `releases/RC3-PREVIEW-REPLACEMENT-20261121.md`；原预览 Git 换行失配已作为失败发行保留。348个安装源码/配置文件一致，r2全部发行Git字节SHA一致。历史全量测试未全绿，局限及定向复测在发行回归报告中披露。

28已登记的精确BLUE组合仍要求可信作用域许可，不能借同名机制放行Native研究路径。阶段2～5其余10项持续公共验收、V2-Lite公共接线、Native真实内存和V4.1/M4尚未完成。现阶段不把历史实验binding证据升级为发行验收。
