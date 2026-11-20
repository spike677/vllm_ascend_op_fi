# 2026-11-20 蓝区实机验收检查点

## 开发提交同步

开发线 `codex/product-integration-20261119`，本轮执行字节以清单而非旧 HEAD 名称认定。18/39/Batch/TP2-18 使用 `e85a886d` 上的审查中源码快照：`ea84055519103b91f970433c248fd47a1a639a36fae8f34e64a2c5ec88ff8da2`。新提交收录实现、独立审查器及小型总结；不提交本轮原始统计、完整张量、vendor 或权重。安装来源是已核对的源码目录，不冒称发行 wheel 安装验收。

## RC1.x / RC2.x 维护

冻结 RC1、RC1.1、RC2 的标签和包不变。未收到可登记的黄区 RC2 双作用域回传，不开 RC2.1 分支，不向黄区发送内容。RC2.2仍未冻结；多卡文件及接线只能进RC3，不能直接以当前开发 HEAD 打RC2.2。

## 逐项蓝区闭环

均为124 Ascend910B4/A2、CANN8.5.1、vLLM/vLLM-Ascend0.18、标准Worker、串行eager。下列总结通过重新读取原始请求及原生资源回读产生，不采信顶层PASS。

|能力|模型 / TP / 卡|实际执行与独立复核|边界|
|---|---|---|---|
|18 api_half|Qwen3-0.6B BF16 / TP1 / 2|同Worker两个owner，各normal/F/OFF/health；首RMS实际返回减半，14336元素变化；hidden/logits变化；全量hash精确恢复，资源0|token未变；不是全部目标覆盖|
|39 output_proxy|Qwen3-0.6B BF16 / TP1 / 5|同Worker两个owner，各四请求；实际MLP投影的一个BF16值变+Inf，SwiGLU消费；native open/apply/restore/release/live闭环；OFF/health精确，资源0|token变化；这是输出代理，不是matmul_range历史越界|
|Batch|Qwen3-0.6B W8A8 / TP1 / B8 / 5|同Worker两个owner；共12个阶段批、4个外部健康批、128请求；首decode qkv的Q 5565元素及scale 7元素变化；OFF Q/scale/hidden/logits精确，资源0|外部health只保留token和模型/Worker身份，没有完整hidden/logits健康捕获；不得称该层级也已验证|
|18 api_half|Qwen3-0.6B BF16 / TP2 / 2+5|同一组Worker两个owner，8请求/16个rank请求观测；rank0命中，rank1选中边界未变；两个rank hidden/logits均变，OFF/health精确，资源0|只验证实际Ascend use_all_gather=True分支；没有直接collective边界记录，不唯一归因某次all-reduce|

原件只留本地私有目录及远端专用attempt；公开摘要：
- [TP1 18](../../experiments/worker_core_v11_20261118/ordinary18_two_scopes_20261120_summary.json)
- [TP1 39](../../experiments/worker_core_v11_20261118/ordinary39_two_scopes_20261120_summary.json)
- [TP1 Batch](../../experiments/worker_core_v11_20261118/ordinary_batch_two_scopes_20261120_summary.json)
- [TP2 18](../../experiments/worker_core_v11_20261118/m1_rms18_tp2_20261120_summary.json)

`review_two_scope_requests.py`、`review_batch_two_scopes.py`、`review_m1_rms18.py`分别重算这些总结。TP2清理以Host历史的finish后分rank状态为准；完成health前后尚挂载的观察hook不能当作finish后泄漏，也不能略过真正清理回读。

### 失败保留与资源授权

初次TP1/M1曾因ATK容器保留设备命名空间触发驱动87/507899，在注入前失败；其他独立尝试不覆盖失败原件。旧门禁只识别`ptq_`，现补充`atk_q_`。用户授权仅停止本批Atan2在卡2/5的实例，包括重派实例；未修改Docker入口、未删除数据、未停止其他批次。原Kubernetes旧分配仍按授权豁免原样记录，不把原始idle=false改成true。

## 发行包交付

本次尚不新发RC，不修改部署准入登记。18/39的TP1双作用域补验可用于后续RC2.2收口；Batch的外部health观测范围需明确。未验收的模型/TP/硬件组合默认关闭。

## 多卡主线进度

M1的18已通过上述限定实际分支。39/22的TP2 provider及Batch/41多卡尚未完成，不能由18或TP1结果推导。M2受0.26资源/驱动门禁及权重预算限制；M3待A3专用实例回传；M4待DeepSeek实际运行证据，蓝区不做DeepSeek实测。

## 现役路径进度

本次没有新黄区材料；0.26完整流式解压总量、精确驱动配套依据未补齐，不拉取镜像或改宿主栈。A2/0.18小模型通过不代表A3、0.26、Qwen3.8或DeepSeek通过。

## 28 限时进度

B1于2026-11-20 02:26:37 UTC开始；真实8.5.1 SDK对象/CPU借用合同通过，新完整host/kernel构建成功，completion SHA `1b1ac86d6d85063ef7fa8a8f9d2ae99eae46d35560645de7bef3d4a482d4fe78`。构建约1.28GB，不占NPU。

首轮研究脚本仍经公共入口，在prepare被拒绝，未进入目标FIA观测请求。这是入口接线缺口，已改为实验目录内显式`create_research_provider`的零F驱动：仅一次observe、禁止公共owner共存、不授普通任务资格。公共Runtime/config拒绝保持不变。第二次真实正常借用在typed绑定前因研究owner不符合Native hex32合同被拒绝；失败及所有权未清理状态保留。已修复owner格式并保留绑定首个返回码，第三次以新进程/作用域验证中；完整stock/private或M128数值桥接、F/OFF以及第二作用域尚未通过。

4小时NPU、两个工作日限制不变。占卡时间须含初始化/失败/健康请求，后续按controller实际起止补记。20/26两变体/33仍为普通适配缺口，未将历史研究证明转为当前普通任务通过。


CPU验证：共享TP1/TP2调用链与设备门禁128项通过；28打包/typed/借用/研究入口与总结审查76项通过；owner/绑定返回码修复后13项专项通过。集合重叠，不累计为独立测试总数。
