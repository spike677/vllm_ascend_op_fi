# r8.2 效果工具 A2 验证完成

> 后续诊断更新：本页记录下述9464df0c那一次注入验收，不代表Batch根因已全部闭合。新增5f1dc24f的三轮无注入对照发现：实际组成相同，冷启动第0请求仍与后两轮不同；不能仅以组成一致推定确定性，也不放宽OFF精确恢复。见 `r8_2_batch_no_injection_20261126/README.md`。rank1身份随后补齐，登记现状见 `yellow_r8_identity_owner_review_20261126.md`。

组合：A2 / Ascend910B4，vLLM0.26.0、vllm-ascend0.26.0rc1、torch2.10.0+cpu、torch_npu2.10.0.post4、CANN9.1.0；社区Qwen3.8 W8A8，TP2，124卡2/5。只记蓝区验证，不赋予A3资格。

最终源码 `9464df0c2bd05eef00759f7cba359fee803a682e`，实机源码清单SHA `7ffa9de5a6fee00ab55b064cf5e64f179f6b2b6f483615a81638307c02c6cd60`。当前代码和这份实机快照一致，之后仅补审查文档。CPU **44/44**；公共机制/运行时 **442文件与冻结r8逐字节一致**。

## 两项结果

| 项目 | 同组Worker两个owner、全rank效果 | 恢复及负例 |
|---|---|---|
|18.variance_under4|每owner每rank1932次eligible/applied/changed/consumed/restored，failed0；API top5变化可见，各位置至少五个独立token-ID键|OFF API精确；产品金丝雀完整输出精确；字节恢复、资源清零通过；原产品请求范围负例通过|
|quant.batch_shared_scale|每owner每rank912次eligible/applied/consumed/restored；rank0 changed912、rank1 changed911，failed0；Q/scale参考及真实QMM消费者通过|B1不注入负例、首轮OFF组成不匹配→有限重发→相同组成且精确恢复；第二owner首次OFF匹配并精确；金丝雀、字节恢复及资源零通过|

changed911小于applied912表示一次写入值未产生新数值变化，不是失败。不将同一批次各请求的参与计数重复累加成调用数。

Batch最终scope0实际轨迹：baseline/fault两rank都是 `[1,8,8,8,7]`；第一次OFF为 `[1,4,4,4,3,1,4,4,4,3]`，判组成不匹配；第二次OFF恢复 `[1,8,8,8,7]`。比较同时核对prompt token SHA、位置、computed/scheduled token数，并非仅看batch size。scope1三轮组成相同。所有尝试原件保留，首次组成匹配即停止重发，随后才比较数值。

under4先用旧工具复现失败，再用服务端已有 `return_tokens_as_token_ids` 参数修复字符串键碰撞。全量logits有限性、请求/位置映射与根因详见 `r8_2_under4_top5_diagnosis_20261126.md`；没有调弱故障、少取候选或放宽恢复条件。

## 原件与独立审查

- under4最终效果原件：32,411,461字节，SHA `c40269cbe881c65c25e24e1d39ac26d67833ce9a443977f59b54243d3210d67d`。
- Batch最终效果原件：7,876,192字节，SHA `dac2f0c3f392740d75e83c3910eb417841fae3908da80c52e07063ae27f58a10`。
- 两项产品原件分别重新运行 `review_product_mode`，均为 `PRODUCT_BLUE_REVIEW_PASSED`；效果原件重建summary并verify，通过且与远端生成页逐字节规范化一致。
- 完整SHA、两owner/rank计数、所有轮次、控制器和隔离记录见 `r8_2_a2_completed_review_20261126.json`。原始大文件留在受控证据目录，未进Git。
- 页面 `registration_ready` 仅表示页面字段校验结果，不代替完整登记的负例/身份门禁，也不等于product_closed。

## 占卡与恢复

按负责人要求先等待10分钟，再检查卡2/5空闲后运行。两个窗口分别为北京时间16:58:04–17:13:12（907.84秒）、17:14:27–17:30:04（937.19秒）。最终under4服务存活340.06秒、Batch587.81秒，含模型加载、预热和控制，不能当纯kernel耗时或性能结论。

两个窗口分别停止4次、8次仅占卡2/5的外部容器，名称、镜像、时间均记录；不删除外部容器/镜像/数据，不改调度器、驱动、CANN或Docker入口。窗口结束即停止并join清理线程。17:31:38只读复查：本项目无计算进程，外部cumsum容器已重新持有卡5；保留它，未在验证结束后继续抢占。

负责人随后再次授权中止/隔离干扰推理的其它进程。后续活动推理窗口延续该保护：核对PID、容器/cgroup和设备归属，保护项目API server、EngineCore和Worker，只处理卡2/5的干扰任务；与其它卡共享设备的容器不能整体停止。容器外进程也须先证明实际归属，不按模糊进程名批量kill。服务/验证结束即解除隔离；不以这项授权长期空占设备。

## 尚需黄区材料

新到22/15页面与产品提取通过；over4页面通过但产品提取哈希不符且CLI流程失败。rank1完整身份、有效负例及Native构建元数据仍缺，r8.1登记分支维持起点e38c4113，不发布空登记版。外部视觉身份文件选择错误由黄区更正。17两份原件的图片/请求关联仍待补；同一蓝区原件两工具均55154，未发现计数定义差异。

33六条采集已回传，但tiling boundary/serializer未证明，N2未完整，不进入窗口二。V4仍按既定证据门禁继续。详情见 `yellow_r8_followup_review_20261126.md`。
