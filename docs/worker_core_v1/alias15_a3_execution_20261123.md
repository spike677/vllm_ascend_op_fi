# Fault15 B档 A3 TP1/TP2执行单

由负责人转交。A2 TP2已通过，见 [产品验收](alias15_product_20261123/README.md)。标签r7不包含15；不要在r7配置中强行启用，也不要删去A2准入检查。本单按A3普查返回后确认现役自然链、再开放受信候选的顺序执行，不能用社区权重结果代替现役。

## 先核对普查与现役身份

Qwen3.8现役W8A8、A3 Ascend910_9382、v0.26.0rc1-a3/CANN9.1.0，TP1和TP2分别记录全部身份字段及config/量化描述/权重索引SHA。沿用 [只读普查说明](alias15_census_handoff_20261123.md)，实际请求4生成token，比较开启观察前后。先确认现役实现确有以下自然边界：

1. Attention.forward真实分配base，创建不同对象、同存储的ND视图。
2. impl实际写入该视图，base按应写值更新；不能仅证明源码存在赋值语句。
3. 实际gate读取base，o_proj读取gate结果；保留设备消费完成证据。若现役无gate，重新审核直接消费者，不套用蓝区gate回执。
4. 写入区间为本请求短生命周期，不是持久cache；真实格式、dtype、几何、源码SHA与拟接线相符。

无合格链、内部不可重定向、持久状态恢复不明均拒绝，换自然候选后重新审查；不添加copy_拼接链。提取结果连同原回执SHA交负责人；Codex核对后单独提交A3受信候选部署适配。目前源码有明确 `ALIAS15_A2_TP2_TRIAL_ONLY` 拒绝，**该门禁尚未解除，不能在黄区自行补丁放开**。

## 候选接线后的验收步骤

仅在上述普查审查通过、收到对应A3候选提交及清单后执行。TP2、TP1分别独立验收；同一组Worker两个独立owner，不通过重启消除残留。

- 使用公开 `configure_product_deployments`、与真实组合和源码清单相符的许可；故障为 `FaultConfig(15, 'alias_detach')`，语义名称为“别名断开语义（ND视图），不含私有格式触发条件”。不填层名、rank或裸地址，不传Native库。
- 宿主仅attach/enable/status/disable/detach；正常金丝雀连续两轮全有限且完整hidden/logits精确一致后开启。
- 开启期间发三条不同纯文本请求，各4token，所有rank所有匹配full-attention调用持续触发；再发5token范围负例，须不注入且明确原因。
- 每次核对独立视图初值、实际应写值、旧base未更新、真实消费者与设备完成；changed为调用数且不超过applied，另报changed_elements。缺少任一证据不判通过。
- disable后区域字节、资源清零、新请求金丝雀三项恢复全部通过；detach释放binding。第二owner重复，身份必须相同。
- 负例：有效profile缺许可、未知profile、未经审核的源码/TP/格式、5token范围。actual_call待观测项按r6.1说明判断，不能掩盖其他身份拒绝。

使用同一候选源码中的 `tools/review_alias15_product.py "$RECEIPT" --tp 2 --output "$REVIEW"`，TP1改1。原件留黄区，回传机理审查全文（小型汇总）、各文件字节数/SHA、两个owner—回执对应表、每rank计数、恢复及负向检查。没有A3实际证据前不登记，产品使用免临时许可属于另一个待批准方案。
