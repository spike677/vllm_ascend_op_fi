# 33：合法分页归约重新设计

## 原失败结论

a27/a28均在768-token输入、控制分组128/替代32下未通过。每rank48次decode调用消费及恢复完成，但rank0仅2个原生等值单元发生变化，rank1仅1个；rank0最终hidden/logits无变化。不能据此断言误差在哪个后续算子消失，也不能把局部变化等同模型验收。旧结果保留，未登记。

## 本轮选择和冻结矩阵

选择方向a+c：增加真实decode历史长度到2304，仍为单请求单chunk；替代分组8（第一attempt）、16（第二attempt），控制分组128不变。固定同一自然文本token规则，两attempt沿用相同输入、种子、模型与FP32计算。不通过时按相同原因最多两次止损；设备冲突单独记录。先冻结矩阵再执行，不根据fault结果挑选输出单元或改误差倍数。

这里的8/16是torch参考online-softmax的分组，不宣称它们是原Native kernel合法tiling。算法对每个实际cache key恰好访问一次，最后不足块大小的尾块完整参与，因果集合不变、分子与分母同步max重标定、FP32累加。不重放分母、不缩放输出。控制重算舍入后与原Native相等的单元才可替换；控制不相等的单元保留原生值。

方向b不采用：目前没有原Native使用低精度部分和的来源证据，禁止人为降精度来扩大差异。方向d暂不宣称实现：已有host/API源码不足以唯一还原实际launch的内部归约顺序；不能把猜测的顺序称为Native一致。后续若需要它，必须先取得精确kernel/launch证据，由负责人决定投入。

## 适用性与公共入口

仅蓝区A2/CANN9.1/0.26/Qwen3.8社区W8A8/TP2，纯文本eager串行，无prefix/MTP/connector。部署侧可信binding显式传入控制/替代分组，公共用户不填写层名或内部字段。最多4096token预算；本轮固定2304输入，max_model_len/max_num_batched_tokens均4096，生成4token。该参数扩展仅用于33研究验收，不扩大已冻结r3/r4其他机制的1024范围。

真实decode页表、当前query、裁剪后的合法K/V、gated o_proj消费者与原租约恢复路径不变。prefill仅观察。要求两作用域同组Worker、每rank全部eligible消费恢复、真实hidden/logits传播、完整OFF/health精确恢复、资源零及负向检查。无模型传播时不包装成通过，继续默认关闭。

## 结果

本轮两项已执行：paged33-long-a1（8）与a2（16），源清单737e11d98d82fcfece79a5c904fbb67f5fba3fe63d3e4497b3eae74d80d20de1。两项均为rank0 CONTINUOUS_NO_MODEL_PROPAGATION；rank1分别有4/3个选中元素变化，48次eligible全部消费及恢复。两次同原因失败，按任务止损，不追加候选；未完成两作用域资格。统一恢复RPC完成，两个控制器均报告isolation_removed=true。r4先冻结已通过的17/41/26/20，不因33未完成改写历史结论；若本轮通过，另行发行候选。
