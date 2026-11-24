# 故障15：黄区只读链普查工具说明

仅观察，无故障注入、无许可安装、无产品资格。工具不改Q/K/V、权重、视图、写入参数或返回值；不添加清零和替代写入。只为阶段0/1收集“真实view→原地写入→相同活跃存储区间被读取”的候选。档位确认之前不得将其改为注入器。

## 环境与使用

使用已匹配的Qwen3.8专用实例环境（torch/torch_npu/vLLM），先TP2；不替换冻结r6目录，不在现役进程热装。工具源码目录加入独立PYTHONPATH，包含 `tools/alias_chain_census.py` 和 `experiments/worker_core_v11_20261118/alias15_readonly_probe.py`。`AliasCensusExtension`是独立测试Worker扩展，可与黄区已有可信Worker扩展组合；不得把无鉴权管理RPC暴露给普通用户。

蓝区脚本 `alias15_readonly_probe.py` 的main只针对隔离容器中的 `/model`、`/evidence`、TP2，不能原样当作黄区服务启动脚本。黄区沿用自己的专用实例启动参数、模型目录和源码身份，扩展方法用于以下已存在executor调用；主项目无需增加产品故障入口。

```python
# 已建立的可信executor；RPC结果仅保存在黄区诊断目录。
executor.collective_rpc('alias_census', args=('start',))
try:
    # 两次相同真实文本请求，各4个生成token；保持greedy/seed与正常对照相同。
    # 用原有HTTP发送方法，不传注册prompt/层名/地址。
    send_normal_text_request()
    send_normal_text_request()
finally:
    rows = executor.collective_rpc('alias_census', args=('stop',))
    # 保存完整rows为alias-census.json；之后再发同输入正常请求作观察关闭对照。
```

开启前跑两次正常请求；观察期间两次；关闭后一次，记录token/text是否相同。该检查不等于hidden/logits完整精确恢复，也不是故障15产品验收。最多16次execute_model；达到预算拒绝继续观察，必须stop。stop撤销本工具所有模块观察hook及execute_model包装。拒绝已有同名execute_model实例包装，防止覆盖其他工具。无调度token的清理迭代也计入execute预算，不能当decode token步数。

```bash
python tools/summarize_alias_census.py alias-census.json > alias-census-text.txt
```

回传每行原样JSON，≤4096字节；原始回执留黄区。补充实际包/模型config/量化描述/索引SHA、硬件与软件版本，以及完整结果的文件SHA。社区权重的量化out_proj结论不自动代替现役BF16 out_proj。

## 观察边界

- 只统计dispatcher声明的alias返回和可写参数。直接Triton/raw-pointer写入、自定义算子内部、编译图融合内部可能不可见；零命中不证明链不存在。
- 使用活着的base弱引用、storage身份及连续字节区间交集关联消费者；不只比data_ptr，不保留/搬移模型张量，不读写裸地址。
- format来自实际 `torch_npu.get_npu_format`，失败为null；源码出现“NZ布局”字样不等于格式29。当前工具不能证明kernel内部格式转换发生独立存储拷贝。
- 消费者是后续值算子的真实输入且覆盖写入区间，属于host调度链候选；没有用这个计数替代设备数值读写证明。
- 每次forward最多96条样本，aggregate计数继续统计；单storage待关联证据上限64，超过时显式记录 `pending_evidence_dropped`。发生截断须保留，不能宣称全覆盖。
- 调度索引/CPU布尔张量即使命中也不是自动推荐的数值故障点。产品阶段须另外验收自然消费者、恢复和租约。

DeepSeek V4-0731仅在黄区匹配0.32/A3/TP8+EP实例具备窗口后另行普查；此工具没有为V4提供已验收资格，也不要求蓝区拉取0.32或运行V4。
