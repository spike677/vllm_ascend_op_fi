# 完整FIA图的独立分析与全请求验收

本轮10组30次完整模型图执行已通过联合验收。分析分为统计定位、真实执行覆盖和完整数组比较三条链，最后交叉核对；不能用host命中、模式或预期profile指定首差。

`graph_evidence.py`只读原始dump/stack/construct、配置、采集器身份、实际FX/GE图和原生CSV。它按图依赖排序所有候选；GraphBoundary原始输出的真实叶子路径映射到NetOutput生产者。对本轮原生 `[[logits,Q0,K0,V0,A0,...]]`，A0是 `$[0][4]`，对应FIA output0；没有改写dump、删除包装层或把module记录重命名为API。

```bash
python analysis/graph_evidence.py \
  --baseline /path/to/B --fault /path/to/F \
  --output /path/to/new_report.json
```

默认GE CSV首个可见差异是Mul_13输入1，保留326个先前不可观测节点。已披露的统一CPU FP64增强及真实输出别名提供更早的FIA输出统计。共享embedding权重多次使用会重复采集Data CSV；同一图/model/iteration且完整行完全相同时，保留全部来源并等价合并观测。不同执行组或不同数值保持歧义并拒绝定位。null、空张量和缺失记录不判相等。

`coverage.py`不把一次JIT回调算成28次命中。它逐个关联全部28个GE FIA节点、共享编译二进制身份、原生静态任务/runtime实例化及三轮84份CSV，再校验B8所有8个真实请求行、输入未变和后端实际绑定。公开ABI独立解码覆盖65范围/63字段/1,576字节；最终编译raw/key与真实public host记录一致，无SplitKV。

`compare_model.py`在124读取私有完整数组，比较所有请求、每层完整Q/K/V和Attention以及151,936维logits。首轮完整元素比较，另两次同步整批及逐请求SHA。输入/权重及请求映射必须一致；首差FIA输入的完整字节摘要也单独核对。它不参与统计首差选择。

`analyze_matrix.py`先保存独立统计推断，再读数值及覆盖；核对最早真实Attention变化层和统计推断一致，完成F/F、OFF、B1和无中间输出别名图的中性对照。第0层末query不变而第1层末query及最终logits改变，是实测结果，不是验收器预设。

`verify_bitwise_controls.py`把完整数组比较的SHA绑定到原始capture全批/逐请求摘要，额外防止将带符号零等数值相等误称为逐bit恢复。

`package_evidence.py`只复制审查白名单。主B8 B/F/F2含全部原生图CSV，其余7组保留全部FIA CSV、原始statistics/mix、完整图和私有全量验算报告；不冒称公开包包含控制组每个算子的全部CSV。多个Data观测均保留，以source_mapping记录原路径和字节SHA。编译器字段摘录与原始文件明确区分。

`verify_public_bundle.py`检查15,119个导出文件及15,096份原始文件映射，重新校验10组public ABI、所有真实任务与840份FIA CSV，再构造物理盲验目录，只放统计、配置、采集器身份及FX/GE图/CSV。case/result、host、策略及完整数值报告物理缺席时，两步×两个F仍独立定位FIA output0。

```bash
python analysis/verify_public_bundle.py \
  --bundle /path/to/public/data \
  --blind-work /path/to/new-empty-review \
  --output /path/to/new-verification.json
```

34项CPU反例测试通过；最终结论还依赖实际NPU图执行和私有完整数组比较。公开复核不能重读未发布的完整张量、原始大日志或机器码，也不能凭四统计唯一认定某条Reduce指令的物理根因。
