# Fault28 原生 FA 因果检查：预注册与独立分析

本目录只读分析新实验；当前没有在本目录宣告 NPU 验收结果。CPU fixture 是分析器回归，不能作为故障证据。旧发布包、旧运行源码及原始采集不改写。

`preregistered_contract.json` 定义运行前矩阵。固定使用旧模型真实捕获的 layer1、batch0 的 Q/K/V 字节；B1 用这一个序列，B8 用其八份等值副本。模型、prompt 或 batch 上游产生的 Q/K/V 差异不进入这个因果比较。

| 阶段 | 组合 | 次数 | 用途 |
| --- | --- | ---: | --- |
| 原生 stock bridge | B1/B8 × 两次独立运行 | 4 | B8 目标输出绑定旧实采 b0；B1 与旧 B8 的差异只记录。 |
| 主矩阵 | B1/B8 × base(128,8)、m_only(64,8)、r_only(128,4)、joint(64,4) × 两次独立运行 | 16 | 分离 batch、M、nRatio 及交互。 |
| 次级受控策略 | B1/B8 × batch_m、batch_r、batch_joint × 两次独立运行 | 12 | B1 实际 base，B8 分别采用 M、ratio、联合策略；按预注册执行，不按主矩阵输出选择。 |

主矩阵的 private BASE 必须与同 batch 的原生 stock bridge 的完整目标输出及实际原生 tiling 字节一致。B1 不被要求与 B8 输出相同。不同 B8 slot 的输出只记录，不假设它们相等。次级策略验证的是新受控的 Batch→tiling 选择链，不是历史 FIA 默认策略。

## 输出与效应

主端点为完整 b0 的 `[16,2048,128]` FP16 Attention 输出。私有完整输出文件逐字节读取并校验 SHA，CPU 按块将每个 FP16 数转换成精确的 `2^-24` 整数单位；19 个对比及交互的差分都使用 int64，避免分析自身的浮点舍入。报告每个 query、每个 head 的差异分布、首批不同坐标、差值 SHA 和幅度。这里的“首坐标”是逻辑张量顺序，不是内核执行时间或故障指令位置。

每个 batch 分别计算 M 在 ratio8/4 下的效应、ratio 在 M128/64 下的效应，以及 `joint−m_only−r_only+base`。另外计算每个 profile 的 B8−B1，以及这些效应的 batch 交互。不能从联合变化反推 M 是唯一原因。所有输出相同是有效结果；重复不稳定则保留观察并报告 `NOT_IDENTIFIED_UNSTABLE_REPEATS`。

固定 FP64 参考 query 为 `[0,63,64,127,128,511,512,1023,1024,2047]`，覆盖预注册分块边界；不根据输出挑选 query。独立 NumPy CPU 参考以真实 FP16 表示值、GQA `head//2` 和 native float32 scale 计算。参考误差方向没有验收阈值，也不表示模型质量变差。

所有六个 native 返回另行记录。FP32 softmaxMax/Sum 等辅助返回改变、主 FP16 输出不变，应分别表述；不得把辅助返回差异自动升级成主输出故障。辅助值同 batch 可比较全 SHA；跨 batch 仅比较实际认证的 b0 切片，不能比较不同形状全量 SHA 后声称数值差异。

## 实际执行与源码计划

`profile_evidence.py` 将一条同步 API marker 窗口、实际 NNOP FA launch、原生日志完整 tiling chunk 与公共 `TilingContext` 记录绑定。只有日志声明长度的 1464 字节前缀与公共 raw 完全相同，才确认 framing。它校验公开 SDK POD 的 205 个字段，包括 BMM1/BMM2 各 50 个 TCube 字段。请求 profile 名称只用于核对预注册条件，不能提供实际 M/nRatio。

`query_work_order.py` 沿用已审查、固定 SHA 的 S1S2 kernel 源码推导，并增加全部目标 batch task 的输出。`causal_core.query_orders` 为每个 query/head 独立重建 computed key groups、合法 causal key groups、masked 数量、query/task/AIV 归属以及动态 Vec1/Softmax shape。合法 key 每个恰好一次、query 不跨核心拆分；额外计算后被 mask 的未来 key 不是非法 key 参与结果，也不是重复合法 key。

M-only 保持 nominal S2 group 不变，仍可能改变早期 query 的 computed tail、带掩码的范围、Vec1/Softmax shape 或 BMM 子 tiling。尾 query2047 的分组相同不能代表全序列。实际内核现场重建 Softmax 子 tiling，不能仅凭 host 保存的 softmax metadata 识别其实际归约计划。报告分开显示合法分组、computed 范围、动态 Softmax shape、行子块分配和 TCube 字段变化；这些是源码派生计划，不是硬件浮点指令轨迹。

## 调用

```sh
python analysis/test_causal_analysis.py
python analysis/verify_causal_matrix.py \
  --cases-dir /path/to/primary_cases --results-dir /path/to/primary_run \
  --stock-cases-dir /path/to/stock_bridge_cases --stock-results-dir /path/to/stock_bridge_run \
  --input-manifest /home/fault28-causal-20261029/inputs/manifest.json \
  --source-root /home/fault28-causal-20261029/source/ops-transformer-v9.1.0 \
  --pod-schema /home/fault28-causal-20261029/build/evidence_a1/fa_pod_schema.json \
  --vendor-manifest /path/to/approved/vendor_manifest.json \
  --previous-analysis /home/fault28-native-tiling-20261029/analysis/tools \
  --output /path/to/new/primary_causal_report.json
```

`--previous-analysis` 必须包含 SHA 固定的 `verify_stock_off.py` 和 `extract_fa_log_profile.py`；根据它们的实际已审查目录传参。源文件/日志/私有 tensor 均只读。主报告没有输出 tensor 值；私有 `.bin` 不能进入统计包或 GitHub。次级 12 次的策略门槛须单独分析，不能用主矩阵结果替代。

### 分批运行的中途门槛与索引

完成 stock4 和 privateBASE4 后，先运行 `verify_base_bridge.py`。参数与上面的主分析命令相同，其中 `--cases-dir` 仍指向完整冻结 primary 配置目录，`--results-dir` 只需包含四个真实 BASE 结果。工具只选择四个 BASE 配置，不读取或伪造其余十二组；`PASS_STOCK_AND_PRIVATE_BASE_BRIDGE` 仅表示同 batch 的 stock/privateBASE 完整输出、实际 raw 与重复稳定性门槛通过，不表示后续交叉策略已成功。

配置的 `case.phase=stock_bridge` 与实际 API 的 `native_call.phase=operator_replay` 各有用途。stock 仍必须 `operator_variant=stock`、`scope_token=null`，且没有 `host_events.jsonl` 或自定义 vendor。

controller 每批最多八个 case。多批完成后，在 Linux 证据主机用下述工具建立新索引，再将索引目录作为分析器 `--results-dir`；原记录不复制或改写：

```sh
python analysis/index_attempts.py --matrix setup/primary/manifest.json \
  --attempt outputs/private_base_a2 --attempt outputs/m_and_r_a2 --attempt outputs/joint_a2 \
  --output analysis/primary_index_a2
```

例子中的 attempt 名称必须换成实际成功记录。`--plan-only` 仅校验并输出索引计划；`--profile base` 可将完整 primary 配置筛选为四个 BASE 索引。只接受已完成且清理成功的明确 attempt，核验 completion/result/冻结配置 SHA，并拒绝重复 case 或不完整覆盖。索引保留原来源映射与 SHA；链接本身不强制目标只读，分析消费者必须保持只读。`verify_secondary_batch_policy.py` 同样接受 automatic 两批的合并索引，并须绑定已完成的主矩阵报告。

本工具不读 msprobe 来赋予原因，也不从 host 命中记录指定 msprobe 首差；后续原生统计独立推断应使用单独流程。当前比较仅限原生 FA replay 与既有模型传播证据，不能升级为历史 38B/FIA 故障、TP2、decode 或模型质量结论。
