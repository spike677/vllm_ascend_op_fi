# 卡 5 释放后的恢复入口

用户要求保留 `gyl_test` 的 8002 推理服务。不得停止服务或外部容器；仅在设备实际空闲、任务资源锁及逐次门禁都通过时执行本任务。卡 2 不在本次 replay 的设备范围内。

新远端目录固定为 `/home/fault28-causal-20261029`。先核验 `handoff_ready.json` 的 READY 状态及所绑定的文件 SHA；不要把 CPU 检查当成任何 NPU 运行回执。没有该文件或状态未就绪时继续准备，不能越过。

## 分批计划

`outputs/cpu_prepare/controller_plan_a1/summary.json` 保存全部 32 个 case 的六批分组与 prepare-only 回执。它们只生成命令，没有 NPU 调用。顺序固定：

1. `stock_bridge`：4 次原生基线；两次重复应稳定，B8 的目标输出必须与冻结的旧模型捕获一致。
2. `private_base`：4 次新 vendor 的 base。用独立 bridge 分析验证同 batch 的完整目标输出与实际 raw tiling 均与 stock 一致；未通过则停止后续实验并报告。
3. `m_and_r`：8 次 M-only / ratio-only。
4. `joint`：4 次联合策略。将前面 private base 与这 12 次结果以只读索引汇总，运行完整主矩阵分析。
5. `automatic_a`：8 次预注册的受控 batch 策略。
6. `automatic_b`：剩余 4 次；随后运行独立次级分析。

每批调用既有 `controller.py`，按 summary 中 `case_files` 逐个传入 `--case-file`，不要改 case JSON。每批使用新 attempt，不覆盖失败或成功记录。示例第一批：

```sh
python3 /home/fault28-causal-20261029/controller.py \
  --card 5 --attempt stock_bridge_a2 --probe replay.py \
  --case-file setup/stock_bridge/stock_bridge_b1_r0.json \
  --case-file setup/stock_bridge/stock_bridge_b1_r1.json \
  --case-file setup/stock_bridge/stock_bridge_b8_r0.json \
  --case-file setup/stock_bridge/stock_bridge_b8_r1.json \
  --capture native-logs --max-seconds 1800 --case-seconds 300 --max-log-mib 512 \
  --allow-idle-container-ID a8eeda3a5db34389fbcc83fdd8cabc99d687ac1db5aa4011a0586f2a9f0ddc4c \
  --allow-idle-container-ID db21bd452c30c64dfd61152b6cf0a98bebfcea468a0f1c8a18452d2cb0e99de1
```

两个 allowlist 是此前逐一核查的空闲后台容器完整 ID，不是允许忽略活跃 NPU 进程。若容器身份改变，先只读核查，不扩大到其他外部容器、不终止它们。`stock_bridge_a1` 已留下门禁失败回执，不能重用该 attempt。

## 分析与结论

各批原始 evidence 不搬移或改写。跨批分析需要按 case ID 创建只读索引，日志、result 与其原生统计文件均指向原始记录，并保留来源/文件 SHA。不要拼接或改写 dump 节点来制造首差。

独立因果分析见 `analysis/README.md`。源码路径为 `source/ops-transformer-v9.1.0`，POD schema 为 `build/evidence_a1/fa_pod_schema.json`，vendor manifest 为 `vendor_stage_a1/vendor_manifest.json`。此前只读工具目录为 `/home/fault28-native-tiling-20261029/analysis/tools`，需按分析器要求核对固定 SHA。

msprobe 首差分析只读 B/F 原生统计及采集回执，不读期望 profile/命中日志来指定位置。主 O 与 Softmax 等辅助返回分开报告；null 为不可观测；跨 batch 形状不同不能直接当作目标请求数值差异。用相同 batch 的 B/F 统计检验输出首差，精确私有输出则支撑完整序列因果分析。

主矩阵的实际 M-only/ratio-only 可能不合法或没有输出差异，应据实保留结果，不更换输入或删掉单元格以追求通过。实际 raw 与请求策略不符、GetTiling/kernel 失败、bridge 不同或重复不稳定时，不能宣称根因链闭合。

通过后再更新总 README 与发布材料。发布从干净的已发布 `297e03d1a03b08b31ec82b26b1940d89d9e49478` 基线或远端后继建立 `codex/` 分支，只复制本次审查后的实现与统计证据，不推送旧可写工作区的未发布历史。私有完整 Q/K/V 和输出 `.bin` 不提交 GitHub。完成后停用本线程等待自动化 `28-5`。
