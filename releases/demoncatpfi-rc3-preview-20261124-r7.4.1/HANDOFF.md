# RC3 preview r7.4.1（2026-11-24）

标签 `demoncatpfi-rc3-preview-20261124-r7.4.1`；源码 `fd62c53828dca1a937c027ca25ae283e9369966e`。原r7.4与其他冻结标签不变。由负责人转交，不直接发送黄区。

许可 source_manifest_sha256 填 **source-review.json** 的SHA256：`9a50e186599fa129d652267e1681c42966cce6603282714ca20f86a78aea9b63`，不是release-manifest.json。

## 本轮交付和门禁

- B1：41行台账以原始描述映射能力；库行号不等于公共编号，不新增41个机制。
- B2：模块输入、实际DynamicQuant dispatch输入、图片Conv3d输出及只读输入/权重、既有采样入口说明；只增加已审核factory与批准的vision路由。
- B3：18.variance_over4（k=4、gain=.5）和18.variance_under4（k=.25、gain=2）固定命名档。两档分别完成同组Worker两个owner，不接收用户数值参数。是RMS均方误差效果代理，不是精确方差kernel。
- B5：失败状态保留双rank原始返回值；真实并发请求排空复核；15的真实base→消费者→hidden只读诊断。限制和未复现项见产品关闭说明及15风险报告。
- r7.4的v2 weight_scale/KV16层独立路径原样保留；既有两owner证据继续引用，不冒充本轮重测。
- CPU：1327项，1325通过，2项既有废弃研究入口带理由跳过；无失败或错误。
- A2组合：Ascend910B4、vLLM0.26.0 / vllm-ascend0.26.0rc1、CANN9.1.0、社区Qwen3.8 W8A8、TP2。每项真实HTTP、全rank匹配调用、消费者、字节恢复/资源清零/新请求金丝雀；A2不等于A3资格。
- 无设备安装：400个源码文件逐字节一致；network=none、无设备映射、未创建Worker、未初始化NPU。
- 相对r7.3.1：472个原有运行时文件中462个逐字节相同；10个批准的目录/路由/证据文件变化。原机制算法文件不变。完整清单见source-delta.json、shared-byte-comparison.json，不能笼统说所有共享路径未改。
- NPU实际源码清单与发布源码另见validation-source-comparison.json：实测包包含的运行时内容一致；CRLF/LF差异单独列出，不冒称测试字节完全相同。未进入本轮NPU源包的45个历史native_execution_strategy文件与r7.3.1逐字节一致，仅保留、不声明本轮重测。
- 三个wheel升级为demoncatpfi 0.4.0rc6.post6、worker-extension和fault39-core 0.3.0rc6.post6。

## 安装和黄区新增验证范围

```bash
sha256sum -c DELIVERY-SHA256SUMS
python -m pip install --no-index --no-deps wheels/*.whl
```

源码包、三个wheel、直接可读docs、factory示例及SHA清单都在发行目录。可信宿主接口、源码SHA注册、permit和真实HTTP入口契约不变，参考docs/worker_core_v1/external_site_matrix_20261124.md。黄区CLI需要显式加入新命名档/外部factory映射；静态目录列出不代表可执行。

黄区只需对本轮新增的两个18命名档、module-input、DynamicQuant dispatch-input、external vision按A3真实组合各做两个owner及负例；受共享路由影响的既有18.api_half与17.spatial_padding作回归。未验收项默认关闭；其他不经过新分支的旧机制沿用对应版本的已审查结果，不新增资格。v2按r7.4交接单单独验收，不能以普通v1注册持久区域。

```bash
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism 18 --variance18 over4 --status-diagnostics
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism 18 --variance18 under4 --status-diagnostics
python experiments/worker_core_v11_20261118/product_http_probe.py --mechanism 17 --external-vision --status-diagnostics
```

每次使用独立evidence目录；以上为受信任验收入口，不是给普通用户自行指定层/rank/数值的接口。输入示例须设置DCAT_EXTERNAL_SITE_EVIDENCE=/evidence，命令见挂载点能力表。TP2图片路径设置VLLM_WORKER_MULTIPROC_METHOD=spawn。

33的候选解码器与A3两个只读窗口执行单以独立补充件 `fault33-evidence-candidate-20261124-r1` 交付；不是注入适配发行。r7.4.1只保留完整采集器，33 A3仍关闭。每组每rank控制重算≥99%、字段边界、三形状含sparseMode3及源码身份缺一不可；任一不通过即停止。

15继续标为B档“别名断开语义（ND视图），不含私有格式触发条件”。empty原base可能保留未初始化旧字节，不保证故障请求有限或逐位稳定；没有填零掩盖。恢复表示原区域字节、资源和新的干净请求，不表示把已污染请求修回原值。

原始实机回执、张量和逐调用sidecar保留在蓝区；Git只含总结、身份与SHA。所有验证窗口结束后恢复派发能力，不修改Docker入口、驱动或CANN。B7的D包与B8/r8登记不因本包冻结而宣称完成，仍依赖各自开发和黄区回传。
