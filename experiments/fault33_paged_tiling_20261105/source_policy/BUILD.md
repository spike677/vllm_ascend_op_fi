# Fault 33 私有 host 策略构建

这是当前固定源码中的受控策略，不代表某个历史 stock 版本。仅在全新任务目录构建和安装；不能向 SDK 或故障 28 的 vendor 写入。根调度器负责实际运行。本文列出的 SDK 编译和 NPU 验收，必须以执行后生成的记录为准。

固定官方 ops-transformer v9.1.0，commit `c77c0829f1f0f0b8f882db1d9b051a758b605472`；镜像固定为 `sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9`。从 `/home/fault28-native-tiling-20261029` 只读获取源码和依赖，复制到 `/home/fault33-paged-tiling-20261105` 的新目录。build_cpu 不挂载旧目录，也不传 NPU device，不启用 privileged。本次CPU构建与续跑安装均使用4 CPU、16 GiB、512 PID、无网络容器，构建上限1800秒、安装上限600秒。

当前a4使用独立 `source_policy_a4`，保持远端a3的source、vendor、原source_policy和拒绝记录不变。由根执行：

```sh
python3 -B /home/fault33-paged-tiling-20261105/source_policy_a4/build_cpu.py --generation a4
python3 -B /home/fault33-paged-tiling-20261105/source_policy_a4/stage_cpu.py --generation a4
```

两步均拒绝覆盖旧 generation。a3 使用 `source_a3`、`dependencies_a3`、`build/python_deps_a3`、`build/evidence_a3`、`vendor_stage_a3`，不写a1/a2目录。证据包含补丁、前后源 SHA、source freeze、CPU selector 测试、真实 CMake compile argv、ABI schema/roundtrip、build log 和 completion。包名 `cann-ops-transformer-fault33_pa_linux-aarch64.run`，host-only `ASCEND_OP_NAME=incre_flash_attention`、vendor `fault33_pa`。私有根为 `vendor_stage_a3/vendors/fault33_pa_transformer`，清单 schema `fault33.pa.private_vendor.v1`。清单检查私有 config 注册 `IncreFlashAttention` 和 tiling so 存在；实际 loader 是否使用这个库，还须用运行时 maps 与 native NNOP 记录独立验证。

a4对应独立的 `source_a4`、`dependencies_a4`、`build/python_deps_a4`、`build/evidence_a4` 和 `vendor_stage_a4`。构建/decoder/stage工具均按自身POLICY目录读取，不覆盖a3共享输入。a4已将后文实证通过的无索引、禁用pip build isolation环境合入完整构建，并在配置前核实际pip26.2.1/setuptools84.2.1/wheel0.48.0及解析选项。正常stage直接绑定 `public_abi` 与同代成功CPU build/source-freeze gate。

a4完整CPU构建已实际 `PASS_CPU_BUILD`，正常私有安装已实际 `PASS_CPU_PRIVATE_STAGE`；两者rc0、容器已移除，freeze与cann-cmake内容未变。753项vendor指纹已冻结，最终manifest SHA为 `f48a82e89fdf7bbfafba5ac3e05334dbde01b422c55dc726f786152db5874cfe`。另行生成的 `stage_package_binding.json` 核实安装前后实际包、build receipt和manifest中的包SHA全部为 `b4c687e2a2e8c97b82402058eee9a00f34b4418cce724876b03045ecaa193c93`。

公开CPU证据位于 [build_a4](../../../artifacts/attention_tiling/fault33_paged_statistics_20261106/build_a4/README.md)，明确allowlist与逐文件原字节SHA见 [index.json](../../../artifacts/attention_tiling/fault33_paged_statistics_20261106/build_a4/index.json)。共21个原始JSON/patch、581958字节；复制前后远端size/SHA一致，manifest的ABI三文件、成功build、source freeze、源补丁清单与包链均复核。未复制任何ELF、so、run、wheel、完整源码树或完整build日志。该目录只记录CPU phase，设备B/F结论由独立实际acceptor记录。

a3桥接4次真实调用完成后，OFF的fixed_inputs、actual_lengths_all_equal均为true，native_scope为false，因此保留reject，F未开始。已定位错误内部状态断言：普通非TND、无rope、非KV-NZ的PA路径只设置 `inputLayout_`，`inputKvLayout_` 留在 `BSH_BSND` 默认值，PA后处理提前返回。a4把该单项改为这一控制流的精确默认枚举断言；public属性BNSD、rank4 KV shape及所有其它原生条件继续严格检查。新增 `native_checks` 分项布尔与 `native_values` 原始enum/尺寸日志，设备门槛以实际每项通过为准。a3已有数值/计划一致记录不替代eligibility门槛。

a4每次scope检查记录以下24个实名判据，聚合值为它们的AND。这里列出源码要求；实际通过值随host事件保存。

| native_checks键 | 固定分支要求 |
|---|---|
| soc_910b | SoC枚举为910B |
| page_attention | PA启用 |
| no_split_kv | 非FlashDecode |
| runtime_tiling | 非workspace-only下沉 |
| no_antiquant | 无伪量化 |
| no_quant | 无量化 |
| no_rope | 无rope |
| no_gqa_kv_nz | 非KV-NZ分支 |
| no_gqa_mtp | 非MTP分支 |
| no_sys_prefix | 无系统prefix |
| no_sliding | 无滑窗 |
| no_balance | 非balance分核 |
| query_layout_bnsd | inputLayout_为BNSD |
| kv_layout_ordinary_pa_default | 此普通PA路径inputKvLayout_保留BSH_BSND默认 |
| query_fp16 | Q为FP16 |
| kv_fp16 | KV为FP16 |
| output_fp16 | 输出为FP16 |
| inner_precise_0 | inner_precise为0 |
| query_seq_1 | Q序列长度为1 |
| query_heads_16 | Q heads为16 |
| kv_heads_8 | KV heads为8 |
| head_dim_128 | QK D为128 |
| head_dim_v_128 | V D为128 |
| block_size_128 | 固定BS128 |

同一检查点另记录12项原值：soc_version、input_layout、input_kv_layout、query_dtype、kv_dtype、output_dtype、head_dim、head_dim_v、query_seq、query_heads、kv_heads、inner_precise。FixedInputs独立检查公开BNSD属性、rank4物理KV、dtype、block table、scale、optional tensors，以及count为1或B且所有actual lengths均为16384。

a3初次构建返回2，原 `build/evidence_a3/build_completion.json` 保留 `FAIL_CPU_BUILD`。实际失败是SDK的 `generate_es_math_whl` 目标执行 `/usr/bin/python3.11 -m pip wheel . --no-deps` 时，PEP517隔离环境尝试从PyPI索引下载 `setuptools>=40.8.0`；无网络容器中DNS失败。此时独立ABI decoder、3000字节布局与72字段CPU往返已经成功，原源码freeze未变。

本次实际采用同一a3源码和已编译对象续跑，新增四个独立helper，不回写任何原有冻结脚本：

```sh
python3 -B /home/fault33-paged-tiling-20261105/source_policy/resume_cpu.py --generation a3 --attempt a1
python3 -B /home/fault33-paged-tiling-20261105/source_policy/resume_stage_cpu.py --generation a3 --attempt a1
```

续跑仅设置打包执行环境 `PIP_NO_INDEX=1 PIP_NO_BUILD_ISOLATION=0 PIP_DISABLE_PIP_VERSION_CHECK=1`。实际pip26.2.1解析记录确认 `build_isolation=0`、`no_index=1`；使用原私有setuptools84.2.1及固定镜像中的wheel0.48.0。先在独立目录用真实es_math源码完成一次离线wheel预验，再继续原CMake目标；主构建日志随后也记录 `Successfully built es_math`。算法、tiling选择和算子源码均未改动。

`build/evidence_a3/resume_a1/build_completion.json` 最终为 `PASS_CPU_BUILD_CONTINUATION_PACKAGE`、rc0、容器移除；`stage_completion.json` 为 `PASS_CPU_PRIVATE_STAGE`、rc0、容器移除，753项vendor文件已逐个指纹记录。新gate与原失败分开保存，原失败日志和成功ABI文件保持原字节。原freeze SHA为 `264c8138ebf322921975b2708169dbb2fb4e93f6ac00869cbcb8eeb54e6f0119`；新增四helper另由 `continuation_source_freeze.json` 逐文件冻结，并在构建和安装前后复核。

最终manifest仍位于 `vendor_stage_a3/vendor_manifest.json`，SHA `58c34ffae2aaa5cf12d612a00b19cc459ae0703dd0c5e94484327f9266dfb05f`。其中 `build_continuation` 绑定本次成功build gate、独立freeze及真实wheel预验；`public_abi` 绑定同代schema、compile receipt、selftest的相对路径与SHA，记录schema文件6832字节、raw_size3000、72字段、`layout_kind=public_host_serializer_projection`、`kernel_argument_raw_observed=false`。stage在schema字段范围/重复、源码与decoder ELF指纹、compile-entry generation和CPU往返结果均通过后创建最终manifest。

本地发布证据副本位于 [build_evidence_a3_resume_a1](build_evidence_a3_resume_a1/)，包含 [build completion](build_evidence_a3_resume_a1/build_completion.json)、[stage completion](build_evidence_a3_resume_a1/stage_completion.json)、[vendor manifest](build_evidence_a3_resume_a1/vendor_manifest.json)、[wheel预验](build_evidence_a3_resume_a1/wheel_preflight.json) 和ABI三文件。本文件是发布后的构建说明，未回写远端原冻结文档。此处记录CPU构建/安装结果；设备phase结论以另行生成的实际acceptor记录为准。

对四helper的审查已核实：超时清理仅接受本次cidfile内的完整ID，并复核固定镜像、名称和随机label；每次运行拒绝覆盖attempt/stage；原freeze和新增helper freeze在各阶段复核；最终manifest创建一次且保持私有路径。该审查未要求修改已执行的helper字节。

a1在离线容器的官方CMake `FetchContent_Populate` 步骤缺少cann-cmake而失败，失败日志保留。a2从 `/home/fault28-causal-20261029/source/ops-transformer-v9.1.0/build/_deps/cann-cmake-src` 只读核对git HEAD `7e5691ce110367662017d9d94114845a7d6a8d8a`（官方声明tag `master-026`）和tracked clean，再 `git clone --local --no-hardlinks` 到本轮的新dependencies。不会复制旧缓存的untracked文件。显式使用 `-DFETCHCONTENT_SOURCE_DIR_CANN-CMAKE:PATH=/deps/cann-cmake`，保持 `--network none`。逐文件SHA与canonical内容树SHA写入本轮cann-cmake清单，构建前后核对；不共享可写旧目录。若复现者提供另一路径，可显式传 `--cann-cmake-source`，仍强制同一个commit。

a2随后因未显式指定原生 `CANN_3RD_LIB_PATH` 而未命中opbase本地目录分支，配置失败，记录保留。a3按固定源码和完整cann-cmake接口审查建立 `offline_cmake_contract.json`：FetchContent仅cann-cmake/opbase/ops-tensor，另有makeself模块的动态声明；json/eigen/abseil/protobuf是不同的ExternalProject接口。a3固定 `CANN_3RD_LIB_PATH=/deps`、`CANN_3RD_PKG_PATH=/deps/pkg`，并显式绑定前三个FetchContent SOURCE_DIR。新source的ops-tensor sibling相对链接只指向本轮副本，命中不调用git fetch的第一原生分支。

JSON使用固定SHA的single header，由 `offline_json.cmake` 提供原生相同include路径合同和`nlohmann=ascend_nlohmann`宏的INTERFACE目标。这样原生json模块的TARGET guard直接返回，避开其本地头命中却仍依赖未定义`third_party_json`的分支。Eigen从旧已建causal任务的 `build/third_party/src/external_eigen` 独立复制并冻结完整内容，使用原生ExternalProject本地目录URL；abseil归档复制到原生实际检查的 `/deps/abseil-cpp`，protobuf归档保留 `/deps/protobuf`，两者SHA均强制核对。没有复制旧第三方已编译库。

打包需要的makeself两脚本来自固定镜像内SDK的 `tools/op_project_templates/op_project_tmpl/cmake/util/makeself`。独立CPU准备容器使用只读根文件系统、无网络、1 CPU、512 MiB、60秒上限，把两脚本复制到本轮 `source/third_party/makeself`，逐个cmp并记录源/目标SHA后才冻结。它们作为SDK打包工具披露，不声称来自gitcode tar归档。主构建仍无网络、无设备。

完整声明清单和实际输入预检在进入主构建前生成；配置成功后还检查真实CMakeCache的路径/开关，以及生成的ExternalProject URL元数据不能含远程获取地址，然后先编译CPU ABI decoder，最后构建package。未出现在启用依赖链的cann-cmake三方模块保留在inventory中；gtest由ENABLE_TEST=OFF排除，device/非open-project的ExternalProject分支不启用。

策略 scope 固定 B8、Q N16/KV N8、Q S1、D128、KV 实长16384、BS128、BNSD、FP16、inner_precise0、无 mask/量化/rope/prefix/padding、910B、非 FlashDecode。除 `off`、`inner1024`、`inner512` 外的 mode 不激活。调用必须设置下列环境，PID/TID 指向执行原生 tiling 的调用线程：

实际长度通过同源原生用法 `Tensor::GetShapeSize()` 和 `GetData<int64_t>()` 读取host metadata，要求count为1或B8，且每项均为16384。仅max等于16384而其他请求更短的mixed-length输入不进入固定scope。该读取不涉及QKV或输出payload。

```text
FAULT33_PA_MODE=off|inner1024|inner512
FAULT33_PA_SCOPE_TOKEN=<token>
FAULT33_PA_CALL_ID=<call-id>
FAULT33_PA_SCOPE_PID=<pid>
FAULT33_PA_SCOPE_TID=<native-tid>
FAULT33_PA_SCOPE_PHASE=decode
FAULT33_PA_CALLER_DETERMINISTIC=1
FAULT33_PA_BATCH=8
FAULT33_PA_LENGTH=16384
FAULT33_PA_BLOCK_SIZE=128
FAULT33_PA_LOG_PATH=<absolute-new-private-jsonl>
```

`CALLER_DETERMINISTIC` 只记录调用者声明，host 本身不能从这个环境变量证明运行时确定性已开启；需 Python 和原生日志另行佐证。

补丁只改一个 host cpp，并增加两个独立33 header。不读写 QKV 或输出 payload，不改 kernel。选择点位于 `CalcInnerSize` 的原生 PA 向下对齐之后、loop/tail/align/UB 之前；requested 必须小于当前计算序列、为BS128倍数且不大于原生值。后续 MM、softmax、workspace、key 和 blocks 都继续由原生代码生成，禁止修补已序列化字节。

`off` 保留原生选择。先将同输入同 BS 的 stock 与 OFF 的完整输出、key、blocks、raw size3000和可得 native SDK 字段核对，再用同一 SHA 的库运行1024/512。stock 当前没有公开raw打印，不能把 OFF raw 称为已直接观测的 stock raw；只能按实际可观测的 stock 字段和输出建立兼容桥接。

末端观察来自 `TilingIncreFlashAttentionAdapter` 内注册分派返回后的公开 `gert::TilingContext`。Typed getter 在原生 `SaveToBuffer/SetDataSize` 之后采集，末端比较全raw字节是否仍相同。workspace 从原生已有 `workSpaces[0]` 观察，不额外调用会改变状态的 `GetWorkspaceSizes`。独立 `ifa_abi_decoder` 使用同一固定公开 host 类型的 `SaveToBuffer`，逐字段0/全1差分导出偏移并做CPU往返；这叫 **public serializer ABI projection**，不声称host类是C++ POD，也不猜kernel POD偏移。schema只投影已列字段，原始3000字节完整保留。decoder输出 fields 应与 host typed 全部相等。

真实执行绑定须满足每个 enter/exit bracket 内唯一 IFA 调用、同 PID/TID、唯一成功 host capture，以及 native NNOP 的 op/type、selected key、launch key、blocks、tiling size 与 capture 一致。公开 host raw 不是设备 kernel argument raw；当前观察不能直接证明设备收到的完整参数字节，结果应保留这个界限。

当前命中的 key `10000000000200000` 在固定 key header 映射为 FP16 PA C1V2，进入 `IncreFlashAttentionAttenSplitBbn2s2Us2` 的 PA、非FD、BNSD 实例。这个 kernel 的 `SoftmaxFlashV2Compute` 根据运行时 `srcShape` 重新生成 softmax 子tiling，因此 host `softmaxFlashTilingData` 是真实 host 输出观察，不能称为直接读取的 kernel 最终 softmax 子tiling。源码可追踪的因果链是 `singleProcessSInnerSize` 决定 loop 和 token offset，在线 softmax 的 sum/max 跨这些循环更新，Vec2 按循环累积并在末轮归一化。
