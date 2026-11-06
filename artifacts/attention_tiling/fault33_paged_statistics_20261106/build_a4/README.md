# Fault 33 a4 CPU构建证据

本目录只记录CPU源码构建、私有安装和元数据核验；设备B/F验收由对应实际acceptor另行记录。

- build为 `PASS_CPU_BUILD`，stage为 `PASS_CPU_PRIVATE_STAGE`，两者rc0、容器已移除，均未执行NPU。
- vendor含753项逐文件指纹；manifest SHA为 `f48a82e89fdf7bbfafba5ac3e05334dbde01b422c55dc726f786152db5874cfe`。
- public serializer ABI schema为6832字节，描述3000字节raw的72字段；schema、compile receipt和selftest均与manifest同代绑定。
- 安装前后实际包、成功build receipt和manifest内package SHA三者一致；本目录只包含包的指纹。
- `index.json` 是明确allowlist。21个原始JSON/patch逐文件核对远端原字节SHA，复制前后远端size/SHA相同；相关manifest、source-freeze、ABI和package链已再次核验。
- 已执行的a4策略代码与仓库 `experiments/fault33_paged_tiling_20261105/source_policy` 对应代码文件逐字节一致，清单见index。构建后的说明文档修订不回写远端冻结树。

未复制exe/ELF、so、run、wheel、完整源码、完整build日志或tensor payload。`official_sources.patch` 是明确允许的单文件host补丁。其余依赖/工具记录均是小型元数据JSON。

`public_abi` 是公开host `SaveToBuffer` 布局投影；`kernel_argument_raw_observed=false`。CPU通过不代表设备phase通过。
