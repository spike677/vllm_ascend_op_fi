# 黄区交付索引 2026-11-26 r3：D73自主收口交接

分支 `codex/yellow-handoff-20261126`。**全部下列文件固定于内容提交 `208485146182e6efd6e66db6532e9f4674c7b7a8`，SHA按Git blob完整字节计算。** 分支HEAD的交接单只会追加交付确认尾注；交接正文SHA核对对象是本内容提交，不是带尾注的HEAD副本，避免索引与交接单互相包含SHA形成循环。

新D73：Qwen与V4由黄区自行决策、执行、验证、登记冻结、合入，蓝区上传后待命，只收升级单。当前r8冻结包不改；本次没有新wheel、没有适配器ZIP、没有NPU任务。所有旧文档历史分析不再构成蓝区主动队列，新权责以本次交接正文和D73为准。

- [交接正文](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/YELLOW-HANDOFF-20261126.md)
- [D65–D73](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/decision-log/20261126.md)
- manifest：`releases/yellow-handoff-20261126/manifest.json`，SHA256 `bdfb6a95e85e93af4a1f90965add1af7d1d5e9d2df3ada850efab60ae1a086d7`。
- SHA256SUMS：`releases/yellow-handoff-20261126/SHA256SUMS`，SHA256 `70c8efae33f99cb59df0f5a783db02ee916169451904b98a9470072251cb91ca`。
- 文件数：47。每个文件完整SHA/大小见下表和manifest；需要工具依赖时取完整内容提交，不只拷单一脚本。Native/N2取既有独立ZIP，已有标准库依赖和自检说明。

## 取件与校验

```sh
git fetch origin codex/yellow-handoff-20261126
mkdir -p handoff-content
git archive 208485146182e6efd6e66db6532e9f4674c7b7a8 | tar -x -C handoff-content
# 将本索引同目录的yellow-handoff-20261126/SHA256SUMS放到新目录根。
# 以下路径应指向实际下载的SHA256SUMS；不从r8 SITE执行。
cd handoff-content
sha256sum -c /absolute/path/to/yellow-handoff-20261126/SHA256SUMS
```

Windows也优先git archive提取，避免core.autocrlf改写文件后误判SHA。黄区PC校验后传`/home/haoran/rc3_r72/incoming/`，保留r8旧SITE。**不得整体安装交接分支**：该分支含尚未发行的C2 Host修复，implementation身份不同。r8.1必须从e38c4113的SOURCE只改登记JSON并独立安装，实际implementation应保持003294ac9e7468b2069719a44f9931514a9ce504c6b12e6c39f64684878f0bee。

## 文件与SHA

### 交接/决策/状态

|文件（固定内容提交）|字节|SHA256|
|---|---:|---|
|[releases/YELLOW-HANDOFF-20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/YELLOW-HANDOFF-20261126.md)|15804|`0623cef3a3c5c7c3211ed7397fafaf794f483594318cca0b0aa4e0c88b2a7ec3`|
|[docs/decision-log/20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/decision-log/20261126.md)|2802|`fc0b9a5659a7f170d5a16ee325d971f5c1607ab3e8d1a5adb1f8a128a6e09fef`|
|[docs/worker_core_v1/yellow_handoff_progress_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/yellow_handoff_progress_20261126.md)|4473|`af227f8d5259d83df5453cbae7ee9c10b97594e0171dc7cb9a2ae1af3a0a0c79`|
|[docs/worker_core_v1/yellow_handoff_progress_20261126.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/yellow_handoff_progress_20261126.json)|17876|`1e74bcff924bb4ae3fa899af487ad0358907e744dab254c6755da683091b5dbb`|
|[docs/worker_core_v1/yellow_handoff_checks_20261126.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/yellow_handoff_checks_20261126.json)|1441|`9075beea12725f6866ff892ab9981f041ea77a3f24eeb535deff6c8b4393fc1c`|
|[docs/worker_core_v1/fault_library_coverage_20261124.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/fault_library_coverage_20261124.md)|12917|`a7266829ad76187131ae5d2cd6e9e5b2d7428ad23b7a753113e10eeeed154253`|
|[docs/worker_core_v1/fault_library_coverage_20261124.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/fault_library_coverage_20261124.json)|18680|`b1059d1fbaf17bd249f9ba85341c5473aee0ba3b67ddf9fbcbe69e62966113cc`|

### 已交付Native/N2工具，非r8冻结包

|文件（固定内容提交）|字节|SHA256|
|---|---:|---|
|[releases/NATIVE-N2-TOOLS-20261126-r1.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/NATIVE-N2-TOOLS-20261126-r1.md)|3403|`7c90282e6408d7ddcee908aed5dc42a08d482dba4637c12538779192a971bde8`|
|[releases/native-n2-tools-20261126-r1/native-n2-tools-20261126-r1.zip](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/native-n2-tools-20261126-r1/native-n2-tools-20261126-r1.zip)|24794|`7d1966a5acdbc03da5df58e693a3775858a7b38e04a2d7e2a629e83c56bd6e8b`|
|[releases/native-n2-tools-20261126-r1/manifest.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/native-n2-tools-20261126-r1/manifest.json)|3027|`91755fd9f2a9f7e2cbeae7c3c20ccf24389c04b7c7e2d2c27e2ca8f271ae1dcd`|
|[releases/native-n2-tools-20261126-r1/DELIVERY-SHA256SUMS](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/native-n2-tools-20261126-r1/DELIVERY-SHA256SUMS)|329|`dfed2fccd7f25b9dbb93509d16db20808a64cf2ef88df3cf3347ab1729707fb9`|
|[releases/native-n2-tools-20261126-r1/README.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/native-n2-tools-20261126-r1/README.md)|4619|`4c3beb6a1266cdec62c8c3f5d284179ac2ef01fafbbaf6a79d42dea3b9ec5215`|
|[releases/native-n2-tools-20261126-r1/cpu.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/releases/native-n2-tools-20261126-r1/cpu.json)|569|`42336a79a9d464b47ae6e7ff5d62d49b9adca440e77fa9b6a6eef9c2a8d67a8b`|
|[docs/worker_core_v1/native_build_metadata_d9aa_20261126/dcat-native-build.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/native_build_metadata_d9aa_20261126/dcat-native-build.json)|989|`c61f1a27a78e27d65cb714f786a5f1015d64754550460027097ad0b04a973ea9`|
|[docs/worker_core_v1/native_build_metadata_d9aa_20261126/README.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/native_build_metadata_d9aa_20261126/README.md)|3104|`dc13a2a4216426b72ebf296f0975c80a04c60d795c00ed44b202bf67db222569`|
|[tools/extract_fault33_launch_host.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/extract_fault33_launch_host.py)|3601|`93d0504f82cbbd7a247fe23803dbeef4c7706afcffc50938ff86341ffaa426df`|
|[tools/extract_fault33_existing.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/extract_fault33_existing.py)|6480|`2c84f71cae4ef00e6b98e94974e5bd57faa7f2edc5f144aa179a16ab4983606c`|
|[tools/review_launch_capture_v2.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/review_launch_capture_v2.py)|4013|`66ee176da78971091b2e2108d7f296123bc6abeb5b909862f99a0fb38268743f`|
|[docs/worker_core_v1/fault33_n2_launch_host_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/fault33_n2_launch_host_20261126.md)|4049|`3691fc34e174627e4efbe905c1bd81c002f8e8e904a813f2fd764297e8b49b28`|
|[docs/worker_core_v1/r8_native_configure_negative_boundary_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_native_configure_negative_boundary_20261126.md)|2974|`26964ef406e571312d40c900e2c57e9ce1c67f136be3bc545646eafba190728e`|

### r8.1自检骨架/身份草稿，非可加载登记

|文件（固定内容提交）|字节|SHA256|
|---|---:|---|
|[docs/worker_core_v1/r8_1_registration_draft_smoke_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_1_registration_draft_smoke_20261126.md)|6515|`e56362847901270d99124c62b90eb37826e1482cb1ea9698f2d24eed81ea2788`|
|[docs/worker_core_v1/r8_1_expected_ranks_draft_20261126.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_1_expected_ranks_draft_20261126.json)|40697|`0e3d8d22e65b7b30dd0906d061ade81a356a4dd7a22267087cd778f18217e468`|
|[tools/product_registration_pipeline.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/product_registration_pipeline.py)|5475|`6874207d4d09ee0fa3e393f17c75abb0fdd3b66b8e649b851907b093a26a41bb`|
|[tools/extract_product_record.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/extract_product_record.py)|3637|`1e6995256fd0a625582fdff1726ce654ff893e1953cd8f177d40ce39160b3080`|
|[tools/review_product_mode.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/review_product_mode.py)|15545|`536cf4166e18250e03f9c56144ca4a3c8fbc4b3271ef4c2bb4d8704f5d053b5e`|
|[tools/extract_registration_record.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/extract_registration_record.py)|9777|`f648e3ff1e9361bf9e23e77852715d647268e89fc1a2bac7b5d91be9df3ca2a2`|
|[tools/review_continuous_public.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/review_continuous_public.py)|20670|`0e972d41f303baa306e7dd81ebe577884e9401ebe0a219d971cc9ce67b33968a`|
|[tests/worker_extension/test_product_registration.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tests/worker_extension/test_product_registration.py)|6029|`2f3cd0e05e9738288426c480df22395970c67f3f3618f2ef42f688786d110e70`|
|[tests/worker_extension/test_product_registration_pipeline.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tests/worker_extension/test_product_registration_pipeline.py)|2901|`d2a90dc9e98af27c4df752c9a95297a7eb3495fff038fa28c1ac9e4b5801d8c3`|
|[tests/worker_extension/test_r81_identity_draft.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tests/worker_extension/test_r81_identity_draft.py)|1089|`a160bee3015915f6a84b8e8031da33f3b555ee8eff02c1216ee9774cdd699634`|

### r8.2开发线未冻结，黄区按需升级索取

|文件（固定内容提交）|字节|SHA256|
|---|---:|---|
|[tools/product_effect_probe.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/product_effect_probe.py)|16545|`86cc87dddeb0c1c0fe49ccf528fb71f3ba297781bb816df56db29094a33c6293`|
|[tools/product_effect_adapter.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/product_effect_adapter.py)|4848|`bc30eee67a99b3443d105310d55f9f526abe1bf78db352c515c699062e38785f`|
|[tools/effect_rounds.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tools/effect_rounds.py)|1547|`5a30b7f1d0c89a40158063c1ab3238268cf39b0bfc4d7fcb33f18a9d71515861`|
|[experiments/worker_core_v11_20261118/effect_schedule_probe.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/experiments/worker_core_v11_20261118/effect_schedule_probe.py)|3478|`20e016ef5f2e0ad895554c7d83673da4966df0fc65752942fe4c93f5c8836555`|
|[experiments/worker_core_v11_20261118/product_http_probe.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/experiments/worker_core_v11_20261118/product_http_probe.py)|26903|`6811514a4104ad343499837ad77a9f0d588fed7bb25c2375595e437ea972a20e`|
|[docs/worker_core_v1/r8_2_effect_round_protocol_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_2_effect_round_protocol_20261126.md)|5274|`f36782a89b2d7aa737a725b4d5b5898b755c75ba7dbffda1356152f852e03d4c`|
|[docs/worker_core_v1/r8_2_under4_top5_diagnosis_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_2_under4_top5_diagnosis_20261126.md)|3303|`f9852723a68e9145845173c37381fab99bfa413a1788273d14170d3134ebe587`|
|[docs/worker_core_v1/r8_2_a2_completion_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_2_a2_completion_20261126.md)|5087|`e6e99c6184c5f3e95c83aa61aade6909b3654d9427192fea7be4d423ac186622`|
|[docs/worker_core_v1/r8_2_a2_completed_review_20261126.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_2_a2_completed_review_20261126.json)|44405|`d06573f332d4c4f8474022891c1198d72654c87b32b014aecd3402aa932c0f44`|
|[docs/worker_core_v1/r8_2_batch_no_injection_20261126/README.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_2_batch_no_injection_20261126/README.md)|4126|`baac36c6e606fc180a23bb5a453c800202daaa58349349d9268415af18159477`|
|[src/demoncat/runtime/vllm_ascend/worker_extension/product_host.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/src/demoncat/runtime/vllm_ascend/worker_extension/product_host.py)|18063|`d1d062b5171eeb65ca5f1552458ec92f08c2195bf27e3ea5605688d9d0a184ed`|
|[tests/worker_extension/test_product_attach_refusal_noop.py](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/tests/worker_extension/test_product_attach_refusal_noop.py)|8175|`d715fecf0cc86adfa0b4ab3231d0bdfea16f922f1879a69e1a076a84642da325`|
|[docs/worker_core_v1/r8_2_owner_refusal_candidate_cpu_20261126.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_2_owner_refusal_candidate_cpu_20261126.json)|1253|`33fd64bbe10e96f867cead5ebec8c20b65697297c64d517f78f3c05fe6de4565`|
|[docs/worker_core_v1/r8_owner_rollback_a2_20261126.md](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_owner_rollback_a2_20261126.md)|7707|`c7291569e8a9ab67eafa9c22f3505b3f098f1e8bff1df8f94cdb2f0663f5ce18`|
|[docs/worker_core_v1/r8_owner_rollback_a2_20261126/runtime-delta.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_owner_rollback_a2_20261126/runtime-delta.json)|623|`937a798d1353fcb2b835e4c0626dd28dd2749978b5deaaebc536d0d9c7ad80a8`|
|[docs/worker_core_v1/r8_owner_rollback_a2_20261126/r8owner1126a2.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_owner_rollback_a2_20261126/r8owner1126a2.json)|4348|`58df8ae8b7e05f7d15b38be4db6382fe51f42dce96193c863d4f53675c019c34`|
|[docs/worker_core_v1/r8_owner_rollback_a2_20261126/r82owner1126a1.json](https://github.com/spike677/vllm_ascend_op_fi/blob/208485146182e6efd6e66db6532e9f4674c7b7a8/docs/worker_core_v1/r8_owner_rollback_a2_20261126/r82owner1126a1.json)|4553|`10094365b21397151ce690119a380086f22e9fa0d642988598ef9299b094fd1e`|

## 使用边界

- 7项expected_ranks草稿含14份实际身份，但negative_controls仍null，不能直接加载。黄区按新D73完成负例、自检、只改登记表提交与自行冻结，无需蓝区复审；来源/完整性门禁不变。
- 外部三项仍用许可，不在公共免许可登记表；over4补齐CLI和产品原件才由黄区决定登记。17差2由黄区按同一事件集原件判断并记Y-D决定。
- r8.2效果工具：token-ID表示修复及Batch按真实组成重试已有A2证据；冷启动无注入不稳定仍是已知限制，不因此放宽精确恢复。本次仅源码候选，黄区需要修复包时提交升级单。
- C2 Host候选：准确ALREADY_INSTALLED路径CPU43/43及A2对照已通过，未发包；不清理他人资源，不把未知错误吞成exit4。不要把它混入r8.1。
- N2 launch_host只处理已有六条原件的host证据，tiling不审定就null；不授予33资格、不代替ABI/布局和控制重算。解码/窗口二、新D包或Native重构建通过升级单请求。
- 登记自检13项通过；完整实现SHA已由冻结源码342个Python文件重新计算，与14份身份一致；41行统计由逐行JSON生成。远端提交可达性记录见yellow_handoff_checks_20261126.json。

本索引没有放行任何新组合，黄区负责按证据决定登记及闭环。里程碑或升级单≤4KB回传；原件留黄区。蓝区交付后待命。
