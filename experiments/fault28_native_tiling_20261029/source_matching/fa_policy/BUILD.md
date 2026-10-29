# 从固定源码重建 FA 私有实验包

本配方用于 Linux aarch64 的 **CPU 构建与私有安装**，不访问 NPU，不复制 SDK、模型权重或构建二进制到 Git。运行实验还须通过独立启动器的逐文件 SHA 验证、stock → OFF 兼容性验收，然后才能尝试 ON。所有构建、依赖和安装目录必须属于新建的任务目录，不能指向系统 SDK 或已有实验。

## 固定材料

| 材料 | 版本或指纹 |
|---|---|
| ops-transformer | 官方 `https://gitcode.com/cann/ops-transformer.git`，`v9.1.0`，commit `c77c0829f1f0f0b8f882db1d9b051a758b605472` |
| opbase | `https://gitcode.com/cann/opbase.git`，commit `cd48c600c4b37146f504217eab2bf3956f7fee8b` |
| ops-tensor | `https://gitcode.com/cann/ops-tensor.git`，commit `b24535e30a6f7a3909c53c8b09713a9c0d08e54b` |
| CANN | 9.1.0 / V100R001C11SPC001B243，OPP timestamp `20260730_231653901` |
| 实际构建镜像 | `sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9` |
| protobuf C++ archive | 25.1，SHA256 `9bd87b8280ef720d3240514f884e56a712f2218f0d693b48050c836028940a42` |
| abseil archive | 20230802.1，SHA256 `987ce98f02eefbaf930d6e38ab16aa05737234d7afbab2d5c4ea7adbe50c28ed` |
| nlohmann json.hpp | SHA256 `9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6` |

SDK 文件只记录指纹，见 a2 的 `dependency_manifest.json`；镜像和 SDK 不随本仓库提供。构建工具为镜像内 GCC 11 / CMake 3.22 / Python 3.11，使用 `_GLIBCXX_USE_CXX11_ABI=0`。本次辅助 Python 包精确版本在 `build_requirements.txt`，安装到任务的 `build_off/python_deps` 后通过 `PYTHONPATH` 使用，不安装到系统。Python protobuf 的版本与 C++ protobuf 源码版本是不同材料。

官方 `cmake/third_party/opbase.cmake:41–51` 在首次获取 opbase 后会删除该依赖副本的 `aicpu_common`、`pkg_inc/op_common/log`、`pkg_inc/op_common/op_host`、`pkg_inc/op_common/op_kernel`；本次依赖副本保留该官方处理结果。不要把这些删除误记成实验修改，也不要对 SDK 目录执行此处理。再次配置时把同一已固定的依赖副本挂回 `/opbase`、`/ops-tensor`；否则容器中原始绝对依赖路径不存在，会触发重新下载或配置失败。

## 准备与源码冻结

以下示例将 `TASK_ROOT` 设为全新任务目录。把本目录的 `apply_policy.py`、`fault28_fa_policy.h`、`scope_cpu_check.cpp` 放在 `$TASK_ROOT/source_matching/fa_policy_a2/`。通过官方 Git 获取固定 release 和上述固定依赖，禁止使用移动的 master 分支。官方 CMake 按 release 自身的下载和校验配置获取其他依赖；下载后核对上表 archive/header SHA，保留依赖清单。

```sh
TASK_ROOT=/home/fault28-native-tiling-20261029
SRC="$TASK_ROOT/source_matching/ops-transformer-fa-a2-v9.1.0"
BUILD_RECORDS="$TASK_ROOT/build_off/fa_policy_a2"
git clone https://gitcode.com/cann/ops-transformer.git "$SRC"
git -C "$SRC" checkout --detach c77c0829f1f0f0b8f882db1d9b051a758b605472
python3 "$TASK_ROOT/source_matching/fa_policy_a2/apply_policy.py" \
  --source "$SRC" --output "$BUILD_RECORDS"
```

`apply_policy.py` 拒绝非固定 commit、已有 tracked diff 或已应用补丁的源目录。它核对唯一源码锚点后修改两个 host `.cpp` 并增加一个 header，生成 `official_sources.patch` 和逐文件前后 SHA 清单。编译开始前冻结该清单；编译期间不再改源码。源码补丁不改 kernel。

本次 a2 的 header SHA 为 `242c97817e1e2578e627ef3c22b5b7a733e3279adec4254e0cb6783d9028cf4c`。a1 存在 rank-0 optional 缺省识别缺陷，仅为探索记录，不能用它替代 a2。这里使用原生 FA 的具体 optional-tensor 缺省规则，不把零维 shape 一概视为任意 API 的空张量。非空 shape、QKV dtype/shape、mask、所有精度属性仍受严格门槛约束。

## 实际 CPU 构建命令

实际容器限制为 4 CPU、16 GiB、512 PIDs、1800 秒上限，无 `--device`、无 `--privileged`。SDK 留在固定镜像中。宿主机 `/usr/bin/patch` 只读挂载用于官方依赖 patch 工具；若换用其他工具，重新记录工具指纹。下方 `DEPS` 指向已经按上述固定 commit 和官方后处理准备好的任务私有依赖目录。

```sh
IMAGE=sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9
DEPS="$TASK_ROOT/source_matching/ops-transformer-fa-v9.1.0/third_party"
timeout 1800 docker run --rm --cpus 4 --memory 16g --pids-limit 512 \
  --name fault28-cpu-build-fa-a2-20261029 \
  -v "$TASK_ROOT/source_matching:/src" \
  -v "$TASK_ROOT/build_off:/build" \
  -v /usr/bin/patch:/buildtools/patch:ro \
  -v "$DEPS/opbase:/opbase" -v "$DEPS/ops-tensor:/ops-tensor" \
  --entrypoint bash "$IMAGE" -lc '
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PYTHONPATH=/build/python_deps:${PYTHONPATH:-}
export PATH=/buildtools:${PATH}
export CMAKE_BUILD_PARALLEL_LEVEL=4
cd /src/ops-transformer-fa-a2-v9.1.0
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I/usr/local/Ascend/cann-9.1.0/include -x c++ -fsyntax-only \
  attention/flash_attention_score/op_host/fault28_fa_policy.h
g++ -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 \
  -I/src/fa_policy_a2 -I/usr/local/Ascend/cann-9.1.0/include \
  /src/fa_policy_a2/scope_cpu_check.cpp \
  -L/usr/local/Ascend/cann-9.1.0/lib64 -lregister \
  -o /build/fa_policy_a2/scope_cpu_check
/build/fa_policy_a2/scope_cpu_check > /build/fa_policy_a2/scope_cpu_check.json
cmake -S . -B build \
  -DENABLE_BUILD_PKG=ON \
  -DBUILD_OPEN_PROJECT=ON -DBUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG=ON \
  -DENABLE_BUILT_IN=OFF -DENABLE_OPS_HOST=ON -DENABLE_OPS_KERNEL=OFF \
  -DENABLE_AICPU=OFF -DENABLE_AICPU_KERNEL=OFF -DENABLE_TEST=OFF \
  -DENABLE_CCACHE=ON -DBUILD_OPS_RTY_KERNEL=OFF \
  -DVENDOR_NAME=fault28_fa -DASCEND_OP_NAME=flash_attention_score \
  -DASCEND_COMPUTE_UNIT=ascend910b \
  -DCUSTOM_ASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.1.0 \
  -DPython3_EXECUTABLE=/usr/bin/python3.11 -DHI_PYTHON=/usr/bin/python3.11 \
  -DASCEND_PYTHON_EXECUTABLE=/usr/bin/python3.11 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/src/ops-transformer-fa-a2-v9.1.0/build_out
cmake --build build --target package --parallel 4
'
```

必须检查 return code、容器移除状态、原始 build 日志，以及冻结源码 SHA 前后相同。单独 `cmake --build ... --target package` 返回 0 仍不代表生成 `.run`：缺少 `ENABLE_BUILD_PKG=ON` 时只会生成 CPack JSON/staging。正式包应位于 `build_out/cann-ops-transformer-fault28_fa_linux-aarch64.run`。若超时，记录失败并只停止本次命名的 CPU 容器，不能继续把残留产物当成通过。

## 私有安装与清单

安装只写新的 task-private stage，不用 `--force`，不挂载系统 SDK 为可写目录：

```sh
docker run --rm --network none --cpus 4 --memory 8g --pids-limit 512 \
  -v "$SRC:/src:ro" -v "$TASK_ROOT/build_off:/build" \
  --entrypoint bash "$IMAGE" -lc '
set -eo pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash /src/build_out/cann-ops-transformer-fault28_fa_linux-aarch64.run \
  --install-path=/build/fa_vendor_stage_a2 --quiet
'
python3 write_vendor_manifest.py \
  --task-root "$TASK_ROOT" \
  --vendor-root "$TASK_ROOT/build_off/fa_vendor_stage_a2/vendors/fault28_fa_transformer" \
  --source "$SRC" --source-manifest "$BUILD_RECORDS/policy_source_manifest.json" \
  --package "$SRC/build_out/cann-ops-transformer-fault28_fa_linux-aarch64.run" \
  --output "$TASK_ROOT/build_off/fa_vendor_stage_a2/vendor_manifest.json"
```

清单生成器只读构建产物，先验证固定源码指纹，再枚举 vendor 内所有文件，验证 symlink 不逃出 vendor，记录大小、SHA 与链接目标。清单必须位于 vendor subtree 外，避免把清单自身纳入循环。运行前使用独立 launcher 再验证所有文件和清单 SHA。

同一个最终 binary 的 `FAULT28_FA_MODE=off` 与 `on` 选择策略，禁止两种条件换库。先验证 stock 与 OFF 的 raw、key、workspace、QKV、attention output、logits 和原生 statistics 观察行为；OFF 目标 B8 还必须真的 `eligible=true`、`policy_applied=false`，guard reports 无失败。然后才允许 ON，且仍需实际 profile、query/key 覆盖和独立数值/统计验收。外部重新构建的 ELF 哈希可能受编译器、构建绝对路径和打包时间影响，应重新冻结自己的指纹并重新验收，不能借用本次二进制的通过记录。
