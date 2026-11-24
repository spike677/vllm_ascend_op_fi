# 33 A3 TP2取证执行单 r1

交负责人转给openchamber。对象：A3、Qwen3.8现役W8A8、TP2、v0.26.0rc1-a3、运行时CANN9.1.0的专用实例。基础包用r7.3.1；本取证补充件不是r7.4发行包，不启用33、不修改冻结包或模型权重。蓝区无法代执行A3取证。

目标依次为：实际库身份→完整launch→匹配源码/serializer→控制重算≥99%→两个owner注入验收。步骤1～5现在执行；步骤6须Codex收到材料并完成适配后执行，不能用A2配置越过。

**已收到的库身份可复用**：黄区9库SHA/3个build-id见[yellow_intake_20261124_fault33_library.md](yellow_intake_20261124_fault33_library.md)。不重采整套库表，只补rank关联、原件SHA及后续切换实例后是否仍相同。现有两卡TP2专用实例足够，无需九卡。当前重点为实际API参数和完整launch；不要把已列FFN/MoE tiling符号当成FIA serializer。

## 1. 准备与停止条件

在黄区既定A组专用设备上运行，不占B组设备。保持TP2、eager、关闭异步调度/prefix cache/MTP/KV connector、max_num_seqs=1，单请求单chunk；max_model_len与max_num_batched_tokens均至少覆盖2304输入和4输出。记录实际值，不改现役生产实例。使用同一tokenizer构造2304个真实token，decode4步。保持已有可信bootstrap和鉴权，不给33增加许可或放开产品门禁。

`WORK`为新建的本次证据目录；`CODE`为解压补充件后的目录。检查磁盘余量至少4GiB（每进程捕获上限1GiB，另留输出空间）。安装Python3、g++、binutils/readelf；编译只使用该实例CANN9.1头文件。原始库路径、地址、Q/K/V、raw文件留黄区，勿提交Git。

```bash
set -euo pipefail
export CODE=/实际路径/fault33-a3-evidence-20261124-r1
export WORK=/实际路径/fault33-evidence-新的轮次
test ! -e "$WORK"
mkdir -p "$WORK/capture" "$WORK/api"
cd "$CODE"
sha256sum -c SHA256SUMS
test -f "$ASCEND_HOME_PATH/include/acl/acl.h"
c++ --version > "$WORK/compiler.txt"
c++ -std=c++17 -shared -fPIC -I "$ASCEND_HOME_PATH/include" \
  experiments/worker_core_v11_20261118/paged33_launch_capture_v2.cpp \
  -ldl -pthread -o "$WORK/capture-v2.so"
sha256sum "$WORK/capture-v2.so" > "$WORK/capture-library.sha256"
readelf -n "$WORK/capture-v2.so" > "$WORK/capture-build-id.txt"
```

编译失败：保留编译器输出和头文件SHA，回传“需退补”；不得换9.2.1 SDK或A2的预编译库。

## 2. 在实际Worker接入只读取证RPC

沿用黄区现有可信Worker RPC子类，增加下列方法。将补充件CODE加入部署的PYTHONPATH。鉴权动作 `fault33_evidence` 仅授权本次操作员，不开放给公共HTTP。该方法不安装故障binding，不改变准入。

```python
def demoncat33_evidence(self, action, payload):
    from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import authority_for
    from experiments.worker_core_v11_20261118.fault33_capture_worker import observe, flush
    try:
        authority = authority_for(self)
        if authority is None or authority.authorize_operator(payload.get("operator"), "fault33_evidence") is not True:
            raise PermissionError("FAULT33_EVIDENCE_OPERATOR_REQUIRED")
        seq = payload["sequence"]
        if not isinstance(seq, str) or not seq or len(seq) > 128:
            raise ValueError("FAULT33_EVIDENCE_SEQUENCE_REQUIRED")
        if action == "observe":
            return observe(self, sequence=seq)
        if action in ("api_start", "api_stop"):
            import os
            if os.environ.get("DCAT_EXTERNAL_MSPROF") != "1" or os.environ.get("DCAT_CAPTURE_PRIVATE_QKV") != "0":
                raise ValueError("FAULT33_METADATA_ONLY_CONFIGURATION_REQUIRED")
            from experiments.worker_core_v11_20261118.paged33_native_probe import census
            return dict(census(self, "start" if action == "api_start" else "stop"), sequence=seq)
        if action == "flush":
            return flush(self, sequence=seq, library_path=payload["library_path"],
                         expected_sha256=payload["library_sha256"])
        raise ValueError("FAULT33_EVIDENCE_ACTION_UNKNOWN")
    except Exception as error:
        return {"rank": getattr(self, "rank", None), "sequence": payload.get("sequence"),
                "status": "EVIDENCE_REFUSED", "reason": str(error)}
```

启动专用实例前设置（使用部署原来的标准启动命令和可信子类，不另写一套模型服务）：

```bash
export DCAT33_CAPTURE_DIR="$WORK/capture"
export DCAT33_CAPTURE_ALL=1
export DCAT33_MAX_RECORDS=65536
export DCAT_EXTERNAL_MSPROF=1
export DCAT_CAPTURE_PRIVATE_QKV=0
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"
export LD_PRELOAD="$WORK/capture-v2.so${LD_PRELOAD:+:$LD_PRELOAD}"
```

确认上述变量确实传到两个Worker，预加载链中的其他库保持原顺序。先正常预热，不创建ARMED文件。原CLI能力33保持关闭，期间不执行fault enable。

API元数据观察器沿用已有0.26只读census，关闭其内置profiler和私有Q/K/V导出；只记录真实算子名、输入输出shape/dtype/stride、scalar参数及request关联。将本轮新的`WORK/api`目录挂载为该专用容器的`/evidence`，不要覆盖已有回执目录；该工具固定写入`/evidence/native-calls-rank0.json`和rank1文件。它不改参数或算子结果；stop会同步设备并移除本次包装。不得直接运行脚本main来替代已有服务。

## 3. 固定一次短窗口，抓取实际库和launch

先使用服务自己的tokenizer把文本重复/截断到恰好2304 token，保存最终input_ids和SHA，固定temperature=0、seed=42、max_tokens=4、ignore_eos=true；记录服务真实生成4 token，不能把max_tokens=4当作实际完成4步。通过服务已有标准completions请求接口发送这份token输入。prompt token数组、HTTP请求/结果原件只在黄区保存。

确保没有其他用户请求，等待预热完成，然后：

```python
rows = executor.collective_rpc("demoncat33_evidence", timeout=120,
    args=("api_start", {"operator": operator, "sequence": "33-r1-api-start-1"}))
assert len(rows) == 2 and {r["rank"] for r in rows} == {0, 1}
assert all(r["sequence"] == "33-r1-api-start-1" and "reason" not in r for r in rows)
```

```bash
touch "$WORK/capture/ARMED"
# 在这里通过现有服务入口发送上述唯一请求，等待HTTP完成和两rank控制边界排空。
```

仍在同一实例，由宿主原有executor调用（这是宿主Python，不是在shell中调用新的Worker）：

```python
import json
rows = executor.collective_rpc("demoncat33_evidence", timeout=120,
    args=("observe", {"operator": operator, "sequence": "33-r1-observe-1"}))
assert len(rows) == 2 and {r["rank"] for r in rows} == {0, 1}
assert all(r["sequence"] == "33-r1-observe-1" for r in rows)
# 使用宿主自己的本地落盘函数，将rows原样保存为WORK/worker-identity.json。
```

observe只读`/proc/self/maps`；对候选库核对映射设备/inode、SHA256、ELF build-id，附完整映射文件清单供发现未匹配名称的库。候选规则不能证明覆盖了所有operator库；若实际算子来自清单中别的库，补上该库身份。CANN运行时版本从现有可信inspect取值及读取来源；helper中的null不能用toolkit/version.info替换。分别给出toolkit版本文件、实际运行时版本，矛盾时保留两者。

请求和两rank控制边界均已完成后，撤下采集标志并显式封存：

```bash
rm -- "$WORK/capture/ARMED"
```

```python
api_rows = executor.collective_rpc("demoncat33_evidence", timeout=180,
    args=("api_stop", {"operator": operator, "sequence": "33-r1-api-stop-1"}))
assert len(api_rows) == 2 and {r["rank"] for r in api_rows} == {0, 1}
assert all(r.get("calls", 0) > 0 and r["sequence"] == "33-r1-api-stop-1" for r in api_rows)
# 原样保存API观察结果；核对包含真实FIA，scalar sparse_mode、mask元数据、布局和实际序列长度。
# library_path是Worker内实际映射路径，library_sha256来自本轮编译产物。
rows = executor.collective_rpc("demoncat33_evidence", timeout=120,
    args=("flush", {"operator": operator, "sequence": "33-r1-flush-1",
                    "library_path": capture_library_path, "library_sha256": capture_library_sha256}))
assert len(rows) == 2 and {r["rank"] for r in rows} == {0, 1}
assert all(r["sequence"] == "33-r1-flush-1" and r["status"] == "CAPTURE_SEALED" for r in rows)
# 原样保存WORK/flush.json，并核对rank到pid的映射与observe一致。
```

**必须在原Worker内flush**：另起Python加载so会是另一份空计数，不能使用。helper用RTLD_NOLOAD，未实际映射或SHA不匹配会拒绝。flush只封存文件与计数，不调用NPU；有ARMED或待写记录时拒绝。封存后本进程不能再开第二窗口；重采使用新目录和新专用实例。显式flush避免Worker用`_exit`退出时析构器不执行导致证据丢失，不要求靠停止服务才落盘。

## 4. 独立核对完整性，处理旧8192截断

按observe中的真实pid逐个执行：

```bash
python tools/review_launch_capture_v2.py "$WORK/capture" --pid 实际rank0的PID --output "$WORK/review-rank0.json"
python tools/review_launch_capture_v2.py "$WORK/capture" --pid 实际rank1的PID --output "$WORK/review-rank1.json"
```

两份均须complete=true、finalized=true、pending_writes=0；selected/written/原始文件数及serial完整对应；unknown、map_drop、oversize、capacity_drop、copy_error、open_error、short_write、close_error、launch_error均为0。非零就回传具体原因，不能忽略或改计数。超过预算：改为同样输入的更短实际观察窗口，或在Codex审查实际key后用DCAT33_CAPTURE_KEYS过滤，仍报告过滤范围；不沿用旧A2 key。缺少manifest、半截文件和没有记录均不通过。

捕获通过只证明**选定窗口记录完整**，不证明FIA归因、serializer正确或33资格。

## 5. 来源链与纯文本回传（本轮交付给Codex）

金丝雀可以并行补：同进程用相同输入和固定参数串行执行两次无注入请求，调用黄区已有的完整logits只读采集路径，逐rank逐生成位置比较全部元素字节；回传token_ids、logits shape/dtype、全部元素数、非有限值数、逐位相等数和原件SHA。先证明实际观察的是完整logits，HTTP的top-k logprobs不是替代品。已有完整logits采集路径若不能使用，标为“完整logits未观测”，不要报告假通过；该缺件和FIA控制重算分开。

不要全文粘贴raw或大JSON。每rank一段≤4KiB，条目太多时分多段并编号，不能静默截断：

```text
FAULT33_A3_EVIDENCE_V1 part=1/N
rank=0 sequence=33-r1-observe-1/33-r1-flush-1
stack=vllm...,vllm_ascend...,torch...,torch_npu...,cann_runtime... [读取来源]
hardware_name=实测原名 tp=2
input_ids_sha256=... prompt_tokens=2304 generated_tokens=4 sparseMode=实际值
operator_library=basename sha256=... build_id=... mapped_status=...
host_library=basename sha256=... build_id=... mapped_status=...
capture_source_sha256=... capture_library_sha256=...
capture_complete=true census={完整计数字段}
launch_keys=[实际key与次数] blocks=[实际核数与次数]
source_package=... source_commit=... source_manifest_sha256=...
serializer_file=... sha256=... fields=已知字段/UNKNOWN
sparse3_branch=源码文件:行号或UNKNOWN template=... S2=实测解码或UNKNOWN
artifacts=[worker-identity.json/flush.json/review-rank0.json:bytes:sha256,...]
missing=[逐项缺失内容] conclusion=需审查
```

所有SHA从现存文件计算。每个raw文件SHA已列在review JSON；原件留黄区。提供对应发行源码包/commit及构建清单，将**实际库SHA/build-id**关联到源码；仅CANN版本号或同名Git标签不够。重点材料：host tiling serializer定义、sparseMode=3的分支、key注册和kernel模板、核数和S2字段。没有匹配源码就注明UNKNOWN并请负责人向安装包供应方取得，不猜布局，不把scheMode当sparseMode。

## 6. 后续控制重算与正式注入（本轮不越过）

Codex根据步骤5材料锁定serializer/分组并提供增量适配与独立审查器后，黄区在同组合执行：固定真实Q/K/V、mask、合法key集合、cache页和序列长度；完整目标输出控制重算逐位一致≥99%，逐rank报告，不能仅比较统计量或挑选通过行。A3实际分组不同就按实测适配，不直接套A2参数。

控制通过后才提供33作用域许可，执行两个owner的baseline/fault/off与恢复检查：全rank覆盖、真实消费者、模型输出变化、区域字节恢复、资源清零和金丝雀。统一一页效果工具回传；CPU模拟、捕获完整、控制重算三者都不等价于这一步通过。材料不齐持续补证并升级负责人，33不按时限结案。

## 执行完成的交接界面

- openchamber：完成步骤1～5，回传两rank取证文本与缺件清单。
- Codex：审查来源链、补A3 serializer/分组适配和控制命令，再交步骤6。
- 负责人：协调无法从部署包获得的匹配源码；转交材料，不要求蓝区直接访问黄区。
