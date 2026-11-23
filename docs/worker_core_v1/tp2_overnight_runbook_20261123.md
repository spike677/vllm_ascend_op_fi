# Qwen A3 TP2隔夜执行总单 v2（统一 r7.1）

交负责人转交；协调者统一接收结果，每次≤4KiB。今晚黄区不做TP1、DeepSeek、旧r6登记、免许可登记；蓝区 r8 开发不下发。各线只操作自己的卡、端口、实例和目录。尚未执行的检查不得填true。

## 1. 分工、安装、部署参数

|线/物理卡|依次执行|发布版本/条件|
|---|---|---|
|L1 / 0,1|18、22、39op，三项负例|r7.1|
|L2 / 2,3|39mr、28、17|r7.1；17必须spawn|
|L3 / 4,5|Batch|r7.1串行金丝雀；故障用户请求并发|
|L4 / 6,7|41、26fia、20|r7.1；实际CANN9.1和已审查库|
|L5 / 8,9|33只观测/控制；15|33等取证审查；15使用r7.1|
|L6 / 10,11；L7 / 12,13；L8 / 14,15|黄区自有量化、数值/算子、KV/通信|由黄区改造/验收，公共故障生效时互斥|

统一现役Qwen3.8 W8A8、A3 Ascend910_9382、v0.26.0rc1-a3、CANN9.1.0、TP2、eager/mp、同步调度、无prefix cache/MTP/KV connector、固定seed42。纯文本实例image/video上限0；17实例image上限1、video0，其他参数相同。每线独立API端口、HCCL master端口及端口段，错峰加载。原host启动器负责其已验证的端口参数，不能依赖不同实例共享默认端口。

```sh
export ASCEND_RT_VISIBLE_DEVICES=0,1  # 每线替换为自己一对卡，容器内逻辑rank仍0/1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$RELEASE" # r7.1完整发行目录，源码和wheel来自同一发行
sha256sum -c SHA256SUMS
python -m pip install --no-index --no-deps --target "$PREVIEW_SITE" wheels/*.whl
python "$SOURCE/tools/check_rc3_preview_install.py" --site "$PREVIEW_SITE" --release "$RELEASE"
```

具名Worker扩展沿用黄区`host/yellow_worker_ext.py`的公共binding包装；不得用蓝区实验bootstrap替换授权。安装输出/文件SHA留存。r7.1许可清单是source-review.json，SHA **77e30525212d2d1e8f46313452bacdd46c81ecb751ce2e052e4d72dabbf6f7e2**；核对 r7.1 交接单。

|机制/profile（建议直接同名）|FaultConfig编号、variant|max_model_len / max_num_batched_tokens / max_num_seqs|请求与效果|
|---|---|---|---|
|18.api_half|18, api_half|1024 / 1024 / 1|norm缩放，真实消费者|
|22.wrong_row|22, wrong_row|1024 / 1024 / 1|真实rotary/QK行变化|
|39.output_proxy|39, output_proxy|1024 / 1024 / 1|BF16 ND投影；新增Inf和已有Inf分别记录|
|39.matmul_range|39, matmul_range|1024 / 1024 / 1|完整K控制、错误累加范围、非目标区不变|
|28.fia.denominator_replay|28, fia.denominator_replay|1024 / 1024 / 1|分母重放torch变体，对应故障库28行；不是历史kernel复刻|
|17.spatial_padding|17, spatial_padding|1024 / 1024 / 1|真实视觉encoder；64×64图，必须spawn|
|quant.batch_shared_scale|quant, batch_shared_scale|1024 / 8192 / 8|实际decode中2～8请求；B1/prefill/混合批不注入|
|41.tail_column|41, tail_column|1024 / 1024 / 1|KV尾列，真实FIA读后恢复|
|26.fia_destination|26, fia_destination|1024 / 1024 / 1|真实FIA值写入活跃residual，消费者后恢复|
|20.buffer_substitute|20, buffer_substitute|1024 / 1024 / 1|真实输入内容覆盖，算子消费后恢复，对应故障库20行|
|15.alias_detach|15, alias_detach|1024 / 1024 / 1|仅r7.1；ND视图别名断开，无私有格式触发条件|
|33.paged_decode|33, paged_decode|4096 / 4096 / 1|仅观测：2304-token prompt、4token；A3未开放|

Batch的FaultConfig编号采用字符串`quant`，其他编号为整数。8在Qwen不适用，不伪造MoE调用。所有请求单chunk；不能为消除警告改用未验证调度策略。

## 2. 部署文件、binding和许可（可信宿主一次接好）

每个profile一个JSON，完整字段如下。canary.prompt：普通文本为固定字符串，Batch为2～8条不同固定字符串组成的有序列表；17用固定图文文本、image=null。

```json
{"schema":"demoncat.product.deployment.v1","profile":"18.api_half","fault":{"fault_id":18,"variant":"api_half","implementation_version":"1"},"model":"<实际模型绝对路径>","tp":2,"mode":"PRODUCT_ACCEPTANCE_ONLY","source":"YELLOW_DEDICATED_INSTANCE","source_manifest":"<SOURCE_REVIEW绝对路径>","source_manifest_sha256":"77e30525212d2d1e8f46313452bacdd46c81ecb751ce2e052e4d72dabbf6f7e2","native_library":null,"native_sha256":null,"canary":{"prompt":"Explain briefly why the sky appears blue.","image":null,"max_tokens":4},"evidence_directory":"<本线私有回执绝对目录>"}
```

N批三项的native_library填本容器现有只读库路径，native_sha256填 `d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274`。无需重建。attach还检查当前Worker映射的libascendcl/libruntime及安装元数据归属9.1；不能拿构建时版本替代，更不能使用9.2.1库。

以下函数均在具名可信Worker RPC内、Worker线程执行；authority绑定该Worker，operator由控制面认证。不要把认证回调改成公网调用者可提供的lambda。

```python
from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
    configure_deployment_acceptance, install_deployment_acceptance_permit,
    remove_deployment_acceptance_permit)
from demoncat.runtime.vllm_ascend.worker_extension.product_config import configure_product_deployments

authority = configure_deployment_acceptance(worker, source='YELLOW_DEDICATED_INSTANCE',
                                          authorize_operator=trusted_authorize_operator)
configure_product_deployments(worker, paths=profile_paths, authority=authority,
    operator=operator, canary_tokens=actual_processor_token_lists_by_profile)
# 每owner前装同一机制的许可；前一owner已清理且移除旧许可后再装下一份。
install_deployment_acceptance_permit(worker, authority=authority, operator=operator,
    mechanisms=[mechanism], attempt=unique_attempt, source_manifest_sha256=source_review_sha)
```

真实token列表由服务tokenizer/processor取得；17须包含实际图像展开tokens。用户不填层、rank、内部request ID或数值扰动参数。许可自动安装属于可信部署功能，不代表正式产品已登记为免许可。

### r7.1 bootstrap 的归属与完整顺序

**bootstrap 是黄区自己的可信 Worker RPC，即 r6 A 组已经使用的包装层；r7.1 没有发行官方 bootstrap RPC。** 不导入蓝区 `product_mode_probe.trusted_bootstrap`，不把 localhost 实验端点暴露给用户。黄区现有 RPC 方法名及鉴权主体沿用原部署；下面 `BOOTSTRAP_RPC` 是其真实方法名变量，不是假定包内存在的新方法。

Worker 包装层的 action→公开函数映射如下（每个 rank 的 Worker 线程执行）：

| action | options 完整字段 | 内部公开调用 |
|---|---|---|
| configure | paths: list[str], canary_tokens: dict[profile, token列表] | authority 不存在时 configure_deployment_acceptance(worker, source="YELLOW_DEDICATED_INSTANCE", authorize_operator=部署认证回调)，随后 configure_product_deployments(worker, authority=authority, operator=已认证主体, **options) |
| permit | mechanisms: list[str], attempt: str, source_manifest_sha256: str | install_deployment_acceptance_permit(worker, authority=authority, operator=已认证主体, **options) |
| remove_permit | 空对象 {} | remove_deployment_acceptance_permit(worker, authority=authority, operator=已认证主体) |

每次调用先校验部署控制面的身份/操作权限。operator 不能由未认证用户自报，认证回调不能来自 JSON。configure 不创建 Worker、不直接启用故障。配置文件必须在所有 Worker 内同路径只读可见，令牌映射来自当前服务 tokenizer/processor，不能复制下面的虚构 token ID。

Native profile 完整 JSON（41；26fia 和20按表替换 fault_id/variant/profile）：

```json
{"schema":"demoncat.product.deployment.v1","profile":"41.tail_column","fault":{"fault_id":41,"variant":"tail_column","implementation_version":"1"},"model":"<容器内实际模型路径>","tp":2,"mode":"PRODUCT_ACCEPTANCE_ONLY","source":"YELLOW_DEDICATED_INSTANCE","source_manifest":"<r7.1/source-review.json绝对路径>","source_manifest_sha256":"77e30525212d2d1e8f46313452bacdd46c81ecb751ce2e052e4d72dabbf6f7e2","native_library":"<只读库目录>/libdcat_postwrite.so","native_sha256":"d9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274","canary":{"prompt":"Explain briefly why the sky appears blue.","image":null,"max_tokens":4},"evidence_directory":"<本线私有回执目录>"}
```

```python
# 可信宿主；各变量由现有部署提供。所有 rank 返回成功才继续。
executor.collective_rpc(BOOTSTRAP_RPC, args=("configure", {
    "paths": profile_paths, "canary_tokens": actual_processor_token_lists_by_profile}))
for owner_index in range(2):
    executor.collective_rpc(BOOTSTRAP_RPC, args=("permit", {
        "mechanisms": [mechanism], "attempt": unique_attempts[owner_index],
        "source_manifest_sha256": "77e30525212d2d1e8f46313452bacdd46c81ecb751ce2e052e4d72dabbf6f7e2"}))
    # session daemon 用下面的 ProductClient 契约服务 CLI；每 owner 建新 client。
    client = ProductClient.from_deployment(executor, deployment_json,
        operator=authenticated_operator, run_canary=run_canary, drain=wait_requests_complete)
    candidates = client.capabilities()  # 内部做 census attach，不安装故障 binding
    # capabilities 已完成 census attach，不能再显式 client.attach()。
    client.enable()
    send_real_user_requests_and_wait()
    receipt = client.status()
    recovery = client.disable()
    client.detach()
    executor.collective_rpc(BOOTSTRAP_RPC, args=("remove_permit", {}))
```

异常走原 session 的 recover/清理，不绕过资源检查移除许可；不能创建新 owner 掩盖残留。若没有先调用 capabilities，则执行 client.attach()→client.enable()；两种入口择一，避免重复 attach。

**可执行性依据：** `ProductClient.capabilities()` 返回当前可信 profile 的候选范围，`fault_injection.product_contract` 定义产品契约，实际 enable 还须通过部署、调用、许可及恢复门禁。`fault_injection/capabilities.py` 是旧研究/结构目录，不能据此禁用已经接线的 Native profile，也不能据此开放候选。PRODUCT_CANDIDATE / product_qualified=false 不表示已登记；更不能把静态目录当运行时资格。

## 3. CLI四动作与两个owner

入口统一为黄区的 `python -m demoncat.cli`。以下语法按负责人回传的实际帮助文本，不增加rank/layer/value参数。SOCKET由可信宿主session daemon创建，每个能力使用绑定相应profile的session；CLI不创建daemon。

```sh
SOCKET=/absolute/path/to/this-line-session.sock
MECHANISM=18.api_half  # 按第1节选择；15需先做下述适配，33不启用
python -m demoncat.cli fault list --socket "$SOCKET"
python -m demoncat.cli fault enable --socket "$SOCKET" --fault-id "$MECHANISM"
python -m demoncat.cli fault status --socket "$SOCKET"
python -m demoncat.cli fault disable --socket "$SOCKET"
```

enable内部先attach再enable；list只是只读候选检查，不能替代attach。disable没有fault-id，处理当前session，执行drain→disable→unregister→detach。退出码0成功、4契约/参数拒绝、5服务端异常。两个owner逐次重复完整循环，daemon保留同组Worker。

**黄区接线增量：**当前CLI白名单只有原12项，未含15。使用r7.1验收15前，宿主把`15.alias_detach`加入显式候选映射并指向同名可信profile；不修改许可、硬件/TP、源码或范围门禁。不能仅换wheel就声称CLI15可用。33静态list即使列出候选也不意味着A3准入；实际范围由Worker返回，不能把静态1024覆盖33的4096限制。Batch的含义是当前DynamicQuant真实调用行共享尺度，不限定专家模块。

下列公开API是CLI应调用的契约：

```python
from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
client = ProductClient.from_deployment(executor, deployment_json, operator=operator,
                                      run_canary=run_canary, drain=wait_requests_complete)
client.capabilities()                # CLI list：只读，不授权
client.enable()                      # 前面的 capabilities 已 census attach
# 此时用户通过正常HTTP推理，无before/after RPC。
status = client.status()            # CLI status，自动发现真实请求
recovery = client.disable(); client.detach()  # CLI disable，先drain，再恢复
```

每能力两个owner，Worker/pid/thread/设备身份相同，owner不同。每owner三条不同文本：`Explain why rain falls.`、`Describe how a bicycle moves.`、`Summarize how forests help rivers.`，temperature=0、seed42、max_tokens=min_tokens=4、ignore_eos=true；另发5token范围负例。每请求至少prefill+3decode。不得因失败重启Worker后继续称同组两个owner。

HTTP文本示例（MODEL_ID为实际served-model-name）：

```sh
curl --fail-with-body "$BASE/v1/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"prompt\":\"Explain why rain falls.\",\"temperature\":0,\"seed\":42,\"max_tokens\":4,\"min_tokens\":4,\"ignore_eos\":true}"
```

17：服务金丝雀调用`fault_injection.vision_input.deterministic_image()`，image=null时不要求外部文件。像素规则(4*x,4*y,4*((x+y)%64))，64×64 RGB；回执记录像素SHA。用户请求通过`/v1/chat/completions`发送真实data-URL图片，至少两张不同64×64图，不能把用户图换成金丝雀图。部署金丝雀用`product_canary_image(spec)`；不恢复完整图片SHA硬门禁。

## 4. 金丝雀、Batch和结束检查

普通项：最多6轮内连续2轮完整hidden/logits全有限且逐位相同；关闭后新金丝雀与参考相同。Batch：固定有序prompt列表，**逐条发送、每条真实完成才发下一条**，正常连续3轮、关闭后2轮，真实调度均B1。不是仅比较生成文本。

```python
def run_canary(spec):
    prompts = spec['prompt'] if isinstance(spec['prompt'], list) else [spec['prompt']]
    for prompt in prompts:
        send_http_and_wait(prompt, max_tokens=4, min_tokens=4, ignore_eos=True)
```

Batch用户请求另外用线程池并发发送2、4、8条不同文本，每组全部完成后status；加一个独立B1请求及一个5token范围负例。并发HTTP数量不等于实际decode批次，必须由Worker回执证明B2～8调用；B1不得注入且原因DECODE_BATCH_SIZE_ONE。

每rank八计数齐全，eligible=attempted=applied=consumed=restored、failed=0、changed≤applied；无变化不等于失败。Batch按rank/frame去重。每owner恢复三项：区域字节恢复、hook/lease/inflight/owner零、新请求金丝雀精确一致。历史累计计数不必变零，活动资源必须零。性能只记录，不作门禁。前向900秒预算，客户端覆盖计算及传输；失败保留原回执，不调阈值，不强行改成功。

恢复失败由原session保留owner执行现有recover/unregister/detach；不能创建新owner掩盖。无法证明安全时隔离并退出实例，记录失败，新起实例只用于新用例。

## 5. 三项负例和正对照

1. **缺许可：**有效18/profile、正确binding/身份，唯独不装许可；enable应因授权拒绝且无注入/残留。随后装正确许可，完全相同配置应成功。ACTUAL_CALL_NOT_YET_OBSERVED是prepare前预期待观测，不影响负例纯度；source/TP/binding错误另列，不能混算。
2. **未知profile：**原有效profile做capabilities正对照，然后只换profile名执行可信RPC：

```python
executor.collective_rpc('demoncat_product_binding', args=('capabilities',
    {'operator': operator, 'profile': 'nonexistent-profile-negative'}))
```

应PRODUCT_PROFILE_UNKNOWN；不通过非法机制代替此项。
3. **未知机制99：**执行`python -m demoncat.cli fault enable --socket "$SOCKET" --fault-id 99`，应退出4并提示`Choose an explicit mechanism variant`，不能退出5。CLI参数本来是TEXT，shell传入的是字符串，不能宣称CLI已测试Python整数类型；若还需整数99的FaultConfig负例，单独用API构造并记录。相同session先用合法18.api_half作正对照。保存命令、退出码、原始错误与owner/session。

## 6. 审查、提取、回传

保存公开ProductClient原始返回，组织与同包product_http_probe一致的entry/scopes/requests/disable/cleanup/detach结构；不能用蓝区bootstrap替代黄区CLI。原件在黄区，身份从实际Worker观测，不补默认值。

```sh
cd "$SOURCE"
PYTHONPATH="$PREVIEW_SITE:$SOURCE" python tools/review_product_mode.py "$RECEIPT" --tp 2 --output "$REVIEW"
# N批41、26fia、20使用这一条：
PYTHONPATH="$PREVIEW_SITE:$SOURCE" python tools/review_product_mode.py "$RECEIPT" --tp 2 --a3-native --output "$REVIEW"
PYTHONPATH="$PREVIEW_SITE:$SOURCE" python tools/extract_product_record.py "$RECEIPT" \
  --runtime-identity "$RUNTIME_IDENTITY" > "$TEXT"
```

每项4页（2scope×2rank）原样回传，页≤4096字节；另附owner—回执文件名/字节/SHA表、三个负例及正对照。summary/完整张量留黄区；null不可补值。15另用r7.1独立机理审查；33未经r7.2候选不得注入。

协调者P0报安装/实例；P1逐项提取；P2报15/33；P3列公共13项和黄区自有的完整状态；P4由黄区自行合master并冒烟；P5手工指南。蓝区只审查与交付，不操作黄区master。未通过能力不合入注入代码。今晚旧A组r6登记被r7重跑替代。
