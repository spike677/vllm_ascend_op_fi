"""Package one accepted all-eligible fault15 B/F attempt; no NPU/network.

The statistics ZIPs contain eight original JSON files each. Exact small values
and injection controls remain separate. Failed attempts and caches are excluded.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import zipfile


STATS = ('config.json', 'receipt.json') + tuple(
    f'observation/step{step}/rank0/{kind}.json' for step in (0, 1) for kind in ('dump', 'stack', 'construct'))
ACCEPTANCE = 'PASS_ALL_ELIGIBLE_MECHANISM_AND_STATISTICS'
ARTIFACT_NAME = 'alias_detach_allhit_statistics_bf_20261028'
REPORTS = ('acceptance.json', 'acceptance.native.json', 'acceptance.joint.json',
           'baseline_gate.json', 'baseline_gate.native.json', 'baseline_gate.joint.json')
CONTROLLER = ('source_manifest.json', 'completion.json', 'before.json',
              'resource_policy.json', 'resource_checks.jsonl',
              'B.controller.json', 'F.controller.json', 'B.log', 'F.log')
SOURCES = {'alias_probe.py', 'alias_injector.py', 'analyze_alias.py',
           'build_reference.py', 'controller.py', 'env.sh',
           'setup/logical_reference.json', 'setup/acceptance_contract.json',
           'tools/post_resource_check.py'}
FORBIDDEN = {'.pt', '.bin', '.npy', '.npz', '.pkl', '.safetensors'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_bytes())


def relative(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and ':' not in name and '\\' not in name,
            'Unsafe relative path: ' + name)
    return Path(*p.parts)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def resource_gate(controller_root, source_root, after_path):
    """Bind the later read-only cleanup check to this run and its frozen checker."""
    completion = load(controller_root / 'completion.json')
    sources = load(controller_root / 'source_manifest.json')
    after = load(after_path)
    require(set(sources) == SOURCES, 'Frozen source manifest must contain the nine agreed files')
    for name, expected in sources.items():
        require(re.fullmatch('[0-9a-f]{64}', expected) and
                digest((source_root / relative(name)).read_bytes()) == expected, 'Frozen source mismatch: ' + name)
    require(completion['status'] == 'PASS' and not completion['cleanup_errors'] and
            completion.get('owned_container_removed') is True and
            completion.get('device_lock_acquired') is True and completion.get('device_lock_released') is True and
            completion.get('external_container_mutations') is False, 'Controller completion/cleanup is incomplete')
    owned = completion.get('owned_container_id')
    require(isinstance(owned, str) and re.fullmatch('[0-9a-f]{64}', owned), 'Completion owned container ID missing/invalid')
    require(after.get('status') in ('PASS_RUN_AND_CLEANUP', 'PASS_RUN_AND_CLEANUP_CURRENT_RESOURCES_CHANGED') and
            after.get('owned_absent') is True and after.get('owned_container_id') == owned,
            'Post-run owned-container cleanup/identity was not verified')
    for field, path in (('completion_sha256', controller_root / 'completion.json'),
                        ('source_manifest_sha256', controller_root / 'source_manifest.json'),
                        ('checker_sha256', source_root / 'tools/post_resource_check.py')):
        require(after.get(field) == digest(path.read_bytes()), 'Post-run provenance mismatch: ' + field)
    checked, end = after['checked_epoch'], completion['end_epoch']
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (checked, end)) and checked >= end,
            'Resource record predates experiment completion or has an invalid time')
    absence = after['absence_checks']
    require(absence['docker_ps_all_rc'] == 0 and absence['inspect_rc'] == 1 and
            absence['inspect_stderr'].strip() in ('Error: No such object: ' + owned,
                'Error: No such container: ' + owned, 'Error response from daemon: No such container: ' + owned),
            'Post-run exact-container absence evidence differs')
    policy = load(controller_root / 'resource_policy.json')
    require(after['idle_review_source'] == policy['review_source'] and
            completion['idle_allowlist_ids'] == policy['idle_allowlist_ids'], 'Resource policy binding differs')
    current = after['current_resources']
    require(current['selected_card'] == completion['card'] and current['idle_allowlist_ids'] == policy['idle_allowlist_ids'] and
            type(current['eligible']) is bool, 'Post-run resource snapshot belongs to another policy/device')
    require(after['status'] == ('PASS_RUN_AND_CLEANUP' if current['eligible'] else
                               'PASS_RUN_AND_CLEANUP_CURRENT_RESOURCES_CHANGED'), 'Post-run status does not match later resource snapshot')
    checks = [json.loads(line) for line in (controller_root / 'resource_checks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    require(checks and checks[0] == load(controller_root / 'before.json'), 'Initial resource check is not the retained before snapshot')
    require(all(row['eligible'] is True and row['selected_card'] == completion['card'] and
                row['idle_allowlist_ids'] == policy['idle_allowlist_ids'] and
                completion['start_epoch'] <= row['epoch'] <= end for row in checks),
            'Resource checks contain a rejected or unrelated observation')
    return sources


def verify(root):
    manifest = load(root / 'SHA256.json')
    files = {p.relative_to(root).as_posix(): p for p in root.rglob('*') if p.is_file()}
    require(set(files) == set(manifest['files']) | {'SHA256.json'}, 'Missing or unlisted package files')
    for name, entry in manifest['files'].items():
        path = root / relative(name)
        require(path.suffix not in FORBIDDEN, 'Forbidden binary payload')
        data = path.read_bytes()
        require(len(data) == entry['bytes'] and digest(data) == entry['sha256'], 'Checksum mismatch: ' + name)
    mapping = load(root / 'source_mapping.json')
    for name, item in mapping['original_files'].items():
        require(digest((root / relative(name)).read_bytes()) == item['source_sha256'], 'Original bytes changed')
    for arm in ('B', 'F'):
        with zipfile.ZipFile(root / (arm + '_msprobe_statistics.zip')) as archive:
            require(archive.namelist() == list(STATS), 'Statistics archive must contain exactly eight JSON files')
            for name in STATS:
                require(archive.read(name) == (root / 'statistics' / arm / relative(name)).read_bytes(),
                        'Statistics archive changed original bytes')
    require(load(root / 'reports/acceptance.json')['status'] == ACCEPTANCE, 'Acceptance is not all-eligible PASS')
    resource_gate(root / 'reports/controller', root / 'source', root / 'reports/controller/after.json')
    return {'status': 'PASS_PACKAGE_INTEGRITY', 'files': len(files),
            'bytes': sum(p.stat().st_size for p in files.values()),
            'statistics_json_files_per_arm': 8, 'tensor_binary_files': 0,
            'scope': 'Packaging integrity only; no new NPU execution or reclassification'}


def package(attempt, output, resource_after, warning_note):
    require(attempt.is_dir(), 'Attempt directory does not exist')
    require(output.name == ARTIFACT_NAME, 'Use the new all-eligible artifact name; do not replace the single-hit delivery')
    require(not output.exists(), 'Output must be a new directory')
    completion = load(attempt / 'completion.json')
    acceptance = load(attempt / 'acceptance.json')
    require(completion['status'] == 'PASS' and not completion.get('cleanup_errors'), 'Controller completion must be PASS')
    require(completion.get('owned_container_removed') is True and completion.get('device_lock_released') is True,
            'Owned experiment cleanup is incomplete')
    require(acceptance['status'] == ACCEPTANCE and not acceptance.get('errors'),
            'Independent acceptance must be ' + ACCEPTANCE)
    require(load(attempt / 'baseline_gate.json')['status'] == 'PASS_BASELINE', 'Healthy baseline gate did not pass')
    stages = completion['stages']
    require([s['arm'] for s in stages] == ['B', 'F'] and all(s['rc'] == 0 and s['independent_acceptance'] for s in stages),
            'Controller did not complete B then F')
    resource_after = resource_after or attempt / 'after.json'
    require(resource_after.is_file(), 'Provide a post-run resource JSON with --resource-after (or attempt/after.json)')
    require(resource_after.resolve() != (attempt / 'before.json').resolve(), 'Post-run resource record cannot be before.json')
    sources = resource_gate(attempt, attempt / 'source', resource_after)
    require(completion['owned_container_id'] == load(attempt / 'container_identity.json')['id'],
            'Completion does not identify the actual recorded experiment container')
    require(acceptance['analyzer_sha256'] == sources['analyze_alias.py'], 'Accepted analyzer does not match frozen source')
    planned, mapping = {}, {}

    def original(source, destination, origin):
        require(destination not in planned, 'Duplicate destination')
        relative(destination)
        require(source.is_file() and source.suffix not in FORBIDDEN, 'Missing or forbidden selected source: ' + str(source))
        data = source.read_bytes()
        data.decode('utf-8')
        if source.suffix == '.json':
            json.loads(data)
        planned[destination] = data
        mapping[destination] = {'source_relative_path': origin, 'source_bytes': len(data), 'source_sha256': digest(data)}

    runtimes, node_counts = {}, {}
    for arm in ('B', 'F'):
        arm_root = attempt / arm
        require(not any(p.is_file() and p.suffix in FORBIDDEN for p in arm_root.rglob('*')), 'Unexpected tensor binary in arm')
        config, receipt = load(arm_root / 'config.json'), load(arm_root / 'receipt.json')
        runtime, control = load(arm_root / 'runtime.json'), load(arm_root / 'control.json')
        runtimes[arm] = runtime
        node_counts[arm] = receipt['dump_nodes']
        require(config['task'] == 'statistics' and config['level'] == 'mix' and
                config['statistics']['summary_mode'] == 'statistics' and not config['statistics'].get('tensor_list'),
                'Not the agreed statistics/mix capture')
        require(config['step'] == [0, 1] and receipt['steps'] == [0, 1] and
                runtime['completed_forwards'] == [0, 1] and len(runtime['cases']) == 6 and len(runtime['guards']) == 4,
                'Both forwards, six matched invocations and four guards must be complete')
        require(receipt['status'] == runtime['status'] == 'completed' and receipt['capture_completed'] and
                receipt['capture_requested'] and receipt['exit_code'] == 0 and not receipt['full_tensor_files'], 'Incomplete arm capture')
        require(control['arm'] == arm and control['hits'] == 6 * int(arm == 'F') and
                control['selected_original_calls'] == 6 and control['unmatched_calls'] == control['unmatched_original_calls'] == 4
                and not control['errors'], 'Wrong all-eligible arm counts')
        for name in ('config', 'runtime', 'control'):
            require(receipt[name + '_sha256'] == digest((arm_root / (name + '.json')).read_bytes()), 'Receipt hash mismatch')
        native = {item['path'].replace('\\', '/'): item for item in receipt['native_files']}
        require(set(native) == set(STATS[2:]), 'Unexpected native file set')
        for name in STATS[2:]:
            payload = (arm_root / relative(name)).read_bytes()
            require(digest(payload) == native[name]['sha256'] and len(payload) == native[name]['bytes'], 'Native receipt mismatch')
        for name, expected in runtime['source_sha256'].items():
            require(sources[name] == expected, 'Runtime source differs from frozen source')
        require(runtime['reference_sha256'] == sources['setup/logical_reference.json'], 'Logical reference source mismatch')
        for name in STATS:
            original(arm_root / relative(name), f'statistics/{arm}/{name}', f'{arm}/{name}')
        for name in ('runtime.json', 'control.json'):
            original(arm_root / name, f'runtime/{arm}/{name}', f'{arm}/{name}')
    require(runtimes['B']['source_sha256'] == runtimes['F']['source_sha256'] and
            runtimes['B']['versions'] == runtimes['F']['versions'], 'B/F sources or versions differ')
    for report in REPORTS:
        original(attempt / report, 'reports/' + report, report)
    for stem in ('acceptance', 'baseline_gate'):
        main = load(attempt / (stem + '.json'))
        for key in ('native', 'joint'):
            require(main['reports'][key]['sha256'] == digest((attempt / (stem + '.' + key + '.json')).read_bytes()),
                    'Native/joint report hash mismatch')
    for name in CONTROLLER:
        original(attempt / name, 'reports/controller/' + name, name)
    original(resource_after, 'reports/controller/after.json', 'provided_post_run_resource/' + resource_after.name)
    if warning_note:
        original(warning_note, 'reports/warning_note.md', 'provided_warning_note/' + warning_note.name)
    for name, expected in sources.items():
        source = attempt / 'source' / relative(name)
        require(source.suffix in {'.py', '.sh', '.json'} and digest(source.read_bytes()) == expected, 'Frozen source mismatch')
        original(source, 'source/' + name, 'source/' + name)
    for arm in ('B', 'F'):
        require(load(attempt / (arm + '.controller.json'))['rc'] == 0, 'Arm controller failure')
    reference = load(attempt / 'source/setup/logical_reference.json')
    require(reference['forward_count'] == 2 and reference['calls_per_forward'] == 3 and
            reference['matched_calls_per_arm'] == 6, 'Logical reference does not describe this matrix')
    target_count = reference['target_count']
    untouched_count = reference['untouched_count']
    versions = ', '.join(f'{key}={value}' for key, value in sorted(runtimes['B']['versions'].items()))
    native_rows = []
    for item in load(attempt / 'acceptance.native.json')['case_comparisons']:
        first = item['trace_comparison_including_observer']['first_aligned_trace_numerical_difference']
        pair = (first['B_name'] + ' / ' + first['F_name']) if first else '无可对齐 API 数值差异'
        native_rows.append('| ' + item['case_id'] + ' | ' + str(item['first_observable_numerical_stage']) +
                           ' | ' + pair + ' |')
    native_table = '\n'.join(native_rows)
    warning = ('已附[警告来源审查说明](reports/warning_note.md)；原始 B/F 日志保留所有警告。' if warning_note else
               '尚未附警告来源审查说明；请阅读原始 B/F 日志，不能将警告自动解释为注入故障或宣称运行无警告。')
    readme = f'''# 故障 15：每次满足条件均触发的 B/F 统计证据

状态：**持续条件命中的机制级 B/F 与统计证据验收通过**。本目录来自一次完整的正式运行 `{attempt.name}`；控制器状态为 PASS，独立验收为 {ACCEPTANCE}。打包没有重新执行 NPU，也没有更改原生 JSON 或验收门槛。旧单次版本及其证据保留，本目录是新的独立交付。

本机制正确编号为**故障 15**；此前单次实验曾误标为故障 27。历史单次源码与原始采集、完整证据包保留原路径、schema 编号和原始字节；公开说明文档另加编号纠错提示。本次新方案使用故障 15，不回写旧原始证据。

B 使用原始 NZ29 BF16 base 的真实 as_strided 视图。F 每次满足注册对象、实际格式/几何和指定入口条件时，执行 base→ND2 格式转换，再调用原始 as_strided；没有首命中后禁用或第二次匹配报错的策略。临时张量创建时等于该次原 base，写后视图与临时存储更新，原 base 保留旧值。

每组在同一进程和持续注入器下运行 **2 个真实 forward × 每个 base 连续 3 次调用**。两个注册 base 提前创建并强引用，forward 之间使用不同 base，同一 forward 的三次调用不重置 base。固定几何为 `[16,16]`、初值 arange(256)，view `[8,8]`、stride `[16,1]`、offset 17；三次写入分别为 `-64-j`、`-128-j`、`-192-j`，`j=0..63`，每次都由同一原 base 计算 `base+1`。

F 的 6 次符合条件调用全部触发，B 的故障命中数为 0；两组都有 6 次匹配的原生调用。每次写入，B 原 base 的 {target_count} 个目标元素相对自身前态全部改变，F 原 base 相对自身前态不变；每次消费者 B/F 差异恰好在这 {target_count} 个目标位置，{untouched_count} 个非目标位置不变。每 forward 另有不同调用位置的已注册 base 只读视图，以及同一选中入口的未注册 NZ base 写入/消费两个 guard；每组 4 次 guard 均原生透传，未注册 base 的写值为 `-16-j`。

第一个调用前，两端新 base 的精确值一致；本次原生语义模块首个被观测数值差异在写后读取原 base。第二、第三次调用前，B/F 已因前次写入具有不同值；已观察到的 ReadBaseBefore 差异属于继承传播，不能冒充新的更早故障点。逐次命中验收依靠每次实际新视图的存储关系及各端自己的 pre/post 转移，不能仅凭累计 hits 或持续输出差异。按 step、invocation 语义模块和调用栈对齐的实际结果见 [native 报告](reports/acceptance.native.json)，精确状态与别名检查见 [joint 报告](reports/acceptance.joint.json)。

| case | 首个被观测数值差异的语义阶段 | 首个对齐 API 数值差异 B / F |
| --- | --- | --- |
{native_table}

call 0 的 API 首差异来自共同观察器写后读回，不能把该诊断格式转换本身当作故障发生位置。原生 dump 节点数：B step0/step1 = {node_counts['B']['0']}/{node_counts['B']['1']}，F = {node_counts['F']['0']}/{node_counts['F']['1']}。

此结果属于构造的最小机制实验，不是模型级调用链通过，也不证明历史 CANN 缺陷或生产事故已复现。统计值不能独立唯一证明别名根因；存储身份、真实写入与精确小规模值来自单独的共同运行观察器，见 [joint 报告](reports/acceptance.joint.json)。

## 数据分区与原始路径

- `statistics/B/`、`statistics/F/`：每组 config、receipt 和 step0–1/rank0 的 dump、stack、construct，共 8 个原始 JSON。配置为 task=statistics、level=mix、summary_mode=statistics，无 tensor_list 全张量采集。step 对应两个真实微基准 forward，不是模型 token 步数。
- `B_msprobe_statistics.zip`、`F_msprobe_statistics.zip`：各自仅打包上述 8 个 JSON，ZIP 内路径从 config.json/observation 开始；内容与解压目录逐字节相同。
- `runtime/B/`、`runtime/F/`：完整小规模逻辑值/BF16 位值、存储身份及单独 control；这些不是 msprobe 统计推断的输入。
- `reports/`：正式 B/F 验收、健康先行门槛、控制器完成/源码指纹、本次资源策略与逐次检查、资源前后快照、每组原始日志。不包含失败 attempt 或缓存。
- `source/`：这次运行冻结且经 SHA 核对的 9 份源码与参考，包括只读结束检查器。版本：{versions}。

冻结的 acceptance_contract.json 仍保留运行前设计状态 pending；本次实测状态以 [acceptance.json](reports/acceptance.json) 为准，不回写冻结文件以维持 source_manifest 对应的原始 SHA。

默认 Python mix API 列表未包含 torch.as_strided / Tensor.copy_；正式运行使用官方 `register_custom_api` 接口额外观察这两个真实调用，并逐项恢复原函数。没有修改安装包或拼造 dump；实际覆盖门槛见 native 报告。结束检查通过容器完整 ID 和 completion、source_manifest、冻结检查器的 SHA 绑定本次运行，确认自有容器已移除；较晚的资源快照可能已变化，不要求整台主机没有其他合法任务，也不据空闲快照声称设备独占。

config/dump 中 `/evidence/B`、`/evidence/F` 等历史路径，以及 receipt 中的相对引用均保持原样。receipt 的 runtime/control 哈希对应本目录 `runtime/{{B,F}}/`；native_files 的相对路径对应 `statistics/{{B,F}}/`。它们不代表需要访问原机器。所有来源映射见 [source_mapping.json](source_mapping.json)。不含 `.pt`、模型权重或其他张量二进制。

{warning}

## 校验

```bash
python package_evidence.py --verify .
```

[SHA256.json](SHA256.json) 覆盖其自身以外的全部交付文件，ZIP 也校验。PASS_PACKAGE_INTEGRITY 只表示文件完整性；机制结论来自原始验收报告。`source/analyze_alias.py` 的原始命令要求每组 config、runtime、control、receipt 与 observation 同目录；本包为清晰分区调整了目录，原文件字节未改。
'''
    planned['README.md'] = readme.encode('utf-8')
    planned['.gitattributes'] = b'* -text\n'
    planned['package_evidence.py'] = Path(__file__).read_bytes()
    planned['source_mapping.json'] = json_bytes({
        'schema': 'fault15.alias_detach.allhit.delivery.v1', 'source_attempt': attempt.name,
        'only_formal_B_F': True, 'original_files': mapping,
        'native_files_byte_identical': True,
        'receipt_mapping': {'native_files': 'statistics/{arm}/<original relative path>',
                            'runtime_sha256': 'runtime/{arm}/runtime.json',
                            'control_sha256': 'runtime/{arm}/control.json'},
        'excluded': ['tensor binaries', 'weights', 'cache', 'failed attempts', 'unselected experiment runs']})
    output.mkdir(parents=True, exist_ok=False)
    for name, data in planned.items():
        path = output / relative(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for arm in ('B', 'F'):
        with zipfile.ZipFile(output / (arm + '_msprobe_statistics.zip'), 'x', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in STATS:
                info = zipfile.ZipInfo(name, date_time=(2026, 10, 28, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(info, planned[f'statistics/{arm}/{name}'])
    checks = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            data = path.read_bytes()
            checks[path.relative_to(output).as_posix()] = {'bytes': len(data), 'sha256': digest(data)}
    (output / 'SHA256.json').write_bytes(json_bytes({'algorithm': 'sha256', 'files': checks}))
    return verify(output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attempt', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resource-after', type=Path)
    parser.add_argument('--warning-note', type=Path)
    parser.add_argument('--verify', type=Path, help='Verify an existing delivery without private source or NPU')
    args = parser.parse_args()
    if args.verify:
        require(not any((args.attempt, args.output, args.resource_after, args.warning_note)), 'Use only --verify')
        result = verify(args.verify)
    else:
        require(args.attempt is not None and args.output is not None, 'Supply --attempt and --output')
        result = package(args.attempt, args.output, args.resource_after, args.warning_note)
    print(json.dumps(result, ensure_ascii=False))
