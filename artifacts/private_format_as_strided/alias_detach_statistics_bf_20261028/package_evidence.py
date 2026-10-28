"""Package one accepted fault27 B/F attempt; standard library, no NPU/network.

The statistics ZIPs contain five original JSON files each. Exact small values
and injection controls remain separate. Failed attempts and caches are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


STATS = ('config.json', 'receipt.json', 'observation/step0/rank0/dump.json',
         'observation/step0/rank0/stack.json', 'observation/step0/rank0/construct.json')
REPORTS = ('acceptance.json', 'acceptance.native.json', 'acceptance.joint.json',
           'baseline_gate.json', 'baseline_gate.native.json', 'baseline_gate.joint.json')
CONTROLLER = ('source_manifest.json', 'completion.json', 'before.json',
              'B.controller.json', 'F.controller.json', 'B.log', 'F.log')
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
            require(archive.namelist() == list(STATS), 'Statistics archive must contain exactly five JSON files')
            for name in STATS:
                require(archive.read(name) == (root / 'statistics' / arm / relative(name)).read_bytes(),
                        'Statistics archive changed original bytes')
    require(load(root / 'reports/acceptance.json')['status'] == 'PASS_MECHANISM_AND_STATISTICS', 'Acceptance is not PASS')
    require(load(root / 'reports/controller/completion.json')['status'] == 'PASS', 'Controller is not PASS')
    return {'status': 'PASS_PACKAGE_INTEGRITY', 'files': len(files),
            'bytes': sum(p.stat().st_size for p in files.values()),
            'statistics_json_files_per_arm': 5, 'tensor_binary_files': 0,
            'scope': 'Packaging integrity only; no new NPU execution or reclassification'}


def package(attempt, output, resource_after, warning_note):
    require(attempt.is_dir(), 'Attempt directory does not exist')
    require(not output.exists(), 'Output must be a new directory')
    completion = load(attempt / 'completion.json')
    acceptance = load(attempt / 'acceptance.json')
    require(completion['status'] == 'PASS' and not completion.get('cleanup_errors'), 'Controller completion must be PASS')
    require(completion.get('owned_container_removed') is True and completion.get('device_lock_released') is True,
            'Owned experiment cleanup is incomplete')
    require(acceptance['status'] == 'PASS_MECHANISM_AND_STATISTICS' and not acceptance.get('errors'),
            'Independent acceptance must be PASS_MECHANISM_AND_STATISTICS')
    require(load(attempt / 'baseline_gate.json')['status'] == 'PASS_BASELINE', 'Healthy baseline gate did not pass')
    stages = completion['stages']
    require([s['arm'] for s in stages] == ['B', 'F'] and all(s['rc'] == 0 and s['independent_acceptance'] for s in stages),
            'Controller did not complete B then F')
    resource_after = resource_after or attempt / 'after.json'
    require(resource_after.is_file(), 'Provide a post-run resource JSON with --resource-after (or attempt/after.json)')
    require(resource_after.resolve() != (attempt / 'before.json').resolve(), 'Post-run resource record cannot be before.json')
    after = load(resource_after)
    require(after.get('status') == 'PASS' and after.get('owned_container_absent') is True and
            after.get('controller_completion_status') == 'PASS', 'Post-run owned-container cleanup was not verified')
    require(after.get('owned_container_id') == load(attempt / 'container_identity.json')['id'],
            'Post-run resource record belongs to a different container')
    require(after['epoch'] >= completion['end_epoch'], 'Resource record predates experiment completion')
    sources = load(attempt / 'source_manifest.json')
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

    runtimes = {}
    for arm in ('B', 'F'):
        arm_root = attempt / arm
        require(not any(p.is_file() and p.suffix in FORBIDDEN for p in arm_root.rglob('*')), 'Unexpected tensor binary in arm')
        config, receipt = load(arm_root / 'config.json'), load(arm_root / 'receipt.json')
        runtime, control = load(arm_root / 'runtime.json'), load(arm_root / 'control.json')
        runtimes[arm] = runtime
        require(config['task'] == 'statistics' and config['level'] == 'mix' and
                config['statistics']['summary_mode'] == 'statistics' and not config['statistics'].get('tensor_list'),
                'Not the agreed statistics/mix capture')
        require(receipt['status'] == runtime['status'] == 'completed' and receipt['capture_completed'] and
                receipt['capture_requested'] and receipt['exit_code'] == 0 and not receipt['full_tensor_files'], 'Incomplete arm capture')
        require(control['arm'] == arm and control['hits'] == int(arm == 'F') and not control['errors'], 'Wrong arm control')
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
    if (attempt / 'authorized_stops.jsonl').exists():
        original(attempt / 'authorized_stops.jsonl', 'reports/controller/authorized_stops.jsonl', 'authorized_stops.jsonl')
    if warning_note:
        original(warning_note, 'reports/warning_note.md', 'provided_warning_note/' + warning_note.name)
    for name, expected in sources.items():
        source = attempt / 'source' / relative(name)
        require(source.suffix in {'.py', '.sh', '.json'} and digest(source.read_bytes()) == expected, 'Frozen source mismatch')
        original(source, 'source/' + name, 'source/' + name)
    for arm in ('B', 'F'):
        require(load(attempt / (arm + '.controller.json'))['rc'] == 0, 'Arm controller failure')
    native_report, joint_report = load(attempt / 'acceptance.native.json'), load(attempt / 'acceptance.joint.json')
    first_module = native_report.get('first_observable_numerical_module_difference')
    target_count = joint_report.get('target_elements')
    untouched_count = joint_report.get('untouched_elements')
    versions = ', '.join(f'{key}={value}' for key, value in sorted(runtimes['B']['versions'].items()))
    warning = ('已附[警告来源审查说明](reports/warning_note.md)；原始 B/F 日志保留所有警告。' if warning_note else
               '尚未附警告来源审查说明；请阅读原始 B/F 日志，不能将警告自动解释为注入故障或宣称运行无警告。')
    readme = f'''# 故障 27：视图别名断开机制的 B/F 统计证据

状态：**机制级 B/F 与统计证据验收通过**。本目录来自一次完整的正式运行 `{attempt.name}`；控制器状态为 PASS，独立验收为 PASS_MECHANISM_AND_STATISTICS。打包没有重新执行 NPU，也没有更改原生 JSON 或验收门槛。

B 使用原始 NZ29 BF16 base 的真实 as_strided 视图。F 只在受控视图入口执行 base→ND2 格式转换，再调用原始 as_strided；它不修改写入或消费结果。两组随后执行相同 view.copy_ 和 original_base+1。共同观察器与独立整数索引参考确认：F 的原 base 保留旧值，消费者差异对应 {target_count} 个目标元素，{untouched_count} 个非目标元素保持不变。

固定几何为 base `[16,16]`、初值 arange(256)，view `[8,8]`、stride `[16,1]`、offset 17，实际写入 `-64-arange(64)`；原 base 消费者为 `base+1`。B/F 都是逻辑 rank0、一个真实微基准 forward 的 step0，不是模型 token 步数。

机制差异是构造视图时失去原存储别名；写入后才出现原 base 的数值差异。原生统计按语义模块对齐得到的首个可观测数值差异为 `{first_module}`。完整 trace 的观察器节点、未配对节点和不可观测统计限制见 [native 报告](reports/acceptance.native.json)；不要将缺失/null 统计判为相等。

此结果属于构造的最小机制实验，不是模型级调用链通过，也不证明历史 CANN 缺陷或生产事故已复现。统计值不能独立唯一证明别名根因；存储身份、真实写入与精确小规模值来自单独的共同运行观察器，见 [joint 报告](reports/acceptance.joint.json)。

## 数据分区与原始路径

- `statistics/B/`、`statistics/F/`：每组 config、receipt 和 step0/rank0 的 dump、stack、construct，共 5 个原始 JSON。配置为 task=statistics、level=mix、summary_mode=statistics，无 tensor_list 全张量采集。
- `B_msprobe_statistics.zip`、`F_msprobe_statistics.zip`：各自仅打包上述 5 个 JSON，ZIP 内路径从 config.json/observation 开始；内容与解压目录逐字节相同。
- `runtime/B/`、`runtime/F/`：完整小规模逻辑值/BF16 位值、存储身份及单独 control；这些不是 msprobe 统计推断的输入。
- `reports/`：正式 B/F 验收、健康先行门槛、控制器完成/源码指纹、资源前后快照、每组原始日志、停止授权执行记录（如有）。不包含失败 attempt 或缓存。
- `source/`：这次运行冻结且经 SHA 核对的源码与参考。版本：{versions}。

默认 Python mix API 列表未包含 torch.as_strided / Tensor.copy_；正式运行使用官方 `register_custom_api` 接口额外观察这两个真实调用，并逐项恢复原函数。没有修改安装包或拼造 dump；实际覆盖门槛见 native 报告。资源结束快照确认本实验容器已移除，不要求整台主机没有其他合法任务。

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
        'schema': 'fault27.alias_detach.delivery.v1', 'source_attempt': attempt.name,
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
