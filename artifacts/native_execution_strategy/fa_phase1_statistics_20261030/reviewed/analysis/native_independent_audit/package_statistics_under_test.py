"""Export reviewed statistics/metadata byte-for-byte; tensor and library payloads stay private."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


FORBIDDEN = {'.bin', '.pt', '.pth', '.npy', '.npz', '.safetensors', '.so', '.o', '.a'}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def build(root, destination, attempts, canary, validation, statistics, reports):
    root = root.resolve(strict=True)
    destination = destination.resolve()
    require(not destination.exists() and len(set(attempts)) == len(attempts) == 4, 'Fresh output and four formal attempts required')
    checked = read(validation)
    independent = read(statistics)
    require(checked['status'] == 'PASS_NATIVE_EXECUTION_STRATEGY_FA_PHASE1', 'No reviewed exact-build acceptance')
    require(set(checked['requested_attempts']) == set(attempts)
            and checked['case_count'] == 20 and checked['native_call_count'] == 56,
            'Passed report belongs to another formal matrix')
    require(checked['independent_msprobe']['summary_sha256'] == sha(statistics),
            'Passed report binds another independent statistics report')
    require(independent['planned_comparisons'] == independent['completed_comparisons'] == 88
            and independent['group'] == 'all', 'Need complete independent88 statistics')
    for name in attempts + [canary]:
        source = (root / 'outputs' / name).resolve(strict=True)
        require(source.is_relative_to(root / 'outputs') and not destination.is_relative_to(source)
                and not source.is_relative_to(destination), 'Escaping or overlapping attempt')
        if name in attempts:
            require(read(source / 'source_manifest.json') == checked['source_generation'],
                    'Formal source generation differs from acceptance')
    verified_evidence = {}
    for relative, item in checked['evidence_files'].items():
        source = (root / relative).resolve(strict=True)
        require(source.is_relative_to(root), 'Acceptance evidence escapes task')
        # Active runtime/catalog may have advanced for the separate canary.
        # The immutable executed-source snapshot retains the exact formal bytes.
        if sha(source) != item['sha256']:
            require(checked['source_generation'].get(relative) == item['sha256'],
                    'Non-source acceptance evidence changed: ' + relative)
            source = (root / 'outputs' / attempts[0] / 'source' / relative).resolve(strict=True)
        require(source.is_relative_to(root) and sha(source) == item['sha256']
                and source.stat().st_size == item['bytes'], 'Reviewed evidence changed: ' + relative)
        verified_evidence[relative] = {'sha256': item['sha256'], 'bytes': item['bytes'], 'verified_source': str(source)}
    origins, inventory, compressed = {}, {}, []
    destination.mkdir(parents=True, exist_ok=False)

    def copy(source, relative, compress=False):
        source = Path(source).resolve(strict=True)
        require(source.is_relative_to(root) and source.is_file() and not source.is_symlink(), 'Unbound source')
        require(source.suffix.lower() not in FORBIDDEN and 'private_output' not in source.relative_to(root).parts,
                'Private tensor/library payload rejected')
        target = destination / relative
        require(target.resolve().is_relative_to(destination) and not target.exists(), 'Duplicate or escaping output')
        target.parent.mkdir(parents=True, exist_ok=True)
        if compress:
            with source.open('rb') as src, target.open('xb') as dst:
                with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as encoded:
                    shutil.copyfileobj(src, encoded, 1024 * 1024)
            with gzip.open(target, 'rb') as decoded:
                digest = hashlib.sha256()
                for block in iter(lambda: decoded.read(1024 * 1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == sha(source), 'Lossy log compression')
            compressed.append({'path': relative, 'original_sha256': sha(source), 'original_bytes': source.stat().st_size})
        else:
            shutil.copyfile(source, target)
            require(sha(source) == sha(target), 'Copy changed original bytes')
        origins[relative] = str(source)

    seen, common_sources, calls = set(), None, 0
    source_generations = set()
    for name in attempts + [canary]:
        attempt = root / 'outputs' / name
        completion = read(attempt / 'completion.json')
        require(completion['status'] == 'CAPTURE_COMPLETED' and completion['owned_container_removed'] is True
                and completion['device_lock_released'] is True and completion['cleanup_errors'] == [], 'Unclean attempt')
        frozen = read(attempt / 'source_manifest.json')
        generation = sha(attempt / 'source_manifest.json')
        if name in attempts:
            require(frozen == checked['source_generation'], 'Formal source generation differs from acceptance')
            require(common_sources in (None, generation), 'Formal capture generations differ')
            common_sources = generation
        if generation not in source_generations:
            for relative, expected in frozen.items():
                source = attempt / 'source' / relative
                require(sha(source) == expected, 'Frozen source SHA differs')
                copy(source, 'source_generations/' + generation + '/' + relative)
            source_generations.add(generation)
        for path in sorted(attempt.iterdir()):
            if path.is_file() and path.suffix in ('.json', '.jsonl'):
                copy(path, 'attempts/' + name + '/' + path.name)
        for row in completion['cases']:
            case_id = row['case_id']
            require(case_id not in seen and row['rc'] == 0, 'Duplicate or failed case')
            seen.add(case_id)
            folder = attempt / case_id
            result = read(folder / 'result.json')
            require(result['status'] == 'CAPTURE_COMPLETED' and sha(folder / 'result.json') == row['result_sha256'], 'Case result differs')
            if name == canary:
                resolution = result['adapter_resolution']
                require(result['original_api_calls'] == 4 and len(result['steps']) == 4
                        and resolution['requested_version'] == 'validated'
                        and resolution['validation_state'] == 'validated'
                        and resolution['resolved_version'] == 'cann9.1-s1s2-v1', 'Validated alias canary incomplete')
            require(sha(folder / 'receipt.json') == result['receipt_sha256'], 'Receipt differs')
            receipt = read(folder / 'receipt.json')
            config = read(folder / 'config.json')
            require(config['task'] == 'statistics' and config['level'] == 'mix'
                    and config['statistics']['summary_mode'] == 'statistics'
                    and config['statistics']['tensor_list'] == [], 'Not statistics/mix')
            require(sha(folder / 'config.json') == receipt['config_sha256'], 'Statistics config differs')
            require(len(receipt['native_files']) == 3 * len(result['steps']), 'Incomplete raw statistics triple')
            for item in receipt['native_files']:
                require(sha(folder / item['path']) == item['sha256']
                        and (folder / item['path']).stat().st_size == item['bytes'], 'Raw statistics receipt mismatch')
            for source in sorted(folder.rglob('*')):
                if not source.is_file() or 'private_output' in source.relative_to(folder).parts:
                    continue
                require(not source.is_symlink(), 'Case symlink rejected')
                relative = 'attempts/' + name + '/' + case_id + '/' + source.relative_to(folder).as_posix()
                if source.suffix in ('.json', '.jsonl'):
                    copy(source, relative)
                elif source.suffix == '.log':
                    copy(source, relative + '.gz', True)
                else:
                    require(source.suffix not in FORBIDDEN, 'Unexpected tensor/library outside private_output')
            log = attempt / (case_id + '.log')
            copy(log, 'attempts/' + name + '/' + log.name + '.gz', True)
            if name in attempts:
                calls += result['original_api_calls']
    require(len(seen) == 21 and calls == 56, 'Need20 formal cases/56calls plus separate4call canary')
    for label, path in [('validation', validation), ('independent_statistics', statistics)]:
        copy(path, 'reports/' + label + '.json')
    for relative in reports:
        path = (root / relative).resolve(strict=True)
        if path.is_file():
            copy(path, 'reviewed/' + relative)
        else:
            for source in sorted(path.rglob('*')):
                if source.is_file() and source.suffix in ('.json', '.jsonl', '.py', '.md', '.patch'):
                    copy(source, 'reviewed/' + source.relative_to(root).as_posix())
    copy(root / 'inputs/manifest.json', 'inputs/manifest.json')
    copy(root / 'native_vendor_stage_a1/vendor_manifest.json', 'build/vendor_manifest.json')
    (destination / '.gitattributes').write_text('* -text\n', encoding='ascii')
    (destination / 'README.md').write_text(
        '# FA 第一阶段 statistics/mix 证据\n\n'
        '正式矩阵20case/56原生API调用；另附validated别名四步canary。步骤不是模型token。\n'
        '仅达到固定FA路径的策略切换和数值现象验收；模型影响未测。\n\n'
        '原始dump/stack/construct/config/receipt和JSONL逐字节保留。源快照按清单SHA去重存放。\n'
        'native日志为确定性gzip，manifest记录解压后原SHA/bytes；解压即可恢复原日志字节。\n'
        '完整目标/QKV二进制留在124；公开精确比较报告不能替代私有完整输出重算。\n'
        '本轮使用已披露的任务本地FP32 high collector，非未修改的官方collector。\n'
        'null与stock/private完整stack不对齐均保留；seed/offset不是主输出故障。\n', encoding='utf-8')
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            require(path.suffix.lower() not in FORBIDDEN and not path.is_symlink(), 'Forbidden public payload')
            inventory[path.relative_to(destination).as_posix()] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    manifest = {'schema': 'nes.statistics_public_package.v1', 'formal_attempts': attempts, 'canary_attempt': canary,
                'formal_cases': 20, 'formal_native_calls': 56, 'canary_native_calls': 4,
                'validation_report_sha256': sha(validation), 'statistics_report_sha256': sha(statistics),
                'files': inventory, 'origins': origins, 'compressed_logs': compressed,
                'reverified_acceptance_evidence': verified_evidence,
                'private_tensor_payloads_published': False, 'model_impact': None}
    target = destination / 'manifest.json'
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return {'files': len(inventory), 'bytes': sum(x['bytes'] for x in inventory.values()), 'manifest_sha256': sha(target)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attempt', action='append', required=True)
    parser.add_argument('--canary', required=True)
    parser.add_argument('--validation', type=Path, required=True)
    parser.add_argument('--statistics', type=Path, required=True)
    parser.add_argument('--reviewed', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output, args.attempt, args.canary, args.validation, args.statistics, args.reviewed)))
