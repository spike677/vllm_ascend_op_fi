"""Recheck supplied source87 metadata and c77 source bytes without a device.

This neither replays inference nor approves the new private implementation.
It leaves private source archives and tensor hashes outside the repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from fault_injection.native_adapters.attention28.fia import apply_tnd28_prefill_normal as recipe
from fault_injection.native_adapters.attention28.fia import tnd28_prefill_geometry as geometry


STOCK_SHA = '3b7143f1b3873211214a79e3d22afc21a86e80b8a86409496bfe6e15dfba3340'
KERNEL_ARCHIVE_SHA = 'e24f9f691af92cd03fd82c49a901d0061458ae782ec7b4703b5ec8055d4ef7df'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def audit_calls(result):
    requests = result['final_cleanup']['evidence']['requests']
    require(len(requests) == 2, 'TND28_TWO_ORIGINAL_REQUESTS_REQUIRED')
    identities, request_ids, candidates = [], [], []
    for request in requests:
        require(len(request['forwards']) == 1, 'TND28_ONE_FRESH_FORWARD_REQUIRED')
        calls = request['forwards'][0]['calls']
        require(len(calls) == 28, 'TND28_COMPLETE_28_SITE_DENOMINATOR_REQUIRED')
        current, ids = [], set()
        for ordinal, call in enumerate(calls):
            require(type(call['ordinal']) is int and call['ordinal'] == ordinal
                and type(call['module_object_id']) is int and call['module_object_id'] > 0
                and call['module_path'] == f'model.layers.{ordinal}.self_attn.attn',
                'TND28_ACTUAL_SITE_ORDER_OR_OBJECT_DIFFERS')
            candidates.append(geometry.public_candidate(call['native_census']['api_inputs_after'], call))
            current.append((call['module_path'], call['module_object_id']))
            ids.add(call['request_id'])
        require(len(ids) == 1 and len(set(current)) == 28, 'TND28_REQUEST_OR_MODULE_ALIAS')
        request_ids.extend(ids)
        identities.append(current)
    require(len(set(request_ids)) == 2 and identities[0] == identities[1],
            'TND28_REPEAT_REQUEST_OR_WORKER_SITE_IDENTITY_DIFFERS')
    return dict(actual_requests=2, actual_public_calls=56, candidate_metadata_matches=56,
                distinct_sites=28, all_original_views_preserved=True,
                public_capture_hashes_sha256=geometry.digest([v['capture_sha256'] for v in candidates]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stock-result', type=Path, required=True)
    parser.add_argument('--host-source-root', type=Path, required=True)
    parser.add_argument('--parser-sources', type=Path, required=True)
    parser.add_argument('--kernel-blobs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'TND28_FRESH_CPU_REPORT_REQUIRED')
    stock, kernel = args.stock_result.read_bytes(), args.kernel_blobs.read_bytes()
    require(sha(stock) == STOCK_SHA and sha(kernel) == KERNEL_ARCHIVE_SHA, 'TND28_ORIGINAL_BYTES_DIFFERS')
    raw = {}
    with tarfile.open(args.kernel_blobs, 'r:gz') as archive:
        members = archive.getmembers()
        require(all(m.isfile() and not m.issym() for m in members), 'TND28_ARCHIVE_NONREGULAR_MEMBER')
        names = [m.name for m in members]
        require(len(names) == len(set(names)), 'TND28_ARCHIVE_DUPLICATE_MEMBER')
        manifest = json.load(archive.extractfile('manifest.json'))
        require(manifest['commit'] == recipe.base.COMMIT and manifest['read_only_git_blobs'] is True
            and manifest['actual_npu_calls'] == 0 and len(manifest['files']) == 5,
            'TND28_PINNED_GIT_BLOBS_REQUIRED')
        for item in manifest['files']:
            name = item['path']
            require(name in recipe.SOURCE_SHA and 'op_kernel/' in name, 'TND28_UNREVIEWED_KERNEL_PATH')
            value = archive.extractfile(name).read()
            require(len(value) == item['bytes'] and sha(value) == item['sha256'] == recipe.SOURCE_SHA[name],
                    'TND28_KERNEL_BLOB_BYTES_DIFFERS')
            raw[name] = value
    parser_raw = args.parser_sources.read_bytes()
    for item in json.loads(parser_raw)['files']:
        if '/source/attention/' not in item['path']:
            continue
        name = 'attention/' + item['path'].split('/source/attention/', 1)[1]
        if name in recipe.SOURCE_SHA:
            value = item['source'].encode()
            require(sha(value) == item['sha256'] == recipe.SOURCE_SHA[name], 'TND28_PARSER_BYTES_DIFFERS')
            raw[name] = value
    for name in recipe.SOURCE_SHA:
        if name not in raw:
            raw[name] = (args.host_source_root/name).read_bytes()
    recipe.verify_source(raw)
    original = {name: raw[name].decode() for name in recipe.PATHS}
    modified = recipe.patched(original)
    restored = dict(modified)
    restored[recipe.PATHS[0]] = restored[recipe.PATHS[0]].replace('#include "tnd28_prefill_normal.h"\n\n','')
    restored[recipe.PATHS[1]] = restored[recipe.PATHS[1]].replace(recipe.CAPTURE,'')
    restored[recipe.ROUTE] = restored[recipe.ROUTE].replace(
        'CheckSpecConditions(context) && !tnd28_prefill_normal::Matches(context)', 'CheckSpecConditions(context)')
    require(restored == original, 'TND28_UNEXPECTED_SOURCE_TRANSFORMATION')
    plan = geometry.normal_coordinate_matrix()
    facts = audit_calls(json.loads(stock))
    implementation = [Path(recipe.__file__), Path(geometry.__file__),
        Path(recipe.__file__).with_name('tnd28_prefill_normal.h'),
        Path(recipe.__file__).with_name('tnd28_prefill_contract.h')]
    report = dict(schema='demoncat.attention28.tnd_normal_cpu_materials.v1',
        status='CPU_SOURCE_AND_PUBLIC_METADATA_CONTRACT_PASS_NOT_NATIVE_ACCEPTANCE',
        stock_result_sha256=sha(stock), kernel_archive_sha256=sha(kernel),
        parser_archive_sha256=sha(parser_raw), source_commit=recipe.base.COMMIT,
        actual_source_files_verified=len(raw), source_sha256=dict(recipe.SOURCE_SHA),
        changed_source_paths=list(modified), arithmetic_restoration_exact=True,
        kernel_bytes_changed=False, normal_key=str(recipe.NORMAL_KEY),
        public_candidate=facts, coordinate_tiles=len(plan['tiles']),
        coordinate_unmasked_pairs_per_kv_head=plan['unmasked_pairs_per_kv_head'],
        implementation_sha256={p.name:sha(p.read_bytes()) for p in implementation},
        sdk_compiled=False, actual_private_normal_calls=0, new_npu_calls=0,
        public_prepared_pointer_stride_join=False, mask_values_verified=False,
        stock_private_exact_proven=False, c1=False, c2=False, c3=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # The published summary hash must match Windows, Linux and Git bytes.
    args.output.write_bytes((json.dumps(report, indent=2, sort_keys=True)+'\n').encode('utf-8'))
    print(json.dumps(dict(status=report['status'], source_files=len(raw), public_calls=facts['actual_public_calls'],
                          output_sha256=sha(args.output.read_bytes()))))


if __name__ == '__main__':
    main()
