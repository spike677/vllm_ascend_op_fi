"""Copy verified real captured Q/K/V into a new task; derive B1/2/4/8 hashes.

Uses only the standard library. It never imports a framework or opens an NPU.
The original fault28 input directory is read only and is never modified.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

SOURCE_MANIFEST_SHA = '509123dc725e351d6b10840919c2640b327a13eecceba2ca4a888c4a094f8e62'
SOURCE_CALL_SHA = '458a683a616da397bd1eeba47b7348d0e2bf78c253ea43c158f7f393bb67c0c8'
BATCHES = (1, 2, 4, 8)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def repeated_sha(payload, count):
    digest = hashlib.sha256()
    for _ in range(count):
        digest.update(payload)
    return digest.hexdigest()


def freeze(source, output):
    source = source.resolve(strict=True)
    output = output.resolve()
    require(not output.exists() and not output.is_relative_to(source)
            and not source.is_relative_to(output), 'Fresh, nonoverlapping output required')
    original = (source / 'manifest.json').read_bytes()
    require(sha(original) == SOURCE_MANIFEST_SHA, 'Original input manifest differs')
    manifest = json.loads(original)
    require(manifest['schema'] == 'fault28.causal.inputs.v1'
            and manifest['source_call_sha256'] == SOURCE_CALL_SHA, 'Unexpected capture provenance')
    require(manifest['source_call_file'] == 'source_call.json', 'Unexpected source-call filename')
    call = (source / 'source_call.json').read_bytes()
    require(sha(call) == SOURCE_CALL_SHA, 'Original source-call metadata differs')
    payloads = {}
    for name in ('query', 'key', 'value'):
        row = manifest['inputs'][name]
        require(row['file'] == name + '.fp16.bin', 'Unexpected input filename')
        path = source / row['file']
        require(path.is_file() and not path.is_symlink(), 'Input must be a direct file')
        payload = path.read_bytes()
        require(len(payload) == row['bytes'] and sha(payload) == row['sha256'],
                'Original captured input differs: ' + name)
        require(row['shape'] == [1, 16 if name == 'query' else 8, 2048, 128]
                and row['dtype'] == 'torch.float16' and row['byteorder'] == 'little',
                'Unexpected captured input geometry')
        require(all(repeated_sha(payload, b) == row['expected_full_sha256'][str(b)] for b in (1, 8)),
                'Original repeated-input hashes differ')
        payloads[row['file']] = payload
    derived = copy.deepcopy(manifest)
    derived.update(schema='nes.captured_inputs.v1',
                   parent_manifest_file='parent_manifest.json',
                   parent_manifest_sha256=SOURCE_MANIFEST_SHA,
                   supported_observation_batches=list(BATCHES),
                   controlled_candidate_batches=[1, 8],
                   discovery_only_batches=[2, 4],
                   preparation='Byte-copy original batch0; derive logical repeated hashes without tensor computation. '
                               'New batch sizes are not controlled-strategy validation.')
    for name, row in derived['inputs'].items():
        row['batch_rule'] = 'B independent byte-identical copies of the originally captured batch0, B in 1/2/4/8; preserve original stride.'
        row['expected_full_sha256'] = {str(b): repeated_sha(payloads[row['file']], b) for b in BATCHES}
    # Finish all validation before creating output; never update an existing input snapshot.
    output.mkdir(parents=True, exist_ok=False)
    for name, payload in {'parent_manifest.json': original, 'source_call.json': call, **payloads}.items():
        with (output / name).open('xb') as stream:
            stream.write(payload)
        require(sha((output / name).read_bytes()) == sha(payload), 'Copy changed bytes')
    with (output / 'manifest.json').open('x', encoding='utf-8') as stream:
        json.dump(derived, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    return {'status': 'FROZEN_INPUT_BYTES_NOT_NATIVE_EXECUTION', 'npu_calls': 0,
            'manifest_sha256': sha((output / 'manifest.json').read_bytes()),
            'parent_manifest_sha256': SOURCE_MANIFEST_SHA,
            'source_call_sha256': SOURCE_CALL_SHA, 'batches': list(BATCHES),
            'controlled_validation_completed': False,
            'private_input_binary_publication': 'EXCLUDED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze(args.source_inputs, args.output)))


if __name__ == '__main__':
    main()
