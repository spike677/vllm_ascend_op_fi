"""Use the actual IFA CMake compiler/include flags for the standalone ABI decoder."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from generation import layout

ROOT = Path('/audit')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',required=True)
    args=parser.parse_args()
    paths=layout(ROOT,args.generation)
    SOURCE=paths['source']
    EVIDENCE=paths['evidence']
    entries = json.loads((SOURCE/'build/compile_commands.json').read_text())
    match = [e for e in entries if Path(e['file']).name == 'incre_flash_attention_tiling.cpp']
    if not match:
        raise RuntimeError('No actual IFA host compile command')
    entry = match[0]
    original = entry.get('arguments') or shlex.split(entry['command'])
    flags = []
    i = 1
    while i < len(original):
        value = original[i]
        if value in ('-o', '-MF', '-MT', '-MQ'):
            i += 2
            continue
        if value in ('-c', '-MD', '-MMD', entry['file']):
            i += 1
            continue
        flags.append(value)
        i += 1
    decoder = Path(__file__).resolve().with_name('ifa_abi_decoder.cpp')
    output = EVIDENCE/'ifa_abi_decoder'
    command = [original[0], *flags, '-I'+str(SOURCE/'attention/incre_flash_attention/op_host'), str(decoder),
        '-L/usr/local/Ascend/cann-9.1.0/lib64', '-lregister', '-o', str(output)]
    subprocess.run(command, cwd=entry['directory'], check=True)
    for argument, name in [('--schema', 'ifa_abi_schema.json'), ('--self-test', 'ifa_abi_selftest.json')]:
        value = subprocess.check_output([str(output), argument], text=True)
        json.loads(value)
        (EVIDENCE/name).write_text(value)
    receipt = {'schema': 'fault33.pa.abi_decoder_compile.v1', 'compiler_argv': command,
        'actual_ifa_compile_entry': entry, 'source_sha256': hashlib.sha256(decoder.read_bytes()).hexdigest(),
        'decoder_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'npu_executed': False,
        'abi_kind': 'public_host_serializer_projection_not_device_argument_read'}
    (EVIDENCE/'ifa_abi_compile.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    main()
