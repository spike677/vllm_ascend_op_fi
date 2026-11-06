"""Verify the actual offline pip configuration before a fresh CPU build."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from generation import layout


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',required=True)
    args=parser.parse_args()
    from pip._internal.commands import create_command
    options,_=create_command('wheel').parse_args(['.','--no-deps'])
    tools={name:importlib.metadata.version(name) for name in ('pip','setuptools','wheel')}
    expected={'pip':'26.2.1','setuptools':'84.2.1','wheel':'0.48.0'}
    if tools!=expected or options.build_isolation or not options.no_index:
        raise RuntimeError('Actual offline wheel toolchain/configuration differs')
    receipt={'schema':'fault28.fia_graph.offline_pip_environment.v1','generation':args.generation,'python':sys.executable,
        'tools':tools,'pip_parsed_build_isolation':options.build_isolation,'pip_parsed_no_index':options.no_index,
        'environment':{k:os.environ.get(k) for k in ('PIP_NO_INDEX','PIP_NO_BUILD_ISOLATION','PIP_DISABLE_PIP_VERSION_CHECK')},
        'network_enabled':False,'npu_executed':False}
    p=layout(Path('/audit'),args.generation)['evidence']/'pip_build_environment.json'
    p.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':
    main()
