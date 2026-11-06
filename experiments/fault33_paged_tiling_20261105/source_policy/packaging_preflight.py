"""Verify pip env semantics and build the actual es_math wheel offline first."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from generation import layout


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--generation',required=True)
    parser.add_argument('--attempt',required=True)
    args=parser.parse_args()
    paths=layout(Path('/audit'),args.generation)
    ev=paths['evidence']/f'resume_{args.attempt}'
    if os.environ.get('PIP_NO_INDEX')!='1' or os.environ.get('PIP_NO_BUILD_ISOLATION')!='0':
        raise RuntimeError('Pinned offline pip environment is missing')
    from pip._internal.commands import create_command
    options,_=create_command('wheel').parse_args(['.','--no-deps'])
    if options.build_isolation or not options.no_index:
        raise RuntimeError('Actual pip did not apply no-build-isolation + no-index')
    tools={}
    for name in ('pip','setuptools','wheel'):
        try:
            dist=importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            tools[name]={'installed':False}
            continue
        files={}
        for item in dist.files or []:
            if str(item).endswith(('/METADATA','/RECORD')):
                p=Path(dist.locate_file(item));files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        tools[name]={'installed':True,'version':dist.version,'distribution_files':files}
    if tools.get('setuptools',{}).get('version')!='84.2.1':
        raise RuntimeError('Private pinned setuptools version differs')
    project=paths['source']/'build/common/stub/op_graph/es_math_build/python_package'
    target=ev/'wheel_preflight'
    target.mkdir(exist_ok=False)
    shutil.copy2(project/'setup.py',target/'setup.py')
    shutil.copytree(project/'es_math',target/'es_math')
    command=[sys.executable,'-m','pip','wheel','.','--no-deps','--wheel-dir='+str(target/'dist')]
    result=subprocess.run(command,cwd=target,capture_output=True,text=True,timeout=120)
    (ev/'wheel_preflight.log').write_text(result.stdout+'\n'+result.stderr)
    wheels=list((target/'dist').glob('*.whl'))
    receipt={'schema':'fault33.pa.offline_wheel_preflight.v1','command':command,'returncode':result.returncode,
        'python':sys.executable,'tools':tools,'environment':{k:os.environ.get(k) for k in ('PYTHONPATH','PIP_NO_INDEX','PIP_NO_BUILD_ISOLATION','PIP_DISABLE_PIP_VERSION_CHECK')},
        'pip_parsed_build_isolation':options.build_isolation,'pip_parsed_no_index':options.no_index,
        'setup_sha256':hashlib.sha256((project/'setup.py').read_bytes()).hexdigest(),
        'wheels':[{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in wheels],
        'npu_executed':False,'network_enabled':False}
    (ev/'wheel_preflight.json').write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode or len(wheels)!=1:
        raise RuntimeError('Actual es_math offline wheel preflight failed')


if __name__=='__main__':
    main()
