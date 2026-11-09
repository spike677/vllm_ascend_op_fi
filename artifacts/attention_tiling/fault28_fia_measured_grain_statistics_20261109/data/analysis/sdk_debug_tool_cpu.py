"""Run the pinned SDK scalar-debug parser in a device-free owned container."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import uuid

ROOT=Path('/home/fault28-fia-grain-20261109')
IMAGE='sha256:8988c3cd3a971a72ebb2e5baae24e0091cde55146f96d2b4c8229448210735b9'
TOOL='/usr/local/Ascend/cann-9.1.0/tools/show_kernel_debug_data/show_kernel_debug_data'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--help-only',action='store_true')
    p.add_argument('--file',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--receipt',type=Path);a=p.parse_args()
    if a.help_only:arguments=['--help']
    else:
        source=a.file.resolve(strict=True);output=a.output.resolve()
        assert source.is_relative_to(ROOT/'outputs') and not output.exists()
        files=[source] if source.is_file() else sorted(source.rglob('*.bin'))
        assert files and len(files)<=64 and all(f.name.startswith('asc_kernel_data_') for f in files)
        assert 'private_device_debug' in source.parts
        assert output.is_relative_to(ROOT/'outputs') and sum(f.stat().st_size for f in files)<64*1024**2
        # This SDK accepts one directory, but does not recurse into its date
        # subdirectory. Require one actual leaf; never treat empty parsing as OK.
        parents={f.parent for f in files};assert len(parents)==1
        sdk_source=source if source.is_file() else next(iter(parents))
        arguments=['/evidence/'+sdk_source.relative_to(ROOT/'outputs').as_posix(),
                   '/evidence/'+output.relative_to(ROOT/'outputs').as_posix()]
    token=uuid.uuid4().hex;name='f28-grain-parser-'+token[:12]
    command=['docker','run','--rm','--name',name,'--label','demoncat.task=fault28-grain-cpu-parser',
        '--label','demoncat.owner='+token,'--network','none','--memory','2g','--cpus','2',
        '--mount',f'type=bind,src={ROOT}/outputs,dst=/evidence','--entrypoint','bash',IMAGE,'-c',
        'source /usr/local/Ascend/cann-9.1.0/set_env.sh && exec "$@"','f28-parser',TOOL,*arguments]
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=120)
        assert len(result.stdout.encode())<=1024**2,'Unbounded scalar parser stdout'
        report={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr[-4000:],
                'npu_devices_mapped':False,'tool':TOOL,'container_image':IMAGE}
        if not a.help_only:
            report.update(source=str(source),source_files=[{'file':str(f),'bytes':f.stat().st_size,
                          'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files],output=str(output))
            # This SDK build emits scalar decoding on stdout. Its advertised
            # parser.log is not materialized; retain the original stream bytes.
            output.mkdir(parents=True,exist_ok=True)
            stdout_file=output/'official_sdk_stdout.txt'
            with stdout_file.open('xb') as stream:stream.write(result.stdout.encode('utf-8'))
            report['saved_original_sdk_stdout']=str(stdout_file)
            report['decoder_stdout_sha256']=hashlib.sha256(result.stdout.encode('utf-8')).hexdigest()
            report['parsed_outputs']=[{'file':str(f),'bytes':f.stat().st_size,
                'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(output.rglob('*')) if f.is_file()]
            report['tool_returncode']=result.returncode
            if 'F28_GRAIN_V1' not in result.stdout:
                report.update(returncode=1,error='No actual F28 device scalar records in official parser output')
        report['command']=command;report['helper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if a.receipt:
            assert a.receipt.resolve().is_relative_to(ROOT/'outputs')
            a.receipt.parent.mkdir(parents=True,exist_ok=True)
            with a.receipt.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
            print(json.dumps({'returncode':report['returncode'],'receipt':str(a.receipt),
                'source_files':len(report.get('source_files',[])),'parsed_outputs':len(report.get('parsed_outputs',[]))}))
        else:print(json.dumps(report))
        return report['returncode']
    finally:
        check=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
        if check.returncode==0:
            obj=json.loads(check.stdout)[0];labels=obj['Config'].get('Labels') or {}
            assert obj['Name']=='/'+name and obj['Image']==IMAGE and labels.get('demoncat.owner')==token
            subprocess.run(['docker','rm','-f',obj['Id']],check=True,capture_output=True,timeout=20)


if __name__=='__main__':raise SystemExit(main())
