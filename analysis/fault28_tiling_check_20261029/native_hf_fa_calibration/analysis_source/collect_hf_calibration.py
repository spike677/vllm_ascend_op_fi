"""Download compact HF a3 metadata, including logged tiling bytes, never tensors."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import base64,hashlib,io,json,pathlib,sys,tarfile
root=pathlib.Path('/home/fault28-native-tiling-20261029/outputs/hf_stock_2048_a3')
cases=['stock_b1','stock_b1_repeat','stock_b8','stock_b8_repeat']
files=['completion.json','source_manifest.json','resource_policy.json','container_identity.json']
files += ['source/'+n for n in ['controller.py','hf_runtime_probe.py','env.sh','build_cases.py','setup/capture_contract.json','setup/cases_2048/manifest.json']]
for case in cases:
 files += ['source/setup/cases_2048/'+case+'.json',case+'/result.json',case+'/native_calls/call_000.json',case+'/native_calls/call_001.json',case+'.fa_log_profile/profile.json']
 for i in range(28):
  files += [case+'.fa_log_profile/call_'+str(i).zfill(3)+'.'+suffix+'.bin' for suffix in ['declared_prefix','aligned_log']]
payloads={};mapping=[]
for relative in files:
 path=root/relative
 assert path.is_file() and path.stat().st_size<3*1024*1024 and 'private_replay' not in path.parts
 data=path.read_bytes();destination='raw/'+relative
 payloads[destination]=data
 mapping.append({'path':destination,'source_attempt_relative_path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_bytes':True,'binary_kind':'logged_tiling_metadata_not_tensor' if path.suffix=='.bin' else None})
for case in cases:
 profile=json.loads(payloads['raw/'+case+'.fa_log_profile/profile.json'])
 wanted={}
 for call in profile['calls']:
  for kind in ['launches','chunks','api_workspaces']:
   for row in call[kind]:
    assert row['line'] not in wanted
    wanted[row['line']]={'call_id':call['enter']['call_id'],'layer_index':call['enter']['layer_index'],'kind':kind,'expected_text':row['text']}
 digest=hashlib.sha256();excerpts=[];total=0
 with (root/(case+'.log')).open('rb') as stream:
  for line_no,raw in enumerate(stream,1):
   digest.update(raw);total+=len(raw)
   if line_no in wanted:
    item=wanted.pop(line_no);text=raw.rstrip(b'\r\n').decode('utf-8','surrogateescape')
    assert text==item.pop('expected_text')
    excerpts.append({**item,'line':line_no,'raw_line_sha256':hashlib.sha256(raw).hexdigest(),'raw_line_base64':base64.b64encode(raw).decode(),'text':text})
 assert not wanted and digest.hexdigest()==profile['source_sha256']['log']
 value={'schema':'fault28.hf_fa.exact_log_excerpts.v1','case_id':case,'source_log':case+'.log','source_log_bytes':total,'source_log_sha256':digest.hexdigest(),'source_log_total_lines':line_no,'full_log_included':False,'extraction':'One binary-line read and whole-log SHA; each requested line equals extractor text after removing only CR/LF. Base64 retains original bytes.','native_lines':excerpts}
 payloads['log_excerpts/'+case+'.json']=(json.dumps(value,ensure_ascii=True,indent=2)+'\n').encode()
payloads['source_mapping.json']=(json.dumps({'schema':'fault28.hf_fa.small_evidence_sources.v1','attempt':'hf_stock_2048_a3','files':mapping,'private_tensor_files_included':False,'full_native_logs_included':False,'tiling_metadata_binary_included':True},indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,data in payloads.items():
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;archive.addfile(info,io.BytesIO(data))
'''


def main():
    destination = HERE / 'native_hf_fa_calibration'
    if destination.exists():
        raise RuntimeError('Fresh evidence directory required')
    result = subprocess.run(['ssh', 'npu-dual', 'python3', '-'], input=REMOTE.encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace')[-8000:])
    destination.mkdir()
    count = total = 0
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        for member in archive:
            path = destination / member.name
            assert member.isfile() and destination.resolve() in path.resolve().parents
            assert path.suffix in ('.json', '.py', '.sh', '.bin') and member.size < 3 * 1024 * 1024
            assert 'private_replay' not in path.parts
            if path.suffix == '.bin':
                assert path.parent.name.endswith('.fa_log_profile') and path.name.endswith(('.declared_prefix.bin', '.aligned_log.bin'))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(archive.extractfile(member).read())
            count += 1; total += member.size
    source = (HERE / 'extract_fa_log_profile.py').read_bytes()
    expected = {json.loads(p.read_bytes())['extractor_sha256'] for p in (destination / 'raw').glob('*.fa_log_profile/profile.json')}
    assert expected == {hashlib.sha256(source).hexdigest()}
    (destination / 'analysis_source').mkdir()
    (destination / 'analysis_source/extract_fa_log_profile.py').write_bytes(source)
    print(json.dumps({'status': 'DOWNLOADED_HF_METADATA_AND_LOGGED_TILING_BYTES', 'files': count + 1,
                      'bytes': total + len(source), 'private_tensor_or_full_log_downloaded': False}))


if __name__ == '__main__':
    main()
