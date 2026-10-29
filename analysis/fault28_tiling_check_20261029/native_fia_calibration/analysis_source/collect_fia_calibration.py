"""Read-only SSH download of small model_stock_a5 evidence; no private tensor data."""
import io
import json
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REMOTE = r'''
import base64, hashlib, io, json, pathlib, re, sys, tarfile
root=pathlib.Path('/home/fault28-native-tiling-20261029/outputs/model_stock_a5')
cases=['stock_b1','stock_b1_repeat','stock_b8','stock_b8_repeat']
files=['completion.json','source_manifest.json','resource_policy.json','container_identity.json']
files += ['source/'+n for n in ['controller.py','runtime_probe.py','model_hook.py','sitecustomize.py','build_cases.py','env.sh','setup/capture_contract.json','setup/cases_1024/manifest.json']]
for case in cases:
 files += [case+'.native_profile.v4.json', 'source/setup/cases_1024/'+case+'.json']
 files += [case+'/'+n for n in ['result.json','receipt.json','logits.json','scheduler.json','native_calls/call_000.json','native_calls/call_001.json']]
payloads={}
mapping=[]
for relative in files:
 path=root/relative
 assert path.is_file() and path.stat().st_size<3*1024*1024
 data=path.read_bytes()
 destination='raw/'+relative
 payloads[destination]=data
 mapping.append({'path':destination,'source_attempt_relative_path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'original_bytes':True})
profiles={case:json.loads(payloads['raw/'+case+'.native_profile.v4.json']) for case in cases}
parser=root.parent.parent/'calibration_log_parser.py'
parser_bytes=parser.read_bytes()
assert {p['parser_sha256'] for p in profiles.values()}=={hashlib.sha256(parser_bytes).hexdigest()}
payloads['analysis_source/calibration_log_parser_v4.py']=parser_bytes
mapping.append({'path':'analysis_source/calibration_log_parser_v4.py','source_task_relative_path':'calibration_log_parser.py','bytes':len(parser_bytes),'sha256':hashlib.sha256(parser_bytes).hexdigest(),'original_bytes':True,'note':'Post-capture parser used for all four v4 reports; distinct from the capture-time source manifest parser.'})
for case,profile in profiles.items():
 wanted={}
 for call in profile['calls']:
  for kind in ['selected_tiling_records','selected_launch_records']:
   for row in call[kind]:
    assert row['line'] not in wanted
    wanted[row['line']]={'call_id':call['call_id'],'layer_index':call['enter']['layer_index'],'kind':kind,'expected_text':row['text']}
 digest=hashlib.sha256(); excerpts=[]; total=0
 log=root/(case+'.log')
 with log.open('r',encoding='utf-8',errors='surrogateescape',newline='') as stream:
  for line_no,text in enumerate(stream,1):
   raw=text.encode('utf-8','surrogateescape')
   digest.update(raw);total+=len(raw)
   if line_no in wanted:
    item=wanted.pop(line_no)
    assert re.sub(r'\x1b\[[0-9;]*[A-Za-z]','',text).rstrip('\r\n')==item.pop('expected_text')
    excerpts.append({**item,'line':line_no,'raw_line_sha256':hashlib.sha256(raw).hexdigest(),'raw_line_base64':base64.b64encode(raw).decode('ascii'),'text':raw.rstrip(b'\r\n').decode('utf-8','surrogateescape')})
 assert not wanted and digest.hexdigest()==profile['inputs']['log_sha256']
 value={'schema':'fault28.fia.exact_log_excerpts.v1','case_id':case,'source_log':case+'.log','source_log_bytes':total,'source_log_sha256':digest.hexdigest(),'source_log_total_lines':line_no,'full_log_included':False,'extraction':'One full read with universal line boundaries but original endings retained; SHA verifies entire source log. Each requested line matches v4 parser text after its documented ANSI removal. Base64 retains the exact original bytes.','selected_and_launch_lines':excerpts}
 payloads['log_excerpts/'+case+'.json']=(json.dumps(value,ensure_ascii=True,indent=2)+'\n').encode()
payloads['source_mapping.json']=(json.dumps({'schema':'fault28.fia.small_evidence_sources.v1','attempt':'model_stock_a5','files':mapping,'private_replay_included':False,'full_native_logs_included':False},indent=2)+'\n').encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
 for name,data in payloads.items():
  info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600
  archive.addfile(info,io.BytesIO(data))
'''


def main():
    destination = HERE / 'native_fia_calibration'
    if destination.exists():
        raise RuntimeError('Fresh evidence directory required')
    result = subprocess.run(['ssh', 'npu-dual', 'python3', '-'], input=REMOTE.encode('utf-8'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=240)
    if result.returncode:
        raise RuntimeError(result.stderr.decode('utf-8', 'replace')[-8000:])
    destination.mkdir()
    count = total = 0
    with tarfile.open(fileobj=io.BytesIO(result.stdout), mode='r:') as archive:
        for member in archive:
            path = destination / member.name
            assert member.isfile() and destination.resolve() in path.resolve().parents
            assert path.suffix in ('.json', '.py', '.sh') and member.size < 3 * 1024 * 1024
            assert 'private_replay' not in path.parts
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                stream.write(archive.extractfile(member).read())
            count += 1
            total += member.size
    print(json.dumps({'status': 'DOWNLOADED_NATIVE_METADATA_AND_EXACT_EXCERPTS', 'files': count,
                      'bytes': total, 'destination': str(destination), 'full_logs_or_tensor_binary_downloaded': False}))


if __name__ == '__main__':
    main()
