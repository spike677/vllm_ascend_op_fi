"""Read-only torch_npu cache source/header/symbol inventory; no runtime import."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import uuid

READER = r'''
import hashlib, importlib.metadata, json, pathlib, re, subprocess
d = importlib.metadata.distribution('torch-npu')
root = pathlib.Path(d.locate_file('torch_npu'))
needles = re.compile(r'ACLNN_CACHE_LIMIT|CheckAclnnCache|GetAclnnCache|Update.*Executor|CachedExecutor|TryCache|try_cache|GetWorkspaceSize')
report = {'schema_version':1, 'distribution_version':d.version, 'root':str(root), 'files':{},
          'headers_searched':0, 'npu_calls':0, 'runtime_imported':False, 'symbols':{}, 'libraries':[]}
version = root / 'version.py'
if version.is_file():
 raw = version.read_bytes()
 report['files']['version.py']={'path':str(version),'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode()}
for path in root.rglob('*'):
 if not path.is_file() or path.suffix not in ('.h','.hpp','.cpp'):
  continue
 report['headers_searched'] += 1
 raw = path.read_bytes()
 if len(raw) > 3000000:
  continue
 source = raw.decode('utf-8', errors='replace')
 if needles.search(source):
  report['files'][str(path.relative_to(root))]={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
    'bytes':len(raw),'text':source}
for path in (root/'lib').glob('*.so'):
 report['libraries'].append({'path':str(path),'bytes':path.stat().st_size})
 if 'plugin' not in path.name and path.name != 'libtorch_npu.so':
  continue
 process = subprocess.run(['nm','-D','-C',str(path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
 selected = [line for line in process.stdout.splitlines() if re.search(
   r'AclnnCache|aclnnCache|PTA|TensorAddr|Update.*Addr|aclnn.*Cache|cache.*Run|npu_quant_matmul|GetOpApiFuncAddr',line)]
 strings = subprocess.run(['strings',str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 strings = sorted(set(line for line in strings.stdout.splitlines() if re.search(
   r'ACLNN.*CACHE|PTA.*Cache|TensorAddr|^aclnnQuantMatmul',line)))
 report['symbols'][path.name]={'path':str(path),'bytes':path.stat().st_size,'nm_exit_code':process.returncode,
                             'selected':selected,'strings':strings,'stderr':process.stderr}
report['cann_headers']={}
for include_root in (pathlib.Path('/usr/local/Ascend/cann-8.5.1/include'),):
 if not include_root.is_dir():
  continue
 for path in include_root.rglob('*.h'):
  raw=path.read_bytes()
  if len(raw)>3000000:
   continue
  source=raw.decode('utf-8',errors='replace')
  if re.search(r'PTAGetExecCache|CanUsePTACache|AddTensorAddr|ACLNN_CACHE_LIMIT|UpdateTensorAddr',source):
   report['cann_headers'][str(path)]={'sha256':hashlib.sha256(raw).hexdigest(),'text':source}
report['top_level_source_candidates']={str(p):sorted(x.name for x in p.iterdir()) for p in
 (pathlib.Path('/vllm-workspace'),pathlib.Path('/opt'),pathlib.Path('/workspace')) if p.is_dir()}
report['cache_libraries']={}
cann_root=pathlib.Path('/usr/local/Ascend/cann-8.5.1')
report['resolver_library_enumeration']=[str(p) for p in (cann_root/'lib64').iterdir()
 if p.is_file() and p.name.startswith('libopapi_') and p.name.endswith('.so')]
torch_library=root/'lib/libtorch_npu.so'
route_strings=subprocess.run(['strings','-t','x',str(torch_library)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
report['resolver_strings']=[line for line in route_strings.stdout.splitlines() if re.search(
 r'lib(opapi|nnopbase|cust_opapi|aclnn)|GetOpApi|default custom|is found in|ASCEND_HOME_PATH|OPS_API|NN_RUNTIME',line)]
local_symbols=subprocess.run(['nm','-a','-C',str(torch_library)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
report['resolver_local_symbols']={'exit_code':local_symbols.returncode,'stderr':local_symbols.stderr,
 'selected':[line for line in local_symbols.stdout.splitlines() if re.search(r'GetOpApi|hit_cache|add_param_to_buf',line)]}
report['opapi_dependencies']={}
for path in [pathlib.Path(p) for p in report['resolver_library_enumeration']]+[cann_root/'lib64/libopapi.so']:
 proc=subprocess.run(['readelf','-d',str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 report['opapi_dependencies'][str(path)]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
   'dynamic':[line.strip() for line in proc.stdout.splitlines() if 'NEEDED' in line or 'SONAME' in line]}
 symbols=subprocess.run(['nm','-D','--defined-only',str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 report['opapi_dependencies'][str(path)]['selected_exports']=[line for line in symbols.stdout.splitlines()
   if re.search(r'\b(?:CanUsePTACache|aclnnAdd|aclnnAddGetWorkspaceSize|aclnnMatmulWeightNz|aclnnMatmulWeightNzGetWorkspaceSize|aclnnQuantMatmulWeightNz|aclnnQuantMatmulWeightNzGetWorkspaceSize)$',line)]
seen=set()
for path in list(cann_root.glob('lib64/lib*op*.so'))+[root/'lib/libtorch_npu.so']:
 path=path.resolve()
 if path in seen or not path.is_file():
  continue
 seen.add(path)
 symbols=subprocess.run(['nm','-D','-C',str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 selected=[line for line in symbols.stdout.splitlines() if re.search(r'PTA.*Cache|CanUsePTA|TensorAddrToCached|CacheLimit|OpExecCache::Run|GetOpApiFuncAddr|add_param_to_buf|CopyTypeV2',line)]
 strings=subprocess.run(['strings',str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 selected_strings=sorted(set(line for line in strings.stdout.splitlines() if re.search(r'CACHE|Cache',line) and re.search(r'PTA|ACLNN|LIMIT|DISABLE|ENABLE|BLACK|WHITE|cacheLimit|CacheLimit',line)))
 if not selected and not selected_strings:
  continue
 notes=subprocess.run(['readelf','-n',str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 item={'path':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'selected_symbols':selected,'selected_strings':selected_strings,'build_id':[line.strip() for line in notes.stdout.splitlines() if 'Build ID:' in line]}
 if any('CanUsePTACache' in line for line in selected):
  item['disassembly']={}
  for symbol in ['CanUsePTACache','PTAGetExecCache','PTAFindExecCache','AddTensorAddrToCachedList']:
   proc=subprocess.run(['objdump','-d','--disassemble='+symbol,str(path)],text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   item['disassembly'][symbol]={'stdout':proc.stdout,'stderr':proc.stderr,'exit_code':proc.returncode}
 report['cache_libraries'][str(path)]=item
print(json.dumps(report))
'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',default='quay.io/ascend/vllm-ascend:v0.18.0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise SystemExit('REFUSE_OVERWRITE')
    nonce=uuid.uuid4().hex[:12]
    name='worker-torch-cache-audit-'+nonce
    image=json.loads(subprocess.check_output(['docker','image','inspect',args.image],text=True))[0]
    command=['docker','run','--rm','--name',name,'--label','demoncat.owner=worker-positive-torch-cache-audit',
             '--label','demoncat.nonce='+nonce,'--runtime','runc','--network','none','--read-only',
             '--cpus','2','--memory','2g','--pids-limit','128','--env','ASCEND_VISIBLE_DEVICES=void',
             '--env','LD_PRELOAD=','--mount','type=bind,src=/dev/null,dst=/etc/ld.so.preload,readonly',
             '--entrypoint','python3',args.image,'-c',READER]
    result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)
    if result.returncode:
        raise SystemExit(result.stderr[-3000:])
    report=json.loads(result.stdout)
    report['image_id']=image['Id']
    report['container']={'name':name,'runtime':'runc','devices':[],'network':'none','rootfs':'read-only','auto_remove':True}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream:
        stream.write(gzip.compress((json.dumps(report)+'\n').encode()))
    print(json.dumps({'output':str(args.output),'version':report['distribution_version'],
                      'matched_files':list(report['files']),'headers_searched':report['headers_searched'],
                      'selected_symbols':{k:len(v['selected']) for k,v in report['symbols'].items()}}))

if __name__=='__main__':
    main()
