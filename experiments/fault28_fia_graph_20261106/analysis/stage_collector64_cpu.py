"""Create a fresh task-local synchronous FP64 statistics observer, no NPU use.

Only four summaries of detached CPU temporaries change. No op input/output,
metadata, tensor storage or system msprobe package is modified or saved.
"""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path('/home/fault28-fia-graph-20261106')
PARENT=Path('/home/fault28-native-tiling-20261029/collector_high_a1')
OFFICIAL=Path('/home/batch-scale-dynamic-msprobe-20261023/python_packages')
PROCESSOR='msprobe/core/dump/data_dump/data_processor/pytorch_processor.py'
STAGE=ROOT/'collector_fp64_a1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not STAGE.exists()
    assert sha(PARENT/'stage_manifest.json')=='743354feff5a5f91a17e68695bad167fa2584a48d257ec227d0973d071f185af'
    original=OFFICIAL/PROCESSOR
    assert sha(original)=='e981aa6fe9afa1e7a1f0654db44166d7c199674b0727d8d7ffa40626b1074011'
    text=original.read_text()
    anchor='        if torch.is_complex(data_clone):'
    assert text.count(anchor)==1
    addition='''        # Task-local synchronous observer: all real floating tensors use
        # detached CPU FP64 temporaries, with no shape/operator/mode selection.
        # Copy first, convert on CPU; never request float64 NPU arithmetic.
        if precision == Const.DUMP_PRECISION_HIGH and data_clone.is_floating_point():
            if async_dump:
                logger.warning("Task-local CPU FP64 statistics require synchronous dump.")
                return tensor_stat
            observed = data_clone.to(device="cpu").to(dtype=torch.float64)
            tensor_stat.max = torch.max(observed).item()
            tensor_stat.min = torch.min(observed).item()
            tensor_stat.mean = torch.mean(observed).item()
            tensor_stat.norm = torch.linalg.vector_norm(observed).item()
            return tensor_stat

'''
    patched=text.replace(anchor,addition+anchor)
    compile(patched,str(STAGE/'packages'/PROCESSOR),'exec')
    # Parent package inventory is fully verified before copying dependencies.
    parent_manifest=json.loads((PARENT/'stage_manifest.json').read_bytes())
    for row in parent_manifest['artifacts']:
        assert sha(PARENT/row['path'])==row['sha256']
    STAGE.mkdir()
    shutil.copytree(PARENT/'packages',STAGE/'packages',symlinks=False)
    destination=STAGE/'packages'/PROCESSOR
    destination.write_text(patched,encoding='utf-8',newline='\n')
    patch=''.join(difflib.unified_diff(text.splitlines(True),patched.splitlines(True),
                                     fromfile='official/'+PROCESSOR,tofile='task_cpu_fp64/'+PROCESSOR))
    (STAGE/'collector.patch').write_text(patch,encoding='utf-8',newline='\n')
    source={'official_processor_sha256':sha(original),'processor_sha256':sha(destination),
            'builder_sha256':sha(Path(__file__)),'parent_stage_manifest_sha256':sha(PARENT/'stage_manifest.json')}
    (STAGE/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n')
    artifacts=[]
    for p in sorted(STAGE.rglob('*')):
        if p.is_file():
            assert not p.is_symlink()
            artifacts.append({'path':p.relative_to(STAGE).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    manifest={'schema':'fault28.msprobe.collector_cpu_fp64.v1','variant':'task_local_cpu_fp64_high_v1',
        'stage_root':'/audit/collector_fp64_a1','precision':'high',
        'original_processor_sha256':sha(original),'patched_processor_sha256':sha(destination),
        'artifacts':artifacts,'npu_executed':False,
        'provenance':'Official msprobe26.1.0 processor plus uniform synchronous CPU FP64 summary branch; dependencies copied from verified prior task-local stage.',
        'policy':{'scope':'all real floating tensors when precision=high, synchronous only',
                  'statistics':['Max','Min','Mean','Norm'],'tensor_metadata_changed':False,
                  'operator_data_changed':False,'tensor_payload_saved':False,
                  'reason':'Actual 81-element FIA difference was hidden at its output by FP32 summary reduction.'}}
    (STAGE/'stage_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'STAGED_NOT_NPU_VALIDATED','manifest_sha256':sha(STAGE/'stage_manifest.json'),
                      'processor_sha256':sha(destination),'files':len(artifacts),'npu_executed':False}))


if __name__=='__main__':main()
