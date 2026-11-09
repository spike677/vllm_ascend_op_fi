"""Fresh copy of an accepted private vendor, adding read-only device scalars.

No NPU call, source-tree mutation, system installation or previous-output copy.
The numerical host and kernel variant remain the published 2026-11-06 ones.
"""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

OLD=Path('/home/fault28-fia-graph-20261106')
ROOT=Path('/home/fault28-fia-grain-20261109')
OLD_MANIFEST='140ca2d1d2a5d1ebe8ae0c03598bc697c556a88ade483d714276f25c67e391dd'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert not ROOT.exists() and sha(OLD/'vendor_stage_a5/vendor_manifest.json')==OLD_MANIFEST
    manifest=json.loads((OLD/'vendor_stage_a5/vendor_manifest.json').read_bytes())
    vendor=OLD/manifest['vendor_root']
    for row in manifest['artifacts']:assert sha(vendor/row['path'])==row['sha256']
    ROOT.mkdir()
    shutil.copytree(OLD/'vendor_stage_a5',ROOT/'vendor_stage_a5',symlinks=True)
    shutil.copytree(OLD/'collector_fp64_a1',ROOT/'collector_fp64_a1',symlinks=True)
    (ROOT/'collector_high_a1').mkdir()
    (ROOT/'analysis').mkdir();(ROOT/'build/evidence_a4').mkdir(parents=True)
    for source in (OLD/'build/evidence_a4').iterdir():
        if source.is_file() and source.suffix in ('.json','.patch','.h'):
            shutil.copyfile(source,ROOT/'build/evidence_a4'/source.name)
    vendor=ROOT/manifest['vendor_root'];changes=[]
    files=sorted(vendor.rglob('fia_block_vec_nonquant.h'));assert len(files)==2
    anchor='    SoftMaxShapeInfo srcShape{dealRowCount, columnCount, dealRowCount, actualColumnCount};'
    code='''    // Diagnostic only: scalars actually consumed by SoftmaxFlashV2 below.
    // Sample request0/KVhead0/G0 and fixed query positions, never payloads.
    if (info.bIdx == 0 && info.n2Idx == 0 && info.gS1Idx < info.actS1Size) {
        const uint32_t f28Start = info.gS1Idx + mSplitInfo.nBufferStartM + mSplitInfo.vecStartM + startRow;
        const uint32_t f28End = f28Start + dealRowCount;
        if ((f28Start <= 63 && 63 < f28End) || (f28Start <= 255 && 255 < f28End) ||
            (f28Start <= 511 && 511 < f28End) || (f28Start <= 767 && 767 < f28End) ||
            (f28Start <= 1023 && 1023 < f28End) || (f28Start <= 1535 && 1535 < f28End) ||
            (f28Start <= 2047 && 2047 < f28End)) {
            AscendC::printf("F28_GRAIN_V1 block=%u m=%u tile=%u row=%u rows=%u s2=%u base=%u columns=%u actual=%u prune=%u brc=%u\\n",
                uint32_t(GetBlockIdx()), uint32_t(info.actMBaseSize), uint32_t(info.gS1Idx),
                f28Start, dealRowCount, uint32_t(info.s2Idx), uint32_t(constInfo.s2BaseSize),
                columnCount, actualColumnCount, uint32_t(FIAT::fault28CausalPrune), uint32_t(SOFTMAX_WITH_BRC));
        }
    }
'''
    for file in files:
        before=file.read_bytes();text=before.decode('utf-8')
        assert text.count(anchor)==1 and 'F28_GRAIN_V1' not in text
        includes='#include "kernel_operator_list_tensor_intf.h"'
        assert text.count(includes)==1
        text=text.replace(includes,'#include "kernel_operator.h" // AscendC::printf diagnostic interface\n'+includes)
        after=text.replace(anchor,code+anchor).encode('utf-8');file.write_bytes(after)
        changes.append({'path':file.relative_to(vendor).as_posix(),'before_sha256':hashlib.sha256(before).hexdigest(),
                        'after_sha256':sha(file),'bytes':len(after)})
    assert len({r['after_sha256'] for r in changes})==1
    patch=''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),
                                     fromfile='a/fia_block_vec_nonquant.h',tofile='b/fia_block_vec_nonquant.h'))
    (ROOT/'build/device_grain_trace.patch').write_text(patch)
    trace={'schema':'fault28.fia.device_scalar_observer.v1','status':'CPU_STAGED_NOT_NPU_VALIDATED',
           'original_vendor_manifest_sha256':OLD_MANIFEST,'changes':changes,
           'observation':'Actual SoftmaxFlashV2Compute scalar columnCount and actualColumnCount before unchanged srcShape/tiling/call.',
           'fixed_queries':[63,255,511,767,1023,1535,2047], 'tensor_payload_written':False,
           'host_binary_changed':False,'numerical_neutrality_requires_actual_NPU_comparison':True,
           'helper_sha256':sha(Path(__file__))}
    (ROOT/'build/device_grain_trace.json').write_text(json.dumps(trace,indent=2)+'\n')
    for row in manifest['artifacts']:
        file=vendor/row['path'];row.update(bytes=file.stat().st_size,sha256=sha(file))
    manifest.update(status='CPU_DEVICE_SCALAR_OBSERVER_STAGED_NOT_NPU_VALIDATED',
                    device_scalar_observer={'manifest':'build/device_grain_trace.json',
                                            'sha256':sha(ROOT/'build/device_grain_trace.json')})
    (ROOT/'vendor_stage_a5/vendor_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assert sha(OLD/'vendor_stage_a5/vendor_manifest.json')==OLD_MANIFEST
    print(json.dumps({'status':trace['status'],'vendor_manifest_sha256':sha(ROOT/'vendor_stage_a5/vendor_manifest.json'),
                      'changed_kernel_source_copies':len(changes),'host_changed':False,'npu_calls':0}))


if __name__=='__main__':main()
