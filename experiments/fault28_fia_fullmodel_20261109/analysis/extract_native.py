"""Original-byte bounded trace excerpts; raw DEBUG log stays private."""
import argparse
import hashlib
import json
from pathlib import Path

TERMS=(b'[GenTilingKey]',b'FillTaskDefAfterGenTask:Set kernel_name[',
       b'StreamLaunchKernelV2: kernel info :',b'KernelTaskInfo Distribute Success, op:')


def extract(source,target):
    digest=hashlib.sha256();offset=0;records=[];target.mkdir(parents=True,exist_ok=False)
    with source.open('rb') as stream,(target/'native_lines.txt').open('xb') as dest:
        for number,line in enumerate(stream,1):
            digest.update(line)
            marker=line.startswith(b'{') and b'fault28.fia_graph.call_marker.v1' in line
            native=(b'FusedInferAttentionScore' in line or b'te_fusedinferattentionscore_' in line) and any(t in line for t in TERMS)
            scalar=b'RUNTIME(' in line and b'ParsePrintToLog: PrintInfo:' in line and b'F28_GRAIN_V1' in line
            if marker or native or scalar:
                dest.write(line);records.append({'original_line':number,'offset':offset,'bytes':len(line),
                    'sha256':hashlib.sha256(line).hexdigest()})
            offset+=len(line)
    receipt={'schema':'fault28.fia_fullmodel.native_line_provenance.v1',
        'source_log':str(source.resolve()),'source_log_sha256':digest.hexdigest(),
        'source_bytes':offset,'original_lines':records,'original_line_bytes_preserved':True,
        'extract_sha256':hashlib.sha256((target/'native_lines.txt').read_bytes()).hexdigest()}
    (target/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=extract(a.source,a.output);print(json.dumps({k:r[k] for k in ('source_log_sha256','source_bytes','extract_sha256')}))
