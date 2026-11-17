"""Publish only normal pretrained admission summaries, never fault qualification."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--case', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--download', type=Path)
args = parser.parse_args()
path = args.case / 'result.json'
result = json.loads(path.read_text())
if result['status'] != 'PASS_PRETRAINED_NPU_INFERENCE_NOT_FAULT_ACCEPTANCE':
    raise RuntimeError('Completed normal inference receipt required')
loaded, = result['loaded']
observation, = result['observation']
requests = [[{key: row[key] for key in ('prompt_token_ids','tokens','text')} for row in group]
            for group in result['requests']]
summary = dict(schema='demoncat.pretrained_admission.summary.v1', status=result['status'],
    source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    actual_model_class=loaded['actual_model_class'], actual_model_type=loaded['actual_model_type'],
    actual_model_source_sha256=loaded['source_sha256'], boundary=loaded['boundary'],
    loaded_from_pretrained=loaded['loaded_from_pretrained'], actual_fault_injection=False,
    requests=requests, complete_logits=observation['complete_logits'],
    actual_matmul_runs=observation['actual_matmul_runs'], native_idle=observation['native_idle'],
    build_profile_sha256=observation['build_profile_sha256'],
    limits=['Two deterministic 4-token generations, TP1/eager/BF16 on 124 A2/CANN8.5.1',
            'No native fault qualification; no model quality or all-layer coverage claim'])
if args.download:
    download = json.loads(args.download.read_text())
    if download['status'] != 'COMPLETE_CHECKSUM_VERIFIED_NOT_NPU_ACCEPTANCE':
        raise RuntimeError('Complete public download checksums required')
    summary['public_checkpoint'] = dict(repository='zai-org/GLM-4-9B-0414',
        revision=download['revision'], download_receipt_sha256=hashlib.sha256(args.download.read_bytes()).hexdigest(),
        files=download['completed'], remote_code_executed=False)
args.out.write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
