"""Assemble a private real capture; does not synthesize observations or tokens."""
import argparse,json
from pathlib import Path
from fault_injection.dsv4_noise import digest,validate_run


def assemble_run(capture,*,tokens,prompt_token_ids,single_chunk,sample_id,process_identity):
    if type(capture) is not list or sorted(r.get('rank',-1) for r in capture)!=list(range(8)):
        raise ValueError('CAPTURE_EXACT_EIGHT_RANKS_REQUIRED')
    if type(prompt_token_ids) is not list or any(type(t) is not int or t<0 for t in prompt_token_ids):
        raise ValueError('ACTUAL_PROMPT_TOKENS_REQUIRED')
    for r in capture:
        if r.get('phase')!='calibration' or r.get('capture_removed') is not True or type(r.get('worker_pid')) is not int:
            raise ValueError('CAPTURE_CALIBRATION_CLEANUP_IDENTITY_REQUIRED')
    run=dict(tokens=tokens,prompt_tokens=len(prompt_token_ids),prompt_sha256=digest(prompt_token_ids),
        single_chunk=single_chunk,sample_id=sample_id,process_identity=process_identity,
        ranks={str(r['rank']):r['steps'] for r in capture},capture_sha256=digest(capture),
        worker_pids={str(r['rank']):r['worker_pid'] for r in capture})
    validate_run(run);return run


def main():
    p=argparse.ArgumentParser();p.add_argument('private_capture_envelope');a=p.parse_args()
    print(json.dumps(assemble_run(**json.loads(Path(a.private_capture_envelope).read_bytes())),allow_nan=False))


if __name__=='__main__':main()
