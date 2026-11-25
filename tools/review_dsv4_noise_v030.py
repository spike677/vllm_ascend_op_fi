"""Run on private full-array observations; print only bounded summary."""
import argparse,json,hashlib
from pathlib import Path
from fault_injection.dsv4_noise_v030 import calibrate,review

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--output');a=p.parse_args()
    raw=Path(a.input).read_bytes();d=json.loads(raw)
    r=review(d['combination'],d['calibration'],d['phases']) if 'phases' in d else calibrate(d['combination'],d['calibration'])
    r['source_receipt_sha256']=hashlib.sha256(raw).hexdigest()
    # Per-rank/step details remain in the local JSON when the stdout envelope is too large.
    text=json.dumps(r,indent=2,allow_nan=False)+'\n'
    if a.output:Path(a.output).write_bytes(text.encode())
    if len(text.encode())>60000:raise ValueError('NOISE_SUMMARY_EXCEEDS_TEXT_LIMIT_SPLIT_BY_PROMPT')
    print(text)
if __name__=='__main__':main()
