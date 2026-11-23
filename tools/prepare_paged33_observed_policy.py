"""Generate a closed A3 candidate policy from reviewed raw observation associations."""
import argparse,json
from pathlib import Path
from fault_injection.sites.paged33_observed_policy import prepare_policy
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('--fault-block',type=int,required=True);p.add_argument('--output',required=True);a=p.parse_args()
 policy=prepare_policy(json.loads(Path(a.input).read_bytes()),fault_block=a.fault_block)
 Path(a.output).write_bytes((json.dumps(policy,indent=2)+'\n').encode())
 print(json.dumps(dict(policy_sha256=policy['policy_sha256'],qualification_granted=False,public_entry_enabled=False)))
