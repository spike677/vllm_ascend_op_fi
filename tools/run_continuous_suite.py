"""Run one fresh dedicated process per mechanism; retain failures and quarantine."""
import argparse,json,subprocess,sys
from pathlib import Path


def run_cases(cases, execute):
    results=[];safe=True
    for case in cases:
        if not safe:
            results.append(dict(case=case,status='NOT_RUN_UNRESOLVED_RECOVERY'));continue
        row=execute(case);results.append(dict(case=case,**row))
        # No inference continues on an unproven failed device/Worker state.
        safe=row['passed'] or row.get('cleanup_verified') is True
    return dict(schema='demoncat.continuous.suite.v1',cases=results,passed=all(r.get('passed') is True for r in results))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mechanisms',nargs='+',required=True)
    p.add_argument('--evidence-root',required=True);p.add_argument('--probe',required=True)
    a,extra=p.parse_known_args();root=Path(a.evidence_root);root.mkdir(parents=True,exist_ok=True)
    def execute(case):
        folder=root/case
        if folder.exists():raise RuntimeError('EVIDENCE_MUST_NOT_BE_OVERWRITTEN')
        folder.mkdir()
        with (folder/'process.log').open('w') as log:
            proc=subprocess.run([sys.executable,a.probe,'--mechanism',case,'--evidence',str(folder),*extra],stdout=log,stderr=subprocess.STDOUT)
        path=folder/'continuous_public.json'
        receipt=json.loads(path.read_text()) if path.exists() else {}
        cleanup=receipt.get('failure_cleanup',[])
        passed=proc.returncode==0 and receipt.get('status')=='PUBLIC_TWO_SCOPES_COMPLETED_PENDING_INDEPENDENT_REVIEW'
        return dict(passed=passed,exit_code=proc.returncode,error=receipt.get('error'),
            cleanup_verified=bool(cleanup) and all(r.get('clean') is True and r.get('leases')==r.get('owners')==0 for r in cleanup),
            receipt=str(path))
    result=run_cases(a.mechanisms,execute);(root/'suite.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result));return 0 if result['passed'] else 2

if __name__=='__main__':raise SystemExit(main())
