"""Serial acceptance wrapper: two fresh owners, exactly one standard Worker.

The reviewed single-scope driver retains its fault/recovery protocol. This
wrapper never performs injection itself and stops after the first failed scope.
"""
import json
from pathlib import Path
import sys
from experiments.worker_core_v11_20261118 import probe_quant_ordinary as single


def main(argv=None):
    root = Path('/evidence')
    result = dict(status='STARTED', scopes=[], qualification_granted=False,
                  main_project_integration_verified=False)
    held = []
    def observe(llm):
        if held and held[0] is not llm:
            raise RuntimeError('SECOND_WORKER_CONSTRUCTION_FORBIDDEN')
        if not held: held.append(llm)
    def save():
        (root/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    try:
        save()
        for index in range(2):
            folder = root/('scope'+str(index));folder.mkdir(exist_ok=False)
            code = single.main(argv, existing_llm=held[0] if held else None,
                               evidence_dir=folder, llm_observer=observe)
            row = json.loads((folder/'result.json').read_bytes())
            host = json.loads((folder/'host-core-receipts.json').read_bytes())
            result['scopes'].append(dict(index=index, result_path=str(folder/'result.json'),
                result_sha256=single.sha_file(folder/'result.json'),
                host_sha256=single.sha_file(folder/'host-core-receipts.json'),
                status=row['status'], owner=host['owner'], worker=host['worker'],
                ownership_retained=host['ownership_retained']))
            save()
            if code != 0 or host['ownership_retained']:
                raise RuntimeError('STOP_AFTER_FIRST_FAILED_SCOPE')
        a,b=result['scopes']
        if a['worker']!=b['worker'] or a['owner']==b['owner'] or len(held)!=1:
            raise RuntimeError('SAME_WORKER_DISTINCT_OWNERS_REQUIRED')
        result.update(status='TWO_SAME_WORKER_SCOPES_READY_FOR_REVIEW',
                      same_worker=True, independent_owners=True)
    except Exception as error:
        result.update(status='FAILED_WITH_EVIDENCE',error=repr(error))
    finally: save()
    return 0 if result['status']=='TWO_SAME_WORKER_SCOPES_READY_FOR_REVIEW' else 2

if __name__=='__main__':raise SystemExit(main(sys.argv[1:]))
