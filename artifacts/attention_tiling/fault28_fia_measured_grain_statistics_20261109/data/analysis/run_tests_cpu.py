"""Run only the analytic/export tests, with a task-owned writable temp folder."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent;root=here.parent
    tmp=root/'outputs/cpu_test_tmp';tmp.mkdir(parents=True,exist_ok=True);tempfile.tempdir=str(tmp)
    sys.path[:0]=[str(here),str(root)]
    modules=['test_device_trace','test_graph_evidence','test_native_plan','test_package_evidence']
    suite=unittest.defaultTestLoader.loadTestsFromNames(modules);stream=io.StringIO()
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'status':'PASS_CPU_ANALYSIS_AND_NEGATIVE_FIXTURES' if result.wasSuccessful() else 'FAIL_CPU_ANALYSIS',
        'tests':result.testsRun,'skipped':len(result.skipped),'errors':len(result.errors),'failures':len(result.failures),
        'test_sources':{m+'.py':hashlib.sha256((here/(m+'.py')).read_bytes()).hexdigest() for m in modules},
        'checked_source_files':{name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in
            ('device_trace.py','graph_evidence.py','native_plan.py','package_evidence.py')},
        'fixture_sha256':hashlib.sha256((here/'fixtures/device_sdk_stdout.txt').read_bytes()).hexdigest(),
        'output':stream.getvalue(),'npu_calls':0}
    with a.output.open('x') as file:json.dump(report,file,indent=2);file.write('\n')
    print(json.dumps({k:report[k] for k in ('status','tests','errors','failures','skipped')}))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
