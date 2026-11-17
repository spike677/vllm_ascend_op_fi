"""Strict independent first difference; only raw statistics and metadata inputs."""
import argparse
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser()
    for flag in ('source','baseline','fault','baseline-config','fault-config',
                 'baseline-collector','fault-collector','out'):
        p.add_argument('--'+flag,type=Path,required=True)
    a=p.parse_args();sys.path[:0]=[str(a.source),str(a.source/'fault39/src')]
    from fault_injection.evidence import independent_statistics_pair
    report=independent_statistics_pair(comparator=a.source/'fault39/tools/compare_msprobe.py',
        baseline=a.baseline,fault=a.fault,baseline_config=a.baseline_config,fault_config=a.fault_config,
        baseline_collector=a.baseline_collector,fault_collector=a.fault_collector)
    with a.out.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('first_difference_status','whole_dump_comparable')},allow_nan=False))


if __name__=='__main__':main()
