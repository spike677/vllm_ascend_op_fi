"""Separate conformance and legacy compatibility receipts, no NPU execution."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest

ROOT=Path(__file__).resolve().parents[2]
LEGACY=('test_capabilities','test_catalog_cli','test_integration_contract','test_lifecycle',
        'test_alias_auto','test_module_faults','test_route8_index22','test_collective25',
        'test_fault_contracts','test_public_interface','test_strategy_bridge','test_fia_masks',
        'test_quant_auto','test_attention_interface','test_native_bindings','test_cleanup_adapters',
        'test_tensor_worker_providers','test_attention28_prefill_boundary')


def run():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    for relative in ('.','src','fault39/src','tests','tests/worker_extension'):sys.path.insert(0,str(ROOT/relative))
    suites={'adapter_conformance':unittest.TestLoader().discover(str(Path(__file__).parent),pattern='test_*.py'),
            'legacy_compatibility':unittest.TestLoader().loadTestsFromNames(LEGACY)}
    results={}
    for label,suite in suites.items():
        start=time.monotonic();result=unittest.TextTestRunner(verbosity=1).run(suite)
        results[label]=dict(tests_run=result.testsRun,passed=result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
            failures=len(result.failures),errors=len(result.errors),
            skipped=[dict(test=str(t),reason=r) for t,r in result.skipped],
            details=[dict(test=str(t),traceback=s) for t,s in result.failures+result.errors],
            success=result.wasSuccessful(),seconds=round(time.monotonic()-start,3))
    sources={}
    for relative in ('fault_registry','demoncat','fault_injection','tests','engineering_closure'):
        for path in sorted((ROOT/relative).rglob('*')):
            if (path.is_file() and path.suffix in ('.py','.json','.yaml')
                    and '__pycache__' not in path.parts and 'validation' not in path.parts):
                sources[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    for relative in ('pyproject.toml','experiments/c2c3_closure_20261117/package_source.py'):
        sources[relative]=hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
    report=dict(schema='demoncat.unified_validation.v1',baseline_commit='9dfdb0c6e0e900f8a67d04dcafc53544bc7fc477',
        recorded_at=datetime.now(timezone.utc).isoformat(),python=sys.version,platform=platform.platform(),
        suites=results,source_sha256=sources,npu_calls=0,qualification_granted=False)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps({name:{k:v for k,v in item.items() if k!='details'} for name,item in results.items()}))
    return 0 if all(r['success'] for r in results.values()) else 1


if __name__=='__main__':raise SystemExit(run())
