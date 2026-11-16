"""Identical raw statistics/mix capture for both arms; no tensor files."""
import hashlib
import json
from pathlib import Path

def capture(folder):
    import msprobe
    from msprobe.pytorch import PrecisionDebugger
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
    config=dict(task='statistics',level='mix',precision='high',rank=[],step=[0,1,2],
        dump_path=str(folder/'observation'),async_dump=False,
        statistics=dict(scope=[],list=[],data_mode=['all'],summary_mode='statistics',tensor_list=[]))
    (folder/'config.json').write_text(json.dumps(config,indent=2))
    package=Path(msprobe.__file__).parent
    files={p.relative_to(package).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in package.rglob('*.py')}
    (folder/'collector_identity.json').write_text(json.dumps(dict(package=str(package),files=files),indent=2))
    return PrecisionDebugger(config_path=str(folder/'config.json'))
