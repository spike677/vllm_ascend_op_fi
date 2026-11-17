"""Collect native msprobe statistics around the actual Provider model replays."""
import hashlib
import json
from pathlib import Path


def main():
    import msprobe
    from msprobe.pytorch import PrecisionDebugger
    from demoncat.runtime.vllm_ascend.worker_extension.providers import tensor_modules
    import tensor_provider_probe

    root = Path('/evidence/statistics')
    root.mkdir(parents=True, exist_ok=False)
    config = dict(task='statistics', level='mix', precision='high', rank=[], step=[0, 3],
                  dump_path=str(root/'observation'), async_dump=False,
                  statistics=dict(scope=[], list=[], data_mode=['all'], summary_mode='statistics', tensor_list=[]))
    path=root/'config.json'; path.write_text(json.dumps(config, indent=2)+'\n')
    package=Path(msprobe.__file__).resolve().parent
    files={p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
           for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml')}
    (root/'collector_identity.json').write_text(json.dumps(dict(package=str(package), files=files,
        edited_by_experiment=False, step_meaning={'0':'baseline','1':'baseline_repeat','2':'lossless','3':'fault','4':'off'},
        complete_model_api_module=True), indent=2)+'\n')
    debugger=PrecisionDebugger(config_path=str(path))
    original=tensor_modules.install_tensor_replay
    actual_phases=[]
    def install(worker, mechanism, **kwargs):
        replay=kwargs['replay']
        def observed(model, phase):
            actual_phases.append(phase)
            debugger.start(model=model,rank_id=0)
            try:
                return replay(model, phase)
            finally:
                debugger.stop()
                debugger.step()
        return original(worker, mechanism, **dict(kwargs,replay=observed))
    tensor_modules.install_tensor_replay=install
    try:
        tensor_provider_probe.main()
    finally:
        tensor_modules.install_tensor_replay=original
        (root/'completion.json').write_text(json.dumps(dict(actual_phases=actual_phases,
            recorded_steps=[0,3], native_logs_used_for_first_difference=False),indent=2)+'\n')


if __name__=='__main__':main()
