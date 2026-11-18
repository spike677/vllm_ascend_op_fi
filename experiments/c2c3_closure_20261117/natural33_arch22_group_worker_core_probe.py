"""Public all-target C2, only after this final package's fresh all-target normal."""
import sys
import natural33_arch22_s2_worker_core_probe as common

PHASES=('reference','lossless','qualification_off')+tuple(
    name+f'_{index:02d}' for index in range(28) for name in ('fault','off'))+('cleanup_health',)


def setup(worker,expected,prompt,proof_args,native_args,execution):
    return common.setup(worker,expected,prompt,proof_args,native_args,execution,group=True)


if __name__=='__main__':raise SystemExit(common.main(protocol=sys.modules[__name__]))
