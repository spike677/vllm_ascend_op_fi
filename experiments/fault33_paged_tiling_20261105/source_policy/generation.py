"""Bound every fresh build/stage generation to distinct task-private paths."""
from pathlib import Path
import re


def layout(root, generation):
    if not re.fullmatch(r'a[1-9][0-9]{0,2}', generation):
        raise ValueError('Generation must be a1..a999')
    root = Path(root)
    return {
        'source': root/f'source_{generation}/ops-transformer-v9.1.0',
        'dependencies': root/f'dependencies_{generation}',
        'python_deps': root/f'build/python_deps_{generation}',
        'evidence': root/f'build/evidence_{generation}',
        'stage': root/f'vendor_stage_{generation}',
    }
