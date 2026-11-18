"""Separate all-object S2-capable host; a build never authorizes F.

Only the original S2 selection changes, before unmodified SplitCore, workspace,
serializer and kernel arithmetic. The final package requires a new complete
stock/private normal pair, then live controls and one owned current-call permit.
"""
import importlib.util
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_group_normal_recipe',HERE/'apply_tnd33_arch22_all_normal.py')
all_normal=importlib.util.module_from_spec(spec);sys.modules[spec.name]=all_normal;spec.loader.exec_module(all_normal)
normal,base=all_normal.normal,all_normal.base
VARIANT='tnd_page16_arch22_group_s2_256'
VENDOR='dcat_f33_arch22_group_s2'
HEADER='paged33_arch22_group_s2.h'
SCHEMA='demoncat.paged33.arch22_group_s2_source.v1'
SELECTED_SCOPE=[512,256]
KEYS,PATHS,SOURCE_SHA=all_normal.KEYS,all_normal.PATHS,all_normal.SOURCE_SHA
verify_source=all_normal.verify_source


def patched(original):
    return {path:text.replace('paged33_arch22_all_normal','paged33_arch22_group_s2')
            for path,text in all_normal.patched(original).items()}


def main():all_normal.main(recipe=sys.modules[__name__])


if __name__=='__main__':main()
