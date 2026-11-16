"""Compare declarations, not marketing versions. No runtime certification."""
import argparse
import json
from pathlib import Path
import re


def parameter_types(declaration):
    # These inspected ACLNN declarations have only scalar and pointer params,
    # not function pointers, templates with commas, or default arguments.
    return tuple(re.sub(r'\s+','',re.sub(r'\b[A-Za-z_]\w*\s*$','',item))
                 for item in declaration.split(','))


def compare(files):
    versions={}
    for file in files:
        for row in json.loads(file.read_text())['toolkits']:
            version=row['installation']['ascend_toolkit_install.info']['version']
            if version in versions:raise ValueError('Duplicate actual version; select one exact installation')
            versions[version]=dict(row,source=file.name)
    symbols=sorted({name for row in versions.values() for name in row['exports']} |
                   {name for row in versions.values() for name in row['signatures']})
    comparison={}
    for symbol in symbols:
        groups={};availability={}
        for version,row in versions.items():
            declarations=row['signatures'].get(symbol,[])
            distinct={parameter_types(x['parameters']) for x in declarations}
            if len(distinct)>1:raise ValueError('Conflicting installed declarations: '+symbol)
            types=next(iter(distinct),())
            groups.setdefault(types,[]).append(version)
            availability[version]=dict(libraries=row['exports'].get(symbol,[]),declaration_found=bool(declarations))
        comparison[symbol]=dict(availability=availability,signature_groups=[dict(parameter_types=list(types),versions=vs)
                                                                          for types,vs in groups.items()])
    return dict(schema='demoncat.cann.interface_comparison.v1',actual_npu_calls=0,
        status='INTERFACE_FACTS_ONLY',versions={v:dict(source=r['source'],installation=r['installation'],abi_facts=r.get('abi_facts'))
                                              for v,r in versions.items()},symbols=comparison,
        limits=['Matching sizeof is not proof of matching field layout or runtime cache semantics.',
                'Header declarations and exports do not prove actual model dispatch or kernel compatibility.',
                'No evidence for uninstalled 8.x/9.x releases; no major-version blanket certification.'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('files',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=compare(a.files)
    with a.output.open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(dict(versions=list(r['versions']),symbols=len(r['symbols']),status=r['status'])))


if __name__=='__main__':main()
