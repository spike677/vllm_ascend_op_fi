"""Disclosed graph-output observer; it does not intercept/relabel a native API.

Output slot 0 is an alias of the actual graph FIA output. Independent FX/GE
edges must prove this mapping before using these statistics to locate FIA.
"""
import hashlib
import importlib
import json
from pathlib import Path
from collector_stage import verify_loaded


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def configure(out, collector):
    from msprobe.pytorch import PrecisionDebugger
    processor = importlib.import_module('msprobe.core.dump.data_dump.data_processor.pytorch_processor')
    collector['actual_loaded'] = verify_loaded(collector, processor)
    save(out/'collector_provenance.json', collector)
    config = {'task':'statistics','level':'mix','rank':[],'step':[0,1],
              'dump_path':str(out/'observation'),'async_dump':False,
              'statistics':{'scope':[],'list':[],'data_mode':['all'],
                            'summary_mode':'statistics','tensor_list':[]}}
    if collector['precision']=='high':
        config['precision']='high'
    save(out/'config.json',config)
    return PrecisionDebugger(config_path=str(out/'config.json'))


def receipt(out):
    files, steps = [], []
    for step in (0,1):
        directory=out/'observation'/f'step{step}'/'rank0'
        objects={}
        for name in ('dump','stack','construct'):
            path=directory/(name+'.json')
            objects[name]=json.loads(path.read_bytes())
            files.append({'path':path.relative_to(out).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)})
        dump=objects['dump']
        assert dump['task']=='statistics' and dump['level']=='mix' and dump.get('dump_data_dir') is None
        boundaries=[key for key in dump['data'] if 'GraphBoundary' in key and key.startswith('Module.')]
        assert len(boundaries)==1, boundaries
        assert boundaries[0] in objects['construct']
        steps.append({'step':step,'graph_call_id':step+1,'boundary_node':boundaries[0],
                      'raw_nodes':len(dump['data'])})
    assert not any(p.suffix.lower() in ('.pt','.pth','.bin','.npy','.npz','.safetensors')
                   for p in (out/'observation').rglob('*') if p.is_file())
    return {'schema':'fault28.fia_graph.statistics_mix.v1','files':files,'steps':steps,
            'config_sha256':sha(out/'config.json'),'collector_sha256':sha(out/'collector_provenance.json'),
            'scope':'GraphBoundary original module statistics. Slot 0 needs independent FX/GE alias verification; not an eager NPU API dump.'}
