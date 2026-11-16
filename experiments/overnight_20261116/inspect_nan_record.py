"""Read the previously blocking NaN record; never remove it from comparison."""
import hashlib
import json
from pathlib import Path

root = Path('/home/provider-adaptation-20261116/provider26-final-a2')
out = {}
for arm in ('baseline','fault'):
    directory=root/f'provider26_qmm_model_{arm}'/arm/'rank1/observation/step0/rank1'
    raw=(directory/'dump.json').read_bytes(); data=json.loads(raw)['data']
    name='Tensor.__setitem__.20.forward'
    keys=list(data); index=keys.index(name)
    stack=json.loads((directory/'stack.json').read_text())
    frames=[value[1] for value in stack.values() if isinstance(value,list) and len(value)==2 and name in value[0]]
    out[arm]=dict(path=str(directory),dump_sha256=hashlib.sha256(raw).hexdigest(),
        record=data[name], stack=frames,
        adjacent_records={k:data[k] for k in keys[max(0,index-2):index+3]})
print(json.dumps(out,indent=2))
