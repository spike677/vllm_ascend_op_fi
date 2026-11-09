"""No device mount: verify checkpoint/source identities and fullgraph export."""
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import inspect


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


def main():
    import torch
    import torch_npu
    from transformers import AutoModelForCausalLM
    import model_adapter
    versions={n:importlib.metadata.version(n) for n in ('torch','torch-npu','transformers')}
    assert versions == {'torch':'2.10.0','torch-npu':'2.10.0.post4','transformers':'5.5.4'}
    manifest=json.loads(Path('/audit/model_manifest.json').read_bytes())
    for name,row in manifest['files'].items():
        p=Path('/models/Qwen3-0.6B')/name
        assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
    modules={}
    for name in ('transformers.models.qwen3.modeling_qwen3','transformers.modeling_utils',
                 'transformers.models.qwen3.configuration_qwen3','torchair'):
        module=importlib.import_module(name);p=Path(module.__file__)
        modules[name]={'path':str(p),'sha256':sha(p)}
    model=AutoModelForCausalLM.from_pretrained('/models/Qwen3-0.6B', dtype=torch.float16,
        attn_implementation='eager', local_files_only=True, trust_remote_code=False).eval()
    model_adapter.install(model)
    wrapper=model_adapter.FullModelGraph(model)
    ids=torch.zeros((1,2048),dtype=torch.long)
    positions=torch.arange(2048,dtype=torch.long).unsqueeze(0)
    mask=torch.triu(torch.ones((2048,2048),dtype=torch.bool),diagonal=1)
    # Dynamo export executes FakeTensor/meta kernels, not model kernels.
    exported=torch._dynamo.export(wrapper, aten_graph=False)(ids,positions,mask)
    nodes=[{'name':n.name,'op':n.op,'target':str(n.target),
            'module_stack':str(n.meta.get('nn_module_stack'))} for n in exported.graph_module.graph.nodes]
    fia=[n for n in nodes if n['target']=='npu.npu_fused_infer_attention_score.default']
    assert len(fia)==28, len(fia)
    payload={'status':'PASS_CPU_FULLGRAPH_EXPORT_NOT_NPU_ACCEPTED','versions':versions,
        'modules':modules,'fia_nodes':fia,'fx_nodes':len(nodes),'npu_calls':0,
        'device_mounts':[], 'model_checkpoint_verified':True,
        'adapter_source_sha256':sha(Path(model_adapter.__file__))}
    Path('/cpu_output/inventory.json').write_text(json.dumps(payload,indent=2)+'\n')
    Path('/cpu_output/fx_code.py').write_text(exported.graph_module.code)
    print(json.dumps(payload))


if __name__=='__main__':
    main()
