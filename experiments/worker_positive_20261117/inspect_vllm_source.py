"""Read installed vLLM source in an owned device-free CPU container.

No vLLM/torch/torch_npu code is imported. This reads files/metadata only.
The container uses runc, no devices, no network and is automatically removed.
Run on the image host; output is a private source audit, not a support certificate.
"""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import uuid


READER = r'''
import hashlib, importlib.metadata, json, pathlib, sys, urllib.parse
roots = [pathlib.Path(p) for p in sys.path if p and pathlib.Path(p).is_dir()]
paths = {
 'vllm': [
  'forward_context.py', 'config/cache.py', 'config/model.py', 'config/compilation.py', 'config/scheduler.py',
  'v1/worker/gpu_input_batch.py', 'v1/worker/gpu_model_runner.py',
  'v1/worker/gpu_worker.py', 'v1/worker/worker_base.py',
  'v1/core/sched/output.py', 'v1/core/sched/scheduler.py',
  'v1/engine/llm_engine.py', 'v1/engine/core.py', 'v1/engine/core_client.py',
  'v1/engine/input_processor.py', 'v1/engine/output_processor.py', 'envs.py',
  'entrypoints/llm.py', 'v1/request.py', 'model_executor/models/registry.py',
  'model_executor/models/glm.py', 'model_executor/models/glm4.py', 'model_executor/models/glm4_moe.py',
  'model_executor/models/deepseek_v2.py', 'model_executor/models/qwen3.py', 'model_executor/models/llama.py',
  'model_executor/layers/logits_processor.py', 'v1/sample/sampler.py',
  'distributed/parallel_state.py', 'v1/outputs.py', 'v1/engine/__init__.py'],
 'vllm_ascend': [
  'worker/model_runner_v1.py', 'worker/model_runner.py', 'worker/worker_v1.py',
  'worker/model_runner_v2.py', 'utils.py', 'compilation/acl_graph.py',
  'attention/attention_v1.py', 'ops/rotary_embedding.py', 'ascend_forward_context.py',
  'worker/npu_input_batch.py', 'worker/worker.py', 'envs.py', 'platform.py',
  'worker/block_table.py', 'quantization/utils.py']}
report = {'schema_version': 1, 'no_runtime_imports': True, 'npu_calls': 0,
          'metadata': {}, 'packages': {}, 'files': {}, 'missing': []}
for distribution in ('vllm', 'vllm-ascend', 'torch', 'torch-npu', 'transformers'):
 try:
  d = importlib.metadata.distribution(distribution)
  report['metadata'][distribution] = {'version': d.version,
    'metadata_path': str(getattr(d, '_path', ''))}
  direct = d.read_text('direct_url.json')
  if direct:
   direct = json.loads(direct)
   report['metadata'][distribution]['direct_url'] = direct
   url = urllib.parse.urlparse(direct.get('url', ''))
   if url.scheme == 'file':
    candidate = pathlib.Path(urllib.parse.unquote(url.path))
    if candidate.is_dir():
     roots.append(candidate)
 except importlib.metadata.PackageNotFoundError:
  report['metadata'][distribution] = {'status': 'NOT_FOUND'}
for package, rels in paths.items():
 root = next((r / package for r in roots if (r / package).is_dir()), None)
 report['packages'][package] = {'root': str(root) if root else None}
 if root is None:
  report['missing'].append(package)
  continue
 # List only package-relative file names: discover moved runner/context files.
 report['packages'][package]['candidate_paths'] = sorted(str(p.relative_to(root))
  for p in root.rglob('*.py') if any(x in p.name.lower() for x in
   ('runner', 'input_batch', 'context', 'worker', 'deepseek', 'glm', 'logits')))
 for rel in rels:
  path = root / rel
  key = package + '/' + rel
  if not path.is_file():
   report['missing'].append(key)
   continue
  raw = path.read_bytes()
  report['files'][key] = {'path': str(path), 'bytes': len(raw),
   'sha256': hashlib.sha256(raw).hexdigest(), 'text': raw.decode('utf-8')}
print(json.dumps(report, ensure_ascii=False))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="quay.io/ascend/vllm-ascend:v0.18.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("REFUSE_OVERWRITE")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", args.image], text=True))[0]
    nonce = uuid.uuid4().hex[:12]
    name = "worker-source-audit-" + nonce
    command = ["docker", "run", "--rm", "--name", name,
               "--label", "demoncat.owner=worker-positive-source-audit",
               "--label", "demoncat.nonce=" + nonce,
               "--runtime", "runc", "--network", "none", "--read-only",
               "--cpus", "2", "--memory", "1g", "--pids-limit", "128",
               "--env", "ASCEND_VISIBLE_DEVICES=void", "--env", "LD_PRELOAD=",
               "--mount", "type=bind,src=/dev/null,dst=/etc/ld.so.preload,readonly",
               "--entrypoint", "python3", args.image, "-c", READER]
    process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if process.returncode:
        raise SystemExit("READ_ONLY_AUDIT_FAILED: " + process.stderr[-4000:])
    report = json.loads(process.stdout)
    report["image"] = {key: image.get(key) for key in ("Id", "RepoDigests", "Architecture", "Created")}
    report["container_policy"] = {"name": name, "owner_nonce": nonce, "runtime": "runc",
        "device_passthrough": False, "network": "none", "rootfs": "read-only", "auto_remove": True}
    report["reader_stderr"] = process.stderr
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, ensure_ascii=False) + "\n").encode("utf-8")
    with args.output.open("xb") as stream:
        stream.write(gzip.compress(payload) if args.output.suffix == ".gz" else payload)
    print(json.dumps({"output": str(args.output), "files": len(report["files"]),
                      "metadata": report["metadata"], "image_id": image["Id"]}))


if __name__ == "__main__":
    main()
