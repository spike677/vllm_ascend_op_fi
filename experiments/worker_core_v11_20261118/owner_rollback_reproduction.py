"""A2 diagnostic overlay. Runtime files come from a separately pinned commit.

The audit never repairs state. Only public ProductClient cleanup is exercised.
This is a trusted, isolated acceptance harness, not a deployment RPC extension.
"""
import argparse
import copy
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import urllib.request

from product_http_probe import AcceptanceExtension, HTTPExecutor


def source_identity():
    result = {}
    for name in ('product_host', 'product_binding', 'continuous_binding'):
        module = importlib.import_module('demoncat.runtime.vllm_ascend.worker_extension.' + name)
        path = Path(module.__file__)
        result[name] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


class AuditExtension(AcceptanceExtension):
    def demoncat_owner_audit(self):
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from demoncat.runtime.vllm_ascend.worker_extension.continuous_binding import (
            ATTRIBUTE, ContinuousBinding, current_binding)
        worker = self.worker if isinstance(self, WorkerWrapperBase) else self
        runner = worker.model_runner
        model = runner.model
        raw = getattr(worker, ATTRIBUTE, {})
        entries = []
        for mechanism, binding in raw.items():
            group = getattr(binding, 'group', {})
            row = dict(mechanism=mechanism, owner=group.get('owner'),
                       generation=group.get('group_generation'), group=group,
                       binding_type=type(binding).__module__ + '.' + type(binding).__qualname__,
                       exact_type=type(binding) is ContinuousBinding,
                       worker_matches=getattr(binding, 'worker', None) is worker,
                       model_matches=getattr(binding, 'model', None) is model,
                       leases=len(getattr(binding, 'leases', ())))
            try:
                row['current_binding_matches'] = current_binding(worker, model, mechanism) is binding
            except Exception as error:
                row['current_binding_error'] = repr(error)
            entries.append(row)
        runtime = getattr(worker, '_demoncat_worker_fault_runtime', None)
        return dict(rank=worker.rank, pid=os.getpid(), worker_id=id(worker), model_id=id(model),
                    bindings=len(raw), entries=entries, leases=len(getattr(runtime, '_leases', {})),
                    owners=len(getattr(runtime, '_tenants', {})),
                    wrappers=sum(n in vars(runner) for n in ('execute_model', 'sample_tokens', '_preprocess')),
                    permit_present=hasattr(worker, '_demoncat_blue_acceptance_permit'),
                    native_maps=[line for line in Path('/proc/self/maps').read_text().splitlines()
                                 if 'libdcat_postwrite' in line], source=source_identity())

    def demoncat_owner_configure(self, options):
        from vllm.v1.worker.worker_base import WorkerWrapperBase
        from demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization import (
            authority_for, configure_deployment_acceptance)
        from demoncat.runtime.vllm_ascend.worker_extension.product_config import configure_product_deployments
        from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_control_reply
        worker = self.worker if isinstance(self, WorkerWrapperBase) else self
        authority = authority_for(worker)
        if authority is None:
            authority = configure_deployment_acceptance(worker, source='BLUE_COMBINATION',
                authorize_operator=lambda principal, operation: principal == 'product-acceptance')
        reply = product_control_reply(lambda: configure_product_deployments(worker,
            authority=authority, operator='product-acceptance', **options))
        return dict(reply, rank=worker.rank, pid=os.getpid())


class JournalExecutor(HTTPExecutor):
    def __init__(self, base, target):
        super().__init__(base)
        self.target, self.rows = target, []

    def collective_rpc(self, method, *, args=(), kwargs=None, timeout=120):
        row = dict(sequence=len(self.rows), time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   method=method, args=copy.deepcopy(args), kwargs=copy.deepcopy(kwargs))
        self.rows.append(row)
        try:
            row['result'] = super().collective_rpc(method, args=args, kwargs=kwargs, timeout=timeout)
            return row['result']
        except Exception as error:
            row['error'] = repr(error)
            raise
        finally:
            self.target.write_text(json.dumps(self.rows, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collision-exit', type=int, choices=(4, 5), required=True)
    args = parser.parse_args()
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))
    root = Path('/evidence')
    report = dict(status='STARTED', source=source_identity(), steps=[],
                  source_manifest_sha256=hashlib.sha256(Path('/source/source_manifest.json').read_bytes()).hexdigest())
    def save():
        (root / 'owner-reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    server = None
    try:
        cmd = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server', '--model', '/model',
               '--served-model-name', 'acceptance', '--host', '127.0.0.1', '--port', '18937',
               '--dtype', 'bfloat16', '--quantization', 'ascend', '--tensor-parallel-size', '2',
               '--distributed-executor-backend', 'mp', '--enforce-eager', '--no-enable-prefix-caching',
               '--no-async-scheduling', '--enable-chunked-prefill', '--max-num-batched-tokens', '1024',
               '--max-model-len', '1024', '--max-num-seqs', '1', '--gpu-memory-utilization', '0.8',
               '--seed', '42', '--trust-remote-code', '--worker-extension-cls',
               'owner_rollback_reproduction.AuditExtension', '--limit-mm-per-prompt', '{"image":0,"video":0}']
        server = subprocess.Popen(cmd, env=dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD='spawn'),
                                  stdout=(root / 'http-server.log').open('w'), stderr=subprocess.STDOUT)
        ex = JournalExecutor('http://127.0.0.1:18937', root / 'rpc-trace.json')
        for _ in range(480):
            if server.poll() is not None:
                raise RuntimeError('HTTP_SERVER_START_FAILED')
            try:
                with urllib.request.urlopen(ex.base + '/health', timeout=2):
                    break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError('HTTP_SERVER_READY_TIMEOUT')
        from demoncat.runtime.vllm_ascend.worker_extension.product_host import ProductClient
        from demoncat.runtime.vllm_ascend.worker_extension.product_errors import product_control_reply
        prompt = 'Explain briefly why the sky appears blue.'
        def infer(text=prompt):
            return ex.post('/v1/completions', dict(model='acceptance', prompt=text, temperature=0,
                seed=42, max_tokens=4, min_tokens=4, ignore_eos=True))
        def audit(label):
            rows = ex.collective_rpc('demoncat_owner_audit')
            report['steps'].append(dict(label=label, audit=rows))
            save()
            return rows
        def zero(rows):
            return len(rows) == 2 and sorted(r['rank'] for r in rows) == [0, 1] and all(
                r[k] == 0 for r in rows for k in ('bindings', 'leases', 'owners', 'wrappers'))
        assert zero(audit('initial'))
        ex.collective_rpc('demoncat_acceptance_bootstrap', args=('capture_start', {}))
        infer()
        captures = ex.collective_rpc('demoncat_acceptance_bootstrap', args=('capture_stop', {}))
        assert len(captures) == 2 and captures[0][0]['tokens'] == captures[1][0]['tokens']
        tokens = captures[0][0]['tokens']
        def deployment(number, variant, mode):
            return dict(schema='demoncat.product.deployment.v1', profile=str(number)+'.'+variant,
                fault=dict(fault_id=number, variant=variant, implementation_version='1'), model='/model', tp=2,
                mode=mode, source='BLUE_COMBINATION', source_manifest='/source/source_manifest.json',
                source_manifest_sha256=report['source_manifest_sha256'], native_library=None, native_sha256=None,
                canary=dict(prompt=prompt, image=None, max_tokens=4), evidence_directory='/evidence')
        paths = {}
        for number, variant, mode in ((18, 'api_half', 'PRODUCT_ACCEPTANCE_ONLY'), (22, 'wrong_row', 'PRODUCT_REGISTERED')):
            path = root / ('deployment-' + str(number) + '.json')
            path.write_text(json.dumps(deployment(number, variant, mode)))
            paths[number] = str(path)
        report['configure'] = ex.collective_rpc('demoncat_owner_configure', args=(dict(paths=list(paths.values()),
            canary_tokens={'18.api_half': tokens, '22.wrong_row': tokens}),))
        assert all(r['exit_code'] == 0 for r in report['configure'])
        def client(number):
            return ProductClient.from_deployment(ex, paths[number], operator='product-acceptance',
                run_canary=lambda spec: infer(spec['prompt']), drain=lambda: None)
        for number, label in ((22, 'registered_rejection'), (18, 'missing_permit')):
            c = client(number)
            row = dict(label=label, owner=c.identity.owner, attach=c.attach(), capabilities=c.capabilities())
            row['enable'] = product_control_reply(c.enable)
            row['cleanup'] = copy.deepcopy(c.cleanup_evidence)
            row['recovery'] = copy.deepcopy(c.recovery)
            report[label] = row
            audit(label + '_immediate_before_cleanup')
            row['disable'] = product_control_reply(c.disable)
            row['detach'] = product_control_reply(c.detach)
            row['audit_after_cleanup'] = audit(label + '_after_public_cleanup')
            save()
        # Always attempt the requested new-owner control, even if the negative leaked.
        report['permit'] = ex.collective_rpc('demoncat_acceptance_bootstrap', args=('permit', dict(
            mechanisms=['18.api_half'], attempt='owner-reproduction', source_manifest_sha256=report['source_manifest_sha256'])))
        c = client(18)
        positive = dict(owner=c.identity.owner, attach=c.attach())
        report['new_owner_positive'] = positive
        positive['enable'] = product_control_reply(c.enable)
        audit('positive_enable')
        if positive['enable']['exit_code'] == 0:
            positive['request'] = infer('Explain briefly why rain falls.')
            positive['status'] = c.status()
            # C2: disable releases observers/leases but retains this owner's
            # binding until detach. This reaches ALREADY_INSTALLED itself,
            # rather than an earlier live-wrapper identity guard.
            positive['disable'] = product_control_reply(c.disable)
            assert positive['disable']['exit_code'] == 0
            challenger = client(18)
            collision = dict(owner=challenger.identity.owner, attach=challenger.attach())
            report['collision'] = collision
            collision['before'] = audit('before_competing_owner')
            collision['enable'] = product_control_reply(challenger.enable)
            collision['cleanup'] = copy.deepcopy(challenger.cleanup_evidence)
            collision['recovery'] = copy.deepcopy(challenger.recovery)
            collision['after'] = audit('after_competing_owner')
            positive['detach'] = product_control_reply(c.detach)
            collision['disable_after_owner_release'] = product_control_reply(challenger.disable)
            collision['detach_after_owner_release'] = product_control_reply(challenger.detach)
        else:
            positive['disable'] = product_control_reply(c.disable)
            positive['detach'] = product_control_reply(c.detach)
        report['remove_permit'] = ex.collective_rpc('demoncat_acceptance_bootstrap', args=('remove_permit', {}))
        final = audit('all_public_cleanup_complete')
        # C3: real trusted deployment file validation, not unknown user RPC fields.
        native_results = []
        for name, sha in (('wrong_sha', '0'*64), ('missing_path', 'd9aa0d9488c5c1e8f7fca10c297e4968687ceb6a24866d8346bd33a58afce274')):
            spec = deployment(41, 'tail_column', 'PRODUCT_ACCEPTANCE_ONLY')
            spec.update(native_library='/evidence/does-not-exist/libdcat_postwrite.so', native_sha256=sha)
            path = root / (name + '.json')
            path.write_text(json.dumps(spec))
            before = audit(name + '_before')
            replies = ex.collective_rpc('demoncat_owner_configure', args=(dict(paths=[str(path)], canary_tokens={'41.tail_column': tokens}),))
            after = audit(name + '_after')
            native_results.append(dict(case=name, before=before, replies=replies, after=after))
        report['native_configure_negatives'] = native_results
        report['healthy_inference'] = infer()
        checks = dict(
            registered_refused=report['registered_rejection']['enable']['exit_code'] == 4,
            missing_permit_refused=report['missing_permit']['enable']['exit_code'] == 4,
            missing_permit_zero=zero(report['missing_permit']['audit_after_cleanup']),
            positive_enabled=positive['enable']['exit_code'] == 0,
            collision_expected=report.get('collision', {}).get('enable', {}).get('exit_code') == args.collision_exit,
            final_zero=zero(final),
            native_refused_unloaded=all(all(r['exit_code'] == 4 for r in n['replies']) and
                all(not r['native_maps'] for r in n['before'] + n['after']) for n in native_results))
        report['checks'] = checks
        report['status'] = 'PASSED' if all(checks.values()) else 'REPRODUCTION_FINDING'
        save()
        assert all(checks.values()), checks
    except Exception as error:
        report.update(status='FAILED', error=repr(error), traceback=traceback.format_exc())
        save()
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)


if __name__ == '__main__':
    main()
