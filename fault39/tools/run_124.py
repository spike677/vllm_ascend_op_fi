"""Bounded, fail-closed 124 runner. Never stops foreign workloads.

Run on the host, not in a device container. One shared lock covers cards 2/5.
The lock coordinates cooperating jobs; it is not host isolation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
from deployment_profile import verify_build

IMAGE = 'quay.io/ascend/vllm-ascend:v0.18.0'
LABEL = 'dcat.task=fault39-20261116'


def command(args, timeout=20):
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT, timeout=timeout)


def dedicated_atk_reservations(container, cards):
    """Conservatively reserve mapped cards for running, dedicated ATK jobs.

    An idle persistent container's device permissions alone are not a
    reservation. A running ptq_ job, however, can retain driver reservation
    before or between visible compute processes; do not compete with it.
    """
    name = container.get('Name', '')
    if (not isinstance(name, str) or not name.lstrip('/').startswith('ptq_')
            or container.get('State', {}).get('Running') is not True):
        return []
    selected = {f'/dev/davinci{card}': card for card in cards if card in (2, 5)}
    mapped = {dev.get('PathOnHost') for dev in container.get('HostConfig', {}).get('Devices') or []}
    return [{'card': selected[path], 'container': name, 'container_id': container.get('Id'),
             'device': path, 'reason': 'DEDICATED_ATK_CONTAINER_RESERVATION',
             'basis': 'running_ptq_name_and_selected_device_mapping'}
            for path in sorted(mapped.intersection(selected))]


def _runtime_option(arguments, names):
    values = []
    for index, argument in enumerate(arguments):
        if argument in names:
            if index + 1 == len(arguments) or arguments[index + 1].startswith('-'):
                raise ValueError('MISSING_RUNTIME_PATH_OPTION')
            values.append(arguments[index + 1])
        elif '=' in argument and argument.split('=', 1)[0] in names:
            values.append(argument.split('=', 1)[1])
    if len(set(values)) > 1:
        raise ValueError('CONFLICTING_RUNTIME_PATH_OPTIONS')
    return values[0] if values else None


def _absolute_runtime_path(value):
    if not isinstance(value, str) or not value or '\x00' in value or not Path(value).is_absolute():
        raise ValueError('ABSOLUTE_RUNTIME_PATH_REQUIRED')
    return Path(value)


def containerd_state_path(arguments):
    """Read the running daemon's path options; never dump its configuration."""
    explicit = _runtime_option(arguments, ('--state',))
    if explicit is not None:
        return _absolute_runtime_path(explicit), {'basis': 'containerd_argv_state'}
    configured = _runtime_option(arguments, ('--config', '-c'))
    path = _absolute_runtime_path(configured or '/etc/containerd/config.toml')
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        if configured is not None:
            raise
        return Path('/run/containerd'), {'basis': 'containerd_default_state_no_default_config'}
    try:
        import tomllib
    except ImportError as error:
        raise ValueError('TOML_READER_UNAVAILABLE') from error
    config = tomllib.loads(raw.decode('utf-8'))
    if config.get('imports'):
        # An imported state override needs its own exact reader; do not guess.
        raise ValueError('CONTAINERD_IMPORTED_STATE_UNRESOLVED')
    return _absolute_runtime_path(config.get('state', '/run/containerd')), dict(
        basis='containerd_config_state', config=str(path), config_sha256=hashlib.sha256(raw).hexdigest())


def checkpoint_allocations(document, cards):
    """Only per-Pod Ascend allocations reserve cards; registration is inventory."""
    if not isinstance(document, dict):
        raise ValueError('CHECKPOINT_OBJECT_REQUIRED')
    data = document.get('Data')
    if isinstance(data, str):
        data = json.loads(data)
    if (not isinstance(data, dict) or not isinstance(data.get('PodDeviceEntries'), list)
            or not isinstance(data.get('RegisteredDevices'), dict)):
        raise ValueError('CHECKPOINT_ALLOCATION_SCHEMA_UNRECOGNIZED')
    blockers, allocations = [], []
    for entry in data['PodDeviceEntries']:
        if not isinstance(entry, dict) or not isinstance(entry.get('ResourceName'), str):
            raise ValueError('CHECKPOINT_RESOURCE_IDENTITY_UNRECOGNIZED')
        resource = entry['ResourceName']
        if not any(token in resource.lower() for token in ('ascend', 'npu')):
            continue
        if resource != 'huawei.com/Ascend910':
            raise ValueError('CHECKPOINT_ACCELERATOR_RESOURCE_UNRECOGNIZED')
        if any(not isinstance(entry.get(key), str) or not entry[key].strip()
               for key in ('PodUID', 'ContainerName')):
            raise ValueError('CHECKPOINT_POD_CONTAINER_IDENTITY_MISSING')
        ids = entry.get('DeviceIDs')
        if isinstance(ids, dict):
            if any(not isinstance(key, str) or not re.fullmatch(r'-?\d+', key)
                   or not isinstance(value, list) for key, value in ids.items()):
                raise ValueError('CHECKPOINT_NUMA_DEVICE_SCHEMA_UNRECOGNIZED')
            ids = [value for group in ids.values() for value in group]
        if not isinstance(ids, list) or not ids:
            raise ValueError('CHECKPOINT_ALLOCATED_DEVICE_IDS_MISSING')
        for device_id in ids:
            match = re.fullmatch(r'Ascend910-(0|[1-9]\d*)', device_id) if isinstance(device_id, str) else None
            if match is None:
                raise ValueError('CHECKPOINT_ALLOCATED_DEVICE_ID_UNRECOGNIZED')
            row = dict(card=int(match[1]), pod_uid=entry['PodUID'], container=entry['ContainerName'],
                resource=resource, device_id=device_id, status='ALLOCATION_RECORDED_RELEASE_UNVERIFIED')
            allocations.append(row)
            if row['card'] in cards:
                blockers.append(dict(row, reason='KUBELET_DEVICE_ALLOCATION'))
    return blockers, allocations


def oci_device_grants(item, devices, *, task, config):
    """Character devices and bind mounts are access grants, not reservations."""
    grants = []
    for dev in item.get('linux', {}).get('devices', []):
        if dev.get('type') == 'c':
            number = os.makedev(dev.get('major', 0), dev.get('minor', 0))
            if number in devices:
                grants.append(dict(task=task, config=config, device=dev.get('path'), card=devices[number],
                                   kind='OCI_DEVICE_PERMISSION'))
    for mount in item.get('mounts', []):
        source = mount.get('source')
        if not isinstance(source, str) or not source.startswith('/'):
            continue
        if source.rstrip('/') == '/dev':
            grants.extend(dict(task=task, config=config, source=source, device=mount.get('destination'), card=card,
                               kind='OCI_DEVICE_DIRECTORY_PERMISSION') for card in devices.values())
        else:
            try:
                value = Path(source).stat()
            except FileNotFoundError:
                # Non-device mounts may legitimately be absent outside the
                # task mount namespace. Declared device sources must be known.
                if source.startswith(('/dev/davinci', '/dev/devmm', '/dev/hisi')):
                    raise
                continue
            if stat.S_ISCHR(value.st_mode) and value.st_rdev in devices:
                grants.append(dict(task=task, config=config, source=source, device=mount.get('destination'),
                    card=devices[value.st_rdev], kind='OCI_DEVICE_MOUNT_PERMISSION'))
    return grants


def kubelet_checkpoint_paths(proc, root):
    """An owned device-plugin socket overrides unrelated kubelet data roots.

    Kubelet can use --root-dir for Pod data while its device plugin manager
    still binds /var/lib/kubelet/device-plugins/kubelet.sock. Check actual socket
    ownership, not merely a socket file left by another process.
    """
    sockets = {}
    for line in (proc / 'net/unix').read_text().splitlines()[1:]:
        row = line.split()
        if len(row) >= 8:
            path = Path(row[7])
            if path.name == 'kubelet.sock' and path.parent.name == 'device-plugins':
                sockets[row[6]] = _absolute_runtime_path(str(path))
    checkpoints = {}
    for fd in (proc / 'fd').iterdir():
        try:
            link = os.readlink(fd)
        except FileNotFoundError:
            continue
        match = re.fullmatch(r'socket:\[(\d+)\]', link)
        if match and match[1] in sockets:
            socket = sockets[match[1]]
            checkpoint = socket.parent / 'kubelet_internal_checkpoint'
            checkpoints[str(checkpoint)] = (checkpoint, dict(basis='owned_device_plugin_socket',
                socket=str(socket), socket_inode=match[1], fd=fd.name))
    if checkpoints:
        return list(checkpoints.values())
    return [(root / 'device-plugins/kubelet_internal_checkpoint', dict(basis='kubelet_root_dir_fallback'))]


def runtime_device_inventory(cards, devices, *, proc_root=Path('/proc'),
                             default_checkpoint=Path('/var/lib/kubelet/device-plugins/kubelet_internal_checkpoint')):
    """Read actual daemon roots and retained kubelet allocations, never clear them."""
    report = dict(blockers=[], device_access_grants=[], containerd_states=[],
                  kubelet_checkpoint_sources=[], kubelet_allocation_checks=[], oci_tasks=[])
    states, checkpoints = {}, {}

    def unknown(reason, **where):
        report['blockers'].append(dict(reason=reason, cards=list(cards), status='UNKNOWN', **where))

    for proc in proc_root.iterdir():
        if not proc.name.isdigit():
            continue
        try:
            name = (proc / 'comm').read_text().strip()
            if name not in ('containerd', 'kubelet'):
                continue
            arguments = [value for value in (proc / 'cmdline').read_bytes().decode('utf-8').split('\x00') if value]
            if name == 'containerd':
                state, basis = containerd_state_path(arguments)
                states[str(state.resolve())] = state
                report['containerd_states'].append(dict(pid=int(proc.name), state=str(state), **basis))
            else:
                root = _absolute_runtime_path(_runtime_option(arguments, ('--root-dir',)) or '/var/lib/kubelet')
                for checkpoint, basis in kubelet_checkpoint_paths(proc, root):
                    checkpoints[str(checkpoint)] = checkpoint
                    report['kubelet_checkpoint_sources'].append(dict(pid=int(proc.name), root_dir=str(root),
                        checkpoint=str(checkpoint), kubeconfig=_runtime_option(arguments, ('--kubeconfig',)), **basis))
        except FileNotFoundError:
            if proc.exists():
                unknown('RUNTIME_ALLOCATION_SOURCE_MISSING', pid=int(proc.name))
        except (OSError, ValueError, TypeError) as error:
            unknown('RUNTIME_ALLOCATION_SOURCE_UNKNOWN', pid=int(proc.name), error=type(error).__name__ + ':' + str(error))

    # A retained allocation still matters after kubelet/workload processes exit.
    if default_checkpoint.exists():
        checkpoints[str(default_checkpoint)] = default_checkpoint
    observed_kubernetes = False
    for state in states.values():
        try:
            if not state.is_dir():
                raise ValueError('RUNNING_CONTAINERD_STATE_DIRECTORY_MISSING')
            for runtime in ('io.containerd.runtime.v1.linux', 'io.containerd.runtime.v2.task'):
                directory = state / runtime
                if not directory.exists():
                    continue
                for namespace in directory.iterdir():
                    if not namespace.is_dir():
                        continue
                    observed_kubernetes |= namespace.name == 'k8s.io'
                    for task in namespace.iterdir():
                        if not task.is_dir():
                            continue
                        config = task / 'config.json'
                        try:
                            raw = config.read_bytes()
                            item = json.loads(raw)
                            if namespace.name == 'k8s.io':
                                annotations = item.get('annotations', {})
                                report['oci_tasks'].append(dict(task=task.name, config=str(config),
                                    config_sha256=hashlib.sha256(raw).hexdigest(),
                                    pod_uid=annotations.get('io.kubernetes.cri.sandbox-uid')))
                            report['device_access_grants'].extend(oci_device_grants(
                                item, devices, task=task.name, config=str(config)))
                        except (OSError, ValueError, TypeError, AttributeError) as error:
                            unknown('OCI_DEVICE_INVENTORY_UNKNOWN', config=str(config), error=type(error).__name__)
        except (OSError, ValueError) as error:
            unknown('CONTAINERD_STATE_INVENTORY_UNKNOWN', state=str(state), error=type(error).__name__)
    if observed_kubernetes and not checkpoints:
        # A CRI task without an observed kubelet path cannot prove no allocation.
        checkpoints[str(default_checkpoint)] = default_checkpoint
    for checkpoint in checkpoints.values():
        try:
            raw = checkpoint.read_bytes()
            blocks, allocations = checkpoint_allocations(json.loads(raw), cards)
            metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                mtime_utc=datetime.fromtimestamp(checkpoint.stat().st_mtime, timezone.utc).isoformat())
            report['kubelet_allocation_checks'].append(dict(metadata, status='READ', allocations=allocations))
            report['blockers'].extend(dict(row, **metadata) for row in blocks)
        except (OSError, ValueError, TypeError) as error:
            unknown('KUBELET_ALLOCATION_UNKNOWN', checkpoint=str(checkpoint), error=type(error).__name__ + ':' + str(error))
    return report


def idle_gate(cards, *, review_stale=False, stale_allocation_scope=None, attempt_scope=None):
    report = {'time': datetime.now(timezone.utc).isoformat(), 'cards': cards, 'blockers': [], 'device_access_grants': []}
    scope = None
    if review_stale:
        try:
            from kubelet_allocations import validate_stale_allocation_scope
            scope = validate_stale_allocation_scope(stale_allocation_scope, attempt_scope)
            report['stale_allocation_scope'] = scope
        except Exception as error:
            report['blockers'].append(dict(reason='STALE_ALLOCATION_SCOPE_INVALID', status='UNKNOWN',
                error=type(error).__name__ + ':' + str(error)[:512]))
    report['npu_smi'] = command(['npu-smi', 'info'])
    rows = report['npu_smi'].splitlines()
    if 'Process id' not in report['npu_smi'] or 'Health' not in report['npu_smi']:
        raise RuntimeError('UNRECOGNIZED_NPU_SMI')
    for card in cards:
        health = next((r for r in rows if re.search(r'\|\s*' + str(card) + r'\s+910', r)), '')
        if not health or not re.search(r'\|\s*OK\s*\|', health):
            report['blockers'].append({'card': card, 'reason': 'HEALTH_UNPROVEN'})
        for row in rows:
            found = re.match(r'\|\s*(\d+)\s+(\d+)\s*\|\s*(\d+)\s*\|', row)
            if found and int(found[1]) == card:
                report['blockers'].append({'card': card, 'pid': int(found[3]), 'reason': 'NPU_PROCESS'})
    devices = {Path('/dev/davinci' + str(c)).stat().st_rdev: c for c in cards}
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    info = fd.stat()
                    if stat.S_ISCHR(info.st_mode) and info.st_rdev in devices:
                        report['blockers'].append({'card': devices[info.st_rdev], 'pid': int(proc.name), 'reason': 'OPEN_DEVICE'})
                        break
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass
    ids = command(['docker', 'ps', '-q']).split()
    for item in json.loads(command(['docker', 'inspect', *ids])) if ids else []:
        report['blockers'].extend(dedicated_atk_reservations(item, cards))
        for dev in item.get('HostConfig', {}).get('Devices') or []:
            path = dev.get('PathOnHost', '')
            if path in ['/dev/davinci' + str(c) for c in cards]:
                report['device_access_grants'].append({'container': item['Name'], 'device': path, 'kind': 'DOCKER_DEVICE_PERMISSION'})
    # Runtime permission inventory alone is not an allocation. Kubelet's
    # retained per-Pod device assignments independently block an apparently idle
    # device until release can be established; processes may not yet exist.
    runtime = runtime_device_inventory(cards, devices)
    report['blockers'].extend(runtime.pop('blockers'))
    report['device_access_grants'].extend(runtime.pop('device_access_grants'))
    report.update(runtime)
    if not review_stale and any(row['reason'] == 'KUBELET_DEVICE_ALLOCATION' for row in report['blockers']):
        report['stale_checkpoint_review'] = dict(status='DISABLED', reason='EXPLICIT_APPROVED_ATTEMPT_SCOPE_REQUIRED')
    if (review_stale and scope is not None and report['blockers']
            and all(row['reason'] == 'KUBELET_DEVICE_ALLOCATION' for row in report['blockers'])):
        try:
            from kubelet_allocations import digest, review_stale_allocations
            proof = review_stale_allocations(report['blockers'], report['kubelet_checkpoint_sources'], report['oci_tasks'],
                stale_allocation_scope=scope, attempt_scope=attempt_scope)
            # Recheck actual compute/FDs, runtime tasks, permissions and retained
            # checkpoints after the live API window. Never recurse into review.
            post = idle_gate(cards, review_stale=False, stale_allocation_scope=scope, attempt_scope=attempt_scope)
            fields = ('blockers', 'device_access_grants', 'oci_tasks', 'containerd_states',
                      'kubelet_checkpoint_sources', 'kubelet_allocation_checks')
            report['stale_checkpoint_post_gate'] = dict(time=post['time'], blockers=post['blockers'],
                npu_smi_sha256=hashlib.sha256(post['npu_smi'].encode()).hexdigest(),
                state_sha256=digest({key: post[key] for key in fields}))
            if any(digest(report[key]) != digest(post[key]) for key in fields):
                raise ValueError('DEVICE_OR_RUNTIME_STATE_CHANGED_DURING_LIVE_REVIEW')
            proof['post_gate'] = report['stale_checkpoint_post_gate']
            proof['evidence_sha256'] = digest({key: value for key, value in proof.items() if key != 'evidence_sha256'})
            report['stale_checkpoint_allocations'] = [dict(original_blocker=row, disposition='STALE_CHECKPOINT_ALLOCATION',
                release_evidence_sha256=proof['evidence_sha256']) for row in report['blockers']]
            report['blockers'] = []
            report['stale_checkpoint_review'] = proof
        except Exception as error:
            report['stale_checkpoint_review'] = dict(status='UNKNOWN_OR_STILL_RESERVED',
                reason=type(error).__name__ + ':' + str(error)[:512])
    report['idle'] = not report['blockers']
    return report


def verify_source(source):
    manifest = json.loads((source / 'source_manifest.json').read_text())
    for item in manifest['files']:
        path = (source / item['path']).resolve()
        if not path.is_relative_to(source.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise RuntimeError('SOURCE_MANIFEST_MISMATCH')
    return hashlib.sha256((source / 'source_manifest.json').read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cards', nargs='+', type=int, choices=[2, 5], required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--model', type=Path, default=Path('/home/gyl/models/Qwen3-0.6B'))
    p.add_argument('--lossless-receipt', type=Path)
    p.add_argument('--timeout', type=int, default=900)
    p.add_argument('--image', default=IMAGE, help='Image containing the exact toolkit used by this build')
    a = p.parse_args()
    profile=verify_build(a.native)
    cases = json.loads(a.cases.read_text())
    if not isinstance(cases, list) or not 1 <= len(cases) <= 8 or len(set(a.cards)) != len(a.cards):
        p.error('Require 1..8 cases and distinct cards')
    for case in cases:
        if not re.fullmatch(r'[a-z0-9_-]{1,48}', case['id']) or case['tool'] not in ('native_probe.py', 'vllm_validate.py'):
            p.error('Unsupported case')
        if any(not isinstance(x, str) for x in case['args']):
            p.error('Arguments must be strings')
    a.attempt.mkdir(parents=True, exist_ok=False)
    receipt = {'schema': 'fault39.runner.v1', 'source_manifest_sha256': verify_source(a.source),
               'image': a.image, 'cards': a.cards, 'cases': [], 'status': 'PREPARING',
               'build_profile_sha256':profile['profile_sha256'],
               'cann_version':profile['installation']['ascend_toolkit_install.info']['version']}
    def save():
        (a.attempt / 'controller.json').write_text(json.dumps(receipt, indent=2))
    save()
    # CPU-only preflight: reject a mixed toolkit before granting any device.
    # The image's device-enabling runtime/entrypoint is deliberately bypassed.
    preflight=['docker','run','--rm','--runtime=runc','--network','none','--label',LABEL,
               '-e','ASCEND_VISIBLE_DEVICES=void',
               '--mount','type=bind,src='+str(a.source.resolve())+',dst=/source,readonly',
               '--mount','type=bind,src='+str(a.native.resolve())+',dst=/native,readonly',
               '--entrypoint','/bin/bash',a.image,'-c','source "$1" >&2; shift; exec python3 "$@"',
               'fault39-preflight',profile['environment_script']['path'],'/source/fault39/tools/deployment_profile.py',
               '--native','/native','--verify-installed','--verify-environment']
    try:
        preflight_result=subprocess.run(preflight,text=True,capture_output=True,timeout=120,check=True)
        receipt['installed_preflight']=json.loads(preflight_result.stdout)
        receipt['environment_setup_log']=preflight_result.stderr
    except Exception as error:
        receipt['status']='BUILD_IMAGE_MISMATCH';receipt['preflight_error']=str(error);save();return 2
    save()
    with open('/var/tmp/fault22_npu_2_5.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            receipt['status'] = 'RESOURCE_LOCK_BUSY'; save(); return 3
        for case in cases:
            gate = idle_gate(a.cards)
            item = {'id': case['id'], 'gate': gate}; receipt['cases'].append(item); save()
            if not gate['idle']:
                receipt['status'] = 'DEVICE_BUSY'; save(); print(json.dumps({'status': receipt['status'], 'blockers': gate['blockers']}), flush=True); return 3
            out = a.attempt / case['id']; out.mkdir()
            name = 'fault39-' + case['id'] + '-' + str(os.getpid())
            args = ['docker', 'run', '--name', name, '--label', LABEL, '--network', 'none',
                    '--memory', '32g', '--cpus', '8', '--pids-limit', '1024', '--shm-size', '2g']
            for card in a.cards:
                args += ['--device', '/dev/davinci' + str(card)]
            for dev in ('davinci_manager', 'devmm_svm', 'hisi_hdc'):
                args += ['--device', '/dev/' + dev]
            mounts=[(a.source, '/source', True), (a.native, '/native', True),
                                 (out, '/evidence', False), (a.model, '/model', True),
                                 (Path('/usr/local/Ascend/driver'), '/usr/local/Ascend/driver', True),
                                 (Path('/usr/local/dcmi'), '/usr/local/dcmi', True),
                                 (Path('/dev/null'), '/etc/ld.so.preload', True),
                                 (Path('/home/batch-scale-dynamic-msprobe-20261023/python_packages'), '/msprobe_packages', True)]
            if profile['proxy_layout']=='main_and_nn':
                mounts.append((a.native/'libopapi_nn.so',profile['toolkit']+'/lib64/libopapi_nn.so',True))
            for src, dst, ro in mounts:
                args += ['--mount', 'type=bind,src=' + str(src.resolve()) + ',dst=' + dst + (',readonly' if ro else '')]
            # Device cgroups expose only the selected physical devices. The
            # driver enumerates them as logical 0..N-1; physical IDs here are
            # not valid ASCEND_RT_VISIBLE_DEVICES logical indices.
            env = {'VLLM_SERVER_DEV_MODE': '1',
                   'VLLM_ENABLE_V1_MULTIPROCESSING': '0', 'PYTHONDONTWRITEBYTECODE': '1',
                   'TASK_QUEUE_ENABLE': '0',
                   'TOKENIZERS_PARALLELISM': 'false', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
            if a.lossless_receipt:
                args += ['--mount', 'type=bind,src=' + str(a.lossless_receipt.resolve()) + ',dst=/validation/lossless.json,readonly']
            for key, value in env.items():
                args += ['-e', key + '=' + value]
            # Preserve the selected image's matched dependencies after our prefixes.
            entry = 'source "$1"; shift; unset ASCEND_RT_VISIBLE_DEVICES; export LD_PRELOAD=/native/libfault39_native_v1.so:/native/libopapi.so; export LD_LIBRARY_PATH=/native:$LD_LIBRARY_PATH; export PYTHONPATH=/source/fault39/src:/msprobe_packages:$PYTHONPATH; exec python3 "$@"'
            args += ['--entrypoint', '/bin/bash', a.image, '-c', entry, 'fault39', profile['environment_script']['path'], '/source/fault39/tools/' + case['tool'], *case['args']]
            item['command'] = args; item['started_utc'] = datetime.now(timezone.utc).isoformat(); save()
            print(json.dumps({'status': 'CASE_START', 'id': case['id'], 'cards': a.cards}), flush=True)
            try:
                with (out / 'process.log').open('w') as log:
                    proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        item['exit_code'] = proc.wait(timeout=a.timeout)
                    except subprocess.TimeoutExpired:
                        item['timeout'] = True
                        # This is exclusively the container created above, never a foreign job.
                        subprocess.run(['docker', 'stop', '--time', '20', name], capture_output=True, timeout=30)
                        item['exit_code'] = proc.wait(timeout=30)
                info = json.loads(command(['docker', 'inspect', name]))[0]
                item['container_state'] = info['State']
                item['image_id'] = info['Image']
            finally:
                try:
                    info = json.loads(command(['docker', 'inspect', name]))[0]
                    if info['Config']['Labels'].get('dcat.task') != 'fault39-20261116':
                        raise RuntimeError('CONTAINER_OWNERSHIP_CHANGED')
                    if info['State']['Running']:
                        command(['docker', 'stop', '--time', '20', name], timeout=30)
                    command(['docker', 'rm', name])
                finally:
                    save()
            print(json.dumps({'status': 'CASE_END', 'id': case['id'], 'exit_code': item.get('exit_code')}), flush=True)
            if item.get('exit_code') != 0:
                receipt['status'] = 'CASE_FAILED'; save(); return 2
        receipt['status'] = 'CASES_COMPLETED'; save(); return 0


if __name__ == '__main__':
    raise SystemExit(main())
