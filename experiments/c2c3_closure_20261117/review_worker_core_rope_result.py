"""Offline source-a50 public Core22 review; standard library, no live authority.

Run on the evidence host. Full observations are read in place, never uploaded.
The source46 reviewer is unchanged. Its exact checks run in a separate module
instance with this review's independently pinned source-a50 entry/implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import struct
import types


SOURCE_COMMIT = '006916a9f9bd37549f0f21fc986ecace23e22e9b'
MANIFEST_SHA256 = 'c87d33a804b46f16ed44536a24863966fd08f0c99b2a21135a367b47d4c714cf'
PROBE_SHA256 = '0e087f1d5996164d5d9c5f843d226354273666230853781302bd7e1f70873dfb'
IMPLEMENTATION_SHA256 = '999051f61294f6f3d196083c4591a31c3442ff2cca8e05de06d2b5852bde5537'
RESULT_SHA256 = '6515fa39db4fc554649aed8265be3611db157bb0487d24661e89df3db20f8900'
BASE_CHECKS_SHA256 = 'ccc30f0ecf6811f49f80059b832dad2ca59bc441580ffdf35511e6d96add29dd'
PROBE_PATH = 'experiments/c2c3_closure_20261117/natural_rope_worker_core_probe.py'
BINDING_PATHS = ('src/demoncat/runtime/vllm_ascend/worker_extension/providers/fault22.py',
                 'src/demoncat/runtime/vllm_ascend/worker_extension/serving22.py')
CONFIG = dict(fault_id=22, variant='wrong_row', implementation_version='1')
DECORATIONS = {'execution_boundary', 'mechanism_semantics', 'phase_rows', 'scheduler_requests_claimed',
               'raw_device_launch_payload_observed', 'independent_statistics'}
COVERAGE_FIELDS = set(('schema session_id scope selection trigger owner generation budget discovery observation '
    'planned_targets planned_targets_hit observed_calls observed_eligible_calls hits missing_outcomes '
    'propagation_added_calls calls recovery contract_satisfied rejection_reasons real_worker_evidence '
    'real_pretrained_model_evidence live_c2_proof public_qualification_granted').split())
AUDIT_FIELDS = {'proof', 'proof_sha256', 'object_binding', 'exact_issued_object_verified',
                'report_hash_scope', 'serializable_authority', 'qualification_granted'}


def _load_frozen_checks():
    path = Path(__file__).with_name('review_natural_rope_result.py')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != BASE_CHECKS_SHA256:
        raise ValueError('FROZEN_COMPLETE_REVIEW_CHECKS_CHANGED')
    module = types.ModuleType('_core22_frozen_complete_checks')
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    # This private instance never changes the source46 module or its CLI.
    # Both new pins are constants reviewed for the exact source-a50 snapshot.
    module.FROZEN_PROBE_SHA256 = PROBE_SHA256
    module.FROZEN_IMPLEMENTATION_SHA256 = IMPLEMENTATION_SHA256
    return module


BASE = _load_frozen_checks()
PHASES = BASE.PHASES + ('cleanup_healthy',)
ACTIONS = ('attach', 'prepare', 'register', 'check', 'enable') + ('advance', 'query') * 10 + (
    'finish', 'disable', 'unregister', 'detach')
sha, is_sha, read_json = BASE.sha, BASE.is_sha, BASE.read_result


def confined_file(root, relative):
    if (not isinstance(relative, str) or not relative or '\\' in relative
            or PurePosixPath(relative).is_absolute() or '..' in PurePosixPath(relative).parts
            or str(PurePosixPath(relative)) != relative):
        raise ValueError('UNSAFE_EVIDENCE_OR_SOURCE_PATH')
    path = root / relative
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError('SYMLINK_EVIDENCE_OR_SOURCE_REFUSED')
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)) or not path.is_file():
        raise ValueError('EVIDENCE_OR_SOURCE_NOT_CONFINED_FILE')
    return path


def implementation_digest(files):
    host = 'src/demoncat/runtime/vllm_ascend/worker_extension/'
    common = [host + 'providers/tensor_natural.py'] + ['fault_injection/' + name + '.py' for name in (
        'rmsnorm_return', 'index22', 'tensor_observer', 'tensor_callable', '_scoped_dispatch',
        'module_faults', 'route8', 'moe_census', 'moe_restore', 'conv3d_natural')] + [
        host + 'serving_tensor.py', host + 'serving_moe.py']
    group_files = ['fault_injection/' + name + '.py' for name in (
        'rope_sites', 'rope_census', 'rope_triton_trace', 'rope_fused22', 'rope_serving')]
    common_hash = sha({PurePosixPath(path).name: files[path]['sha256'] for path in common})
    group_hash = sha(dict(common=common_hash,
        files={PurePosixPath(path).name: files[path]['sha256'] for path in group_files},
        provider=files['fault_injection/rope_group.py']['sha256']))
    bindings = {PurePosixPath(path).name: files[path]['sha256'] for path in BINDING_PATHS}
    return sha(dict(group=group_hash, worker_rpc_binding=bindings)), bindings


def verify_source(root, *, _cpu_contract=False):
    root = Path(root).resolve(strict=True)
    manifest_path = confined_file(root, 'source_manifest.json')
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest, _ = read_json(manifest_path, digest if _cpu_contract else MANIFEST_SHA256)
    if (set(manifest) != {'schema', 'base_commit', 'files', 'contains_model_weights',
                        'contains_native_binaries', 'contains_tensor_payloads'}
            or manifest['schema'] != 'demoncat.c2c3.source.v1'
            or manifest['base_commit'] != SOURCE_COMMIT
            or any(manifest[key] is not False for key in ('contains_model_weights',
                'contains_native_binaries', 'contains_tensor_payloads'))):
        raise ValueError('FROZEN_SOURCE_MANIFEST_CONTRACT_CHANGED')
    files = {}
    if not isinstance(manifest['files'], list) or not manifest['files']:
        raise ValueError('COMPLETE_SOURCE_MANIFEST_REQUIRED')
    for row in manifest['files']:
        if (set(row) != {'path', 'sha256', 'bytes'} or row['path'] in files
                or not is_sha(row['sha256']) or type(row['bytes']) is not int or row['bytes'] < 0):
            raise ValueError('INVALID_OR_DUPLICATE_SOURCE_FILE')
        path = confined_file(root, row['path'])
        if path.stat().st_size != row['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('FROZEN_SOURCE_BYTES_CHANGED:' + row['path'])
        files[row['path']] = row
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob('*') if path.is_file()}
    if actual_files != set(files) | {'source_manifest.json'}:
        raise ValueError('UNMANIFESTED_OR_MISSING_FROZEN_SOURCE_FILE')
    implementation, bindings = implementation_digest(files)
    if not _cpu_contract and (files[PROBE_PATH]['sha256'] != PROBE_SHA256 or implementation != IMPLEMENTATION_SHA256):
        raise ValueError('SOURCE_MANIFEST_ENTRY_OR_IMPLEMENTATION_CHANGED')
    return dict(base_commit=manifest['base_commit'], manifest_sha256=digest, verified_files=len(files),
                probe_sha256=files[PROBE_PATH]['sha256'], implementation_sha256=implementation,
                binding_sources=bindings, actual_frozen_source_verified=not _cpu_contract)


def read_artifacts(result_path, expected_sha256):
    result_path = Path(result_path)
    root = result_path.parent.resolve(strict=True)
    result, digest = read_json(confined_file(root, result_path.name), expected_sha256)
    core = result['worker_core']
    host_record = core['final_host_receipt']
    if set(host_record) != {'path', 'sha256'} or host_record['path'] != 'host_receipt.json':
        raise ValueError('EXACT_FINAL_HOST_RECEIPT_REQUIRED')
    host, _ = read_json(confined_file(root, host_record['path']), host_record['sha256'])
    receipts = []
    if len(core['receipts']) != len(ACTIONS):
        raise ValueError('EXACT_29_ORIGINAL_RPC_RECEIPTS_REQUIRED')
    expected_paths = []
    for index, (record, action) in enumerate(zip(core['receipts'], ACTIONS)):
        expected = 'worker_core_rpc/' + str(index).zfill(3) + '-' + action + '.json'
        if set(record) != {'action', 'path', 'sha256'} or record['action'] != action or record['path'] != expected:
            raise ValueError('ORIGINAL_RPC_INDEX_OR_PATH_CHANGED')
        row, _ = read_json(confined_file(root, expected), record['sha256'])
        receipts.append(row); expected_paths.append(expected)
    actual = {path.relative_to(root).as_posix() for path in (root / 'worker_core_rpc').iterdir()}
    if actual != set(expected_paths) or host['history'] != receipts:
        raise ValueError('FULL_HOST_HISTORY_OR_RPC_FILE_SET_CHANGED')
    return result, host, receipts, digest


class CoreReview(BASE.Review):
    def __init__(self, result, host, receipts, *, source=None, expected_targets=None, _cpu_contract=False):
        super().__init__(result, expected_targets=expected_targets, _cpu_contract=_cpu_contract)
        self.host, self.receipts, self.source = host, receipts, source

    def envelope(self):
        super().envelope()
        r, core = self.result, self.result['worker_core']
        self.need(r['status'] == 'STANDARD_WORKER_ROPE_GROUP_CONTRACT_COMPLETE'
                  and r['public_qualification_granted'] is False
                  and core['schema'] == 'demoncat.rope22.public_probe.v1'
                  and core['config'] == CONFIG and core['qualification_granted'] is False
                  and not r.get('recovery_requests'), 'PUBLIC_CORE_ENTRY_OR_FAILED_ATTEMPT_CHANGED')
        if not self.cpu:
            self.need(r['public_entrypoint'] == 'WorkerFaultClient+FaultConfig(22,wrong_row)'
                      and self.source['actual_frozen_source_verified'] is True
                      and self.source['base_commit'] == SOURCE_COMMIT
                      and self.source['manifest_sha256'] == MANIFEST_SHA256
                      and r['probe_sha256'] == self.source['probe_sha256'] == PROBE_SHA256
                      and self.source['implementation_sha256'] == IMPLEMENTATION_SHA256,
                      'FROZEN_SOURCE_OR_PUBLIC_ENTRY_NOT_VERIFIED')

    def live_proof(self):
        c2, c3 = self.result['c2'], self.result['c3']
        for group in (c2, c3):
            self.need(set(group) == COVERAGE_FIELDS | DECORATIONS
                      and group['schema'] == 'demoncat.coverage_session.v1'
                      and group['public_qualification_granted'] is False,
                      'FROZEN_COVERAGE_REPORT_SCHEMA_CHANGED')
            audit = group['live_c2_proof']
            self.need(set(audit) == {'schema', 'issued', 'consumed', 'qualification_granted', 'serializable_authority'}
                      and audit['schema'] == 'demoncat.live_c2_proof_audit.v1'
                      and audit['qualification_granted'] is False and audit['serializable_authority'] is False,
                      'LIVE_PROOF_AUDIT_SCHEMA_OR_AUTHORITY_CHANGED')
        a, b = c2['live_c2_proof'], c3['live_c2_proof']
        self.need(len(a['issued']) == 1 and a['consumed'] is None and b['issued'] == []
                  and isinstance(b['consumed'], dict), 'EXACT_ONE_ISSUANCE_AND_CONSUMPTION_REQUIRED')
        issued, consumed = a['issued'][0], b['consumed']
        self.need(set(issued) == AUDIT_FIELDS and set(consumed) == AUDIT_FIELDS | {'consumer'},
                  'LIVE_PROOF_RECEIPT_FIELDS_CHANGED')
        for receipt in (issued, consumed):
            self.need(receipt['exact_issued_object_verified'] is True
                      and receipt['serializable_authority'] is False and receipt['qualification_granted'] is False
                      and receipt['report_hash_scope'] == 'coverage_report_without_live_c2_proof'
                      and receipt['proof_sha256'] == sha(receipt['proof'])
                      and type(receipt['proof']['generation']) is int
                      and set(receipt['object_binding']) == {'pid', 'python_object_id'}
                      and type(receipt['object_binding']['pid']) is int
                      and receipt['object_binding']['pid'] == self.worker['pid']
                      and type(receipt['object_binding']['python_object_id']) is int
                      and receipt['object_binding']['python_object_id'] > 0,
                      'LIVE_PROOF_FIELDS_DIGEST_OR_PROCESS_BINDING_CHANGED')
        self.need(issued == {key: consumed[key] for key in AUDIT_FIELDS},
                  'ISSUED_AND_CONSUMED_ORIGINAL_OBJECT_RECEIPTS_DIFFER')
        covered = {key: value for key, value in c2.items() if key not in DECORATIONS | {'live_c2_proof'}}
        expected = dict(scope=c2['scope'], candidates_sha256=sha(self.result['qualification']['candidate_set']),
            target_ids=sorted(self.targets), owner=c2['owner'], generation=c2['generation'],
            session_id=c2['session_id'], report_sha256=sha(covered),
            discovery_sha256=sha(self.rows[:4]), observation_sha256=sha(c2['phase_rows']),
            recovery_sha256=sha(c2['recovery']))
        self.need(issued['proof'] == expected, 'ISSUED_C2_PROOF_NOT_BOUND_TO_ORIGINAL_FULL_REPORT')
        self.need(consumed['consumer'] == dict(session_id=c3['session_id'], scope=c3['scope'],
            owner=c3['owner'], generation=c3['generation'], candidates_sha256=expected['candidates_sha256'],
            target_ids=expected['target_ids']) and c3['generation'] > c2['generation'],
            'CONSUMER_SCOPE_OWNER_GENERATION_OR_TARGETS_CHANGED')

    def rpc(self):
        h, core = self.host, self.result['worker_core']
        self.need(h['schema'] == 'demoncat.worker_core.host_receipt.v1' and h['worker'] == self.worker
                  and h['config'] == core['config'] == CONFIG and h['mechanism'] == '22.wrong_row'
                  and h['history'] == self.receipts and tuple(row['action'] for row in self.receipts) == ACTIONS,
                  'COMPLETE_ORIGINAL_RPC_SEQUENCE_OR_WORKER_CHANGED')
        self.need(all(set(row) == {'action', 'response'} for row in self.receipts), 'RPC_HISTORY_RECORD_CHANGED')
        attach = self.receipts[0]['response']
        self.need(attach['status'] == 'ATTACHED' and set(attach['ticket']) == {'tenant_id', 'attach_nonce', 'worker'}
                  and attach['ticket']['worker'] == self.worker and attach['ticket']['tenant_id'] == h['tenant_id']
                  and isinstance(attach['ticket']['attach_nonce'], str) and bool(attach['ticket']['attach_nonce']),
                  'ACTUAL_ATTACHMENT_SCOPE_CHANGED')
        ticket = self.receipts[1]['response']['ticket']
        self.need(set(ticket) == set(attach['ticket']) | {'mechanism', 'implementation_version', 'owner', 'generation'}
                  and all(ticket[key] == value for key, value in attach['ticket'].items())
                  and ticket['owner'] == h['owner'] and isinstance(ticket['owner'], str) and bool(ticket['owner'])
                  and ticket['mechanism'] == '22.wrong_row' and ticket['implementation_version'] == '1'
                  and type(ticket['generation']) is int and ticket['generation'] == 1,
                  'RPC_OWNER_GENERATION_TENANT_OR_IMPLEMENTATION_CHANGED')
        expected_phases = ['PREPARED', 'REGISTERED', 'CHECKED', 'ATTEMPTED'] + ['ATTEMPTED'] * 20 + [
            'FINISHED', 'STOPPED', 'UNREGISTERED']
        provider_owner = self.result['c2']['owner']
        for index, (entry, phase) in enumerate(zip(self.receipts[1:-1], expected_phases), 1):
            action, response = entry['action'], entry['response']
            state = response['provider_state']
            self.need(response['status'] == ('QUERY_OK' if action == 'query' else 'ACTION_COMPLETED')
                      and (action == 'query' or response['action'] == action)
                      and response['phase'] == phase and response['ticket'] == ticket
                      and response['recovery_required'] is False and response['native'] is None
                      and response.get('quarantined', False) is False, 'RPC_FAILURE_OR_SCOPE_DRIFT')
            self.need(state['owner'] == provider_owner and type(state['generation']) is int
                      and state['generation'] == ticket['generation']
                      and state['execution_boundary'] == 'python_hook' and state['state_source'] == 'OWNED_PYTHON_HOOKS'
                      and state['hook_state_available'] is True and state['native_loaded'] is False
                      and state['native_registered'] is False and state['enabled'] is False
                      and BASE.zeros(state, ('inflight', 'leases')) and state['replay_failed'] is False
                      and state['registered'] is (index not in (1, 27))
                      and state['replay_pending'] is (4 <= index <= 24)
                      and state['mechanism'] == '22.wrong_row' and state['legacy_lookup_fallback'] is False
                      and state['public_qualification_granted'] is False
                      and state['E5'] == 'NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED',
                      'RPC_PROVIDER_IDENTITY_RESOURCE_OR_QUALIFICATION_CHANGED')
            serving, coverage = state['serving'], state['coverage']
            self.need(serving['error'] is None and serving['failures'] == [] and serving['recovered'] is False,
                      'RPC_FAILED_ATTEMPT_OR_RECOVERY_HIDDEN')
            if index >= 4:
                deployment = coverage['deployment']
                self.need(deployment['worker_rpc_owner'] == provider_owner
                          and type(deployment['worker_rpc_generation']) is int
                          and deployment['worker_rpc_generation'] == ticket['generation']
                          and deployment['simulation'] is self.cpu and deployment['qualification_granted'] is False,
                          'REVIEWED_DEPLOYMENT_OWNER_OR_GENERATION_CHANGED')
                if not self.cpu:
                    self.need(deployment['binding_sources'] == self.source['binding_sources'],
                              'FROZEN_DEPLOYMENT_BINDING_SOURCES_CHANGED')
                self.need(serving['provenance'] == self.result['setup']['provenance'],
                          'ACTUAL_SERVING_PROVENANCE_CHANGED')
            for key, first in (('discovery', 13), ('C2', 17), ('C3', 23), ('same_worker_healthy', 24)):
                expected = (self.result['qualification'] if key == 'discovery' else
                    self.result['health_request']['worker'] if key == 'same_worker_healthy' else self.result[key.lower()])
                self.need((key in coverage) is (index >= first)
                          and (index < first or coverage[key] == expected), 'RPC_ORIGINAL_EVIDENCE_TIMING_OR_BYTES_CHANGED')
        for index, phase in enumerate(PHASES):
            advanced = self.receipts[5 + index * 2]['response']['provider_state']
            queried = self.receipts[6 + index * 2]['response']['provider_state']
            item = self.result['requests'][index] if index < 9 else self.result['health_request']
            request_id = item['worker']['request_id']
            self.need(all(state['serving']['pending_request'] == [request_id, phase]
                          and state['serving']['next_phase'] == phase
                          and state['serving']['complete'] is False for state in (advanced, queried)),
                      'RPC_ADVANCE_NOT_BOUND_TO_ACTUAL_ENQUEUE_PHASE')
            if index < 9:
                self.need(queried['serving']['last_request'] == item['worker'],
                          'QUERY_ROW_DIFFERS_FROM_COMPLETE_RECORDED_RETURN')
            if phase in ('c2_fault', 'c3_fault_0'):
                group = self.result['c2_open' if index == 4 else 'c3_open']
                closed = self.result['c2' if index == 4 else 'c3']
                self.need(type(group['generation']) is int and type(group['targets']) is int
                    and advanced['serving']['group'] == group and group == dict(
                    session_id=closed['session_id'], owner=closed['owner'], generation=closed['generation'],
                    targets=len(self.paths), trigger=closed['trigger']), 'RPC_GROUP_OPEN_SCOPE_CHANGED')
        for index in (25, 26, 27):
            state = self.receipts[index]['response']['provider_state']
            self.need(state['serving']['complete'] is True and state['serving']['next_phase'] == 'complete'
                      and state['serving']['pending_request'] is None and state['hook_owned'] is False,
                      'FINAL_PROTOCOL_OR_HOOK_CLEANUP_INCOMPLETE')
        self.need(self.receipts[-1]['response'] == dict(status='DETACHED', recoveries=[])
                  and h['status'] == 'DETACHED' and h['attachment'] is h['ticket'] is h['census_ticket'] is None
                  and h['ownership_retained'] is core['ownership_retained'] is False
                  and h['provider_check'] == 'COMPLETED' and h['inner_framework_integration_verified'] is False
                  and all(h['assessments'][name] == 'NOT_ASSESSED' for name in ('C2', 'C3', 'E5'))
                  and h['assessments']['Recovery'] == dict(status='LEASE_RELEASE_CONFIRMED', healthy_output='NOT_ASSESSED')
                  and core['last_scope'] == {key: value for key, value in h.items()
                      if key not in ('history', 'response', 'capability_enumeration')}
                  and h['response'] == self.receipts[-1]['response'], 'FINAL_DETACH_OR_ASSESSMENT_SCOPE_CHANGED')

    def health(self):
        item = self.result['health_request']; row, mapping = item['worker'], item['mapping']
        baseline = self.rows[0]
        request_id = row['request_id']
        self.need(item['phase'] == 'cleanup_healthy' and row['worker'] == self.worker
                  and request_id not in [entry['request_id'] for entry in self.rows]
                  and mapping['internal_request_id'] == request_id
                  and mapping['external_request_id'] not in [entry['mapping']['external_request_id'] for entry in self.result['requests']]
                  and isinstance(mapping['external_request_id'], str) and bool(mapping['external_request_id'])
                  and type(mapping['request_state_object_id']) is int and mapping['request_state_object_id'] > 0
                  and mapping['source'] == 'ACTUAL_PUBLIC_ENQUEUE_AND_OUTPUT_PROCESSOR'
                  and row['cleanup_before_request'] is True and row['full_hidden_logits_exact'] is True
                  and type(row['fault_hooks']) is int and row['fault_hooks'] == 0
                  and row['qualification_granted'] is False and row['input'] == baseline['input']
                  and row['output'] == baseline['output']
                  and item['output_tokens'] == self.result['healthy_before'] == self.result['healthy_after'],
                  'TENTH_ACTUAL_SAME_WORKER_FULL_HEALTH_REQUEST_UNPROVEN')
        scheduler = row['scheduler']
        if self.cpu:
            self.need(scheduler == dict(cpu_contract=True, request_id=request_id), 'HEALTH_CPU_SCHEDULER_CHANGED')
        else:
            self.need(scheduler['internal_request_id'] == request_id and scheduler['num_requests'] == 1
                      and scheduler['num_computed_tokens'] == 0
                      and scheduler['num_scheduled_tokens'] == scheduler['num_prompt_tokens'] == len(scheduler['positions'])
                      and scheduler['positions'] == list(range(len(scheduler['token_ids'])))
                      and scheduler['token_ids'] == baseline['scheduler']['token_ids']
                      and sha(scheduler['token_ids']) == scheduler['token_ids_sha256']
                      and type(scheduler['request_object_id']) is int and scheduler['request_object_id'] > 0,
                      'TENTH_HEALTH_SCHEDULER_MAPPING_CHANGED')
            for name, values in (('input_ids', scheduler['token_ids']), ('positions', scheduler['positions'])):
                captured = row['input'][1][name]
                code = {'torch.int32': 'i', 'torch.int64': 'q'}[captured['dtype']]
                self.need(captured['shape'] == [len(values)] and captured['sha256'] == hashlib.sha256(
                    struct.pack('<' + code * len(values), *values)).hexdigest(), 'TENTH_HEALTH_INPUT_BYTES_CHANGED')
        final = dict(self.result['cleanup'], requests=[entry['request_id'] for entry in self.rows] + [request_id])
        self.need(self.result['worker_core']['final_cleanup'] == final
                  and self.receipts[23]['response']['provider_state']['coverage']['cleanup'] == self.result['cleanup']
                  and all(self.receipts[index]['response']['provider_state']['coverage']['cleanup'] == final
                          for index in (24, 25, 26, 27)), 'CLEANUP_BEFORE_HEALTH_OR_FINAL_SCOPE_CHANGED')

    def run(self):
        self.section('envelope', self.envelope)
        if 'envelope' in self.completed:
            self.section('requests', self.requests)
            for phase, row in zip(BASE.PHASES, self.rows):
                self.section(phase + '.trace', lambda p=phase, r=row: self.trace(p, r))
            self.section('candidates', self.candidates)
            if 'candidates' in self.completed:
                self.section('c2', lambda: self.group('c2', ['c2_fault']))
                self.section('c3', lambda: self.group('c3', ['c3_fault_0', 'c3_fault_1']))
                self.section('cross_group', self.cross_group)
                self.section('live_c2_issuance_use', self.live_proof)
            self.section('public_rpc', self.rpc)
            self.section('tenth_health', self.health)
        all_valid = not self.issues and len(self.completed) == 18
        c2_sections = {'envelope', 'requests', 'candidates', 'c2'} | {phase + '.trace' for phase in BASE.PHASES[:6]}
        c2_valid = c2_sections.issubset(self.completed)
        c3_valid = c2_valid and {'c3', 'cross_group'}.issubset(self.completed) and all(
            phase + '.trace' in self.completed for phase in BASE.PHASES[6:])
        public_valid = {'public_rpc', 'tenth_health'}.issubset(self.completed)
        return dict(schema='demoncat.worker_core_rope_independent_review.v1',
            status='EVIDENCE_VERIFIED' if all_valid else 'EVIDENCE_REJECTED',
            actual_natural_targets=len(self.paths), checked_conditions=self.checks,
            c2_execution_evidence_valid=c2_valid, c3_execution_evidence_valid=c3_valid,
            c2_review_passed=c2_valid and public_valid and not self.cpu,
            c3_review_passed=c3_valid and public_valid and 'live_c2_issuance_use' in self.completed and not self.cpu,
            live_c2_issuance_use_evidence_valid='live_c2_issuance_use' in self.completed,
            live_capability_reconstructed=False, public_qualification_granted=False,
            raw_device_payload_qualified=False, integer_overflow_qualified=False,
            E5='NOT_ASSESSED_SEPARATE_RAW_STATISTICS_REQUIRED', cpu_contract=self.cpu,
            source=self.source, issues=self.issues,
            limits=['Original issued/consumed object-binding receipts audited; JSON is never live authority.',
                    'Full reader/alias/byte/call checks reuse the pinned source46 evaluator unchanged.',
                    'No raw statistics, device ABI payload, yellow-zone adoption or CANN 9.x/A3 acceptance.'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--expected-targets', type=int)
    args = parser.parse_args(argv)
    try:
        if args.expected_targets is not None and args.expected_targets <= 0:
            raise ValueError('POSITIVE_EXPECTED_TARGET_COUNT_REQUIRED')
        source = verify_source(args.source)
        result, host, receipts, digest = read_artifacts(args.result, RESULT_SHA256)
        report = CoreReview(result, host, receipts, source=source, expected_targets=args.expected_targets).run()
        report['result_sha256'] = digest
    except (OSError, ValueError, TypeError, KeyError, IndexError, AttributeError, RecursionError) as error:
        report = dict(schema='demoncat.worker_core_rope_independent_review.v1', status='EVIDENCE_REJECTED',
            c2_review_passed=False, c3_review_passed=False, public_qualification_granted=False,
            live_capability_reconstructed=False, issues=[dict(section='input', code=str(error)[:180])])
    print(json.dumps(report, sort_keys=True, separators=(',', ':'), allow_nan=False))
    return 0 if report.get('c2_review_passed') and report.get('c3_review_passed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
