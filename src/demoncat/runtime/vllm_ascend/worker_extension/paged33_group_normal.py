"""Normal-only execution of the separate group package; no fault admission."""
import json
import os
import threading
import time
import uuid

from .paged33_all_normal import AllNormalWorkerSession
from .paged33_arch22_bridge import Arch22NormalNativeCall,validate_normal_call,validate_scalar_geometry
from .paged33_arch22_s2_normal import verify_bundle as _verify_bundle
from .paged33_logical_binding import LogicalPagedInput
from .paged33_normal_targets import NormalTargetScope
from .paged33_boundary_probe import read_scoped_launches
from .serving_context import require
from fault_injection.native_adapters.attention28.launch import verify_launch
from fault_injection.native_adapters.attention28.fia.arch22_s2_build_evidence import verify_scope_geometry

VARIANT='tnd_page16_arch22_group_s2_256'


def verify_bundle(completion,completion_sha256,vendor):
    return _verify_bundle(completion,completion_sha256,vendor,group_s2=True)


def validate_group_host(host,boundary,scope,*,owner,generation,token,phase,admission,coverage_generation):
    tile=256 if phase=='fault' else 512
    ordinal=scope.check(scope.worker,boundary['module_path'],scope.model.get_submodule(boundary['module_path']),
                        scope.owner,generation)
    require(phase in ('baseline','off','fault') and host.get('schema')=='demoncat.paged33.arch22_group_s2_host.v1'
        and host.get('mode')==('arch22_group_s2_256' if tile==256 else 'arch22_group_s2_off')
        and host.get('phase')==phase and host.get('owner')==owner and host.get('token')==token
        and type(host.get('generation')) is int and host['generation']==generation
        and host.get('target_path')==boundary['module_path'] and host.get('module_object_id')==boundary['module_object_id']
        and host.get('target_set_sha256')==scope.fingerprint and type(host.get('target_ordinal')) is int
        and host['target_ordinal']==ordinal and (host.get('pid'),host.get('tid'))==(os.getpid(),threading.get_native_id())
        and host.get('normal_only') is (tile==512) and host.get('policy_applied') is (tile==256)
        and host.get('native_s2')==512 and host.get('selected_s2')==tile
        and host.get('actual_observer_scope_matched') is True and host.get('eligible') is True
        and host.get('route')=='FiaTilingNonQuant' and host.get('normal_route_preference_changed') is True
        and host.get('admission_sha256')==admission and type(host.get('coverage_generation')) is int
        and host['coverage_generation']==coverage_generation,'GROUP_ACTUAL_HOST_SCOPE_DIFFERS')


class GroupNormalNativeCall(Arch22NormalNativeCall):
    bundle_verifier=staticmethod(verify_bundle)

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        from fault_injection.native_adapters.attention28.runtime_log_index import RuntimeLogIndex
        self.runtime_index=RuntimeLogIndex(self.runtime_dir);self.validation_seconds=0.

    def _admission(self,phase,binding,scope,owner,generation):
        require(phase in ('baseline','off'),'GROUP_NORMAL_DRIVER_CANNOT_FAULT')
        return '0'*64,0

    def execute(self,actual,boundary,phase,invoke_once,*,binding,scope,owner,generation):
        require(type(binding) is LogicalPagedInput and type(scope) is NormalTargetScope
            and binding.target_scope is scope and binding.owner is owner and binding.generation==generation
            and binding.actual is actual and boundary['module_path']==binding.module_path
            and boundary['module_object_id']==id(binding.module) and boundary['sequence_lengths']==[1025],
            'GROUP_CURRENT_ACTUAL_INPUT_REQUIRED')
        ordinal=scope.check(binding.worker,binding.module_path,binding.module,owner,generation)
        geometry=validate_normal_call(actual,boundary)
        admission,coverage_generation=self._admission(phase,binding,scope,owner,generation)
        from .paged33_loading import validate_environment
        validate_environment(self.build['vendor'],platform_ready=True)
        token=uuid.uuid4().hex;log=self.folder/'host.jsonl';tile=256 if phase=='fault' else 512
        env=dict(DCAT33_TND_MODE='arch22_group_s2_256' if tile==256 else 'arch22_group_s2_off',
            DCAT33_TND_TOKEN=token,DCAT33_TND_LOG=str(log),DCAT33_TND_PID=str(os.getpid()),
            DCAT33_TND_TID=str(threading.get_native_id()),DCAT33_TND_OWNER=self.owner_token,
            DCAT33_TND_GENERATION=str(generation),DCAT33_TND_PHASE=phase,DCAT33_TND_TARGET=binding.module_path,
            DCAT33_TND_TARGET_SET=scope.fingerprint,DCAT33_TND_OBJECT=str(id(binding.module)),
            DCAT33_TND_ORDINAL=str(ordinal),DCAT33_TND_ADMISSION=admission,
            DCAT33_TND_COVERAGE_GENERATION=str(coverage_generation))
        def verify():
            started=time.monotonic()
            hosts=[json.loads(line) for line in log.read_text().splitlines()]
            hosts=[row for row in hosts if row.get('token')==token]
            require(len(hosts)==1,'GROUP_ONE_ACTUAL_HOST_CALLBACK_REQUIRED');host=hosts[0]
            validate_group_host(host,boundary,scope,owner=self.owner_token,generation=generation,token=token,
                                phase=phase,admission=admission,coverage_generation=coverage_generation)
            launch=verify_launch(self.runtime_dir,token,schema=self.schema,host=host,
                kernel_sha256=self.fingerprint['kernel_sha256'],kernel_metadata=self.metadata,kernel_abi=self.abi,
                runtime_index=self.runtime_index)
            require(launch['tiling_key']==103000000010300003 and launch['schedule_mode']==1
                and launch['block_dim']==host['block_dim'],'GROUP_ACTUAL_ENTRY_DIFFERS')
            tokens=validate_scalar_geometry(launch['typed'],actual,boundary,selected_s2=tile)
            rows=self.build['s2_cpu_matrix']['first_decode_rows'][tile]
            require(host['aic'] in rows,'GROUP_ACTUAL_CPU_MATRIX_ROW_MISSING')
            split=verify_scope_geometry(launch['typed'],host,rows[host['aic']])
            self.validation_seconds+=time.monotonic()-started
            return dict(call_id=token,token=token,phase=phase,module_path=binding.module_path,
                module_object_id=id(binding.module),scheduler=boundary['scheduler'],virtual_engine=boundary['virtual_engine'],
                target_ordinal=ordinal,target_set_sha256=scope.fingerprint,owner=self.owner_token,generation=generation,
                admission_sha256=admission,coverage_generation=coverage_generation,native=launch,host=host,scope=geometry,
                geometry=dict(selected_s2=tile,complete_cpu_split_join=split,public_to_native_token_window=tokens),
                original_calls=1,selection_applied=tile==256,normal_only=tile==512,cache_repacked=False,
                api_arguments_changed=False,output_mutated=False,mechanism=self.mechanism,logical_input_binding=binding.report())
        return self._run(token,env,lambda:binding.execute(invoke_once,owner=owner,generation=generation),verify)


class GroupNormalWorkerSession(AllNormalWorkerSession):
    native_type=GroupNormalNativeCall
    capture_flag='arch22_group_normal'

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from fault_injection.native_adapters.attention28.runtime_log_index import RuntimeLogIndex
        self.runtime_index=self.driver.runtime_index if self.private else RuntimeLogIndex(self.driver.runtime_dir)
        self.capture_validation_seconds=0.

    def _runtime_observation(self,token,capture):
        started=time.monotonic()
        try:return read_scoped_launches(self.driver.runtime_dir,token,expected_pid=capture.pid,
            expected_native_tid=capture.native_tid,runtime_index=self.runtime_index)
        finally:self.capture_validation_seconds+=time.monotonic()-started

    def collect(self):
        result=super().collect()
        result['runtime_validation_cost']=dict(index=self.runtime_index.report(),
            indexed_history_validation_seconds=self.capture_validation_seconds+getattr(self.driver,'validation_seconds',0.),
            raw_log_kept_complete=True,all_prior_registration_conflicts_retained=True)
        (self.folder/'runtime_result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        return result

    def capture_artifact(self,capture):
        from .paged33_tnd_probe import payload_identity
        from .paged33_boundary_probe import sha
        path=self.folder/'runtime_result.json'
        require(0<path.stat().st_size<=128*1024**2,'GROUP_NORMAL_CAPTURE_FILE_BOUND')
        return dict(schema='demoncat.paged33.group_normal_capture_ref.v1',path=str(path),
            sha256=sha(path),bytes=path.stat().st_size,payload_identity=payload_identity(capture))
