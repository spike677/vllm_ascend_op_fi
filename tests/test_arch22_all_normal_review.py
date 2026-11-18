"""Constructed CPU byte fixtures; the all-target reader never creates proof."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/c2c3_closure_20261117'))
import review_arch22_all_normal as reader
import test_arch22_normal_bridge_review as files
import test_arch22_normal_repeat_review as repeats


def scope_fixture():
    rows=[dict(path=p,module_object_id=100+i,impl_object_id=200+i) for i,p in enumerate(reader.N.LAYERS)]
    identity=dict(owner_object_id=11,generation=1,worker_object_id=12,model_object_id=13,
                  pid=14,python_thread_ident=15,native_tid=16)
    scope=dict(schema='demoncat.paged33.all_normal_targets.v1',targets=rows,sha256=reader.E.digest(rows),
        normal_only=True,serializable_authority=False,C1=False,C2=False,C3=False,**identity)
    audit={r['path']:dict(module_object_id=r['module_object_id'],impl_object_id=r['impl_object_id']) for r in rows}
    saved={p:dict(logical_input_binding=dict(identity=dict(identity,**row))) for p,row in audit.items()}
    return dict(target_scope=scope,all_decode_boundaries=saved),audit


class AllNormalReaderContracts(unittest.TestCase):
    def test_raw_payloads_must_join_actual_final_model_and_api_returns(self):
        tensor=dict(shape=[1,1024],dtype='torch.float16',sha256='a'*64,finite=True)
        file=dict(tensor,bytes=2048)
        capture=dict(complete_tensor_readback_during_statistics=False,hidden=file,logits_hidden=deepcopy(file),
            logits=deepcopy(file),attention={reader.N.FIRST:dict(output=deepcopy(file))},
            natural_request=dict(forwards=[{},dict(hidden=deepcopy(tensor),logits=deepcopy(tensor),
                calls=[dict(module_path=reader.N.FIRST,output=[deepcopy(tensor),{}])])]))
        reader.actual_output_join(capture)
        for change in (lambda c:c['logits'].__setitem__('sha256','b'*64),
                       lambda c:c['logits_hidden'].__setitem__('sha256','b'*64),
                       lambda c:c['attention'][reader.N.FIRST]['output'].__setitem__('sha256','b'*64),
                       lambda c:c.__setitem__('complete_tensor_readback_during_statistics',True)):
            bad=deepcopy(capture);change(bad)
            with self.assertRaises(ValueError):reader.actual_output_join(bad)

    def test_complete_target_set_is_recomputed_and_layer0_evidence_cannot_fill_another_layer(self):
        capture,audit=scope_fixture()
        self.assertEqual(reader.target_set(capture,audit),list(reader.N.LAYERS))
        original=deepcopy(capture)
        changes=(lambda c:c['target_scope']['targets'].pop(),
            lambda c:c['all_decode_boundaries'].pop(reader.N.LAYERS[-1]),
            lambda c:c['all_decode_boundaries'].__setitem__(reader.N.LAYERS[1],deepcopy(c['all_decode_boundaries'][reader.N.FIRST])),
            lambda c:c['target_scope'].__setitem__('sha256','f'*64),
            lambda c:c['target_scope'].__setitem__('generation',2),
            lambda c:c['target_scope'].__setitem__('generation',True),
            lambda c:c['target_scope']['targets'][1].__setitem__('module_object_id',100))
        for index,change in enumerate(changes):
            bad=deepcopy(original);change(bad)
            with self.subTest(index=index),self.assertRaises((ValueError,KeyError)):reader.target_set(bad,audit)

    def test_nonfirst_boundary_reads_its_own_raw_files_and_keeps_strict_writer_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);phase=repeats.make_boundary(root)
            repeats.read_boundary(root,phase)  # Make the ordinary source-compatible byte fixture complete.
            capture=phase['finish']['capture'];saved=capture.pop('first_decode_boundary');path=reader.N.LAYERS[17]
            saved['boundary']['module_path']=saved['boundary']['writer']['module_path']=path
            capture['attention'][path]=capture['attention'].pop(reader.N.FIRST)
            capture['all_decode_boundaries']={path:saved}
            folder=root/'baseline_before/boundaries/17';folder.mkdir(parents=True)
            for filename in ('query.bin','output-0.bin','output-1.bin'):
                (folder/filename).write_bytes((root/'baseline_before/first_boundary'/filename).read_bytes())
            prefix='/evidence/run/baseline_before/boundaries/17/'
            saved['inputs']['query']['snapshot']['path']=prefix+'query.bin'
            for index,item in enumerate(saved['outputs']):item['path']=prefix+f'output-{index}.bin'
            (folder/'boundary.json').write_text(json.dumps(saved))
            bound=deepcopy(saved['boundary']);bound['consumer_returned']=True
            checked=repeats.boundary.first_boundary(root,phase,bound,difference_reader=repeats.differences,
                target_path=path,relative_folder='boundaries/17')
            proof=reader.N.page_relocation_evidence(saved,checked,bound,target_path=path)
            self.assertEqual(proof['summary']['writer_consumer_slot'],1024)
            with self.assertRaises(ValueError):reader.N.page_relocation_evidence(saved,checked,bound)
            (folder/'query.bin').write_bytes(b'not the same target query')
            with self.assertRaises(ValueError):repeats.boundary.first_boundary(root,phase,bound,
                difference_reader=repeats.differences,target_path=path,relative_folder='boundaries/17')

    def test_all_target_cross_arm_signature_is_strict_and_relocation_only_within_arm(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);left_phase=repeats.make_boundary(root/'left');right_phase=repeats.make_boundary(root/'right')
            repeats.relocate(right_phase,list(range(1,66))+[0]*15)
            left=repeats.read_boundary(root/'left',left_phase);right=repeats.read_boundary(root/'right',right_phase)
            a=dict(payloads=left['payloads'],targets={p:deepcopy(left['boundary']) for p in reader.N.LAYERS})
            b=dict(payloads=right['payloads'],targets={p:deepcopy(right['boundary']) for p in reader.N.LAYERS})
            helper=NS(boundary=repeats.boundary,differences=repeats.differences)
            # Payload comparison uses rehashed fixtures once, not once per target.
            with patch.object(helper.differences,'compare',wraps=helper.differences.compare) as compare:
                cross=reader.compare(a,b,helper)
                self.assertEqual(compare.call_count,1)
            self.assertFalse(cross['exact']);self.assertEqual(len(cross['targets']),28)
            within=reader.compare(a,b,helper,relocated=True);self.assertTrue(within['normal_repeat_valid'])
            self.assertFalse(within['exact'])
            b['targets'].pop(reader.N.LAYERS[17])
            with self.assertRaises(ValueError):reader.compare(a,b,helper)

    def test_new_normal_host_schema_rejects_fault_tile_and_old_package(self):
        source=files.source();helper=reader.E.helpers(source)
        event,resolved,host,bundle=files.pod_fixture(helper)
        bundle['completion']={'variant':reader.E.ALL_VARIANT}
        host.update(schema='demoncat.paged33.arch22_all_normal_host.v1',mode='arch22_all_normal')
        checked=reader.N.final_private_pod(event,resolved,host,bundle,helper.abi)
        self.assertEqual(checked['typed']['innerSplitParams.s2BaseSize'],512)
        with self.assertRaisesRegex(ValueError,'EXPLICIT_NEW_S2'):
            reader.N.final_private_pod(event,resolved,host,bundle,helper.abi,selected_s2=256)
        host['schema']='demoncat.paged33.arch22_s2_host.v1'
        with self.assertRaisesRegex(ValueError,'EXPLICIT_SELECTION'):
            reader.N.final_private_pod(event,resolved,host,bundle,helper.abi)

    def test_preparer_uses_new_package_scope_and_only_two_normal_cases(self):
        source=files.source();source['variant']=reader.E.ALL_VARIANT
        args=NS(card=5,timeout=1800,source=Path('.'),installed=Path('.'),model=Path('.'),build_root=Path('.'))
        bundle=dict(completion_sha256='a'*64,completion={'arch22_public_schema_sha256':'b'*64},summary={'variant':reader.E.ALL_VARIANT})
        observer=dict(sha256='c'*64,folder=Path('.'),alias='/build/attention-preflight',receipt={'observer_library':{'sha256':'d'*64}})
        cases=files.prepare.make_cases(args,source,bundle,observer,{}, {})
        self.assertEqual(len(cases),2)
        for case in cases:
            self.assertIn('--arch22-all-normal-bridge',case['command'])
            self.assertNotIn('--phase fault',case['command'])
            self.assertFalse(case['review']['fault_authorized'] or case['review']['c2'] or case['review']['c3'])
        self.assertIn(reader.E.ALL_VENDOR,cases[1]['command'])
        self.assertNotIn(reader.E.S2_VENDOR,cases[1]['command'])

    def test_fresh_all_normal_build_controller_cannot_be_an_old_s2_controller(self):
        frozen=files.source();frozen['variant']=reader.E.ALL_VARIANT
        frozen['pins'].update({p:reader.E.sha(files.ROOT/p) for p in reader.E.S2_REQUIRED+reader.E.ALL_REQUIRED})
        helper=reader.E.helpers(frozen)
        temporary=tempfile.TemporaryDirectory()
        if os.name=='nt':temporary.name='\\\\?\\'+str(Path(temporary.name).resolve())
        with temporary as temp,patch.dict(reader.E.CANONICAL_TOKEN_SOURCE_SHA256,
                {name:files.hash_bytes(raw) for name,raw in files.TOKEN_SOURCE_FIXTURES.items()},clear=True):
            root=Path(temp);build,data=files.bundle_fixture(root,frozen,helper)
            # Synthetic ELF/schema byte fixture, never a real build. Matrix
            # verification has its own full-row contracts; isolate namespace.
            old=build/'stage/vendors'/reader.E.VENDOR;new=old.with_name(reader.E.ALL_VENDOR);old.rename(new)
            data['variant']=reader.E.ALL_VARIANT
            data['arch22_s2_recipe_sha256']=frozen['pins'][reader.E.NATIVE+'fia/apply_tnd33_arch22_all_normal.py']
            receipt=reader.E.read(root/'patch/policy_source_manifest.json')
            receipt.update(variant=reader.E.ALL_VARIANT,recipe_sha256=data['arch22_s2_recipe_sha256'],
                selection_before_original_split=True,m_policy_changed=False,old_normal_certificate_reused=False,loading_allowed=False)
            files.write(root/'patch/policy_source_manifest.json',receipt)
            data['patch_receipt_sha256']=reader.E.sha(root/'patch/policy_source_manifest.json')
            files.write(build/'evidence/completion.json',data)
            controller=reader.E.read(root/'controller.json');controller['strategy']='arch22-all-normal'
            files.write(root/'controller.json',controller)
            matrix=dict(receipt_sha256='a'*64,matrix_sha256='b'*64,cases=22528,qualification_granted=False)
            with patch.object(helper.arch22_s2_build_evidence,'verify_inputs',return_value=matrix) as verify:
                actual=reader.E.build_bundle(build,frozen,helper)
                self.assertEqual(actual['summary']['variant'],reader.E.ALL_VARIANT)
                self.assertTrue(verify.call_args.kwargs['all_normal'])
                controller['strategy']='arch22-s2';files.write(root/'controller.json',controller)
                with self.assertRaisesRegex(ValueError,'BUILD_CONTROLLER'):reader.E.build_bundle(build,frozen,helper)


if __name__=='__main__':unittest.main()
