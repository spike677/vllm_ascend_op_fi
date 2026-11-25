"""Y1 control-path regressions. CPU fixtures do not simulate Native execution."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_product_automatic import Runner, Session
from demoncat.runtime.vllm_ascend.worker_extension.product_observer import AutomaticRequests
from demoncat.runtime.vllm_ascend.worker_extension import product_config, product_binding, product_native


class OwnerLifecycleTests(unittest.TestCase):
    def binding(self, runner, owner):
        return NS(worker=NS(model_runner=runner),check=Mock(),prompts=((1,2),),
                  warmups=[],group={'owner':owner})

    def test_two_owners_and_delayed_idempotent_close_preserve_new_owner(self):
        runner=Runner()
        old=AutomaticRequests(self.binding(runner,'old'),factory=Session)
        old.install();old.close()
        new=AutomaticRequests(self.binding(runner,'new'),factory=Session)
        new.install()
        installed=dict(new.patches)
        old.close()  # A late duplicate old cleanup cannot remove new wrappers.
        self.assertTrue(all(vars(runner)[n] is f for n,f in installed.items()))
        new.close()
        self.assertEqual(vars(runner),{})

    def test_missing_previous_close_rejects_at_install_not_second_disable(self):
        runner=Runner()
        old=AutomaticRequests(self.binding(runner,'old'),factory=Session);old.install()
        new=AutomaticRequests(self.binding(runner,'new'),factory=Session)
        with self.assertRaisesRegex(RuntimeError,'BOUNDARY_ALREADY_OWNED:execute_model'):new.install()
        old.close()
        self.assertFalse(new.patches)

    def test_foreign_replacement_refuses_without_deleting_any_wrapper(self):
        runner=Runner()
        observer=AutomaticRequests(self.binding(runner,'current'),factory=Session);observer.install()
        original=runner.sample_tokens
        foreign=lambda:original()
        runner.sample_tokens=foreign
        with self.assertRaisesRegex(RuntimeError,'PRODUCT_AUTOMATIC_WRAPPER_OWNER_CHANGED'):observer.close()
        self.assertIs(runner.sample_tokens,foreign)
        self.assertEqual(len(observer.patches),2)
        # Explicit fixture restoration, never a proposed recovery for production.
        runner.sample_tokens=original;observer.close()


class NativeConfigureTests(unittest.TestCase):
    def test_three_native_variants_share_configure_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);library=root/'cpu-fixture.so';library.write_bytes(b'CPU fixture only')
            sha=hashlib.sha256(library.read_bytes()).hexdigest()
            manifest=root/'manifest.json';manifest.write_bytes(b'{}')
            for mechanism in ('41.tail_column','26.fia_destination','20.buffer_substitute'):
                with self.subTest(mechanism=mechanism):
                    number,variant=mechanism.split('.',1)
                    authority=NS(source='YELLOW_DEDICATED_INSTANCE',authorize_operator=lambda *a:True)
                    worker=NS(model_runner=NS(vllm_config=NS(model_config=NS(model='/model'),
                        parallel_config=NS(tensor_parallel_size=2))))
                    deployment=dict(schema='demoncat.product.deployment.v1',profile=mechanism,
                        fault=dict(fault_id=int(number),variant=variant,implementation_version='1'),
                        model='/model',tp=2,mode='PRODUCT_ACCEPTANCE_ONLY',source=authority.source,
                        source_manifest=str(manifest),source_manifest_sha256=hashlib.sha256(b'{}').hexdigest(),
                        native_library=str(library),native_sha256=sha,
                        canary=dict(prompt='known canary',image=None,max_tokens=4),evidence_directory=tmp)
                    path=root/'deployment.json';path.write_text(json.dumps(deployment))
                    with patch('demoncat.runtime.vllm_ascend.worker_extension.deployment_authorization.authority_for',return_value=authority), \
                         patch.object(product_binding,'authority_for',return_value=authority), \
                         patch.object(product_config,'validate_blue_native_worker'), \
                         patch.object(product_binding,'validate_blue_native_worker'), \
                         patch.object(product_native,'BLUE_NATIVE_LIBRARY_SHA256',sha):
                        result=product_config.configure_product_deployments(worker,[str(path)],
                            authority=authority,operator='trusted',canary_tokens={mechanism:[1,2]})
                        self.assertEqual(result['status'],'PRODUCT_DEPLOYMENT_FILES_CONFIGURED')
                        profile=worker._demoncat_product_profiles[mechanism]
                        self.assertEqual(profile['native_library'],(str(library),sha))
                        self.assertEqual(profile['mechanism'],mechanism)
                        deployment['native_sha256']='0'*64;path.write_text(json.dumps(deployment))
                        with self.assertRaisesRegex(RuntimeError,'REVIEWED_LIBRARY_REQUIRED'):
                            product_config.load_product_deployment(path)


if __name__=='__main__':unittest.main()
