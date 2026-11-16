import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path=Path(__file__).resolve().parents[1]/'tools/deployment_profile.py'
spec=importlib.util.spec_from_file_location('deployment_profile',path)
deployment=importlib.util.module_from_spec(spec);spec.loader.exec_module(deployment)


class BuildProfileTests(unittest.TestCase):
    def make_build(self,root,version='9.2.0',layout='main_and_nn',adapter=None):
        profile=dict(schema='fault39.cann.build_profile.v1',toolkit='/opt/cann/aarch64-linux',proxy_layout=layout,
                     installation={'ascend_toolkit_install.info':{'version':version}},libraries={})
        if adapter is not None:
            profile['adapter'] = adapter
        profile['profile_sha256']=hashlib.sha256(json.dumps(profile,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for name in deployment.artifacts(profile):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixture, not a native build')
        (root/'version_profile.json').write_text(json.dumps(profile))
        manifest=dict(schema='fault39.native.build_manifest.v1',profile_sha256=profile['profile_sha256'],
                      files={name:deployment.sha(root/name) for name in deployment.artifacts(profile)})
        (root/'build_manifest.json').write_text(json.dumps(manifest))
        return profile

    def test_identity_not_version_string_controls_build_check(self):
        for version in ('8.3.RC2','8.5.1','9.0.0','9.1.0','9.2.0'):
            with self.subTest(version=version),tempfile.TemporaryDirectory() as d:
                root=Path(d);p=self.make_build(root,version)
                self.assertEqual(deployment.verify_build(root)['profile_sha256'],p['profile_sha256'])

    def test_modified_library_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_build(root)
            (root/'libfault39_native_v1.so').write_bytes(b'other version')
            with self.assertRaisesRegex(ValueError,'BUILD_ARTIFACT_MISMATCH'):deployment.verify_build(root)

    def test_modified_profile_is_rejected_even_with_same_version(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=self.make_build(root);p['toolkit']='/opt/another-cann'
            (root/'version_profile.json').write_text(json.dumps(p))
            with self.assertRaisesRegex(ValueError,'BUILD_PROFILE_IDENTITY_MISMATCH'):deployment.verify_build(root)

    def test_main_only_layout_does_not_require_unavailable_split_proxy(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.make_build(root,layout='main_only')
            self.assertEqual(deployment.verify_build(root)['proxy_layout'],'main_only')
            self.assertFalse((root/'libopapi_nn.so').exists())

    def test_isolated_fault20_bundle_has_distinct_identity_and_requires_split(self):
        adapter = 'fault20-isolated-cache-abi0-v1'
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.make_build(root, version='8.5.1', adapter=adapter)
            self.assertEqual(deployment.verify_build(root)['adapter'], adapter)
            self.assertTrue((root/'libdcat20_qmm.so').is_file())
            self.assertFalse((root/'libfault39_native_v1.so').exists())
            with self.assertRaisesRegex(ValueError, 'UNSUPPORTED_BUILD_ADAPTER'):
                deployment.artifacts(dict(adapter=adapter, proxy_layout='main_only'))

    def test_installed_library_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'actual.so';p.write_bytes(b'other installation')
            with self.assertRaisesRegex(ValueError,'INSTALLED_LIBRARY_MISMATCH'):
                deployment.verify_installed({'upstream_libraries':{'nn':{'path':str(p),'file_sha256':'0'*64}}})


if __name__=='__main__':unittest.main()
