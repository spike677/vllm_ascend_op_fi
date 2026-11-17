"""CPU file/transaction tests. Fixtures never represent a native qualification."""
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from fault_injection.native_build import BuildError, NativeBuildManager, recipes, resolve_recipe
from fault_injection.native_build.manager import _elf, _hash, _headers, _identity, _json_write, _source_inventory


def elf_fixture(path, machine=183, flags=5):
    # Minimal CPU parser fixture, not executable compiler output.
    data = bytearray(128)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<HH", data, 16, 3, machine)
    struct.pack_into("<Q", data, 32, 64)
    struct.pack_into("<HH", data, 54, 56, 1)
    struct.pack_into("<IIQQQQQQ", data, 64, 1, flags, 0, 0, 0, 128, 128, 4096)
    path.write_bytes(data)


class RecipeTests(unittest.TestCase):
    def test_operator_and_mechanism_have_distinct_recipe_identity(self):
        self.assertEqual(len(recipes()), 7)
        self.assertEqual(len({r.id for r in recipes()}), 7)
        weight = resolve_recipe("20.buffer_substitute", "aclnnMatmulWeightNz")
        qmm = resolve_recipe("20.buffer_substitute", "aclnnQuantMatmulWeightNz")
        self.assertNotEqual(weight.id, qmm.id)
        self.assertNotEqual(weight.runtime_symbols, qmm.runtime_symbols)
        with self.assertRaisesRegex(ValueError, "ACTUAL_OPERATOR_REQUIRED"):
            resolve_recipe("20.buffer_substitute")
        with self.assertRaisesRegex(ValueError, "NATIVE_RECIPE_NOT_ADAPTED"):
            resolve_recipe("39.matmul_range", "aclnnQuantMatmulWeightNz")

    def test_public_acl_output_proxy_does_not_inherit_private_range_profile(self):
        proxy = resolve_recipe("39.output_proxy")
        self.assertEqual(proxy.backend, "public_acl_postwrite")
        self.assertEqual(proxy.operator, "aten.mm.default")
        self.assertNotEqual(proxy.runtime, resolve_recipe("39.matmul_range").runtime)
        self.assertEqual(resolve_recipe("20.buffer_substitute", "aclnnMatmulWeightNz").allowed_versions, ("8.5.1",))

    def test_unknown_platform_refuses_before_tool_or_device_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = NativeBuildManager(repository=tmp, cache=Path(tmp) / "cache")
            with patch("fault_injection.native_build.manager.sys.platform", "win32"), \
                    patch("fault_injection.native_build.manager._toolchain") as toolchain:
                with self.assertRaisesRegex(BuildError, "NATIVE_BUILD_REQUIRES_LINUX_AARCH64"):
                    manager.inspect("39.matmul_range")
                toolchain.assert_not_called()


class FileIdentityTests(unittest.TestCase):
    def test_elf_architecture_executable_segment_and_exact_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.so"
            elf_fixture(path)
            actual = _elf(path)
            self.assertEqual(actual["sha256"], _hash(path))
            self.assertIsNone(actual["elf_build_id"])
            elf_fixture(path, machine=62)
            with self.assertRaisesRegex(BuildError, "AARCH64"):
                _elf(path)
            elf_fixture(path, flags=7)
            with self.assertRaisesRegex(BuildError, "WRITABLE"):
                _elf(path)
            path.write_bytes(b"not an ELF")
            with self.assertRaises(BuildError):
                _elf(path)

    def test_source_and_header_mutation_change_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "src").mkdir()
            source = root / "src" / "entry.cpp"
            source.write_text("first")
            before = _source_inventory(root, ("src",))
            source.write_text("changed")
            after = _source_inventory(root, ("src",))
            self.assertNotEqual(_identity(before), _identity(after))
            with self.assertRaisesRegex(BuildError, "SOURCE_MISSING"):
                _source_inventory(root, ("missing",))
            with self.assertRaisesRegex(BuildError, "HEADER_MISSING"):
                _headers(root, ("include/acl/acl.h",))

    def test_exclusive_lock_is_not_silently_stolen(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = NativeBuildManager(repository=tmp, cache=Path(tmp) / "cache")
            key = "a" * 64
            with manager._lock(key):
                with self.assertRaisesRegex(BuildError, "BUILD_CACHE_BUSY"):
                    with manager._lock(key):
                        self.fail("stole active lock")
            with manager._lock(key):
                pass
            self.assertFalse((manager.cache / (key + ".lock")).exists())
            with self.assertRaisesRegex(BuildError, "INVALID_BUILD_KEY"):
                with manager._lock("../escape"):
                    self.fail("accepted path traversal")

    def test_artifact_path_cannot_escape_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../escape", str(root.resolve()), None):
                with self.subTest(name=name), self.assertRaises(BuildError):
                    NativeBuildManager._contained(root, name)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        self.artifact = self.root / "attempt"
        self.native = self.artifact / "native"
        self.native.mkdir(parents=True)
        self.recipe = resolve_recipe("39.output_proxy")
        self.sources = {"entry.cpp": "cpu"}
        inputs = dict(recipe=json.loads(json.dumps(self.recipe.as_dict())), sources=self.sources,
                      libraries={}, toolkit=str(self.root), headers={},
                      environment_script=dict(path=str(self.root / "set_env.sh"), sha256="cpu"))
        self.inspection = dict(inputs=inputs, build_key=_identity(inputs))
        _json_write(self.artifact / "inputs.json", self.inspection)
        path = self.native / self.recipe.runtime
        elf_fixture(path)
        self.receipt = dict(schema="demoncat.native.build_receipt.v1", status="BUILD_VERIFIED_NO_NPU_QUALIFICATION",
                            build_key=self.inspection["build_key"], recipe=self.recipe.id,
                            mechanism=self.recipe.mechanism, operator=self.recipe.operator,
                            inputs_sha256=_hash(self.artifact / "inputs.json"),
                            artifacts={"native/" + path.name: _hash(path)},
                            images={"native/" + path.name: _elf(path)}, actual_npu_calls=0, native_qualified=False)
        _json_write(self.artifact / "receipt.json", self.receipt)
        self.manager = NativeBuildManager(repository=self.repo, cache=self.root / "cache")
        self.source_patch = patch("fault_injection.native_build.manager._source_inventory", return_value=self.sources)
        self.source_patch.start()
        self.addCleanup(self.source_patch.stop)

    def verify(self, **kwargs):
        return self.manager.verify(self.artifact, installed=False, **kwargs)

    def test_build_receipt_never_becomes_native_pass(self):
        result = self.verify(expected_inputs=self.inspection)
        self.assertFalse(result["native_qualified"])
        self.assertEqual(result["actual_npu_calls"], 0)
        self.assertEqual(result["native_directory"], str(self.native.resolve()))

    def test_mutated_artifact_is_rejected(self):
        with (self.native / self.recipe.runtime).open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(BuildError, "BUILD_ARTIFACT_MISMATCH"):
            self.verify()

    def test_unlisted_or_missing_artifact_is_rejected(self):
        (self.native / "extra.so").write_bytes(b"unexpected")
        with self.assertRaisesRegex(BuildError, "ARTIFACT_SET_MISMATCH"):
            self.verify()

    def test_cached_build_cannot_be_reused_for_different_inputs(self):
        with self.assertRaisesRegex(BuildError, "CACHE_INPUTS_MISMATCH"):
            self.verify(expected_inputs={"build_key": "b" * 64})
        with patch("fault_injection.native_build.manager._source_inventory", return_value={}):
            with self.assertRaisesRegex(BuildError, "REPOSITORY_SOURCES_CHANGED"):
                self.verify()

    def test_claiming_npu_execution_in_build_receipt_is_rejected(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["native_qualified"] = True
        (self.artifact / "receipt.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(BuildError, "RECEIPT_MISMATCH"):
            self.verify()

    def test_recipe_schema_cannot_be_changed_by_artifact(self):
        receipt = copy.deepcopy(self.receipt)
        receipt["recipe"] = "fault39.matmul.abi0.v1"
        (self.artifact / "receipt.json").write_text(json.dumps(receipt))
        with self.assertRaisesRegex(BuildError, "RECIPE_CHANGED"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
