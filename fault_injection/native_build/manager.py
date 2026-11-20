"""Execute bounded, reviewed builds and bind artifacts to their real inputs."""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import uuid

from ..environment import (_elf_build_id, _loaded_library, _mapped_images, _manifest,
                           _toolkit, discover_installations, inspect_environment)
from .recipes import resolve_recipe


class BuildError(RuntimeError):
    def __init__(self, code, detail=""):
        self.code = code
        super().__init__(code + (": " + str(detail) if detail else ""))


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _json_write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def _run(argv, *, env=None, timeout=60):
    try:
        out = subprocess.run([str(a) for a in argv], capture_output=True, text=True,
                             env=env, check=False, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BuildError("BUILD_TOOL_UNAVAILABLE", str(error)) from error
    if out.returncode:
        raise BuildError("BUILD_TOOL_FAILED", {"argv": [str(a) for a in argv],
                                               "returncode": out.returncode, "stderr": out.stderr[-4000:]})
    return out.stdout


def _elf(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        header = stream.read(64)
        if (len(header) != 64 or header[:6] != b"\x7fELF\x02\x01"
                or struct.unpack_from("<H", header, 18)[0] != 183
                or struct.unpack_from("<H", header, 16)[0] != 3):
            raise BuildError("ELF64_AARCH64_SHARED_OBJECT_REQUIRED", path)
        offset = struct.unpack_from("<Q", header, 32)[0]
        size, count = struct.unpack_from("<HH", header, 54)
        length = path.stat().st_size
        if size < 56 or count == 0 or count > 1024 or offset + size * count > length:
            raise BuildError("MALFORMED_ELF_PROGRAM_TABLE", path)
        executable = False
        for index in range(count):
            stream.seek(offset + index * size)
            kind, flags, start, _, _, file_bytes, memory_bytes, _ = struct.unpack("<IIQQQQQQ", stream.read(56))
            if start + file_bytes > length or file_bytes > memory_bytes:
                raise BuildError("MALFORMED_ELF_SEGMENT", path)
            if kind == 1 and flags & 1:
                if flags & 2 or not flags & 4:
                    raise BuildError("WRITABLE_OR_UNREADABLE_ELF_CODE", path)
                executable = True
        if not executable:
            raise BuildError("ELF_EXECUTABLE_SEGMENT_REQUIRED", path)
        build_id = _elf_build_id(stream, path.stat().st_size)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=_hash(path), elf_build_id=build_id)


def _exports(path, nm):
    return {line.split()[-1].split("@")[0]
            for line in _run([nm, "-D", "--defined-only", path]).splitlines() if line.split()}


def _require_exports(path, symbols, nm):
    missing = set(symbols) - _exports(path, nm)
    if missing:
        raise BuildError("REQUIRED_NATIVE_SYMBOL_MISSING", {"path": str(path), "symbols": sorted(missing)})


def _source_inventory(root, source_roots):
    paths = set()
    for item in source_roots:
        path = root / item
        if not path.exists():
            raise BuildError("BUILD_SOURCE_MISSING", item)
        candidates = path.rglob("*") if path.is_dir() else [path]
        for file in candidates:
            if file.is_file() and file.suffix in (".py", ".sh", ".h", ".hpp", ".cpp", ".cc", ".inc", ".json"):
                resolved = file.resolve()
                if not resolved.is_relative_to(root):
                    raise BuildError("BUILD_SOURCE_OUTSIDE_REPOSITORY", file)
                paths.add(resolved)
    return {p.relative_to(root).as_posix(): _hash(p) for p in sorted(paths)}


def _headers(toolkit, required):
    hashes = {p.relative_to(toolkit).as_posix(): _hash(p)
              for base in ("include", "pkg_inc") for p in sorted((toolkit / base).rglob("*"))
              if p.is_file() and p.suffix in (".h", ".hpp", ".inc")}
    for path in required:
        if path not in hashes:
            raise BuildError("REQUIRED_NATIVE_HEADER_MISSING", path)
    return hashes


def _toolchain(*, require_patchelf=True):
    tools = {}
    names = ("bash", "g++", "nm", "readelf") + (("patchelf",) if require_patchelf else ())
    for name in names:
        found = shutil.which(os.environ.get("CXX", name) if name == "g++" else name)
        if not found:
            raise BuildError("BUILD_TOOL_UNAVAILABLE", name)
        path = Path(found).resolve(strict=True)
        tools[name] = dict(path=str(path), sha256=_hash(path), version=_run([path, "--version"]).splitlines()[0])
    tools["python3"] = dict(path=str(Path(sys.executable).resolve()), sha256=_hash(sys.executable),
                            version=platform.python_version())
    target = _run([tools["g++"]["path"], "-dumpmachine"]).strip()
    if not target.startswith("aarch64"):
        raise BuildError("COMPILER_TARGET_NOT_AARCH64", target)
    tools["g++"]["target"] = target
    # GCC's driver alone is insufficient: cc1plus, assembler and linker are
    # separately installed files and may change without replacing g++.
    for name in ("cc1plus", "collect2", "as", "ld"):
        reported = _run([tools["g++"]["path"], "-print-prog-name=" + name]).strip()
        resolved = Path(reported) if Path(reported).is_absolute() else Path(shutil.which(reported) or "/missing")
        if not resolved.is_file():
            raise BuildError("COMPILER_COMPONENT_UNAVAILABLE", name)
        tools[name] = dict(path=str(resolved.resolve()), sha256=_hash(resolved), version="compiler-resolved-component")
    return tools


def _select_toolkit(toolkit=None):
    if toolkit is not None:
        root = Path(toolkit)
        if (root / "aarch64-linux" / "include").is_dir():
            root /= "aarch64-linux"
        root = root.resolve(strict=True)
    else:
        selected = _toolkit()
        if selected["status"] == "FOUND":
            root = Path(selected["installations"][0]["toolkit"])
        else:
            if selected["ascend_home_path"]:
                raise BuildError("SELECTED_TOOLKIT_UNAVAILABLE")
            installed = discover_installations()["installations"]
            if len(installed) != 1:
                raise BuildError("TOOLKIT_ENVIRONMENT_AMBIGUOUS", [i["toolkit"] for i in installed])
            root = Path(installed[0]["toolkit"])
    info = _manifest(root / "ascend_toolkit_install.info")
    if not info or not re.fullmatch(r"(?:8|9)\.\d+\.\d+(?:[.\w-]*)", info["fields"].get("version", "")):
        raise BuildError("TOOLKIT_METADATA_NOT_REVIEWED", str(root))
    env = (root.parent if root.name == "aarch64-linux" else root) / "set_env.sh"
    if not env.is_file():
        raise BuildError("TOOLKIT_ENV_SCRIPT_MISSING", env)
    return root, info, dict(path=str(env.resolve()), sha256=_hash(env))


def _custom_api():
    """Find the installed vLLM-Ascend custom artifact, never choose a caller's .so."""
    roots = [Path("/vllm-workspace/vllm-ascend")]
    try:
        spec = util.find_spec("vllm_ascend")
    except (ValueError, ImportError):
        spec = None
    if spec and spec.origin:
        package = Path(spec.origin).parent.resolve()
        roots.extend((package, package.parent))
    pairs = {}
    for root in roots:
        # Bounded known package/build layouts; do not scan unrelated filesystems.
        layouts = (("csrc/build/libcust_opapi.so", "csrc/build/autogen/aclnn_add_rms_norm_bias.h"),
                   ("_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so",
                    "_cann_ops_custom/vendors/vllm-ascend/op_api/include/aclnn_add_rms_norm_bias.h"))
        for library, header in layouts:
            if (root / library).is_file() and (root / header).is_file():
                pairs[(root / library).resolve()] = (root / header).resolve()
    if not pairs:
        raise BuildError("CUSTOM_API_IDENTITY_REQUIRED")
    # Prefer the library actually mapped into this Worker. Without a Worker,
    # byte-identical build/install copies are equivalent; different copies are ambiguous.
    mapped = inspect_environment(additional_libraries=("libcust_opapi.so",))["loaded_libraries"]
    live = [Path(row["path"]).resolve() for row in mapped if row["name"] == "libcust_opapi.so"
            and row["status"] == "MAPPED_FILE_VERIFIED"]
    if live:
        if len(live) != 1 or live[0] not in pairs:
            raise BuildError("CUSTOM_API_MAPPED_IDENTITY_UNKNOWN")
        library = live[0]
    else:
        if len({(_hash(lib), _hash(header)) for lib, header in pairs.items()}) != 1:
            raise BuildError("CUSTOM_API_ARTIFACTS_AMBIGUOUS", sorted(map(str, pairs)))
        library = sorted(pairs, key=lambda p: ("/op_api/" not in p.as_posix(), str(p)))[0]
    header = pairs[library]
    return dict(library=str(library), header=str(header), header_sha256=_hash(header))


def _package_identity():
    out = {}
    for name in ("torch", "torch-npu", "vllm", "vllm-ascend"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


class NativeBuildManager:
    """Build only registered recipes; return discovered paths and exact manifests.

    The cache is immutable after success. A failed attempt remains available for
    diagnosis. Retrying creates a new attempt under the same exclusive cache lock.
    """
    def __init__(self, *, repository=None, cache=None):
        self.repository = Path(repository or Path(__file__).resolve().parents[2]).resolve(strict=True)
        self.cache = Path(cache or Path.home() / ".cache" / "demoncatpfi" / "native").resolve()

    def inspect(self, mechanism, operator=None, *, toolkit=None):
        recipe = resolve_recipe(mechanism, operator)
        if sys.platform != "linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise BuildError("NATIVE_BUILD_REQUIRES_LINUX_AARCH64")
        root, info, script = _select_toolkit(toolkit)
        version = info["fields"]["version"]
        if recipe.allowed_versions and version not in recipe.allowed_versions:
            raise BuildError("RECIPE_ABI_NOT_ADAPTED", {"recipe": recipe.id, "version": version})
        public_acl = recipe.backend == "public_acl_postwrite"
        toolchain = _toolchain(require_patchelf=not public_acl)
        libraries = {}
        required = ["libascendcl.so"] if public_acl else ["libopapi.so", "libnnopbase.so", "libascendcl.so"]
        if recipe.requires_pta_wheel or recipe.requires_custom:
            required.append("libopapi_nn.so")
        elif not public_acl and (root / "lib64/libopapi_nn.so").is_file():
            required.append("libopapi_nn.so")
        if recipe.requires_pta_wheel:
            required.append("libopapi_math.so")
        install_root = root.parent if root.name == "aarch64-linux" else root
        for name in required:
            path = (root / "lib64" / name).resolve(strict=True)
            if not path.is_relative_to(install_root):
                raise BuildError("MIXED_TOOLKIT_LIBRARIES", str(path))
            libraries[name] = _elf(path)
        nm = toolchain["nm"]["path"]
        if public_acl:
            _require_exports(Path(libraries["libascendcl.so"]["path"]), recipe.operator_exports, nm)
        else:
            _require_exports(Path(libraries["libopapi.so"]["path"]), recipe.operator_exports, nm)
            from fault39.tools.version_profile import PRIVATE, METADATA, ACL
            _require_exports(Path(libraries["libnnopbase.so"]["path"]), PRIVATE + METADATA, nm)
            _require_exports(Path(libraries["libascendcl.so"]["path"]), ACL, nm)
        if recipe.requires_pta_wheel:
            from fault39.tools.version_profile import AUDITED_PTA_RESOLVER_WHEEL
            try:
                distribution = metadata.distribution("torch-npu")
                wheel = Path(distribution.locate_file("torch_npu/lib/libtorch_npu.so"))
                libraries["libtorch_npu.so"] = _elf(wheel)
            except (metadata.PackageNotFoundError, OSError) as error:
                raise BuildError("PTA_RESOLVER_WHEEL_UNAVAILABLE") from error
            if libraries["libtorch_npu.so"]["sha256"] != AUDITED_PTA_RESOLVER_WHEEL:
                raise BuildError("PTA_RESOLVER_WHEEL_NOT_AUDITED")
        custom = _custom_api() if recipe.requires_custom else None
        if custom:
            libraries["libcust_opapi.so"] = _elf(custom["library"])
            _require_exports(Path(custom["library"]), ("aclnnAddRmsNormBias", "aclnnAddRmsNormBiasGetWorkspaceSize"), nm)
        environment = inspect_environment()
        inputs = dict(recipe=recipe.as_dict(), sources=_source_inventory(self.repository, recipe.source_roots),
                      toolkit=str(root), installation=info, environment_script=script,
                      headers=_headers(root, recipe.api_headers), libraries=libraries, custom=custom,
                      toolchain=toolchain, packages=_package_identity(), cxx11_abi=0,
                      soc=environment["soc"], soc_source=environment["soc_source"], driver=environment["driver"])
        return dict(schema="demoncat.native.build_inputs.v1", status="INPUTS_INSPECTED_ABI_COMPILE_REQUIRED",
                    build_key=_identity(inputs), inputs=inputs, actual_npu_calls=0,
                    native_qualified=False, runtime_soc_qualification_required=True)

    @contextmanager
    def _lock(self, key):
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{64}", key):
            raise BuildError("INVALID_BUILD_KEY")
        self.cache.mkdir(parents=True, exist_ok=True)
        path = self.cache / (key + ".lock")
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise BuildError("BUILD_CACHE_BUSY", path) from error
        try:
            with os.fdopen(descriptor, "w") as stream:
                stream.write(json.dumps(dict(pid=os.getpid(), build_key=key)))
            yield
        finally:
            path.unlink()

    def build(self, mechanism, operator=None, *, toolkit=None, timeout=1800):
        if type(timeout) is not int or not 1 <= timeout <= 7200:
            raise ValueError("Build timeout must be 1..7200 seconds")
        inspection = self.inspect(mechanism, operator, toolkit=toolkit)
        key, inputs = inspection["build_key"], inspection["inputs"]
        recipe = resolve_recipe(mechanism, operator)
        with self._lock(key):
            entry = self.cache / key
            entry.mkdir(exist_ok=True)
            index = entry / "completed.json"
            if index.exists():
                pointer = json.loads(index.read_text())
                artifact = self._contained(entry, pointer.get("attempt"))
                result = self.verify(artifact, expected_inputs=inspection)
                return dict(result, cache_hit=True)
            attempt = entry / ("attempt-" + uuid.uuid4().hex)
            attempt.mkdir()
            _json_write(attempt / "inputs.json", inspection)
            tools = attempt / "toolchain"
            tools.mkdir()
            for name, record in inputs["toolchain"].items():
                (tools / name).symlink_to(record["path"])
            native = attempt / "native"
            env = {k: v for k, v in os.environ.items()
                   if not k.startswith(("FAULT20_", "FAULT26_", "FAULT39_", "DEMONCAT_", "DCAT26_"))
                   and k not in ("LD_PRELOAD", "CXX", "CXXFLAGS", "CPPFLAGS", "LDFLAGS")}
            env.update({recipe.output_variable: str(native), "FAULT39_ASCEND_TOOLKIT": inputs["toolkit"],
                        "DCAT_BUILD_TOOLKIT": inputs["toolkit"]})
            if "patchelf" in inputs["toolchain"]:
                env["FAULT39_PATCHELF"] = str(tools / "patchelf")
            if inputs["custom"]:
                env.update(DCAT_BUILD_CUSTOM_LIBRARY=inputs["custom"]["library"],
                           DCAT_BUILD_CUSTOM_HEADER=inputs["custom"]["header"])
            command = [inputs["toolchain"]["bash"]["path"], "-c",
                       'set -euo pipefail; source "$1"; export PATH="$2:$PATH"; export ASCEND_HOME_PATH="$3"; exec bash "$4"',
                       "demoncat-build", inputs["environment_script"]["path"], str(tools), inputs["toolkit"],
                       str(self.repository / recipe.script)]
            started = datetime.now(timezone.utc).isoformat()
            try:
                with (attempt / "build.log").open("w", encoding="utf-8") as log:
                    process = subprocess.Popen(command, cwd=self.repository, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        returncode = process.wait(timeout=timeout)
                    except BaseException:
                        # Stop this build's process group only; never leave a timed-out compiler behind.
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGTERM)
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(process.pid, signal.SIGKILL)
                                process.wait()
                        raise
                if returncode:
                    raise BuildError("NATIVE_BUILD_FAILED", {"returncode": returncode, "log": str(attempt / "build.log")})
                # Inputs are reread after compilation; a concurrent source/library edit invalidates the result.
                after = self.inspect(mechanism, operator, toolkit=inputs["toolkit"])
                if after["build_key"] != key:
                    raise BuildError("BUILD_INPUTS_CHANGED_DURING_COMPILATION")
                _require_exports(native / recipe.runtime, recipe.runtime_symbols, inputs["toolchain"]["nm"]["path"])
                if recipe.requires_custom:
                    facts = json.loads((native / "abi_facts.json").read_text())
                    if facts.get("custom_signature") != "verified" or facts.get("actual_npu_calls") != 0:
                        raise BuildError("CUSTOM_ABI_PROBE_NOT_PASSED")
                elif recipe.backend != "public_acl_postwrite":
                    from fault39.tools.deployment_profile import verify_build, verify_installed
                    profile = verify_build(native)
                    verify_installed(profile)
                files = {p.relative_to(attempt).as_posix(): _hash(p) for p in sorted(native.rglob("*")) if p.is_file()}
                images = {name: _elf(attempt / name) for name in files if name.endswith(".so")}
                receipt = dict(schema="demoncat.native.build_receipt.v1", status="BUILD_VERIFIED_NO_NPU_QUALIFICATION",
                               build_key=key, recipe=recipe.id, mechanism=recipe.mechanism, operator=recipe.operator,
                               inputs_sha256=_hash(attempt / "inputs.json"), artifacts=files, images=images,
                               started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                               log_sha256=_hash(attempt / "build.log"), actual_npu_calls=0, native_qualified=False)
                _json_write(attempt / "receipt.json", receipt)
                _json_write(index, dict(attempt=attempt.name, build_key=key))
                return dict(receipt, artifact_directory=str(attempt), native_directory=str(native), cache_hit=False)
            except BaseException as error:
                _json_write(attempt / "failure.json", dict(status="BUILD_FAILED", error_type=type(error).__name__,
                            reason=str(error), build_key=key, log=str(attempt / "build.log"), actual_npu_calls=0))
                raise

    @staticmethod
    def _contained(root, relative):
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise BuildError("INVALID_ARTIFACT_PATH", relative)
        path = (root / relative).resolve(strict=True)
        if not path.is_relative_to(root.resolve()):
            raise BuildError("ARTIFACT_ESCAPES_BUILD", relative)
        return path

    def verify(self, artifact, *, expected_inputs=None, installed=True):
        artifact = Path(artifact).resolve(strict=True)
        receipt = json.loads((artifact / "receipt.json").read_text())
        inspection = json.loads((artifact / "inputs.json").read_text())
        inputs = inspection["inputs"]
        if (receipt.get("schema") != "demoncat.native.build_receipt.v1"
                or receipt.get("status") != "BUILD_VERIFIED_NO_NPU_QUALIFICATION"
                or type(receipt.get("actual_npu_calls")) is not int or receipt["actual_npu_calls"] != 0
                or receipt.get("native_qualified") is not False
                or receipt.get("inputs_sha256") != _hash(artifact / "inputs.json")
                or receipt.get("build_key") != _identity(inputs) or inspection.get("build_key") != receipt["build_key"]):
            raise BuildError("NATIVE_BUILD_RECEIPT_MISMATCH")
        recipe = resolve_recipe(receipt["mechanism"], receipt["operator"])
        if receipt["recipe"] != recipe.id or inputs["recipe"] != json.loads(json.dumps(recipe.as_dict())):
            raise BuildError("NATIVE_RECIPE_CHANGED")
        if expected_inputs is not None and expected_inputs["build_key"] != receipt["build_key"]:
            raise BuildError("CACHE_INPUTS_MISMATCH")
        if _source_inventory(self.repository, recipe.source_roots) != inputs["sources"]:
            raise BuildError("REPOSITORY_SOURCES_CHANGED")
        files = receipt.get("artifacts", {})
        expected_names = {p.relative_to(artifact).as_posix() for p in (artifact / "native").rglob("*") if p.is_file()}
        if not files or set(files) != expected_names or "native/" + recipe.runtime not in files:
            raise BuildError("BUILD_ARTIFACT_SET_MISMATCH")
        for name, digest in files.items():
            path = self._contained(artifact, name)
            if _hash(path) != digest:
                raise BuildError("BUILD_ARTIFACT_MISMATCH", name)
            if name.endswith(".so"):
                actual = _elf(path)
                recorded = receipt["images"].get(name, {})
                if any(recorded.get(k) != actual[k] for k in ("sha256", "bytes", "elf_build_id")):
                    raise BuildError("BUILD_ELF_IDENTITY_MISMATCH", name)
        if installed:
            for name, record in inputs["libraries"].items():
                if _hash(record["path"]) != record["sha256"]:
                    raise BuildError("INSTALLED_LIBRARY_CHANGED", name)
            if _hash(inputs["environment_script"]["path"]) != inputs["environment_script"]["sha256"]:
                raise BuildError("INSTALLED_ENVIRONMENT_CHANGED")
            if _headers(Path(inputs["toolkit"]), recipe.api_headers) != inputs["headers"]:
                raise BuildError("INSTALLED_HEADERS_CHANGED")
            custom = inputs.get("custom")
            if custom and _hash(custom["header"]) != custom["header_sha256"]:
                raise BuildError("CUSTOM_API_HEADER_CHANGED")
        return dict(receipt, artifact_directory=str(artifact), native_directory=str(artifact / "native"))

    def verify_loaded(self, artifact, *, pid):
        """Read the actual process mappings. Provider runtime attestation stays separate."""
        if type(pid) is not int or pid <= 0:
            raise ValueError("Expected a positive PID")
        result = self.verify(artifact)
        artifact = Path(result["artifact_directory"])
        inputs = json.loads((artifact / "inputs.json").read_text())["inputs"]
        recipe = resolve_recipe(result["mechanism"], result["operator"])
        unused26 = ({"libfault26_qmm_output_underflow_abi0.so", "libfault26_fia_writer_proxy_abi0.so"}
                    - {recipe.runtime}) if recipe.requires_custom else set()
        expected = {Path(name).name: record for name, record in result["images"].items()
                    if Path(name).name not in unused26}
        for name in ("libnnopbase.so", "libascendcl.so", "libtorch_npu.so"):
            if name in inputs["libraries"]:
                expected[name] = inputs["libraries"][name]
        proc = Path("/proc") / str(pid)
        def start_identity():
            stat = (proc / "stat").read_text()
            return stat.rsplit(")", 1)[1].split()[19]
        try:
            start = start_identity()
            before = _mapped_images((proc / "maps").read_text(errors="surrogateescape"), set(expected))
            rows = [_loaded_library(key, offsets) for key, offsets in before.items()]
            after = _mapped_images((proc / "maps").read_text(errors="surrogateescape"), set(expected))
            if start_identity() != start or before != after:
                raise BuildError("WORKER_MAPPINGS_CHANGED")
        except OSError as error:
            raise BuildError("WORKER_MAPPINGS_UNAVAILABLE", error) from error
        for name, record in expected.items():
            matching = [row for row in rows if row["name"] == name]
            if (len(matching) != 1 or matching[0]["status"] != "MAPPED_FILE_VERIFIED"
                    or matching[0]["sha256"] != record["sha256"]):
                raise BuildError("WORKER_LOADED_LIBRARY_MISMATCH", name)
        return dict(schema="demoncat.native.loaded_receipt.v1", status="MAPPED_FILES_MATCH_BUILD",
                    pid=pid, process_start_ticks=start, build_key=result["build_key"], libraries=rows,
                    has_native_code_attestation=False, native_qualified=False,
                    required_next="PROVIDER_RUNTIME_ATTESTATION_AND_LOSSLESS_REAL_EXECUTION")
