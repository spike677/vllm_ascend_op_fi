"""Verify public JSON evidence or restore its fixed allowlist from a private archive.

No torch, NPU, network, or Git operations are performed. Original selected bytes
are preserved. Completion receipts deliberately omit prompts and generated data.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parent
SEEDS = {"README.md", "restore_subset.py", "verify_publication.py", "selection.json", ".gitattributes"}
ALLOWED_SUFFIXES = {".json", ".md", ".py", ".cpp"}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise ValueError("Unsafe relative path")
    return Path(*path.parts)


def write_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def verify(directory):
    checks = json.loads((directory / "SHA256.json").read_bytes())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != set(checks["files"]) | {"SHA256.json"}:
        raise ValueError("Missing or extra public files")
    total = 0
    for name, info in checks["files"].items():
        path = directory / safe_relative(name)
        if path.suffix not in ALLOWED_SUFFIXES and name != ".gitattributes":
            raise ValueError("Binary or unsupported file extension")
        data = path.read_bytes()
        if sha(data) != info["sha256"] or len(data) != info["bytes"]:
            raise ValueError("Public file checksum mismatch: " + name)
        if path.suffix == ".json":
            json.loads(data)
        else:
            data.decode("utf-8")
        total += len(data)
    selection = json.loads((directory / "selection.json").read_bytes())
    for item in selection["original_files"]:
        data = (directory / safe_relative(item["published_path"])).read_bytes()
        if sha(data) != item["source_sha256"] or len(data) != item["source_bytes"]:
            raise ValueError("Original selected bytes changed")
    frozen = json.loads((directory / "source/SHA256.json").read_bytes())
    for name, expected in frozen.items():
        if sha((directory / "source" / safe_relative(name)).read_bytes()) != expected:
            raise ValueError("Frozen source checksum mismatch")
    return {"status": "PASS", "checked_files": len(checks["files"]),
            "checked_bytes_excluding_checksum_manifest": total,
            "byte_identical_original_files": len(selection["original_files"]),
            "tensor_binary_files": 0}


def restore(package, output, populate):
    selection_bytes = (ROOT / "selection.json").read_bytes()
    selection = json.loads(selection_bytes)
    manifest_bytes = (package / "manifest.json").read_bytes()
    if sha(manifest_bytes) != selection["private_manifest_sha256"]:
        raise ValueError("Unexpected private source manifest")
    manifest = json.loads(manifest_bytes)
    archives = {}

    def read_original(name, expected_sha, expected_size):
        info = manifest["files"][name]
        if info["sha256"] != expected_sha or info["size"] != expected_size:
            raise ValueError("Source selection does not match manifest")
        chunks = []
        for digest in info["segments"]:
            blob = manifest["blobs"][digest]
            shard = blob["shard"]
            if Path(shard).name != shard or ":" in shard or "\\" in shard:
                raise ValueError("Unsafe archive path")
            if shard not in archives:
                archive_path = package / shard
                shard_info = manifest["shards"][shard]
                with archive_path.open("rb") as stream:
                    actual_sha = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual_sha != shard_info["sha256"] or archive_path.stat().st_size != shard_info["size"]:
                    raise ValueError("Private archive checksum mismatch")
                archives[shard] = zipfile.ZipFile(archive_path)
            data = archives[shard].read("blobs/" + digest)
            if sha(data) != digest or len(data) != blob["size"]:
                raise ValueError("Private chunk checksum mismatch")
            chunks.append(data)
        data = b"".join(chunks)
        if sha(data) != expected_sha or len(data) != expected_size:
            raise ValueError("Reconstructed original checksum mismatch")
        return data

    if output.exists():
        present = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
        if not populate or present - SEEDS:
            raise ValueError("Output must be new, or contain only seed files with --populate")
    else:
        output.mkdir(parents=True)
    try:
        for item in selection["original_files"]:
            destination = output / safe_relative(item["published_path"])
            if destination.suffix not in {".json", ".py", ".cpp"}:
                raise ValueError("Selection includes unsupported content")
            data = read_original(item["source_path"], item["source_sha256"], item["source_bytes"])
            if destination.suffix == ".json":
                json.loads(data)
            else:
                data.decode("utf-8")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(data)
        receipts = []
        for item in selection["derived_receipt_sources"]:
            original = json.loads(read_original(item["source_path"], item["source_sha256"], item["source_bytes"]))
            if len(original["runs"]) != 1:
                raise ValueError("Expected one original run")
            run = original["runs"][0]
            receipts.append({"run_id": item["run_id"], "published_role": item["published_role"],
                "source_path": item["source_path"], "source_sha256": item["source_sha256"],
                "status": original["status"], "backend": original["backend"], "observe": original["observe"],
                "max_tokens": original["max_tokens"], "observation_window": original["observation_window"],
                "condition": run["condition"], "stage": run["stage"], "observation_step": run["observation_step"],
                "rank_contract_shapes": {rank: {"shape": value["shape"], "dtype": value["dtype"]}
                                         for rank, value in run["contracts"].items()},
                "backend_hits": run["backend_hits"]})
        write_json(output / "records/derived_completion_receipts.json", {
            "format": "derived-completion-receipts-v1", "derived": True,
            "note": "Selected completion fields from private result.json; not byte-identical originals. Prompts, token IDs, generated text and logprobs are omitted.",
            "runs": receipts})
        for name in SEEDS:
            data = (ROOT / name).read_bytes()
            if (output / name).resolve() != (ROOT / name).resolve():
                (output / name).write_bytes(data)
        checks = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.name != "SHA256.json":
                data = path.read_bytes()
                checks[path.relative_to(output).as_posix()] = {"bytes": len(data), "sha256": sha(data)}
            elif path.is_file() and path.parent != output:
                data = path.read_bytes()
                checks[path.relative_to(output).as_posix()] = {"bytes": len(data), "sha256": sha(data)}
        write_json(output / "SHA256.json", {"algorithm": "sha256", "files": checks})
    finally:
        for archive in archives.values():
            archive.close()
    return verify(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", help="Verify this public directory without private archives")
    parser.add_argument("--package", type=Path, help="Private source package directory containing manifest.json")
    parser.add_argument("--output", type=Path, help="New destination directory")
    parser.add_argument("--populate", action="store_true", help="Allow a destination containing only seed files")
    args = parser.parse_args()
    if args.verify:
        if args.package or args.output or args.populate:
            parser.error("--verify cannot be combined with restoration options")
        result = verify(ROOT)
    else:
        if not args.package or not args.output:
            parser.error("Use --verify, or supply --package and --output")
        result = restore(args.package, args.output, args.populate)
    print(json.dumps(result, ensure_ascii=False))
