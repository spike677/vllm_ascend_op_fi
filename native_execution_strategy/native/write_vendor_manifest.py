#!/usr/bin/env python3
"""Fingerprint an already built private vendor. Does not build/install/run it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    for key in ("task-root", "vendor-root", "source", "source-manifest", "package", "output"):
        ap.add_argument("--" + key, type=Path, required=True)
    args = ap.parse_args()
    root = args.task_root.resolve()
    vendor = args.vendor_root.resolve()
    source = args.source.resolve()
    package = args.package.resolve()
    for p in (vendor, source, args.source_manifest.resolve(), package, args.output.resolve()):
        if not p.is_relative_to(root):
            raise ValueError(f"Path outside task root: {p}")
    if args.output.exists() or args.output.resolve().is_relative_to(vendor):
        raise ValueError("Use a new output outside the vendor subtree")
    manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != manifest["source_commit"]:
        raise ValueError("Source HEAD differs from frozen manifest")
    for item in manifest["files"]:
        p = (source / item["path"]).resolve()
        if not p.is_relative_to(source) or sha(p) != item["after_sha256"]:
            raise ValueError("Frozen source differs: " + item["path"])
    artifacts = []
    for p in sorted(vendor.rglob("*")):
        if not p.is_file() and not p.is_symlink():
            continue
        if not p.resolve().is_relative_to(vendor) or not p.is_file():
            raise ValueError("Unsafe or broken vendor symlink: " + str(p))
        artifacts.append({"path": p.relative_to(vendor).as_posix(), "bytes": p.stat().st_size,
                          "sha256": sha(p), "symlink": os.readlink(p) if p.is_symlink() else None})
    if not artifacts:
        raise ValueError("Empty vendor")
    result = {"schema": "nes.fa.private_vendor.v1", "vendor_root": vendor.relative_to(root).as_posix(),
              "artifacts": artifacts, "source_commit": commit,
              "policy_source_manifest_sha256": sha(args.source_manifest),
              "package": {"path": package.relative_to(root).as_posix(), "sha256": sha(package),
                          "bytes": package.stat().st_size},
              "status": "CPU_BUILT_AND_STAGED_NOT_NPU_VALIDATED", "global_sdk_modified": False,
              "native_replay_output_and_raw_abi_verified": False, "same_binary_native_and_controlled": True}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"path": str(args.output), "sha256": sha(args.output), "files": len(artifacts)}))


if __name__ == "__main__":
    main()
