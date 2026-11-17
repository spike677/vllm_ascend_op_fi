"""Download one pinned public model as preparation only; never imports model code."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
import urllib.request

REPO = "Qwen/Qwen3-VL-2B-Instruct"
REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
WEIGHT_SHA256 = "7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0"
NAMES = (
    "chat_template.json", "config.json", "generation_config.json", "merges.txt",
    "model.safetensors", "preprocessor_config.json", "tokenizer.json",
    "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url, path, expected_size):
    partial = path.with_name(path.name + ".part")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_size:
            break
        if offset > expected_size:
            raise ValueError("PARTIAL_FILE_LARGER_THAN_PINNED_SIZE")
        request = urllib.request.Request(url, headers={"Range": "bytes=%d-" % offset})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    if not response.headers.get("Content-Range", "").startswith("bytes %d-" % offset):
                        raise ValueError("RANGE_RESPONSE_MISMATCH")
                    mode = "ab"
                elif response.status == 200:
                    offset, mode = 0, "wb"
                else:
                    raise ValueError("UNEXPECTED_DOWNLOAD_STATUS")
                report_at = offset + 256 * 1024 * 1024
                with partial.open(mode) as output:
                    while True:
                        block = response.read1(8 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
                        offset += len(block)
                        if offset > expected_size:
                            raise ValueError("DOWNLOAD_EXCEEDS_PINNED_SIZE")
                        if offset >= report_at:
                            print(json.dumps({"file": path.name, "downloaded_bytes": offset}), flush=True)
                            report_at = offset + 256 * 1024 * 1024
            if partial.stat().st_size != expected_size:
                raise ValueError("DOWNLOAD_SIZE_MISMATCH")
            break
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)
    return partial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"),
                        default="https://huggingface.co")
    parser.add_argument("--metadata", type=Path, help="Previously fetched official API response")
    parser.add_argument("--metadata-sha256", help="Root-verified official response SHA256")
    parser.add_argument("--only-file", choices=NAMES,
                        help="Prepare one public file; this does not declare a complete model")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(p.name not in set(NAMES) | {n + ".part" for n in NAMES} | {"download_receipt.json"}
           for p in output.iterdir()):
        raise ValueError("OUTPUT_CONTAINS_UNRELATED_FILES")
    metadata_url = "https://huggingface.co/api/models/%s/revision/%s?blobs=true" % (REPO, REVISION)
    if args.metadata is not None:
        if not args.metadata_sha256 or digest(args.metadata) != args.metadata_sha256:
            raise ValueError("OFFICIAL_METADATA_DIGEST_MISMATCH")
        metadata = json.loads(args.metadata.read_bytes())
        metadata_sha256 = args.metadata_sha256
    else:
        if args.metadata_sha256 is not None or args.endpoint != "https://huggingface.co":
            raise ValueError("MIRROR_REQUIRES_PINNED_OFFICIAL_METADATA")
        with urllib.request.urlopen(metadata_url, timeout=30) as response:
            raw_metadata = response.read()
        metadata = json.loads(raw_metadata)
        metadata_sha256 = hashlib.sha256(raw_metadata).hexdigest()
    if metadata.get("sha") != REVISION or metadata.get("gated") is not False:
        raise ValueError("MODEL_NOT_PUBLIC_OR_REVISION_MISMATCH")
    entries = {item["rfilename"]: item for item in metadata["siblings"]}
    if entries["model.safetensors"].get("lfs", {}).get("sha256") != WEIGHT_SHA256:
        raise ValueError("WEIGHT_IDENTITY_MISMATCH")
    selected = (args.only_file,) if args.only_file else NAMES
    total = sum(entries[name]["size"] for name in selected)
    if shutil.disk_usage(output).free < total + 1024 ** 3:
        raise ValueError("INSUFFICIENT_DISK_FOR_PINNED_MODEL")
    files = []
    for name in selected:
        entry, path = entries[name], output / name
        url = "%s/%s/resolve/%s/%s" % (args.endpoint, REPO, REVISION, name)
        source = path if path.exists() else fetch(url, path, entry["size"])
        sha = digest(source)
        if source.stat().st_size != entry["size"]:
            raise ValueError("FILE_SIZE_MISMATCH: " + name)
        if name == "model.safetensors" and sha != WEIGHT_SHA256:
            raise ValueError("WEIGHT_DIGEST_MISMATCH")
        if name != "model.safetensors":
            data = source.read_bytes()
            git_blob = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
            if entry.get("blobId") != git_blob:
                raise ValueError("PINNED_CONFIG_BLOB_MISMATCH: " + name)
        if source != path:
            source.replace(path)
        files.append({"name": name, "bytes": entry["size"], "sha256": sha, "url": url})
        print(json.dumps({"file": name, "bytes": entry["size"], "sha256": sha}), flush=True)
    receipt = {"schema": "demoncat.public_model_preparation.v1", "repository": REPO,
               "revision": REVISION, "gated": False, "remote_code_downloaded": False,
               "official_metadata_url": metadata_url, "official_metadata_sha256": metadata_sha256,
               "download_endpoint": args.endpoint,
               "preparation_script_sha256": digest(Path(__file__)),
               "complete_model_files_prepared": args.only_file is None,
               "files": files, "npu_calls": 0, "model_qualification": False}
    (output / "download_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    status = ("PINNED_PUBLIC_MODEL_DOWNLOADED_NOT_VALIDATED" if args.only_file is None else
              "PINNED_PUBLIC_FILE_PREPARED_NOT_COMPLETE_MODEL")
    print(json.dumps({"status": status, "bytes": total}), flush=True)


if __name__ == "__main__":
    main()
