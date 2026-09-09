"""Read-only export of existing evidence. Stdout is base64 ZIP; no remote writes.

Run this file through SSH stdin when the evidence is on a separate host.
Original bytes, SHA256 and source paths are retained. It never uses NPU or RPC.
"""

import argparse
import base64
import hashlib
import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    paths = [root / "manifest.json"]
    for cycle in sorted((root / "cycles").glob("cycle_*")):
        for phase in ("baseline", "fault", "recovery"):
            for category in ("gsm8k", "short", "long", "wikitext"):
                paths.append(cycle / phase / f"{category}_responses.jsonl")
        paths.append(cycle / "cycle_summary.json")
    paths.append(root / "final_cleanup.json")
    destination, inventory = io.BytesIO(), []
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in paths:
            if not path.is_file():
                inventory.append({"path": path.relative_to(root).as_posix(), "missing": True})
                continue
            data = path.read_bytes()
            name = path.relative_to(root).as_posix()
            inventory.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
            archive.writestr(name, data)
        archive.writestr("export_manifest.json", json.dumps({
            "format": "RAW_DISTORTION_REPLAY_ZIP_V1", "source_root": str(root),
            "exported_utc": datetime.now(timezone.utc).isoformat(), "files": inventory,
            "original_evidence_modified": False,
        }, ensure_ascii=False, indent=2))
    sys.stdout.write(base64.b64encode(destination.getvalue()).decode("ascii"))


if __name__ == "__main__":
    main()
