"""Documentation-only generated audit; never deletes/moves source or evidence."""

import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit


def main():
    root = Path(__file__).resolve().parents[1]
    docs = root / "docs"
    inventory = json.loads((docs / "document_inventory_v1.json").read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    git = ["git", "-c", f"safe.directory={root.as_posix()}", "-C", str(root)]
    tracked = subprocess.check_output(git + ["ls-files", "*.md"], text=True, encoding="utf-8").splitlines()
    groups = defaultdict(list)
    for path in tracked:
        groups[sha(root / path)].append(path)
    duplicates = [{"sha256": digest, "paths": paths} for digest, paths in groups.items() if len(paths) > 1]
    checks = []
    for row in inventory["documents"]:
        checks.append({"source": row["source_repository_or_worktree"], "path": row["path"], "sha256": row["sha256"],
                       "inventory_action": row["action"], "duplicate_of": row.get("duplicate_of"),
                       "v2_action": "KEEP_CROSS_WORKTREE_READONLY" if row.get("duplicate_of") else "KEEP_SOURCE_REFERENCE",
                       "reason": "inventory spans independent Git trees; do not mutate other branches or legacy clones"})
    audit = {"schema": "DOCUMENTATION_CLEANUP_V2", "generated_utc": datetime.now(timezone.utc).isoformat(),
             "inventory_counts": {k: inventory[k] for k in ("markdown_count", "exact_duplicate_group_count", "exact_duplicate_document_count")},
             "current_tree_tracked_markdown": len(tracked), "current_tree_exact_duplicates": duplicates,
             "canonical_docs_markdown": len(list(docs.rglob("*.md"))),
             "deleted": [], "physically_archived": [], "new_stubs": [], "source_decisions": checks}
    (docs / "document_cleanup_v2.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    migration_path = docs / "document_migration_manifest.json"
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    for entry in migration["mappings"]:
        path = root / entry["new_canonical_path"]
        if path.is_file():
            current = sha(path)
            if current != entry.get("canonical_sha256"):
                entry.setdefault("v1_canonical_sha256", entry.get("canonical_sha256"))
                entry["canonical_sha256"] = current
    for name in ("docs/evaluation/distortion_evaluation_v1.md", "docs/document_cleanup_v2.md"):
        if not any(e["new_canonical_path"] == name for e in migration["mappings"]):
            migration["mappings"].append({"old_path": None, "source_branch_or_worktree": "feature/distortion-evaluation-v1",
                                          "new_canonical_path": name, "canonical_sha256": sha(root/name),
                                          "action": "KEEP_CANONICAL", "reason": "V2 addition; no historical evidence move"})
    migration["v2_cleanup_audit"] = "docs/document_cleanup_v2.json"
    migration_path.write_text(json.dumps(migration, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    links = []
    for path in docs.rglob("*.md"):
        text = re.sub(r"```.*?```", "", path.read_text(encoding="utf-8"), flags=re.S)
        targets = re.findall(r"\[[^\]]*\]\(([^)]+)\)", text)
        targets += re.findall(r"(?m)^\[[^\]]+\]:\s*(\S+)", text)
        for target in targets:
            target = target.strip().strip("<>")
            if re.match(r"(?:[a-z]+:|#|//)", target, re.I):
                continue
            url = unquote(urlsplit(target).path)
            candidate = (root / url.lstrip("/")) if url.startswith("/") else (path.parent / url)
            links.append({"source": path.relative_to(root).as_posix(), "target": target,
                          "exists": candidate.exists()})
    result = {"schema": "DOCUMENT_LINK_VALIDATION_V2", "checked_links": len(links),
              "broken_links": sum(not x["exists"] for x in links), "links": links,
              "scope": "canonical docs Markdown local file links; remote/legacy prose not dependencies"}
    (docs / "document_link_validation_v1.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"current_tree_duplicate_groups": len(duplicates), "docs_markdown": audit["canonical_docs_markdown"],
                      "checked_links": len(links), "broken": [x for x in links if not x["exists"]]}, ensure_ascii=False, indent=2))
    if result["broken_links"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
