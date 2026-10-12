"""Hash-compare two pipeline output roots: `python -m docpipe.qa.compare_runs <rootA> <rootB>`.
Final outputs must match byte for byte; intermediate files are listed as diagnostics."""
import hashlib, sys
from pathlib import Path

FINAL = ["data/processed/text/*.jsonl", "data/processed/metadata.csv", "data/manifest/sources.csv",
         "data/manifest/image_tables.csv", "data/processed/text_image_tables/*.jsonl"]
INTERIM = ["data/interim/triage/*.jsonl", "data/interim/pages/index.jsonl", "data/interim/pages_clean/index.jsonl",
           "data/interim/language/*.jsonl", "data/interim/confidence/*.jsonl", "data/interim/normalise/*.json",
           "data/interim/image_tables/*.png"]


def digest(root, pats):
    out = {}
    for pat in pats:
        for p in sorted(Path(root).glob(pat)):
            out[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def compare(a, b, pats):
    da, db = digest(a, pats), digest(b, pats)
    diff = sorted(k for k in set(da) | set(db) if da.get(k) != db.get(k))
    return len(da), len(db), diff


if __name__ == "__main__":
    a, b = sys.argv[1:3]
    for title, pats in (("FINAL outputs", FINAL), ("intermediate (diagnostic)", INTERIM)):
        na, nb, diff = compare(a, b, pats)
        print(f"{title}: {na} vs {nb} files, {len(diff)} differ" + ("" if not diff else ":"))
        for k in diff[:20]:
            print("  DIFF", k)
    sys.exit(0 if not compare(a, b, FINAL)[2] else 1)
