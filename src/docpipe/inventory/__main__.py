"""Stage 1: hash, dedupe, page-count, reconcile raw PDFs with the manifest. Rerunnable alone."""
import csv, hashlib, re, sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import unquote

import pymupdf
from docpipe.config import load

COLS = ["doc_id", "source_url", "licence_note", "group", "locality_as_published", "year",
        "sha256", "path", "pages", "match_score", "status"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unquote(s).lower().removesuffix(".pdf"))


def page_info(p: Path):
    """-> (pages, status). status: ok | encrypted | unreadable"""
    try:
        with pymupdf.open(p) as d:
            if d.needs_pass:
                return 0, "encrypted"
            return d.page_count, "ok"
    except Exception:
        return 0, "unreadable"


def match(path: Path, rows, c):
    """Best manifest row for a file: Local File suffix or ID in filename = 1.0, else similarity to URL basename.
    `c` = the manifest.columns mapping (logical name -> header)."""
    for r in rows:
        if r.get(c["file"]) and path.as_posix().endswith(r[c["file"]]):
            return r, 1.0
        if r.get(c["id"]) and r[c["id"]] in path.name:
            return r, 1.0
    stem, best, score = norm(path.name), None, 0.0
    for r in rows:
        u = norm((r.get(c["url"]) or "").rsplit("/", 1)[-1])
        if u and (s := SequenceMatcher(None, stem, u).ratio()) > score:
            best, score = r, s
    return best, round(score, 3)


def run(cfg):
    raw = Path(cfg["raw_dir"])
    c, mp = cfg["_manifest"]["columns"], cfg["_manifest"]["path"]
    rows = list(csv.DictReader(open(mp, encoding="utf-8"))) if mp and Path(mp).exists() else []   # no manifest: every file gets a generated id
    files = sorted(p for p in raw.rglob("*") if p.suffix.lower() == ".pdf")
    out, unmatched, used = [], [], defaultdict(list)
    for p in files:
        h = sha256(p)
        pages, status = page_info(p)
        r, score = match(p, rows, c)
        ok = r is not None and score >= cfg["match_threshold"]
        gm = cfg["_manifest"]["group_from"]
        folder = p.relative_to(raw).parts[0] if len(p.relative_to(raw).parts) > 1 else ""
        slug = re.sub(r"[^a-z0-9]+", "-", p.stem.lower()).strip("-")[:40]
        rec = dict(
            doc_id=r[c["id"]] if ok else "-".join(x for x in (folder, slug, h[:8]) if x),
            source_url=r.get(c["url"], "") if ok else "",
            licence_note=r.get(c["licence_note"], "") if ok else "",   # operator-supplied via the manifest column
            group=(r.get(c["group"], "") if gm == "column" and ok else folder) if gm != "none" else "",
            locality_as_published=r.get(c["locality"], "") if ok else "",
            year=(r.get(c["year"]) or "").removesuffix(".0") if ok else "",
            sha256=h, path=str(p), pages=pages, match_score=score if r else "", status=status)
        if ok:
            used[r[c["id"]]].append(p.name)
        elif rows:
            unmatched.append({**rec, "best_guess_id": r.get(c["id"], "") if r else "", "best_guess_title": r.get(c["title"], "") if r else ""})
        out.append(rec)

    # duplicates
    by_hash = defaultdict(list)
    for rec in out:
        by_hash[rec["sha256"]].append(rec["path"])
    dups = {h: v for h, v in by_hash.items() if len(v) > 1}
    # two files matched to one manifest row -> doc_id collision; fall back to own id for the later ones
    for rid, names in used.items():
        if len(names) > 1:
            print(f"WARN: manifest {rid} matched {len(names)} files: {names}", file=sys.stderr)

    Path(cfg["manifest_out"]).parent.mkdir(parents=True, exist_ok=True)
    for path, data, cols in [(cfg["manifest_out"], out, COLS),
                             (cfg["unmatched_csv"], unmatched, COLS + ["best_guess_id", "best_guess_title"])]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, cols)
            w.writeheader()
            w.writerows(data)

    bad = [r for r in out if r["status"] != "ok"]
    no_file = [r for r in rows if r.get(c["id"]) not in used]
    L = [f"# Inventory report", "",
         f"- PDFs on disk: {len(files)}  |  total pages: {sum(r['pages'] for r in out)}",
         f"- Manifest rows: {len(rows)}  |  matched to a file: {len(used)}  |  rows with no file: {len(no_file)}",
         f"- Files matched (>= {cfg['match_threshold']}): {len(out) - len(unmatched)}  |  need confirmation: {len(unmatched)} (see `reports/inventory_unmatched.csv`)",
         f"- Exact-duplicate groups: {len(dups)}  |  unreadable/encrypted: {len(bad)}", ""]
    top = lambda r: (Path(r["path"]).relative_to(raw).parts[0] if len(Path(r["path"]).relative_to(raw).parts) > 1 else "(root)")
    L += ["## Files per top-level folder"] + [f"- {k}: {sum(1 for r in out if top(r) == k)}" for k in sorted({top(r) for r in out})]
    L += ["", "## Exact duplicates"] + ([f"- {h[:12]}: " + "; ".join(v) for h, v in dups.items()] or ["- none"])
    L += ["", "## Unreadable / encrypted"] + ([f"- {r['path']} ({r['status']})" for r in bad] or ["- none"])
    L += ["", "## Manifest rows with no matching file", ""] + [f"- {r.get(c['id'])}: {(r.get(c['title']) or '')[:80]}" for r in no_file]
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:8]))


if __name__ == "__main__":
    full = load()
    run({**full["inventory"], "manifest_in": full["manifest"]["path"], "_manifest": full["manifest"]})
