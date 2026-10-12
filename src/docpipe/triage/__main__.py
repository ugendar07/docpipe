"""Stage 2: label every page born_digital / scanned / hybrid / blank_or_vector / needs_review. Rerunnable alone."""
import csv, json, random, unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from docpipe.scripts import share
import pymupdf
from docpipe.config import load

LABELS = ["born_digital", "hybrid", "scanned", "blank_or_vector", "needs_review"]
GRID = 64


def garbage_score(text: str) -> float:
    """Share of non-space chars that are private-use/control/unassigned (broken font encoding)."""
    chars = [c for c in text if not c.isspace()]
    bad = sum(1 for c in chars if unicodedata.category(c) in ("Co", "Cn", "Cc", "Cs"))
    return bad / len(chars) if chars else 0.0


def stray_count(text: str, stray: str = "") -> int:
    return sum(1 for c in text if c in stray)


def control_count(text: str) -> int:
    """C0 control chars other than ordinary whitespace/form feed: glyph-offset fonts turn spaces/digits into these.
    C1 chars (0x80-0x9f, e.g. a lone symbol-font bullet) are NOT counted: the surrounding text is usually fine."""
    return sum(1 for c in text if ord(c) < 32 and c not in "\t\n\r\x0b\x0c")


def image_coverage(page) -> float:
    """Fraction of the page covered by images (union, on a coarse grid; good to ~2%)."""
    r, mask = page.rect, np.zeros((GRID, GRID), bool)
    for b in page.get_image_info():
        x0, y0, x1, y1 = b["bbox"]
        mask[int(max(0, (y0 - r.y0) / r.height) * GRID):int(np.ceil(min(1, (y1 - r.y0) / r.height) * GRID)),
             int(max(0, (x0 - r.x0) / r.width) * GRID):int(np.ceil(min(1, (x1 - r.x0) / r.width) * GRID))] = True
    return float(mask.mean())


def classify(chars, cov, garbage, stray, cfg, control=0):
    """-> (label, reason)"""
    if chars < cfg["min_chars"]:
        return ("scanned", "no text layer, image") if cov >= cfg["image_scanned_cov"] else ("blank_or_vector", "no text layer, no image")
    if chars < cfg["review_chars"]:
        return "needs_review", f"{chars} chars"
    if garbage > cfg["garbage_max"]:
        return "scanned", f"garbage text layer (control/PUA share {garbage:.3f})"
    if stray > cfg["stray_max"]:
        return "scanned", f"garbage text layer ({stray} stray glyphs)"
    if control > cfg["control_max"]:
        return "scanned", f"garbage text layer ({control} control chars, glyph-offset font)"
    if cov >= cfg["image_hybrid_cov"]:
        return "hybrid", "text layer + full-page image"
    return "born_digital", "text layer"


def triage_pdf(path, doc_id, cfg):
    out = []
    with pymupdf.open(path) as d:
        for i, p in enumerate(d, 1):
            t = p.get_text()
            chars, cov, g, st, ctl = len(t.strip()), image_coverage(p), garbage_score(t), stray_count(t, cfg["stray_glyphs"]), control_count(t)
            label, reason = classify(chars, cov, g, st, cfg, ctl)
            out.append(dict(doc_id=doc_id, page=i, chars=chars, words=len(t.split()), image_cov=round(cov, 3),
                            garbage=round(g, 4), stray=st, control=ctl, script_share={s: round(share(t, s)[0], 3) for s in cfg["_languages"]["scripts"][1:]}, label=label, reason=reason))
    return out


def run(cfg):
    docs = list(csv.DictReader(open(cfg["sources"], encoding="utf-8")))
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for r in docs:
        recs = triage_pdf(r["path"], r["doc_id"], cfg)
        (out_dir / f"{r['doc_id']}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in recs), encoding="utf-8")
        pages += recs

    tot, per_doc = Counter(x["label"] for x in pages), defaultdict(Counter)
    for x in pages:
        per_doc[x["doc_id"]][x["label"]] += 1
    L = ["# Triage report", "", f"Pages: {len(pages)} in {len(docs)} documents", "", "## Totals", "", "| label | pages | % |", "|---|---|---|"]
    L += [f"| {l} | {tot[l]} | {100 * tot[l] / len(pages):.1f} |" for l in LABELS]
    L += ["", "## By document", "", "| doc_id | pages | " + " | ".join(LABELS) + " |", "|---|---|" + "---|" * len(LABELS)]
    L += [f"| {d} | {sum(c.values())} | " + " | ".join(str(c[l]) for l in LABELS) + " |" for d, c in sorted(per_doc.items())]
    gar = [x for x in pages if x["reason"].startswith("garbage")]
    L += ["", f"## Text layer rejected as garbage ({len(gar)} pages)", ""]
    L += [f"- {d}: pages {[x['page'] for x in gar if x['doc_id'] == d]}" for d in sorted({x['doc_id'] for x in gar})] or ["- none"]
    rev = defaultdict(list)
    for x in pages:
        if x["label"] == "needs_review":
            rev[x["doc_id"]].append(x["page"])
    L += ["", f"## needs_review ({sum(map(len, rev.values()))} pages)", ""] + [f"- {d}: {v}" for d, v in sorted(rev.items())]
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")

    # thumbnails for eyeballing, seeded
    rng, sd = random.Random(cfg["seed"]), Path(cfg["samples_dir"])
    sd.mkdir(parents=True, exist_ok=True)
    for f in sd.glob("*.png"):
        f.unlink()
    path_of = {r["doc_id"]: r["path"] for r in docs}
    for l in LABELS:
        cand = sorted((x["doc_id"], x["page"]) for x in pages if x["label"] == l)
        for d, n in rng.sample(cand, min(cfg["samples_per_label"], len(cand))):
            with pymupdf.open(path_of[d]) as doc:
                doc[n - 1].get_pixmap(matrix=pymupdf.Matrix(0.4, 0.4)).save(sd / f"{l}__{d[:30]}__p{n}.png")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    run(load()["triage"])
