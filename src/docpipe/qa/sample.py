"""P2-12: seeded random sample of processed pages for manual review. `python -m docpipe.qa.sample`
Writes reports/qa_sample.csv (verdict column to fill) and reports/qa_sample/index.html (page image next to the text)."""
import csv, html, json, random
from pathlib import Path

import pymupdf
from docpipe.config import load


def pick(cfg):
    rows = []
    for f in sorted(Path(cfg["text_dir"]).glob("*.jsonl")):
        rows += [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()]
    live = [r for r in rows if r["text"].strip()]
    rng = random.Random(cfg["seed"])
    tl = sorted((r for r in live if r["source"] == "text_layer"), key=lambda r: (r["doc_id"], r["page"]))
    ocr = sorted((r for r in live if r["source"] not in ("text_layer", "blank", "dropped", "ocr_missing", "ocr_failed")), key=lambda r: (r["doc_id"], r["page"]))
    # NEW in docpipe: small corpora have fewer pages than the sample size
    return rng.sample(tl, min(cfg["n_text_layer"], len(tl))) + rng.sample(ocr, min(cfg["n_ocr"], len(ocr)))


def run(cfg):
    path_of = {r["doc_id"]: r["path"] for r in csv.DictReader(open(cfg["sources"], encoding="utf-8"))}
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    sample = pick(cfg)
    cards = []
    sheet = Path(cfg["csv"])
    if sheet.exists() and any(r[cfg["_review"]["qa_column"]].strip() for r in csv.DictReader(open(sheet, encoding="utf-8"))):
        sheet = sheet.with_suffix(".regenerated.csv")   # never overwrite verdicts the user has already entered
        print(f"NOTE: {cfg['csv']} already holds verdicts; writing {sheet} instead")
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["n", "doc_id", "page", "source", "confidence", "language", "flags", cfg["_review"]["qa_column"], "NOTES"])
        for i, r in enumerate(sample, 1):
            w.writerow([i, r["doc_id"], r["page"], r["source"], r["confidence"], r["language"], ";".join(r["flags"]), "", ""])
            img = f"{r['doc_id']}_p{r['page']:04d}.png"
            with pymupdf.open(path_of[r["doc_id"]]) as d:
                d[r["page"] - 1].get_pixmap(matrix=pymupdf.Matrix(cfg["dpi"] / 72, cfg["dpi"] / 72), alpha=False).save(out / img)
            cards.append(f"<section><h3>#{i} {r['doc_id']} p{r['page']} &middot; {r['source']} &middot; conf {r['confidence']} &middot; "
                         f"lang {r['language']} &middot; flags {html.escape(', '.join(r['flags']))}</h3><div class='row'>"
                         f"<img src='{img}'><pre>{html.escape(r['text'])}</pre></div></section>")
    (out / "index.html").write_text("<!doctype html><meta charset=utf-8><title>QA sample</title><style>body{font-family:sans-serif;margin:16px}"
                                    ".row{display:flex;gap:16px}img{width:48%;border:1px solid #888}pre{width:48%;white-space:pre-wrap;font-size:12px}"
                                    "section{margin-bottom:32px;border-bottom:2px solid #ccc}</style><h1>QA sample</h1>" + "".join(cards), encoding="utf-8")
    print(f"{len(sample)} pages -> {sheet}, {out}/index.html")


if __name__ == "__main__":
    run(load()["qa_sample"])
