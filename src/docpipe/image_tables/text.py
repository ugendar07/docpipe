"""Phase 2b output: one JSONL per document in data/processed/text_image_tables/, one record per OCR'd table region, keyed by
doc_id/page/region. This is an enrichment layer kept apart from the primary text (data/processed/text/). Rerunnable alone."""
import csv, json
from pathlib import Path

from docpipe.config import load

from docpipe.image_tables.select import selection


def run(full):
    cfg, oc = full["image_tables"], full["image_tables_ocr"]
    chosen, skipped = selection(cfg); out = Path(oc["text_dir"]); out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.jsonl"):
        f.unlink()   # regenerate from the raw OCR so a changed selection leaves no stale file
    recs = {}
    for r in chosen:
        p = Path(oc["out_dir"]) / r["doc_id"] / f"{r['region_id']}.json"
        if not p.exists():
            raise SystemExit(f"missing OCR output for {r['region_id']}: run the image_tables_ocr stage")
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not raw["ok"]:
            raise SystemExit(f"OCR failed for {r['region_id']}: {raw.get('error')}")
        tables = raw["tables"] or []
        recs.setdefault(r["doc_id"], []).append(dict(
            doc_id=r["doc_id"], page=int(r["page"]), region=int(r["region"]), region_id=r["region_id"], source=oc["engine"] + "_page",
            region_source=r["source"], page_type=r["page_type"], bbox=[float(r[k]) for k in ("x0", "y0", "x1", "y1")],
            kind="table" if tables else "no_table_found", n_tables=len(tables), tables_html=tables, text=raw["text"]))
    for d, rs in recs.items():
        rs.sort(key=lambda x: (x["page"], x["region"]))
        (out / f"{d}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rs), encoding="utf-8")
    report(full, chosen, skipped, recs)
    return recs


def report(full, chosen, skipped, recs):
    allr = [x for rs in recs.values() for x in rs]
    found = sum(x["kind"] == "table" for x in allr)
    reasons = {}
    for v in skipped.values():
        reasons[v] = reasons.get(v, 0) + 1
    L = ["# Phase 2b: text from table images", "",
         f"Regions in the inventory: **{len(chosen) + len(skipped)}**. OCR'd (marked `table` by the user): **{len(chosen)}**, on "
         f"{len({(x['doc_id'], x['page']) for x in allr})} pages in {len(recs)} documents. Engine: {full['image_tables_ocr']['engine']}.",
         f"Table found in the output: **{found}**; no table found (text only, kind `no_table_found`): **{len(allr) - found}**.", "",
         "Not OCR'd, by reason: " + ", ".join(f"{k} {v}" for k, v in sorted(reasons.items())) + ".", "",
         "Output: `data/processed/text_image_tables/<doc_id>.jsonl`, keyed by doc_id/page/region; separate from `data/processed/text/`. "
         "Tables are kept as the engine's HTML; parsing them is Phase 3.", "",
         "| region_id | page type | kind | tables |", "|---|---|---|---|"]
    L += [f"| {x['region_id']} | {x['page_type']} | {x['kind']} | {x['n_tables']} |" for rs in recs.values() for x in rs]
    Path(full["image_tables_ocr"]["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"{len(allr)} regions written; {found} with a table; report {full['image_tables_ocr']['report']}")


if __name__ == "__main__":
    run(load())
