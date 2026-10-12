"""Phase 2b: contact sheet of the table-like image regions for manual review. `python -m docpipe.qa.image_tables_sheet`
Writes reports/image_tables_review.csv (verdict column to fill) and reports/image_tables_sheet/index.html (thumbnails)."""
import csv, html
from pathlib import Path

from docpipe.config import load
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def run(cfg):
    VERDICT = cfg["_review"]["verdict_column"]
    rows = [r for r in csv.DictReader(open(cfg["manifest"], encoding="utf-8")) if r["table_like"] == "True"]
    out = Path(cfg["sheet_dir"]); out.mkdir(parents=True, exist_ok=True)
    sheet = Path(cfg["sheet_csv"])
    if sheet.exists() and any(r[VERDICT].strip() for r in csv.DictReader(open(sheet, encoding="utf-8"))):
        sheet = sheet.with_suffix(".regenerated.csv")   # never overwrite verdicts the user has already entered
        print(f"NOTE: {cfg['sheet_csv']} already holds verdicts; writing {sheet} instead")
    cards = []
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["n", "region_id", "doc_id", "page", "page_type", "source", "rows", VERDICT, "GOLD (y)", "NOTES"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["region_id"], r["doc_id"], r["page"], r["page_type"], r["source"], r["rows"], "", "", ""])
            with Image.open(r["crop_path"]) as im:
                im.thumbnail((cfg["thumb_px"], cfg["thumb_px"])); im.convert("RGB").save(out / f"{r['region_id']}.png")
            cards.append(f"<section><h3>#{i} {r['region_id']} &middot; {r['page_type']} &middot; {r['source']} &middot; {r['rows']} rows "
                         f"&middot; {r['width_px']}x{r['height_px']}px</h3><img src='{r['region_id']}.png'></section>")
    (out / "index.html").write_text("<!doctype html><meta charset=utf-8><title>Table-image regions</title><style>body{font-family:sans-serif;margin:16px}"
                                    "img{border:1px solid #888;max-width:100%}section{margin-bottom:28px;border-bottom:2px solid #ccc}</style>"
                                    f"<h1>Table-like image regions ({len(rows)})</h1>" + "".join(cards), encoding="utf-8")
    print(f"{len(rows)} regions -> {sheet}, {out}/index.html")


if __name__ == "__main__":
    run(load()["image_tables"])
