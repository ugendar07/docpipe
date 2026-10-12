"""Create gold templates: `python -m docpipe.qa.gold_build`. Renders each pick at 300 DPI (full page or crop) into
eval/gold_ocr/images/ (git-ignored) and creates EMPTY .txt / _t1.csv files to type into. Never overwrites typed files."""
import csv
from pathlib import Path

import pymupdf
from docpipe.config import load

from docpipe.imaging.rasterise import render
from PIL import Image


def run(cfg, picks):
    root = Path(cfg["dir"]); (root / "images").mkdir(parents=True, exist_ok=True)
    cand = {int(r["n"]): r for r in csv.DictReader(open(cfg["candidates_csv"], encoding="utf-8"))}
    path_of = {r["doc_id"]: r["path"] for r in csv.DictReader(open(cfg["sources"], encoding="utf-8"))}
    rows = []
    for p in picks:
        c = cand[p["n"]]; doc, page = c["doc_id"], int(c["page"])
        gid = f"{doc}_p{page:04d}" + ("_crop" if p["mode"] == "crop" else "")
        with pymupdf.open(path_of[doc]) as d:
            pm = render(d[page - 1], 300)
        im = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
        if p["mode"] == "crop":
            x0, y0, x1, y1 = p["box"]
            im = im.crop((int(x0 * im.width), int(y0 * im.height), int(x1 * im.width), int(y1 * im.height)))
        im.save(root / "images" / f"{gid}.png")
        for name in [f"{gid}.txt"] + ([f"{gid}_t1.csv"] if p["table"] else []):
            if not (root / name).exists():
                (root / name).write_text("", encoding="utf-8")
        rows.append([gid, doc, page, p["mode"], " ".join(map(str, p.get("box", []))), p["script"], p["table"], p["note"], p["n"]])
    with open(root / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["gold_id", "doc_id", "page", "mode", "box", "script", "has_table", "notes", "candidate_n"])
        w.writerows(rows)
    print(f"{len(rows)} gold items: {sum(r[3] == 'full' for r in rows)} full, {sum(r[3] == 'crop' for r in rows)} crops, "
          f"{sum(r[6] for r in rows)} with tables")


if __name__ == "__main__":
    c = load()["gold"]
    run(c, c["picks"])
