"""Build contact sheets of candidate gold pages: `python -m docpipe.qa.gold_candidates`.
Candidates come from pages that will actually be OCRed (non-blank image pages), stratified by triage label and crop,
plus the forced Telugu/mixed pages."""
import csv, json, random
from collections import defaultdict
from pathlib import Path

import pymupdf
from docpipe.config import load
from PIL import Image, ImageDraw

W, H, COLS, ROWS = 300, 400, 5, 4


def pick(cfg):
    rng = random.Random(cfg["seed"])
    forced = [tuple(x) for x in cfg["forced"]]
    pool = defaultdict(list)
    dense = {(x["doc_id"], x["page"]) for f in Path(cfg["language_dir"]).glob("*.jsonl")
             for x in map(json.loads, open(f, encoding="utf-8")) if x["chars"] >= cfg["min_ocr_chars"]}  # skip covers/dividers
    for x in map(json.loads, open(cfg["pages_index"], encoding="utf-8")):
        if not x["blank"] and (x["doc_id"], x["page"]) not in forced and (x["doc_id"], x["page"]) in dense:
            pool[(x["label"], x["doc_id"])].append((x["doc_id"], x["page"], x["label"]))
    groups = sorted(pool)
    chosen = []
    while len(chosen) < cfg["n_candidates"] - len(forced) and groups:   # round-robin over (label, doc) -> spread
        for g in list(groups):
            if not pool[g]:
                groups.remove(g); continue
            chosen.append(pool[g].pop(rng.randrange(len(pool[g]))))
            if len(chosen) >= cfg["n_candidates"] - len(forced):
                break
    return [(d, p, "forced") for d, p in forced] + sorted(chosen)


def run(cfg):
    path_of = {r["doc_id"]: r["path"] for r in csv.DictReader(open(cfg["sources"], encoding="utf-8"))}
    cands = pick(cfg)
    Path(cfg["candidates_csv"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["candidates_csv"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["n", "doc_id", "page", "label"])
        w.writerows([i + 1, d, p, l] for i, (d, p, l) in enumerate(cands))
    per = COLS * ROWS
    for s in range(0, len(cands), per):
        sheet = Image.new("RGB", (W * COLS, H * ROWS), "white")
        dr = ImageDraw.Draw(sheet)
        for k, (d, p, l) in enumerate(cands[s:s + per]):
            with pymupdf.open(path_of[d]) as doc:
                pm = doc[p - 1].get_pixmap(matrix=pymupdf.Matrix(0.4, 0.4), alpha=False)
            im = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            im.thumbnail((W - 6, H - 22))
            x, y = (k % COLS) * W, (k // COLS) * H
            sheet.paste(im, (x + 3, y + 20))
            dr.rectangle([x, y, x + W - 1, y + H - 1], outline=(150, 150, 150))
            dr.text((x + 5, y + 4), f"#{s + k + 1}  {d} p{p} [{l}]", fill=(200, 0, 0))
        sheet.save(f"{cfg['sheet_prefix']}_{s // per + 1}.png")
    print(f"{len(cands)} candidates -> {cfg['candidates_csv']}, {cfg['sheet_prefix']}_*.png")


if __name__ == "__main__":
    run(load()["gold"])
