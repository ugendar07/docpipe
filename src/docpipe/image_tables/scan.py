"""Phase 2b step 1: inventory every embedded image region that looks like a table, and crop it at full resolution.
Two region sources: (a) images embedded in the PDF page (born-digital / hybrid pages), (b) `Picture`/`Figure` blocks
that Surya found (and skipped) on OCR'd pages, e.g. screenshots pasted inside a scan. A region is "table-like" when
Tesseract finds enough rows of several words with several numbers. Crops are saved only for pages that have at least
one table-like region, but every region on such a page is listed. Rerunnable alone; deterministic."""
import csv, json, re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
NUM = re.compile(r"^[\d.,%±-]+$")
FIELDS = ["region_id", "doc_id", "page", "region", "source", "page_type", "x0", "y0", "x1", "y1", "width_px", "height_px",
          "rows", "table_like", "crop_path"]


def table_rows(img, cfg):
    """Number of rows with >= min_words confident words of which >= min_numeric are numeric."""
    t = pytesseract.image_to_data(img, lang=cfg["lang"], config=f"--psm {cfg['psm']}", output_type=pytesseract.Output.DICT)
    lines = {}
    for w, c, k in zip(t["text"], t["conf"], zip(t["block_num"], t["par_num"], t["line_num"])):
        if w.strip() and float(c) > cfg["conf_min"]:
            lines.setdefault(k, []).append(w)
    return sum(1 for l in lines.values() if len(l) >= cfg["min_words"] and sum(bool(NUM.match(w)) for w in l) >= cfg["min_numeric"])


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def doc_regions(job):
    """-> list of region dicts for one document (crops held in memory only long enough to score them)."""
    doc_id, pdf_path, labels, cfg, ocr_dir = job
    out = []
    pdf = pymupdf.open(pdf_path)
    for pno, pg in enumerate(pdf, 1):
        W, H = pg.rect.width, pg.rect.height
        regs = []   # (source, fraction bbox, image-getter)
        for info in pg.get_image_info():
            r = pymupdf.Rect(info["bbox"]) & pg.rect
            if r.is_empty or r.width < cfg["min_pt"] or r.height < cfg["min_pt"] or r.width * r.height / (W * H) > cfg["max_page_frac"]:
                continue   # too small (logos, rules) or the full-page scan itself
            regs.append(("pdf_image", (r.x0 / W, r.y0 / H, r.x1 / W, r.y1 / H), r))
        sj = Path(ocr_dir) / doc_id / f"p{pno:04d}.json"
        if sj.exists():
            d = json.loads(sj.read_text(encoding="utf-8"))
            blocks = d.get("blocks")
            for b in blocks if isinstance(blocks, list) else []:   # only layout-aware engines (Surya) report Picture/Figure blocks
                if not isinstance(b, dict) or b.get("label") not in ("Picture", "Figure"):
                    continue
                x0, y0, x1, y1 = b["bbox"]
                if (x1 - x0) < cfg["min_pt"] * cfg["dpi"] / 72 or (y1 - y0) < cfg["min_pt"] * cfg["dpi"] / 72:
                    continue
                with Image.open(d["image"]) as im:
                    iw, ih = im.size
                fb = (x0 / iw, y0 / ih, x1 / iw, y1 / ih)
                if any(iou(fb, f) > cfg["dedup_iou"] for _, f, _ in regs):
                    continue   # same picture already found as an embedded image: keep the native one
                regs.append(("surya_picture", fb, (d["image"], (x0, y0, x1, y1))))
        regs.sort(key=lambda r: (round(r[1][1], 4), round(r[1][0], 4)))
        for i, (src, fb, ref) in enumerate(regs, 1):
            if src == "pdf_image":
                pix = pg.get_pixmap(dpi=cfg["dpi"], clip=ref)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            else:
                with Image.open(ref[0]) as im:
                    img = im.convert("RGB").crop(tuple(int(v) for v in ref[1]))
            rows = table_rows(img, cfg)
            out.append(dict(region_id=f"{doc_id}_{pno:04d}_{i}", doc_id=doc_id, page=pno, region=i, source=src,
                            page_type=labels.get((doc_id, pno), ""), x0=round(fb[0], 4), y0=round(fb[1], 4), x1=round(fb[2], 4),
                            y1=round(fb[3], 4), width_px=img.width, height_px=img.height, rows=rows,
                            table_like=rows >= cfg["min_rows"], crop_path=""))
    return out


def save_crop(row, cfg, pdf_path, ocr_dir):
    """Re-render one region and save it as PNG (same pixels as were scored)."""
    pdf = pymupdf.open(pdf_path); pg = pdf[row["page"] - 1]; W, H = pg.rect.width, pg.rect.height
    box = (row["x0"], row["y0"], row["x1"], row["y1"])
    if row["source"] == "pdf_image":
        pix = pg.get_pixmap(dpi=cfg["dpi"], clip=pymupdf.Rect(box[0] * W, box[1] * H, box[2] * W, box[3] * H))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    else:
        d = json.loads((Path(ocr_dir) / row["doc_id"] / f"p{row['page']:04d}.json").read_text(encoding="utf-8"))
        with Image.open(d["image"]) as im:
            iw, ih = im.size
            img = im.convert("RGB").crop((round(box[0] * iw), round(box[1] * ih), round(box[2] * iw), round(box[3] * ih)))
    p = Path(cfg["crop_dir"]) / f"{row['region_id']}.png"
    p.parent.mkdir(parents=True, exist_ok=True); img.save(p)
    return str(p)


def run(cfg, ocr_dir, triage_dir, sources):
    labels = {}
    for f in sorted(Path(triage_dir).glob("*.jsonl")):
        for l in open(f, encoding="utf-8"):
            x = json.loads(l); labels[(x["doc_id"], x["page"])] = x["label"]
    docs = sorted((r["doc_id"], r["path"]) for r in csv.DictReader(open(sources, encoding="utf-8")) if r["path"])
    jobs = [(d, p, labels, cfg, ocr_dir) for d, p in docs]
    with ProcessPoolExecutor(cfg["workers"]) as ex:
        rows = [r for part in ex.map(doc_regions, jobs) for r in part]   # map keeps input order -> deterministic
    pdfs = dict(docs)
    keep = {(r["doc_id"], r["page"]) for r in rows if r["table_like"]}
    rows = [r for r in rows if (r["doc_id"], r["page"]) in keep]
    for r in rows:
        r["crop_path"] = save_crop(r, cfg, pdfs[r["doc_id"]], ocr_dir)
    Path(cfg["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg["manifest"], "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS, lineterminator="\n"); w.writeheader(); w.writerows(rows)
    report(rows, cfg)
    return rows


def report(rows, cfg):
    tl = [r for r in rows if r["table_like"]]
    pages = {(r["doc_id"], r["page"]) for r in rows}
    L = ["# Phase 2b: embedded table-image inventory", "",
         f"Pages with at least one table-like image region: **{len(pages)}** in {len({r['doc_id'] for r in rows})} documents.",
         f"Regions listed (all regions on those pages): **{len(rows)}**; table-like regions: **{len(tl)}**.", "",
         "Table-like = at least %d rows with %d+ confident words, %d+ numeric (Tesseract %s, psm %d). This is a heuristic, not a verdict:"
         " charts with numeric labels can pass; tables smaller than %d pt, drawn as vector graphics, or in low contrast are missed,"
         " so the count is a lower bound for tables and an upper bound for how many are real." % (
             cfg["min_rows"], cfg["min_words"], cfg["min_numeric"], cfg["lang"], cfg["psm"], cfg["min_pt"]), ""]
    for title, key in (("Table-like regions by region source", "source"), ("Table-like regions by page type", "page_type")):
        L += [f"## {title}", "", "| value | regions | pages |", "|---|---|---|"]
        for v in sorted({r[key] for r in tl}):
            L.append(f"| {v} | {sum(r[key] == v for r in tl)} | {len({(r['doc_id'], r['page']) for r in tl if r[key] == v})} |")
        L.append("")
    L += ["## By document (table-like regions)", "", "| doc_id | regions | pages |", "|---|---|---|"]
    c = Counter(r["doc_id"] for r in tl)
    for d in sorted(c):
        L.append(f"| {d} | {c[d]} | {len({r['page'] for r in tl if r['doc_id'] == d})} |")
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
