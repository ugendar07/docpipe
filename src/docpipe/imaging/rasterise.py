"""Stage 3: rasterise pages to PNG at config DPI. `--estimate` reports disk use without writing images."""
import csv, json, random, shutil, sys
from pathlib import Path

import numpy as np
import pymupdf
from docpipe.config import load


def selected(cfg):
    """[(doc_id, pdf_path, page_no)] for pages whose triage label is in cfg['labels']."""
    path_of = {r["doc_id"]: r["path"] for r in csv.DictReader(open(cfg["sources"], encoding="utf-8"))}
    out = []
    for f in sorted(Path(cfg["triage_dir"]).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            if x["label"] in cfg["labels"]:
                out.append((x["doc_id"], path_of[x["doc_id"]], x["page"], x["label"]))
    return out


def render(page, dpi):
    s = dpi / 72
    return page.get_pixmap(matrix=pymupdf.Matrix(s, s), colorspace=pymupdf.csRGB, alpha=False)


def ink_fraction(pm, below):
    a = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, 3).mean(axis=2)
    return float((a < below).mean())


def estimate(cfg):
    pages = selected(cfg)
    rng = random.Random(cfg["seed"])
    sample = rng.sample(pages, min(cfg["estimate_sample"], len(pages)))
    px_total, bpp = 0, []
    docs = {}
    for doc, pdf, n, _ in pages:
        docs.setdefault(pdf, []).append(n)
    s = cfg["dpi"] / 72
    for pdf, ns in docs.items():
        with pymupdf.open(pdf) as d:
            px_total += sum(int(d[n - 1].rect.width * s) * int(d[n - 1].rect.height * s) for n in ns)
    for doc, pdf, n, _ in sample:
        with pymupdf.open(pdf) as d:
            pm = render(d[n - 1], cfg["dpi"])
            bpp.append(len(pm.tobytes(cfg["format"])) / (pm.width * pm.height))
    mean, mx = float(np.mean(bpp)), float(np.max(bpp))
    free = shutil.disk_usage(cfg["out_dir"] if Path(cfg["out_dir"]).exists() else ".").free
    L = ["# Rasterise: disk estimate", "", f"- pages selected ({', '.join(cfg['labels'])}): {len(pages)}",
         f"- total pixels at {cfg['dpi']} DPI: {px_total / 1e6:.0f} Mpx (mean {px_total / len(pages) / 1e6:.1f} Mpx/page)",
         f"- sampled {len(sample)} pages: mean {mean:.3f} bytes/px, worst {mx:.3f}",
         f"- estimated {cfg['format']} size: {px_total * mean / 1e9:.2f} GB expected, {px_total * mx / 1e9:.2f} GB worst case",
         f"- free disk: {free / 1e9:.0f} GB"]
    print("\n".join(L))


def run(cfg, force=False):
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    idx, ext = [], cfg["format"]
    old = {}   # reuse existing renders (identical by construction) so images in use by a running OCR job are never rewritten
    if not force and (out_dir / "index.jsonl").exists():
        old = {(x["doc_id"], x["page"]): x for x in map(json.loads, (out_dir / "index.jsonl").read_text(encoding="utf-8").splitlines())}
    by_pdf = {}
    for doc, pdf, n, label in selected(cfg):
        by_pdf.setdefault((doc, pdf), []).append((n, label))
    for (doc, pdf), ns in by_pdf.items():
        (out_dir / doc).mkdir(exist_ok=True)
        with pymupdf.open(pdf) as d:
            for n, label in ns:
                p = out_dir / doc / f"p{n:04d}.{ext}"
                if (doc, n) in old and p.exists():
                    idx.append(dict(old[(doc, n)], label=label)); continue
                pm = render(d[n - 1], cfg["dpi"])
                pm.save(p)
                ink = ink_fraction(pm, cfg["ink_gray_below"])
                idx.append(dict(doc_id=doc, page=n, label=label, path=str(p), width=pm.width, height=pm.height,
                                bytes=p.stat().st_size, ink=round(ink, 5), blank=ink <= cfg["blank_ink_max"]))
    tmp = out_dir / "index.jsonl.tmp"
    tmp.write_text("".join(json.dumps(x) + "\n" for x in idx), encoding="utf-8"); tmp.replace(out_dir / "index.jsonl")
    blank = [x for x in idx if x["blank"]]
    L = ["# Rasterise report", "", f"- pages rendered: {len(idx)}  |  disk: {sum(x['bytes'] for x in idx) / 1e9:.2f} GB",
         f"- truly blank (ink <= {cfg['blank_ink_max']}): {len(blank)}  |  to OCR: {len(idx) - len(blank)}", "",
         "## Blank pages by label"] + [f"- {l}: {sum(1 for x in blank if x['label'] == l)}" for l in cfg["labels"]]
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    c = load()["rasterise"]
    estimate(c) if "--estimate" in sys.argv else run(c, force="--force" in sys.argv)
