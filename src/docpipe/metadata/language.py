"""Stage 4: per-page script/language survey over the WHOLE corpus (not just gold pages).
Text-layer pages use the Unicode script shares from triage; image pages use Tesseract OCR output (languages.ocr).
A seeded sample of born-digital pages is also OCRed to catch legacy-font text of another script hidden in text layers that look primary-script."""
import csv, json, os, random, tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pymupdf
import pytesseract
from docpipe.config import load
from PIL import Image

from docpipe.imaging.rasterise import render
from docpipe.scripts import share

os.environ.setdefault("OMP_THREAD_LIMIT", "1")  # one thread per tesseract process, parallelism comes from workers


def script_ratios(text, cfg):
    """-> ({non-primary script: share of non-space chars}, nchars)"""
    sc = cfg["_languages"]["scripts"][1:]
    n = sum(not x.isspace() for x in text)
    return {s: share(text, s)[0] for s in sc}, n


def label(ratios, n, cfg):
    """primary script name | the dominant other script | mixed | unknown.
    ponytail: generalises the old english/telugu/mixed rule to N scripts by looking at the single dominant non-primary script."""
    sc = cfg["_languages"]["scripts"]
    if n < cfg["min_chars"]:
        return "unknown"
    top, r = max(ratios.items(), key=lambda kv: kv[1], default=(None, 0.0))
    return sc[0] if r < cfg["primary_below"] else top if r > cfg["secondary_above"] else "mixed"


def ocr_image(im, cfg):
    """-> (ratio, nchars, osd_script, osd_conf)"""
    r, n = script_ratios(pytesseract.image_to_string(im, lang=cfg["_languages"]["ocr"]), cfg)
    try:
        o = pytesseract.image_to_osd(im, output_type=pytesseract.Output.DICT)
        return r, n, o.get("script"), o.get("script_conf")
    except pytesseract.TesseractError:   # no OSD result on this page; other errors are real and must surface
        return r, n, None, None


def run(cfg):
    path_of = {r["doc_id"]: r["path"] for r in csv.DictReader(open(cfg["sources"], encoding="utf-8"))}
    tri = {}
    for f in sorted(Path(cfg["triage_dir"]).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            x = json.loads(line); tri[(x["doc_id"], x["page"])] = x
    img = {(x["doc_id"], x["page"]): x for x in map(json.loads, open(cfg["pages_index"], encoding="utf-8"))}

    recs, todo = {}, []
    for k, t in tri.items():
        i = img.get(k)
        if i and i["blank"]:
            recs[k] = dict(doc_id=k[0], page=k[1], method="skipped_blank", chars=0, script_share={}, label="blank")
        elif i:  # scanned / blank_or_vector / hybrid with a rendered image -> OCR
            todo.append(k)
        else:    # born_digital / needs_review -> text layer
            n = t["chars"]
            recs[k] = dict(doc_id=k[0], page=k[1], method="text_layer", chars=n, script_share=t["script_share"],
                           label=label(t["script_share"], n, cfg))

    def job(k):
        r, n, s, c = ocr_image(Image.open(img[k]["path"]), cfg)
        d = dict(doc_id=k[0], page=k[1], method="ocr", chars=n, script_share={k: round(v, 4) for k, v in r.items()}, label=label(r, n, cfg),
                 osd_script=s, osd_conf=c)
        if tri[k]["label"] == "hybrid":
            d["text_layer_share"] = tri[k]["script_share"]
        return k, d

    with ThreadPoolExecutor(cfg["workers"]) as ex:
        for k, d in ex.map(job, sorted(todo)):
            recs[k] = d

    # legacy-font check: OCR a seeded sample of text-layer pages, compare script label with the text layer
    cand = sorted(k for k, r in recs.items() if r["method"] == "text_layer" and tri[k]["label"] == "born_digital")
    sample = random.Random(cfg["seed"]).sample(cand, min(cfg["validation_sample"], len(cand)))
    dis, ocr_n = [], 0
    with tempfile.TemporaryDirectory() as tmp:
        def vjob(k):
            with pymupdf.open(path_of[k[0]]) as d:
                png = Path(tmp) / f"{k[0]}_{k[1]}.png"; render(d[k[1] - 1], 300).save(png)
            r, n, _, _ = ocr_image(Image.open(png), cfg)
            return k, r, n
        with ThreadPoolExecutor(cfg["workers"]) as ex:
            for k, r, n in ex.map(vjob, sample):
                ocr_n += 1
                lo, lt = label(r, n, cfg), recs[k]["label"]
                if lo != lt and not (lo == "unknown" or lt == "unknown"):
                    dis.append((k, lt, recs[k]["script_share"], lo, {a: round(b, 4) for a, b in r.items()}))

    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    by_doc = defaultdict(list)
    for k in sorted(recs):
        by_doc[k[0]].append(recs[k])
    for d, rs in by_doc.items():
        (out / f"{d}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs), encoding="utf-8")

    labs = cfg["_languages"]["scripts"] + ["mixed", "unknown", "blank"]
    tot = Counter(r["label"] for r in recs.values())
    L = ["# Language survey (all pages, all documents)", "", f"Pages: {len(recs)}", "", "| label | pages |", "|---|---|"]
    L += [f"| {l} | {tot[l]} |" for l in labs]
    L += ["", "| method | pages |", "|---|---|"] + [f"| {m} | {c} |" for m, c in Counter(r['method'] for r in recs.values()).items()]
    flagged = defaultdict(list)
    for r in recs.values():
        if r["label"] in labs[1:-2]:
            flagged[r["doc_id"]].append((r["page"], r["label"], r["script_share"]))
    L += ["", f"## Non-primary-script / mixed pages ({sum(map(len, flagged.values()))} pages in {len(flagged)} documents)", ""]
    L += [f"- {d}: " + ", ".join(f"p{p} {l} {t}" for p, l, t in v) for d, v in sorted(flagged.items())] or ["- none"]
    osd = Counter(r.get("osd_script") for r in recs.values() if r["method"] == "ocr")
    L += ["", "## OSD script on OCRed pages", ""] + [f"- {s}: {c}" for s, c in osd.most_common()]
    L += ["", f"## Legacy-font check: {ocr_n} born-digital pages OCRed vs text layer; disagreements: {len(dis)}", ""]
    L += [f"- {k[0]} p{k[1]}: text layer {a} ({b}) vs OCR {c} ({d})" for k, a, b, c, d in dis] or ["- none"]
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    run(load()["language"])
