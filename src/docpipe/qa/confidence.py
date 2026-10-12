"""Confidence for OCR pages: agreement between Surya and a Tesseract pass on the same image (1 = identical words),
plus hard flags. Surya itself reports no confidence. `uv run python -m docpipe.qa.confidence`"""
import csv, json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytesseract
from docpipe.config import load
from PIL import Image
from rapidfuzz.distance import Levenshtein

from docpipe.qa.score import normalise
from docpipe.scripts import share

Image.MAX_IMAGE_PIXELS = None


def agreement(a, b):
    """Order-insensitive word agreement (multiset Dice, 0..1). A different reading order of the same words
    (e.g. two columns) scores 1.0; use order_agreement() to see reading-order differences."""
    ta, tb = Counter(normalise(a).split()), Counter(normalise(b).split())
    na, nb = sum(ta.values()), sum(tb.values())
    if not na and not nb:
        return 1.0
    return 2 * sum((ta & tb).values()) / (na + nb)


def order_agreement(a, b):
    """Word-sequence similarity (normalised Levenshtein over words), sensitive to reading order."""
    ta, tb = normalise(a).split(), normalise(b).split()
    if not ta and not tb:
        return 1.0
    return float(Levenshtein.normalized_similarity(ta, tb))


def has_loop(text, n, min_rep):
    """Generation loop: the same n-gram of words repeated min_rep or more times."""
    t = normalise(text).split()
    if len(t) < n * min_rep:
        return False
    c = Counter(tuple(t[i:i + n]) for i in range(len(t) - n + 1))
    return c.most_common(1)[0][1] >= min_rep


_LATEX = re.compile(r"\\(?:text|mathrm|mu|bar|frac|sum)\b|\^\{|_\{")


def flags_for(text, ink, cfg, tess_words=0):
    """Hard flags. Empty output only counts if Tesseract found text Surya did not; a loop only if Surya wrote far
    more words than Tesseract (repeated table/list rows are normal); out-of-scope-script pages are not failures."""
    f = []
    if len(text.strip()) < cfg["empty_chars"] and ink >= cfg["empty_ink_min"] and tess_words >= cfg["empty_tess_words"]:
        f.append("empty_output")
    if has_loop(text, cfg["loop_ngram"], cfg["loop_min"]) and len(text.split()) > cfg["loop_word_ratio"] * max(1, tess_words):
        f.append("repetition_loop")
    lg = cfg["_languages"]
    for s in lg["out_of_scope"]:
        if share(text, s)[0] > lg["out_of_scope_share"]:
            f.append(f"out_of_scope:{s}")
    if _LATEX.search(text):
        f.append("latex_markup")
    return f


def run(cfg):
    ink = {(x["doc_id"], x["page"]): x["ink"] for x in map(json.loads, open(cfg["pages_index"], encoding="utf-8"))}
    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(cfg["ocr_dir"]).glob("*/*.json"))]

    def job(r):
        key = (r["doc_id"], r["page"])
        if not r.get("ok"):
            return dict(doc_id=r["doc_id"], page=r["page"], confidence=0.0, flags=["ocr_failed"])
        if r.get("engine") == "tesseract":
            # the cross-check IS Tesseract: agreement would always be 1.0 and prove nothing, so no score is given
            fl = flags_for(r["text"], ink.get(key, 0.0), cfg, 0)
            return dict(doc_id=r["doc_id"], page=r["page"], confidence=None, flags=fl, crosscheck="skipped_same_engine")
        cache = Path(cfg["tess_cache"]) / r["doc_id"] / f"p{r['page']:04d}.txt"
        if cache.exists():
            t = cache.read_text(encoding="utf-8")
        else:
            t = pytesseract.image_to_string(Image.open(r["image"]), lang=cfg["_languages"]["ocr"])
            cache.parent.mkdir(parents=True, exist_ok=True); cache.write_text(t, encoding="utf-8")
        conf = round(agreement(r["text"], t), 4)
        fl = flags_for(r["text"], ink.get(key, 0.0), cfg, len(t.split()))
        if conf < cfg["low_conf_below"] and not any(x.startswith("out_of_scope:") for x in fl):
            fl.append("low_agreement")
        return dict(doc_id=r["doc_id"], page=r["page"], confidence=conf, order_agreement=round(order_agreement(r["text"], t), 4), flags=fl, tess_words=len(t.split()),
                    surya_words=len(r["text"].split()))

    with ThreadPoolExecutor(cfg["workers"]) as ex:
        out = list(ex.map(job, recs))
    by = defaultdict(list)
    for o in out:
        by[o["doc_id"]].append(o)
    d = Path(cfg["out_dir"]); d.mkdir(parents=True, exist_ok=True)
    for doc, rs in by.items():
        (d / f"{doc}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in sorted(rs, key=lambda x: x["page"])), encoding="utf-8")
    REVIEW = {"low_agreement", "empty_output", "repetition_loop", "ocr_failed"}
    low = sorted((o for o in out if REVIEW & set(o["flags"])), key=lambda o: o["confidence"] or 0.0)
    info = Counter(f for o in out for f in o["flags"])
    cs = sorted(o["confidence"] for o in out if o["confidence"] is not None)
    skipped = sum(o["confidence"] is None for o in out)
    L = ["# OCR confidence (OCR engine vs Tesseract agreement)", "", f"- pages: {len(out)}; scored: {len(cs)}; **no score (engine is Tesseract, cross-check would compare it with itself): {skipped}**",
         (f"- agreement quantiles: min {cs[0]:.2f}, p10 {cs[len(cs) // 10]:.2f}, median {cs[len(cs) // 2]:.2f}, max {cs[-1]:.2f}" if cs
          else "- agreement quantiles: none (use a different OCR engine than Tesseract to get an agreement score; only loop/script flags apply)"),
         f"- pages needing a fix/drop/accept decision: {len(low)} (low agreement < {cfg['low_conf_below']}, empty output, repetition loop, failed)",
         f"- informational flags: " + ", ".join(f"{k} {v}" for k, v in sorted(info.items()) if k.startswith("out_of_scope:") or k == "latex_markup"), "",
         "| doc | page | agreement | flags |", "|---|---|---|---|"]
    L += [f"| {o['doc_id']} | {o['page']} | {'-' if o['confidence'] is None else format(o['confidence'], '.2f')} | {', '.join(o['flags'])} |" for o in low]
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    # NEW in docpipe (the Telangana repo had no generator): decision sheet for the operator; never overwritten once it exists
    t = Path(cfg["decisions_template"])
    if not t.exists() and low:
        t.parent.mkdir(parents=True, exist_ok=True)
        with open(t, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n"); w.writerow(["doc_id", "page", "agreement", "flags", cfg["_review"]["decision_column"]])
            w.writerows([o["doc_id"], o["page"], "" if o["confidence"] is None else f"{o['confidence']:.2f}", ";".join(o["flags"]), ""] for o in low)
    print("\n".join(L[:8]))


if __name__ == "__main__":
    run(load()["confidence"])
