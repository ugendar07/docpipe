"""Stage 6: normalise text. `uv run python -m docpipe.normalise` (rerunnable; rebuild after OCR finishes)."""
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

import pymupdf
from docpipe.config import load

from docpipe.normalise.clean import (build_vocab, check_invariant, dehyphenate, find_edges, unicode_clean, whitespace)


def page_languages(cfg, doc):
    """{page: 'en'|'te'|'mixed'|None} from the language survey (unknown/blank -> None)."""
    f = Path(cfg["language_dir"]) / f"{doc}.jsonl"
    codes = cfg["_languages"]["codes"]   # script label -> language code; unknown/blank -> None
    return {x["page"]: (codes.get(x["label"], x["label"]) if x["label"] not in ("unknown", "blank") else None)
            for x in map(json.loads, f.read_text(encoding="utf-8").splitlines())} if f.exists() else {}


def load_decisions(cfg):
    """{(doc, page): 'fix'|'drop'|'accept'} from the user's low-confidence decision sheet."""
    f = Path(cfg["decisions_csv"])
    if not f.exists():
        return {}
    col = cfg["_review"]["decision_column"]
    return {(r["doc_id"], int(r["page"])): r[col].strip().lower()
            for r in csv.DictReader(open(f, encoding="utf-8")) if r[col].strip()}


def conf_flags(cfg, doc):
    f = Path(cfg["confidence_dir"]) / f"{doc}.jsonl"
    return {x["page"]: x.get("flags", []) for x in map(json.loads, f.read_text(encoding="utf-8").splitlines())} if f.exists() else {}


def load_pages(cfg):
    """-> {doc_id: {page: dict(raw, source, label, flags)}} for ALL pages (3310)."""
    path_of = {r["doc_id"]: r["path"] for r in csv.DictReader(open(cfg["sources"], encoding="utf-8"))}
    blank = {(x["doc_id"], x["page"]) for x in map(json.loads, open(cfg["pages_index"], encoding="utf-8")) if x["blank"]}
    docs, dec = defaultdict(dict), load_decisions(cfg)
    for doc in sorted(path_of):
        cflags = conf_flags(cfg, doc)
        tri = [json.loads(l) for l in (Path(cfg["triage_dir"]) / f"{doc}.jsonl").read_text(encoding="utf-8").splitlines()]
        pdf = pymupdf.open(path_of[doc]) if any(t["label"] in cfg["text_layer_labels"] for t in tri) else None
        for t in tri:
            p, flags = t["page"], []
            if (doc, p) in blank:
                rec = dict(raw="", source="blank", flags=["blank_skipped"])
            elif t["label"] in cfg["text_layer_labels"]:
                raw = pdf[p - 1].get_text()
                flags = (["short_text"] if t["label"] == "needs_review" else [])
                if any(ord(c) < 32 and c not in "\t\n\r\x0b\x0c" for c in raw):   # 1-2 control chars: kept, but flagged for QA
                    flags.append("stray_control_chars")
                rec = dict(raw=raw, source="text_layer", flags=flags)
            else:
                f = Path(cfg["ocr_dir"]) / doc / f"p{p:04d}.json"
                r = json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
                d = dec.get((doc, p))
                oos = next((x for x in cflags.get(p, []) if x.startswith("out_of_scope:")), None)
                if d == "drop" or oos:
                    why = "decision_drop" if d == "drop" else oos
                    rec = dict(raw="", source="dropped", flags=[f"dropped:{why}"])
                elif d == "fix":      # user decision: use the Tesseract text instead of Surya's for this page
                    tf = Path(cfg["tess_cache"]) / doc / f"p{p:04d}.txt"
                    rec = dict(raw=tf.read_text(encoding="utf-8"), source="tesseract_fallback", flags=["fixed_by_user_decision"])
                elif r and r.get("ok"):
                    rec = dict(raw=r["text"], source=r.get("engine", "ocr"), flags=[])
                else:
                    rec = dict(raw="", source="ocr_missing" if r is None else "ocr_failed", flags=["ocr_missing" if r is None else "ocr_failed"])
            rec["label"] = t["label"]
            if "garbage" in t["reason"]:
                rec["flags"].append("garbage_text_layer_replaced_by_ocr")
            docs[doc][p] = rec
        if pdf:
            pdf.close()
    return docs


def run(cfg):
    docs, log = load_pages(cfg), Counter()
    # pass 1: unicode clean everything, build corpus vocabulary
    for d in docs.values():
        for r in d.values():
            r["uni"] = unicode_clean(r["raw"], log)
    vocab = build_vocab(r["uni"] for d in docs.values() for r in d.values())
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    removed_examples, src_count, checked = defaultdict(Counter), Counter(), 0
    for doc, pages in sorted(docs.items()):
        lines = {p: r["uni"].split("\n") for p, r in pages.items()}
        edges = find_edges(lines, cfg)
        recs, removed_all, plang = [], [], page_languages(cfg, doc)
        for p in sorted(pages):
            r, ls, rem = pages[p], lines[p], []
            keep = []
            for i, l in enumerate(ls):
                if i in edges[p]:
                    rem.append(l); removed_all.append(dict(page=p, line=l.strip()))
                    kind = "page_number" if not any(c.isalpha() for c in l) else "header_footer"
                    log[f"edge_line_removed:{kind}"] += 1; removed_examples[doc][l.strip()[:60]] += 1
                else:
                    keep.append(l)
            keep = dehyphenate(keep, vocab, cfg["dehyph_min_vocab"], log)
            text = whitespace("\n".join(keep), log)
            check_invariant(r["raw"], rem, text, f"{doc} p{p}", cfg["keep_symbols"]); checked += 1
            src_count[r["source"]] += 1
            conf, flags = None, list(r["flags"])
            cf = Path(cfg["confidence_dir"]) / f"{doc}.jsonl"
            if cf.exists():
                c = {x["page"]: x for x in map(json.loads, cf.read_text(encoding="utf-8").splitlines())}.get(p)
                if c:
                    conf = c.get("confidence"); flags += c.get("flags", [])
            oos = next((f.split(":", 1)[1] for f in flags if f.startswith("out_of_scope:")), None)
            lang = (cfg["_languages"]["codes"].get(oos, oos) if oos and r["source"] == "dropped" else plang.get(p))
            recs.append(dict(doc_id=doc, page=p, source=r["source"], label=r["label"], language=lang, text=text, confidence=conf, flags=flags))
        side = Path(cfg["removed_dir"]); side.mkdir(parents=True, exist_ok=True)
        (side / f"{doc}.json").write_text(json.dumps(removed_all, ensure_ascii=False), encoding="utf-8")
        (out_dir / f"{doc}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs), encoding="utf-8")

    L = ["# Normalisation report", "", f"- pages written: {sum(src_count.values())} (invariant checked on {checked})",
         "- invariant: digits and the configured keep_symbols identical before/after, except lines removed as headers/page numbers", "",
         "## Pages by source", ""] + [f"- {k}: {v}" for k, v in sorted(src_count.items())]
    L += ["", "## Transformations (type: count)", ""] + [f"- {k}: {v}" for k, v in sorted(log.items())]
    L += ["", "## Removed header/footer/page-number lines per document (top, for review)", ""]
    for doc, c in sorted(removed_examples.items()):
        L.append(f"- {doc}: " + "; ".join(f"{k!r} x{v}" for k, v in c.most_common(3)))
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    run(load()["normalise"])
