"""OCR scoring: CER/WER, protected-token recall (numbers, units, chemical terms), table fidelity, bootstrap CIs.
Gold and engine output get identical normalisation. Nothing here repairs or forgives engine output."""
import csv, difflib, io, random, re, unicodedata
from collections import Counter
from html.parser import HTMLParser

import jiwer
import numpy as np

_SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻₀₁₂₃₄₅₆₇₈₉", "0123456789-0123456789")
_TAG = re.compile(r"<[^>]+>")


def normalise(text, html=False):
    """NFC, sub/superscripts -> plain chars (scoring only), HTML stripped, whitespace collapsed. No case folding."""
    if html:
        text = _TAG.sub(" ", text)
    text = unicodedata.normalize("NFC", text).translate(_SUP)
    return " ".join(text.split())


def edits(ref, hyp, unit):
    """-> (edit_count, ref_len) for unit in {'char','word'}"""
    f = jiwer.process_characters if unit == "char" else jiwer.process_words
    o = f(ref, hyp)
    return o.substitutions + o.deletions + o.insertions, o.substitutions + o.deletions + o.hits


# ---------- protected tokens ----------
_NUM = re.compile(r"\d+(?:[.,:/-]\d+)*")
_WORD = re.compile(r"[A-Za-z%][A-Za-z0-9%/-]*")


def protected(text, units, terms):
    """-> {'numbers': Counter, 'units': Counter, 'terms': Counter} from normalised text."""
    low_terms = {t.lower() for t in terms}
    words = _WORD.findall(text)
    return dict(numbers=Counter(_NUM.findall(text)),
                units=Counter(w for w in words if w in units),
                terms=Counter(w.lower() for w in words if w.lower() in low_terms))


def protected_recall(gold, hyp, units, terms):
    """Multiset recall per kind; None when the gold has none of that kind."""
    g, h, out = protected(gold, units, terms), protected(hyp, units, terms), {}
    for k in g:
        n = sum(g[k].values())
        out[k] = (n, sum((g[k] & h[k]).values())) if n else (0, 0)
    return out


# ---------- tables ----------
def grid_from_csv(text):
    return [[normalise(c) for c in row] for row in csv.reader(io.StringIO(text))]


class _Tables(HTMLParser):
    """HTML -> list of grids, expanding rowspan/colspan (spanned cells repeat their text)."""

    def __init__(self):
        super().__init__(); self.tables, self.stack = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self.stack.append(dict(rows=[], cur=None, cell=None, pend={}))
        elif self.stack and tag == "tr":
            self.stack[-1]["cur"] = []
        elif self.stack and tag in ("td", "th"):
            s = self.stack[-1]
            s["cell"] = dict(text=[], rs=int(a.get("rowspan", 1) or 1), cs=int(a.get("colspan", 1) or 1))

    def handle_data(self, d):
        if self.stack and self.stack[-1]["cell"] is not None:
            self.stack[-1]["cell"]["text"].append(d)

    def handle_endtag(self, tag):
        if not self.stack:
            return
        s = self.stack[-1]
        if tag in ("td", "th") and s["cell"] is not None and s["cur"] is not None:
            c, r = s["cell"], s["rows"]
            row = s["cur"]
            while (len(r), len(row)) in s["pend"]:      # skip slots filled by an earlier rowspan
                row.append(s["pend"].pop((len(r), len(row))))
            txt = normalise(" ".join(c["text"]))
            for j in range(c["cs"]):
                row.append(txt)
                for i in range(1, c["rs"]):
                    s["pend"][(len(r) + i, len(row) - 1)] = txt
            s["cell"] = None
        elif tag == "tr" and s["cur"] is not None:
            row = s["cur"]
            while (len(s["rows"]), len(row)) in s["pend"]:
                row.append(s["pend"].pop((len(s["rows"]), len(row))))
            s["rows"].append(row); s["cur"] = None
        elif tag == "table":
            self.tables.append(s["rows"]); self.stack.pop()


def grids_from_html(html):
    p = _Tables(); p.feed(html)
    return p.tables


def table_metrics(gold, pred, hyp_text):
    """Per gold table. pred=None when the engine produced no matching table."""
    toks = Counter(normalise(hyp_text).split())
    cells = [c for r in gold for c in r if c]
    gt = Counter(t for c in cells for t in c.split())
    out = dict(cells=len(cells), flat_cell_recall=sum((gt & toks).values()) / max(1, sum(gt.values())))
    if pred is None:
        return dict(out, detected=False, shape_match=False, cell_acc=0.0, row_sim=0.0)
    n = sum(len(r) for r in gold)
    ok = sum(1 for i, r in enumerate(gold) for j, c in enumerate(r) if i < len(pred) and j < len(pred[i]) and pred[i][j] == c)
    rs = difflib.SequenceMatcher(None, [tuple(r) for r in gold], [tuple(r) for r in pred], autojunk=False).ratio()
    shape = len(gold) == len(pred) and all(len(a) == len(b) for a, b in zip(gold, pred))
    return dict(out, detected=True, shape_match=shape, cell_acc=ok / max(1, n), row_sim=rs)


def match_tables(gold_grids, pred_grids):
    """Each gold table takes the unused predicted table with the best row similarity (None if no tables left)."""
    left, out = list(pred_grids or []), []
    for g in gold_grids:
        if not left:
            out.append(None); continue
        k = max(range(len(left)), key=lambda i: difflib.SequenceMatcher(
            None, [tuple(r) for r in g], [tuple(r) for r in left[i]], autojunk=False).ratio())
        out.append(left.pop(k))
    return out


# ---------- per page ----------
def score_page(gold_text, hyp_text, gold_tables=(), hyp_tables=None, units=(), terms=(), hyp_is_html=False):
    """CER/WER use the gold text only; protected tokens also cover the gold table cells (order-free)."""
    g, h = normalise(gold_text), normalise(hyp_text, html=hyp_is_html)
    ce, cn = edits(g, h, "char"); we, wn = edits(g, h, "word")
    g_all = normalise(" ".join([gold_text] + [c for t in gold_tables for r in t for c in r]))
    r = dict(char_edits=ce, char_ref=cn, word_edits=we, word_ref=wn, protected=protected_recall(g_all, h, set(units), terms))
    preds = match_tables(list(gold_tables), [t for html in (hyp_tables or []) for t in grids_from_html(html)])
    r["tables"] = [table_metrics(gt, pt, h) for gt, pt in zip(gold_tables, preds)]
    return r


# ---------- aggregate ----------
def _agg(pages):
    d = dict(cer=sum(p["char_edits"] for p in pages) / max(1, sum(p["char_ref"] for p in pages)),
             wer=sum(p["word_edits"] for p in pages) / max(1, sum(p["word_ref"] for p in pages)))
    for k in ("numbers", "units", "terms"):
        n = sum(p["protected"][k][0] for p in pages)
        d[f"{k}_recall"] = sum(p["protected"][k][1] for p in pages) / n if n else None
    t = [x for p in pages for x in p["tables"]]
    if t:
        for k in ("detected", "shape_match"):
            d[f"table_{k}"] = sum(x[k] for x in t) / len(t)
        for k in ("cell_acc", "row_sim", "flat_cell_recall"):
            d[f"table_{k}"] = sum(x[k] for x in t) / len(t)
    return d


def aggregate(pages, n_boot=1000, seed=0):
    """-> {metric: (value, lo95, hi95)}; bootstrap resamples pages."""
    base, rng = _agg(pages), random.Random(seed)
    boots = [_agg([pages[rng.randrange(len(pages))] for _ in pages]) for _ in range(n_boot)]
    out = {}
    for k, v in base.items():
        xs = sorted(b[k] for b in boots if b.get(k) is not None)
        out[k] = (v, xs[int(0.025 * len(xs))], xs[int(0.975 * len(xs)) - 1]) if xs and v is not None else (v, None, None)
    return out
