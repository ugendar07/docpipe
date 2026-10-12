"""Text normalisation rules. NEVER alters numbers, units or chemical names: no NFKC, no reflow, and a per-page
invariant check on digits and unit symbols (raises InvariantError instead of writing a changed page)."""
import re, unicodedata
from collections import Counter

MAP = {" ": " ", " ": " ", " ": " ", "​": "", "‌": "", "­": "", "﻿": "",
       "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
       "": "•", "": "•", "": "•", "": "•"}   # symbol-font bullets


class InvariantError(Exception):
    pass


def unicode_clean(text, log):
    out = []
    for ch in unicodedata.normalize("NFC", text):
        if ch in MAP:
            log[f"unicode:U+{ord(ch):04X}->{MAP[ch]!r}"] += 1; out.append(MAP[ch])
        elif ch not in "\n\t" and unicodedata.category(ch) in ("Cc", "Cs", "Co", "Cn"):
            log[f"unicode:control_or_private U+{ord(ch):04X} -> space"] += 1; out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


def signature(line):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.lower())).strip()


def _edge_ok(line, n_pages, cfg):
    """Could this edge line be a header/footer/page number? Bare numbers must be plausible page numbers
    (<= doc page count + slack); other lines must be 3+ chars (a lone bullet glyph is not a header)."""
    t = line.strip()
    if len(t) > cfg["header_max_len"]:
        return False
    if re.fullmatch(r"\d+", t):
        return int(t) <= n_pages + cfg["header_pagenum_slack"]
    return len(t) >= 3


def find_edges(pages, cfg):
    """pages: {page_no: [lines]} for one document -> {page_no: set(line_index)} of header/footer/page-number lines."""
    n_text = sum(1 for ls in pages.values() if any(l.strip() for l in ls))
    cands, count = {}, Counter()
    for p, ls in pages.items():
        idx = [i for i, l in enumerate(ls) if l.strip()]
        edge = idx[:cfg["header_edge_lines"]] + idx[-cfg["header_edge_lines"]:]
        cands[p] = {i: signature(ls[i]) for i in set(edge) if _edge_ok(ls[i], len(pages), cfg)}
        for s in set(cands[p].values()):
            count[s] += 1
    need = max(cfg["header_min_pages"], cfg["header_min_ratio"] * n_text)
    bad = {s for s, c in count.items() if c >= need and s}
    return {p: {i for i, s in c.items() if s in bad} for p, c in cands.items()}


def build_vocab(texts):
    v = Counter()
    for t in texts:
        v.update(re.findall(r"[a-z]{3,}", t.lower()))
    return v


_HY = re.compile(r"(?<![A-Za-z0-9])([A-Z]?[a-z]{2,})-$")


def dehyphenate(lines, vocab, min_count, log):
    out, i = [], 0
    while i < len(lines):
        cur = lines[i]
        while i + 1 < len(lines):
            m, nxt = _HY.search(cur), lines[i + 1]
            m2 = re.match(r"[a-z]+", nxt)
            if not (m and m2):
                break
            if vocab[(m.group(1) + m2.group(0)).lower()] >= min_count:
                cur = cur[:-1] + nxt; i += 1; log["dehyphen:joined"] += 1
            else:
                log["dehyphen:kept_hyphen"] += 1; break
        out.append(cur); i += 1
    return out


def whitespace(text, log):
    lines = [re.sub(r"[ ]{2,}", " ", l).rstrip(" \t") for l in text.split("\n")]
    s = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")
    if s != text:
        log["whitespace:changed_pages"] += 1
    return s


def invariant(text, keep):
    return Counter(c for c in text if c.isdecimal() or c in keep)


def check_invariant(raw, removed_lines, final, where, keep):
    diff = invariant(raw, keep) - invariant("\n".join(removed_lines), keep)
    if diff != invariant(final, keep):
        d = (diff - invariant(final, keep)) + (invariant(final, keep) - diff)
        raise InvariantError(f"{where}: digits/unit symbols changed by normalisation: {dict(d)}")
