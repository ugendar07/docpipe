"""Stage 7: per-document metadata from the processed text. Unknown stays null; derived values carry their evidence.
Localities are kept as published (no remapping). `uv run python -m docpipe.metadata`"""
import csv, json, random, re
from collections import Counter
from pathlib import Path

from docpipe.config import load

def entities_from_text(text, entities, exclude, min_mentions, min_share, short_min=5, short_density=2.0):
    """-> (sorted entity list or None, counts). Matches of `exclude` (e.g. institute names) are not mentions.
    An entity needs `min_mentions`, or (for short documents) `short_min` mentions at >= `short_density` per 1000 words."""
    t = re.sub(exclude, " ", text, flags=re.I) if exclude else text
    cnt = {e: sum(len(re.findall(rx, t, re.I)) for rx in rxs) for e, rxs in entities.items()}
    tot, words = sum(cnt.values()), max(1, len(t.split()))
    keep = sorted(e for e, n in cnt.items() if tot and n / tot >= min_share
                  and (n >= min_mentions or (n >= short_min and 1000 * n / words >= short_density)))
    return (keep or None), cnt


RANGE = re.compile(r"(?<!\d)((?:19|20)\d\d)\s*[-–]\s*((?:19|20)?\d\d)(?!\d)")
YEAR = re.compile(r"(?<!\d)((?:19|20)\d\d)(?!\d)")


def years_in(text, lo, hi):
    """-> list of (start_year, published_string). Ranges like 2017-18 / 2013 - 2014 count once, as the range."""
    out = []
    for m in RANGE.finditer(text):
        a, b = m.group(1), m.group(2)
        end = a[:2] + b if len(b) == 2 else b
        if lo <= int(a) <= hi and int(a) < int(end) <= int(a) + 1:
            out.append((int(a), f"{a}-{end[2:]}"))
    rest = RANGE.sub(" ", text)
    out += [(int(y), y) for y in YEAR.findall(rest) if lo <= int(y) <= hi]
    return out


def pick_year(title, front, headers, manifest_year, cfg):
    """Score candidates by evidence; null if nothing found or the top two years tie. -> (year, published, evidence)."""
    score, pub, ev = Counter(), {}, []
    def add(items, w, label):
        for y, s in items:
            score[y] += w; pub.setdefault(y, s)
            ev.append(f"{label}:{s}")
    lo, hi = cfg["year_min"], cfg["year_max"]
    if manifest_year:
        add([(int(float(manifest_year)), str(int(float(manifest_year))))], cfg["w_manifest"], "manifest")
    add(years_in(title, lo, hi), cfg["w_title"], "title")
    hc = Counter(years_in(" \n".join(headers), lo, hi))
    add([k for k, n in hc.items() if n >= 3], cfg["w_header"], "header")
    add(years_in(front, lo, hi), cfg["w_front"], "front")
    if not score:
        return None, None, ""
    top = score.most_common(2)
    if len(top) == 2 and top[0][1] == top[1][1]:
        return None, None, "tie:" + ",".join(str(y) for y, _ in top)
    y = top[0][0]
    return y, pub[y], ";".join(dict.fromkeys(ev))


def year_confidence(evidence):
    """high: backed by the manifest, the title or repeated page headers; low: only mentions on the first pages."""
    if not evidence or evidence.startswith("tie"):
        return ""
    return "high" if any(evidence.startswith(k) or f";{k}" in evidence for k in ("manifest:", "title:", "header:")) else "low"


def doc_type(title, front, pages, cfg):
    t, f = title.lower(), front.lower()
    for r in cfg["doc_type_rules"]:
        if re.search(r["title"], t):
            return r["name"], f"title:{r['title']}"
        if r.get("front") and re.search(r["front"], f, re.S):
            return r["name"], f"front:{r['front']}"
    sd = cfg["short_doc"]
    if sd and pages <= sd["max_pages"] and sd.get("not_in_front", "\0") not in f:
        return sd["name"], f"pages<={sd['max_pages']} and no {sd.get('not_in_front')}"
    return None, ""


def doc_language(pages_lang, primary, primary_share, other_share):
    """primary language if >= primary_share of pages, else the top other language if >= other_share, else 'mixed'."""
    c = Counter(l for l in pages_lang if l)
    n = sum(c.values())
    if not n:
        return None, dict(c)
    if c[primary] / n >= primary_share:
        return primary, dict(c)
    other = [(k, v) for k, v in c.most_common() if k not in (primary, "mixed")]
    if other and other[0][1] / n >= other_share:
        return other[0][0], dict(c)
    return "mixed", dict(c)


def locality(manifest_locality, title, cfg):
    d = (manifest_locality or "").strip()
    if d and d.lower() not in [x.lower() for x in cfg["locality_ignore"]]:
        return d, "manifest"
    pat = cfg["locality_title_pattern"]
    m = re.search(pat, title) if pat else None
    return (m.group(1), "title") if m else (None, "")


def run(cfg):
    src = {r["doc_id"]: r for r in csv.DictReader(open(cfg["sources"], encoding="utf-8"))}
    mc, mp = cfg["_manifest"]["columns"], cfg["_manifest"]["path"]
    man = {r[mc["id"]]: r for r in csv.DictReader(open(mp, encoding="utf-8"))} if mp and Path(mp).exists() else {}
    lg = cfg["_languages"]; primary = lg["codes"].get(lg["scripts"][0], lg["scripts"][0])
    rows = []
    for doc, s in sorted(src.items()):
        m = man.get(doc, {})
        pages = [json.loads(l) for l in (Path(cfg["text_dir"]) / f"{doc}.jsonl").read_text(encoding="utf-8").splitlines()]
        live = [p for p in pages if p["text"].strip()]
        full = "\n".join(p["text"] for p in live)
        front = "\n".join(p["text"] for p in live[:cfg["year_front_pages"]])
        title = m.get(mc["title"], "")
        rem = Path(cfg["removed_dir"]) / f"{doc}.json"
        headers = [x["line"] for x in json.loads(rem.read_text(encoding="utf-8"))] if rem.exists() else []
        ents, cnt = entities_from_text(full, cfg["entities"], cfg["entity_exclude"], cfg["entity_min_mentions"], cfg["entity_min_share"],
                                       cfg["entity_short_min"], cfg["entity_short_density"])
        year, ypub, yev = pick_year(title, front, headers, s["year"], cfg)
        loc, lev = locality(s["locality_as_published"], title, cfg)
        lang, lc = doc_language([p.get("language") for p in pages], primary, cfg["lang_primary_share"], cfg["lang_secondary_share"])
        dt, dtev = doc_type(title, front[:1500], len(pages), cfg)
        rows.append(dict(doc_id=doc, title=title, entities=";".join(ents) if ents else "", entity_counts=json.dumps(cnt),
                         group=s["group"], locality_as_published=loc or "", locality_evidence=lev,
                         year=year or "", year_published=ypub or "", year_evidence=yev, year_confidence=year_confidence(yev), year_manifest=s["year"],
                         language=lang or "", language_pages=json.dumps(lc), document_type=dt or "", document_type_evidence=dtev,
                         document_type_manifest=m.get(mc["doc_type"], ""), pages=len(pages), pages_with_text=len(live),
                         source_url=s["source_url"], licence_note=s["licence_note"] or cfg["licence_default"] or ""))
    out = Path(cfg["out_csv"]); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    unk = lambda k: sum(1 for r in rows if not r[k])
    L = ["# Metadata report", "", f"Documents: {len(rows)}. Empty = unknown (null).", "", "| field | unknown | known |", "|---|---|---|"]
    L += [f"| {k} | {unk(k)} | {len(rows) - unk(k)} |" for k in ("entities", "locality_as_published", "year", "language", "document_type")]
    L += ["", "## Values", ""]
    for k in ("entities", "language", "document_type"):
        L.append(f"- {k}: {dict(Counter(r[k] or 'null' for r in rows))}")
    lowy = [r for r in rows if r["year_confidence"] == "low"]
    L += ["", f"## Years with low confidence (front-page mentions only): {len(lowy)}", ""] + [f"- {r['doc_id']}: {r['year']} ({r['year_evidence'][:60]})" for r in lowy]
    conf = [r for r in rows if r["year"] and r["year_manifest"] and int(r["year"]) != int(float(r["year_manifest"]))]
    L += ["", f"## Year differs from manifest ({len(conf)})", ""] + [f"- {r['doc_id']}: derived {r['year']} ({r['year_evidence']}) vs manifest {r['year_manifest']}" for r in conf]
    L += ["", "## Spot check (seeded sample; compare with the first page)", ""]
    for r in random.Random(cfg["seed"]).sample(rows, min(cfg["spot_check"], len(rows))):
        first = next((json.loads(l)["text"] for l in (Path(cfg["text_dir"]) / f"{r['doc_id']}.jsonl").read_text(encoding="utf-8").splitlines()
                      if json.loads(l)["text"].strip()), "")
        L.append(f"- **{r['doc_id']}** {r['title'][:70]!r} | entities={r['entities'] or None} | year={r['year'] or None} ({r['year_evidence'][:40]}) | "
                 f"lang={r['language'] or None} | type={r['document_type'] or None} | locality={r['locality_as_published'] or None}\n  - page 1: {first[:140]!r}")
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:20]))


if __name__ == "__main__":
    full = load()
    run({**full["metadata"], "_manifest": full["manifest"]})
