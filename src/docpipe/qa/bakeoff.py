"""Bake-off driver and report: `uv run python -m docpipe.qa.bakeoff [--score-only]`.
1) clean gold images (core env)  2) run each enabled engine in its own interpreter (resumable)  3) score + report.
Only typed gold items are scored; untyped (empty) items are skipped with a note."""
import csv, json, os, subprocess, sys
from collections import defaultdict
from pathlib import Path

import cv2
from docpipe.config import load

from docpipe.imaging.clean import clean_image
from docpipe.qa.score import aggregate, grid_from_csv, score_page


def read_gold(gdir):
    """-> list of dicts(gold_id, script, has_table, text, tables, typed)"""
    items = []
    for r in csv.DictReader(open(Path(gdir) / "index.csv", encoding="utf-8")):
        g = r["gold_id"]
        txt = (Path(gdir) / f"{g}.txt").read_text(encoding="utf-8")
        tables = [grid_from_csv(p.read_text(encoding="utf-8")) for p in sorted(Path(gdir).glob(f"{g}_t*.csv"))
                  if p.read_text(encoding="utf-8").strip()]
        items.append(dict(gold_id=g, script=r["script"], has_table=r["has_table"] == "True", text=txt, tables=tables,
                          typed=bool(txt.strip()) or bool(tables)))
    return items


def clean_gold(cfg):
    gdir = Path(cfg["bakeoff"]["gold_dir"]); out = gdir / "images_clean"; out.mkdir(exist_ok=True)
    for p in sorted((gdir / "images").glob("*.png")):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        cv2.imwrite(str(out / p.name), clean_image(img, cfg["clean"])[0])


def run_engines(cfg):
    b = cfg["bakeoff"]
    for eng, e in b["engines"].items():
        if not e["enabled"]:
            continue
        py = cfg["pipeline"]["interpreters"].get(eng) or sys.executable
        for cond in b["conditions"]:
            subprocess.run([py, "-m", "docpipe.ocr.run_gold", eng, cond], check=False, env=dict(os.environ))


def load_raw(raw_dir, engine, cond):
    d = Path(raw_dir) / engine / f"gold_{cond}"
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in d.glob("*.json")} if d.exists() else {}


def score_engine(items, raw, cfg):
    """-> (per-item scores {gold_id: dict}, failures [gold_id], seconds [..])"""
    sc, fails, secs = {}, [], []
    tf = cfg["score"]["terms_file"]
    terms = [t.strip() for t in Path(tf).read_text(encoding="utf-8").splitlines() if t.strip() and not t.startswith("#")] if tf else []
    for it in items:
        r = raw.get(it["gold_id"])
        if r is None:
            continue
        secs.append(r["seconds"])           # timing and failures cover every item, typed or not
        if not r["ok"]:
            fails.append(it["gold_id"]); continue
        if not it["typed"]:
            continue
        sc[it["gold_id"]] = score_page(it["text"], r["text"], it["tables"], r.get("tables"), cfg["score"]["units"], terms)
    return sc, fails, secs


def summarise(items, sc, cfg):
    s = cfg["score"]; by = {it["gold_id"]: it for it in items}
    notab = [sc[g] for g in sc if not by[g]["has_table"]]
    out = {}
    if notab:
        # groups come from the `script` column of the gold index: "all" plus one per distinct value (was hard-wired english / telugu_mixed)
        for name in ["all"] + sorted({by[g]["script"] for g in sc if not by[g]["has_table"]}):
            sub = [sc[g] for g in sc if not by[g]["has_table"] and (name == "all" or by[g]["script"] == name)]
            if sub:
                a = aggregate(sub, s["bootstrap_n"], s["seed"]); out[f"cer_{name}"] = a["cer"]; out[f"wer_{name}"] = a["wer"]; out[f"n_{name}"] = len(sub)
    if sc:
        a = aggregate(list(sc.values()), s["bootstrap_n"], s["seed"])
        for k, v in a.items():
            if k not in ("cer", "wer"):
                out[k] = v
    return out


def fmt(v, pct=True):
    if v is None:
        return "-"
    x, lo, hi = v if isinstance(v, tuple) else (v, None, None)
    if x is None:
        return "-"
    f = (lambda y: f"{100 * y:.1f}") if pct else (lambda y: f"{y:.1f}")
    return f"{f(x)}" + (f" [{f(lo)}-{f(hi)}]" if lo is not None else "")


def report(cfg):
    b = cfg["bakeoff"]; items = read_gold(b["gold_dir"])
    typed = [i for i in items if i["typed"]]
    L = ["# OCR bake-off", "", f"Gold items typed: {len(typed)} of {len(items)}. CER/WER use items without tables only "
         f"({sum(1 for i in typed if not i['has_table'])} typed); protected-token recall also covers table cells. "
         "Values in %, with seeded bootstrap 95% CI in brackets.", ""]
    rows = []
    for eng, e in b["engines"].items():
        if not e["enabled"]:
            continue
        for cond in b["conditions"]:
            raw = load_raw(b["raw_dir"], eng, cond)
            sc, fails, secs = score_engine(items, raw, cfg)
            if not raw:
                continue
            rows.append((eng, cond, summarise(items, sc, cfg), fails, secs, raw, len(sc)))
    scripts = sorted({i["script"] for i in typed if not i["has_table"]})
    cols = [("CER all", "cer_all")] + [(f"CER {s}", f"cer_{s}") for s in scripts] + [("WER all", "wer_all"),
            ("numbers", "numbers_recall"), ("units", "units_recall"), ("terms", "terms_recall"),
            ("tbl detected", "table_detected"), ("tbl cell acc", "table_cell_acc"), ("tbl row sim", "table_row_sim"),
            ("tbl flat recall", "table_flat_cell_recall")]
    L += ["| engine | cleanup | items | " + " | ".join(c for c, _ in cols) + " | s/page | fails |",
          "|---|---|---|" + "---|" * (len(cols) + 2)]
    for eng, cond, m, fails, secs, raw, n in rows:
        mean = sum(secs) / len(secs) if secs else None
        L.append(f"| {eng} | {cond} | {n} | " + " | ".join(fmt(m.get(k)) for _, k in cols) +
                 f" | {fmt(mean, pct=False)} | {len(fails)} |")
    L += ["", "## Hard filter (max seconds per page)", ""]
    for eng, cond, m, fails, secs, raw, n in rows:
        mean = sum(secs) / len(secs) if secs else 0
        L.append(f"- {eng}/{cond}: mean {mean:.1f}s, max {max(secs, default=0):.1f}s, peak RSS "
                 f"{max((r.get('peak_rss_gb', 0) for r in raw.values()), default=0):.2f} GB (process) -> "
                 f"{'PASS' if mean <= b['max_seconds_per_page'] and not fails else 'FAIL'}")
    L += ["", "## Failures", ""] + ([f"- {eng}/{cond}: {fails}" for eng, cond, m, fails, *_ in rows if fails] or ["- none"])
    Path(b["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    cfg = load()
    if "--score-only" not in sys.argv:
        clean_gold(cfg); run_engines(cfg)
    report(cfg)
