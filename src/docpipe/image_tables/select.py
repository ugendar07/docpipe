"""Which image regions go to OCR: those the user marked `table` in the review sheet (a user-owned input, like lowconf_decisions.csv).
Stdlib only, so the OCR interpreter can import it."""
import csv, os

def selection(cfg):
    """-> (selected manifest rows, {region_id: reason} for every other region). Aborts if the sheet doesn't match the manifest."""
    VERDICT = cfg["_review"]["verdict_column"]
    man = {r["region_id"]: r for r in csv.DictReader(open(cfg["manifest"], encoding="utf-8"))}
    # NEW in docpipe: no sheet yet (first run) = no verdicts = nothing is OCR'd; the sheet stage writes the template to fill in
    verdict = ({r["region_id"]: r[VERDICT].strip().lower() for r in csv.DictReader(open(cfg["sheet_csv"], encoding="utf-8"))}
               if os.path.exists(cfg["sheet_csv"]) else {})
    stale = sorted(set(verdict) - set(man))
    if stale:
        raise SystemExit(f"review sheet has {len(stale)} region ids not in {cfg['manifest']} (e.g. {stale[0]}): the manifest changed since the review")
    chosen, skipped = [], {}
    for rid, r in man.items():
        v = verdict.get(rid, "")
        if v == "table":
            chosen.append(r)
        else:
            skipped[rid] = f"verdict {v}" if v else ("no verdict" if r["table_like"] == "True" else "not table-like")
    return chosen, skipped
