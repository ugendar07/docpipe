"""P2-09: OCR every non-blank rasterised page with the chosen engine, inside that engine's own interpreter:
`PYTHONPATH=src .venvs/surya/bin/python -m docpipe.ocr.run_corpus`. Resumable: pages with an ok result are skipped,
failed pages are retried. One JSON per page, engine output untouched."""
import json, resource, time
from pathlib import Path

from docpipe.config import load
from PIL import Image

from docpipe.ocr.base import get_engine

Image.MAX_IMAGE_PIXELS = None


def pages(cfg):
    clean = {(x["doc_id"], x["page"]): x["path"] for x in map(json.loads, open(cfg["clean_index"], encoding="utf-8"))}
    out = []
    for x in map(json.loads, open(cfg["pages_index"], encoding="utf-8")):
        if not x["blank"]:
            out.append((x["doc_id"], x["page"], clean.get((x["doc_id"], x["page"]), x["path"]), x["label"]))
    return sorted(out)


def done(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("ok") is True
    except Exception:
        return False


def main():
    full = load()
    cfg = full["ocr_corpus"]; out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    Path(cfg["log"]).parent.mkdir(parents=True, exist_ok=True)
    allp = pages(cfg)
    todo = [p for p in allp if not done(out / p[0] / f"p{p[1]:04d}.json")]
    log = open(cfg["log"], "a", encoding="utf-8")
    def say(m):
        print(m, flush=True); log.write(m + "\n"); log.flush()
    say(f"{time.strftime('%F %T')} start: {len(todo)} to do of {len(allp)}")
    if not todo:
        return
    eng = get_engine(cfg["engine"], full["ocr"])
    t0, fails, consec = time.time(), 0, 0
    for i, (doc, page, path, label) in enumerate(todo, 1):
        (out / doc).mkdir(exist_ok=True)
        rec = dict(doc_id=doc, page=page, image=path, triage_label=label, engine=cfg["engine"])
        t = time.time()
        try:
            r = eng.ocr(Image.open(path))
            rec.update(ok=True, seconds=round(time.time() - t, 2), text=r["text"], confidence=r["confidence"],
                       tables=r["tables"], blocks=r["blocks"], version=r["version"], raw=r.get("raw"))
            consec = 0
        except Exception as e:
            rec.update(ok=False, seconds=round(time.time() - t, 2), error=f"{type(e).__name__}: {e}")
            fails += 1; consec += 1
        (out / doc / f"p{page:04d}.json").write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
        el = time.time() - t0
        say(f"{time.strftime('%T')} [{i}/{len(todo)}] {doc} p{page} {rec['seconds']}s ok={rec['ok']} "
            f"eta {(el / i) * (len(todo) - i) / 60:.0f} min rss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30:.2f}GB")
        if consec >= cfg["max_consecutive_failures"]:
            say(f"STOP: {consec} consecutive failures, engine looks dead; rerun to resume"); return
    say(f"{time.strftime('%F %T')} finished: {len(todo) - fails} ok, {fails} failed in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
