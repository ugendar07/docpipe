"""Run ONE engine on the gold images (inside that engine's own interpreter):
`PYTHONPATH=src <python> -m docpipe.ocr.run_gold <engine> <off|on>`. Resumable; one JSON per gold item."""
import csv, json, resource, sys, time
from pathlib import Path

from docpipe.config import load
from PIL import Image

from docpipe.ocr.base import get_engine

Image.MAX_IMAGE_PIXELS = None


def main(engine, cond):
    cfg = load()
    b, gdir = cfg["bakeoff"], Path(cfg["bakeoff"]["gold_dir"])
    imgs = gdir / ("images" if cond == "off" else "images_clean")
    out = Path(b["raw_dir"]) / engine / f"gold_{cond}"; out.mkdir(parents=True, exist_ok=True)
    ids = [r["gold_id"] for r in csv.DictReader(open(gdir / "index.csv", encoding="utf-8"))]
    todo = [g for g in ids if not (out / f"{g}.json").exists()]
    if not todo:
        print(f"{engine}/{cond}: all {len(ids)} done"); return
    t0 = time.time(); eng = get_engine(engine, cfg["ocr"]); load = time.time() - t0
    for g in todo:
        im = Image.open(imgs / f"{g}.png")
        rec = dict(gold_id=g, engine=engine, condition=cond, load_seconds=round(load, 1))
        t = time.time()
        try:
            r = eng.ocr(im)
            rec.update(seconds=round(time.time() - t, 2), ok=True, text=r["text"], confidence=r["confidence"],
                       tables=r["tables"], version=r["version"], raw=r.get("raw"))
        except Exception as e:  # a crash on one item must not lose the others
            rec.update(seconds=round(time.time() - t, 2), ok=False, error=f"{type(e).__name__}: {e}")
        rec["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30, 2)
        (out / f"{g}.json").write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"{engine}/{cond} {g}: {rec['seconds']}s ok={rec['ok']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
