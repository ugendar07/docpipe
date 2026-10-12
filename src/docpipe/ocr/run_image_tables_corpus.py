"""Phase 2b production OCR: Surya (page mode) on every region the user marked `table`, in the OCR interpreter:
`PYTHONPATH=src .venvs/surya/bin/python -m docpipe.ocr.run_image_tables_corpus`. Resumable; one raw JSON per region."""
import json, time
from pathlib import Path

from docpipe.config import load
from PIL import Image

from docpipe.image_tables.select import selection
from docpipe.ocr.base import get_engine

Image.MAX_IMAGE_PIXELS = None


def done(p):
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("ok") is True
    except Exception:
        return False


def main():
    full = load(); cfg = full["image_tables"]; oc = full["image_tables_ocr"]
    chosen, _ = selection(cfg); out = Path(oc["out_dir"])
    todo = [r for r in chosen if not done(out / r["doc_id"] / f"{r['region_id']}.json")]
    print(f"{len(todo)} to do of {len(chosen)}", flush=True)
    if not todo:
        return
    eng = get_engine(oc["engine"], full["ocr"]); fails = 0
    for r in todo:
        t = time.time(); rec = dict(region_id=r["region_id"], doc_id=r["doc_id"], page=int(r["page"]), region=int(r["region"]), engine=oc["engine"])
        try:
            res = eng.ocr(Image.open(r["crop_path"]))
            rec.update(ok=True, seconds=round(time.time() - t, 2), text=res["text"], tables=res["tables"], version=res["version"], raw=res.get("raw"))
            fails = 0
        except Exception as e:  # one bad region must not lose the others
            rec.update(ok=False, seconds=round(time.time() - t, 2), error=f"{type(e).__name__}: {e}"); fails += 1
        (out / r["doc_id"]).mkdir(parents=True, exist_ok=True)
        (out / r["doc_id"] / f"{r['region_id']}.json").write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
        print(f"{r['region_id']}: {rec['seconds']}s ok={rec['ok']}", flush=True)
        if fails >= oc["max_consecutive_failures"]:
            raise SystemExit(f"{fails} consecutive failures: engine looks dead, stopping")


if __name__ == "__main__":
    main()
