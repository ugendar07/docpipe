"""`docpipe run -c config.yaml [--from STAGE] [--only STAGE] [--until STAGE]`: the whole pipeline from raw with one command.
Every stage is rerunnable alone and writes its saved outputs; engine stages run in the engine's own interpreter if configured."""
import argparse, os, subprocess, sys, time
from pathlib import Path

from docpipe.config import load

# name -> (module, engine whose interpreter runs it, or None for the current interpreter)
STAGES = {"inventory": ("docpipe.inventory", None), "triage": ("docpipe.triage", None),
          "rasterise": ("docpipe.imaging.rasterise", None), "clean": ("docpipe.imaging.clean", None),
          "ocr": ("docpipe.ocr.run_corpus", "ocr_corpus"), "language": ("docpipe.metadata.language", None),
          "confidence": ("docpipe.qa.confidence", None), "normalise": ("docpipe.normalise", None),
          "image_tables": ("docpipe.image_tables", None), "image_tables_sheet": ("docpipe.qa.image_tables_sheet", None),
          "image_tables_ocr": ("docpipe.ocr.run_image_tables_corpus", "image_tables_ocr"),
          "image_tables_text": ("docpipe.image_tables.text", None), "metadata": ("docpipe.metadata", None),
          "qa_sample": ("docpipe.qa.sample", None)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="docpipe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the pipeline")
    r.add_argument("-c", "--config", required=True)
    r.add_argument("--from", dest="start"); r.add_argument("--only"); r.add_argument("--until")
    a = ap.parse_args(argv)
    os.environ["DOCPIPE_CONFIG"] = str(Path(a.config).resolve())
    cfg = load(a.config)
    names = cfg["pipeline"]["stages"]
    for n in names + [x for x in (a.start, a.only, a.until) if x]:
        if n not in STAGES:
            raise SystemExit(f"unknown stage {n!r}; known: {list(STAGES)}")
    todo = [a.only] if a.only else names[names.index(a.start) if a.start else 0:(names.index(a.until) + 1) if a.until else None]
    log = Path(cfg["pipeline"]["log"]); log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)   # inherit everything: e.g. a missing TMPDIR breaks pytesseract's temp files
    for name in todo:
        mod, key = STAGES[name]
        engine = cfg[key]["engine"] if key else None
        py = cfg["pipeline"]["interpreters"].get(engine) or sys.executable
        t = time.time()
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"{time.strftime('%F %T')} START {name}\n")
        print(f"== {name}", flush=True)
        rc = subprocess.run([py, "-m", mod], env=env).returncode
        with open(log, "a", encoding="utf-8") as lf:
            lf.write(f"{time.strftime('%F %T')} END {name} rc={rc} {time.time() - t:.0f}s\n")
        if rc:
            print(f"STAGE FAILED: {name} (rc={rc}); fix and rerun with --from {name}", file=sys.stderr); return rc
    print("pipeline complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
