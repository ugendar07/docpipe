"""Config loading: packaged defaults deep-merged with the user's YAML. Stages run with project.dir as working directory,
so every path in the config is relative to it. The config file location travels to stage subprocesses in DOCPIPE_CONFIG."""
import os
from pathlib import Path

import yaml

from docpipe.scripts import RANGES

DEFAULTS = Path(__file__).with_name("defaults.yaml")


def _merge(a, b):
    for k, v in (b or {}).items():
        a[k] = _merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return a


def load(path=None):
    """-> full config dict. Chdirs into project.dir. Raises SystemExit on missing/invalid required fields."""
    path = Path(path or os.environ["DOCPIPE_CONFIG"]).resolve()
    user = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    cfg = _merge(yaml.safe_load(DEFAULTS.read_text(encoding="utf-8")), user)
    root = (path.parent / cfg["project"]["dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.chdir(root)
    os.environ["DOCPIPE_CONFIG"] = str(path)
    lg = cfg["languages"]
    if not lg["ocr"]:
        raise SystemExit("config: languages.ocr is required (Tesseract language string, e.g. 'eng')")
    for s in lg["scripts"] + lg["out_of_scope"]:
        if s not in RANGES:
            raise SystemExit(f"config: unknown script {s!r}; known: {sorted(RANGES)}")
    cfg["ocr"]["tesseract"]["lang"] = cfg["ocr"]["tesseract"]["lang"] or lg["ocr"]
    cfg["image_tables"]["lang"] = cfg["image_tables"]["lang"] or lg["ocr"]
    for name, sec in cfg.items():     # stages see languages/review/seed without having to know the whole config
        if isinstance(sec, dict) and name not in ("languages", "project"):
            sec["_languages"], sec["_review"] = lg, cfg["review"]
    return cfg
