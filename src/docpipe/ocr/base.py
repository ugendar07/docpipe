"""Common OCR engine interface. An engine is any object with .name and .ocr(PIL.Image) -> dict:
   {text, confidence (0-1 or None), blocks (engine-native, JSON-able), tables (None or list), engine, version}
Timing is added by the caller. Raw engine output goes into `raw` untouched."""
import importlib


def get_engine(name, cfg):
    """Lazy import so a missing optional dependency only breaks its own engine."""
    return importlib.import_module(f"docpipe.ocr.{name}").Engine(cfg.get(name) or {})
