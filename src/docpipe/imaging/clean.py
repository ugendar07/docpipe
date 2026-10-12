"""Stage 4: image cleanup. Each step is a config toggle; originals are never touched; only changed pages are written.
`python -m docpipe.imaging.clean`"""
import json, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(1 << 40))
import cv2
import numpy as np
import pytesseract
from docpipe.config import load
from PIL import Image


def fill_colour(img):
    return tuple(int(v) for v in np.median(img.reshape(-1, img.shape[-1]), axis=0)[:3])


def orientation(img, cfg):
    """-> (img, params). Rotates by 90/180/270 only when OSD is confident."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = cfg["osd_max_side"] / max(g.shape)
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    try:
        o = pytesseract.image_to_osd(Image.fromarray(small), output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractError as e:   # e.g. "Too few characters": a legitimate "no OSD result"; anything else is a real error
        return img, dict(rotate=None, conf=None, osd_error=str(e)[:80])
    rot, conf = int(o["rotate"]), float(o["orientation_conf"])
    p = dict(rotate=rot, conf=conf)
    if rot and conf >= cfg["osd_min_conf"]:
        img = np.ascontiguousarray(np.rot90(img, k=-rot // 90))  # OSD 'rotate' = clockwise degrees needed
        p["applied"] = True
    return img, p


def skew_angle(img, cfg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s = 1000 / max(g.shape)
    b = cv2.threshold(cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 0, 255,
                      cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = b.shape

    def score(a):
        r = cv2.warpAffine(b, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h))
        return float(np.var(r.sum(axis=1)))
    coarse = np.arange(-cfg["deskew_max_deg"], cfg["deskew_max_deg"] + 1e-9, 0.25)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.25, best + 0.2501, 0.05)
    best = max(fine, key=score)
    return float(best), score(best) / max(score(0.0), 1e-9)


def deskew(img, cfg):
    a, gain = skew_angle(img, cfg)
    p = dict(angle=round(a, 2), gain=round(gain, 3))
    # an angle at the edge of the search range means the fit failed (graphic-heavy page), not a real skew
    if cfg["deskew_min_deg"] <= abs(a) < cfg["deskew_max_deg"] and gain >= cfg["deskew_min_gain"]:
        h, w = img.shape[:2]
        img = cv2.warpAffine(img, cv2.getRotationMatrix2D((w / 2, h / 2), a, 1), (w, h), flags=cv2.INTER_CUBIC,
                             borderValue=fill_colour(img))
        p["applied"] = True
    return img, p


def _run(dark_line, max_px):
    """Length of the leading run of border lines; 0 if the run is longer than max_px (dark page design)."""
    n = 0
    for v in dark_line:
        if not v:
            break
        n += 1
        if n > max_px:
            return 0
    return n


def border(img, cfg):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    d = g < cfg["border_dark_below"]
    h, w = g.shape
    rows, cols = d.mean(axis=1) >= cfg["border_dark_frac"], d.mean(axis=0) >= cfg["border_dark_frac"]
    t, b = _run(rows, int(h * cfg["border_max_frac"])), _run(rows[::-1], int(h * cfg["border_max_frac"]))
    l, r = _run(cols, int(w * cfg["border_max_frac"])), _run(cols[::-1], int(w * cfg["border_max_frac"]))
    t, b, l, r = (v if v >= cfg["border_min_px"] else 0 for v in (t, b, l, r))
    p = dict(crop=[t, b, l, r])
    if t or b or l or r:
        img = img[t:h - b, l:w - r]; p["applied"] = True
    return img, p


def denoise(img, cfg):
    return cv2.fastNlMeansDenoisingColored(img, None, cfg["denoise_h"], cfg["denoise_h"], 7, 21), dict(applied=True)


def shadow(img, cfg):
    k = cfg["shadow_kernel"]
    ch = []
    for c in cv2.split(img):
        bg = cv2.medianBlur(cv2.dilate(c, np.ones((7, 7), np.uint8)), k)
        ch.append(cv2.divide(c, bg, scale=255))
    return cv2.merge(ch), dict(applied=True)


STEPS = [("orientation", orientation), ("deskew", deskew), ("border", border), ("denoise", denoise), ("shadow", shadow)]


def clean_image(img, cfg):
    """-> (img, {step: params}) applying the enabled steps in order."""
    log = {}
    for name, fn in STEPS:
        if cfg["steps"].get(name):
            img, log[name] = fn(img, cfg)
    return img, log


def run(cfg):
    out = Path(cfg["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    pages = [x for x in map(json.loads, open(cfg["pages_index"], encoding="utf-8")) if not x["blank"]]

    def job(x):
        img = cv2.imread(x["path"], cv2.IMREAD_COLOR)
        res, log = clean_image(img, cfg)
        changed = any(v.get("applied") for v in log.values())
        path = x["path"]
        if changed:
            d = out / x["doc_id"]; d.mkdir(exist_ok=True)
            path = str(d / Path(x["path"]).name)
            if not Path(path).exists():   # never rewrite an image that may be in use by a running OCR job
                cv2.imwrite(path, res)
        return dict(doc_id=x["doc_id"], page=x["page"], path=path, changed=changed, steps=log)

    with ThreadPoolExecutor(cfg["workers"]) as ex:
        idx = list(ex.map(job, pages))
    idx.sort(key=lambda r: (r["doc_id"], r["page"]))
    (out / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in idx), encoding="utf-8")
    L = ["# Cleanup report", "", f"- pages: {len(idx)}  |  changed: {sum(r['changed'] for r in idx)}",
         f"- steps enabled: {[k for k, v in cfg['steps'].items() if v]}", "", "| step | pages changed |", "|---|---|"]
    L += [f"| {n} | {sum(1 for r in idx if r['steps'].get(n, {}).get('applied'))} |" for n, _ in STEPS if cfg["steps"].get(n)]
    errs = sum(1 for r in idx if r["steps"].get("orientation", {}).get("osd_error"))
    osd = [r for r in idx if r["steps"].get("orientation", {}).get("rotate")]
    L += ["", f"- pages with no OSD result (too little text): {errs}"]
    L += ["", f"## OSD suggested a rotation on {len(osd)} pages (applied only if confidence >= {cfg['osd_min_conf']})", ""]
    L += [f"- {r['doc_id']} p{r['page']}: rotate {r['steps']['orientation']['rotate']}, conf {r['steps']['orientation']['conf']}, "
          f"applied={bool(r['steps']['orientation'].get('applied'))}" for r in osd]
    Path(cfg["report"]).write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    run(load()["clean"])
