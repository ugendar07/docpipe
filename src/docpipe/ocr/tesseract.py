import pytesseract
from pytesseract import Output


class Engine:
    name = "tesseract"

    def __init__(self, cfg):
        self.lang, self.psm = cfg["lang"], cfg.get("psm", 3)

    def ocr(self, image):
        d = pytesseract.image_to_data(image, lang=self.lang, config=f"--psm {self.psm}", output_type=Output.DICT)
        words = [(t, float(c)) for t, c in zip(d["text"], d["conf"]) if t.strip() and float(c) >= 0]
        text = pytesseract.image_to_string(image, lang=self.lang, config=f"--psm {self.psm}")
        return dict(text=text, confidence=(sum(c for _, c in words) / len(words) / 100) if words else None,
                    blocks=d, tables=None, engine=self.name, version=str(pytesseract.get_tesseract_version()),
                    raw=d)
