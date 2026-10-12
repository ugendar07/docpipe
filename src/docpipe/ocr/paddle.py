import numpy as np


class Engine:
    name = "paddle"

    def __init__(self, cfg):
        from paddleocr import PaddleOCR
        self.lang = cfg["lang"]
        self.ocr_ = PaddleOCR(lang=self.lang, use_doc_orientation_classify=False, use_doc_unwarping=False,
                              use_textline_orientation=False)

    def ocr(self, image):
        import importlib.metadata as md
        res = self.ocr_.predict(np.array(image.convert("RGB")))[0]
        texts, scores = list(res["rec_texts"]), [float(s) for s in res["rec_scores"]]
        boxes = [np.asarray(p).tolist() for p in res["rec_polys"]]
        return dict(text="\n".join(texts), confidence=(sum(scores) / len(scores)) if scores else None,
                    blocks=[dict(text=t, score=s, poly=b) for t, s, b in zip(texts, scores, boxes)], tables=None,
                    engine=self.name, version=md.version("paddleocr"),
                    raw=dict(texts=texts, scores=scores, polys=boxes))
