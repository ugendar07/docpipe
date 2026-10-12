from html.parser import HTMLParser


class _Text(HTMLParser):
    def __init__(self):
        super().__init__(); self.out = []

    def handle_data(self, d):
        self.out.append(d)

    def handle_endtag(self, tag):
        if tag in ("p", "div", "tr", "li", "h1", "h2", "h3", "br", "table"):
            self.out.append("\n")
        elif tag in ("td", "th"):
            self.out.append("\t")


def html_text(h):
    p = _Text(); p.feed(h)
    return "".join(p.out).strip()


def _seed_requests(seed):
    """Surya retries a failed/looping generation at escalating temperature (0.2-0.6, top_p 0.95) with NO seed, so those pages
    are random across runs. Give every chat-completion request a fixed seed so the retries are reproducible."""
    from openai.resources.chat.completions import Completions
    orig = Completions.create
    if getattr(orig, "_seeded", False):
        return
    def create(self, *a, **k):
        k.setdefault("seed", seed)
        return orig(self, *a, **k)
    create._seeded = True
    Completions.create = create


class Engine:
    name = "surya"

    def __init__(self, cfg):
        if cfg.get("seed") is not None:
            _seed_requests(cfg["seed"])
        from surya.inference import SuryaInferenceManager
        from surya.recognition import RecognitionPredictor
        self.manager = SuryaInferenceManager()
        self.pred = RecognitionPredictor(self.manager)

    def ocr(self, image):
        import importlib.metadata as md
        page = self.pred([image.convert("RGB")])[0]
        blocks = [b.model_dump() for b in sorted(page.blocks, key=lambda b: b.reading_order)]
        text = "\n\n".join(html_text(b["html"]) for b in blocks if b["html"] and not b["skipped"])
        tables = [b["html"] for b in blocks if b["label"].lower() == "table" and b["html"]] or None
        return dict(text=text, confidence=None, blocks=blocks, tables=tables, engine=self.name,
                    version=md.version("surya-ocr"), raw=page.model_dump())
