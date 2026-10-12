"""Surya in block mode for a crop that is already one table: one BLOCK_PROMPT request on the whole image (no layout step)."""
from docpipe.ocr.surya import Engine as PageEngine, html_text


class Engine(PageEngine):
    name = "surya_block"

    def ocr(self, image):
        import importlib.metadata as md
        from surya.inference.parsers import clean_block_html
        from surya.inference.prompts import PROMPT_TYPE_BLOCK
        from surya.inference.schema import BatchInputItem
        from surya.settings import settings
        out = self.manager.generate([BatchInputItem(image=image.convert("RGB"), prompt_type=PROMPT_TYPE_BLOCK,
                                                    max_tokens=settings.SURYA_MAX_TOKENS_BLOCK_CEILING)])[0]
        if out.error:
            raise RuntimeError("surya block request failed")
        html = clean_block_html(out.raw)
        return dict(text=html_text(html), confidence=out.mean_token_prob, blocks=None,
                    tables=[html] if "<table" in html else None,   # anything else (e.g. an <img alt=...>) is "not a table"
                    engine=self.name, version=md.version("surya-ocr"), raw=out.raw)
