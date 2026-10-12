# docpipe

Turns a folder of PDFs into clean, page-addressed text plus document metadata, from one config file and one command. It triages every page (text layer vs scan), rasterises and cleans the pages that need OCR, runs a pluggable OCR engine, normalises the text without altering numbers or units, derives metadata with evidence, and recovers tables embedded as images.

```
docpipe run -c config.yaml
```

## Results on two corpora

| | Corpus 1: Telangana (source project) | Corpus 2: Maharashtra |
|---|---|---|
| Domain / language | Agriculture documents, English + Telugu (+ Hindi dropped) | Agriculture documents English only |
| Documents / pages | 101 / 3,310 | 34 / 1,175 |
| Input manifest | Yes | None (generated ids) |
| born_digital / hybrid | 2,499 (75.5%) / 92 (2.8%) | 1,061 (90.3%) / 0 |
| scanned / blank_or_vector / needs_review | 303 (9.2%) / 233 (7.0%) / 183 (5.5%) | 69 (5.9%) / 6 (0.5%) / 39 (3.3%) |
| Pages OCR'd | 574 | 71 |
| OCR engine | Surya 0.22.1 | Tesseract 5 |
| OCR time | about 3.5 h full run | 91 s (OCR stage) |
| OCR accuracy | CER 2.9% English, 17.6% Telugu/mixed (11 gold items) | **Not measured** (no gold set) |
| Confidence score | Surya vs Tesseract agreement | **Not available** (engine = cross-check) |
| Digit/unit invariant | Held on 3,310 pages | Held on 1,175 pages |
| Reproducible from raw | 103/103 final, 350/350 intermediate files identical | 37/37 final, 156/156 intermediate files identical |
| QA sample | 20/20 OK (user review) | Not yet reviewed |
| Config needed | Large project config | 3 lines |

Details: [reports/gate_report_corpus2.md](reports/gate_report_corpus2.md). Corpus 1 numbers are copied from the source project's gate report; its pages were processed by the original code, with a 4-document subset (18 pages) rerun through docpipe: 17 of 18 pages identical (the other used a user decision sheet not present in the test copy).

## Known limitations

- **The second corpus is the same document family as the first**, in a different state. It does not test a different domain or a different script mix. Treat "dataset-agnostic" as unproven until it has been run on one that does.
- **Reproducibility is determinism, not correctness.** OCR digit errors pass through; normalisation never alters numbers.
- **Without a different, second engine there is no OCR confidence score.** With Tesseract as the engine only loop and script flags apply.
- **The garbage-text-layer rule may send good pages to OCR** (symbol-font bullets count as private-use characters). Thresholds were tuned on Telangana only.
- **Metadata needs a manifest title or configured rules.** Entities, locality and document type are empty by default, and nothing reads the title from page 1.
- A fixed character map in `normalise/clean.py` (ligatures, private-use bullets) is not yet configurable.
- Dehyphenation only handles Latin text.
- The image-table inventory is a lower bound (small, vector or low-contrast tables are missed) and every candidate needs a human verdict. The image-table bake-off code was not extracted.
- Phase 3 structure parsing (tables as rows, section paths) is not included.
- Tested install: base package only. The `[surya]` extra has not been installed from scratch here.

## Quick start

```
uv venv .venv --python 3.12 && uv pip install -e ".[dev]"
```

Put PDFs under `<project dir>/data/raw/`, then a minimal `config.yaml`:

```yaml
project: {dir: .}
languages: {ocr: eng}      # Tesseract language pack(s); the only required field
manifest: {path: null}     # or the path of your input manifest CSV
```

```
.venv/bin/docpipe run -c config.yaml [--from STAGE | --only STAGE | --until STAGE]
```

Defaults for everything else are in `src/docpipe/defaults.yaml`; `examples/telangana.yaml` shows a full corpus-specific config. To compare OCR engines on your own pages, see [docs/GOLD_SET.md](docs/GOLD_SET.md).
