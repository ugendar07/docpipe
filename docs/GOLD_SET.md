# Creating a gold set for a new corpus

A gold set is a small, hand-typed ground truth. It is what lets you say which OCR engine to use on *your* documents and how wrong it is.
Without one, "the OCR works" is an opinion. Budget: about 15-25 items, 2-4 hours of typing.

## 1. Run the pipeline up to the OCR stage
```
docpipe run -c my.yaml --until language
```
Gold pages are drawn from pages that will actually be OCR'd, and the language survey tells the picker which pages have enough text.

## 2. Get candidates
Set `gold.n_candidates` (default 40) and, optionally, `gold.forced` (`[[doc_id, page], ...]`, pages you want in regardless: e.g. every non-primary-script page you know of). Then:
```
python -m docpipe.qa.gold_candidates          # DOCPIPE_CONFIG=my.yaml must be set
```
Output: `gold.candidates_csv` plus contact sheets `gold.sheet_prefix_*.png`. Candidates are spread round-robin over (triage label, document), so no single document dominates; pages under `gold.min_ocr_chars` are excluded (covers, dividers).

## 3. Pick (the selection rule)
Look at the sheets and choose roughly 20, aiming for:
- a mix of scanned and born-digital-with-image pages (the labels are on the sheet);
- every script you listed in `languages.scripts`, at least 2-3 pages each;
- pages **with** tables (aim for 8 of 20 if the corpus has them) and pages without;
- at least one page with sub/superscripts, units or other symbols you must not lose;
- no more than a quarter of the items from one document.

Record each pick in `gold.picks`: `{n: <candidate number>, mode: full|crop, box: [x0,y0,x1,y1] (page fractions, crop only), script: <name>, table: true|false, note: "..."}`.
`script` is what the bake-off report groups CER by, so use the script names from `languages.scripts`, or `mixed`.

## 4. Build templates
```
python -m docpipe.qa.gold_build
```
Renders each pick at 300 DPI to `<gold.dir>/images/` and creates **empty** `<gold_id>.txt` (plus `<gold_id>_t1.csv` for table items). Typed files are never overwritten.

## 5. Type it
- `.txt`: all text in reading order, exactly as printed; UTF-8; no spelling corrections; each script in its own script; units and sub/superscripts as printed. Text inside tables goes only in the CSV.
- `_tN.csv`: one row per table row; repeat merged header cells across the columns they span and merged row labels down the rows they span.
- Skip page numbers, running headers, logos and photos.
- `[?]` for illegible text. It counts as an error for every engine equally.
- Leave an item **empty** if you cannot type it; empty items are skipped by the scorer and reported.

## 6. Protected terms
Set `score.units` and (optionally) `score.terms_file` from your typed pages: the units and domain terms (chemical names, part numbers, drug names...) whose corruption would matter. Have the corpus owner approve the list. Recall of numbers, units and terms is reported next to CER.

## 7. Score the engines
Enable the engines to compare under `bakeoff.engines` (and give each its interpreter in `pipeline.interpreters`), then:
```
python -m docpipe.qa.bakeoff            # cleans images, runs each engine off/on cleanup, writes bakeoff.report
python -m docpipe.qa.bakeoff --score-only
```
Read the table with the caveats on: with ~10 typed items the bootstrap intervals are wide, and a winner differing by less than the interval is a tie. Pick the engine, set `ocr_corpus.engine`, and write the decision and its evidence in your own decisions log.

## 8. Table gold (image tables)
After the `image_tables_sheet` stage, fill `VERDICT (table/chart/other)` in `reports/image_tables_review.csv` from the contact sheet. Real tables can be typed as one CSV each. The image-table **bake-off** code of the original project was not extracted (see README, limitations).
