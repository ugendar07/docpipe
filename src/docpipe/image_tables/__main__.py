"""Phase 2b step 1 entry point (worker functions live in scan.py so multiprocessing can import them)."""
from docpipe.config import load

from docpipe.image_tables.scan import run

if __name__ == "__main__":
    full = load()
    rows = run(full["image_tables"], full["ocr_corpus"]["out_dir"], full["triage"]["out_dir"], full["inventory"]["manifest_out"])
    print(f"{len(rows)} regions listed; {sum(r['table_like'] for r in rows)} table-like; manifest {full['image_tables']['manifest']}")
