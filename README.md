# Engineering drawing extraction pipeline

This repository contains an offline-first pipeline for raster P&IDs and table-bearing engineering drawings. It renders PDFs, performs tiled OCR, detects instrument bubbles and table grids geometrically, normalizes domain tags, and writes traceable structured output.

## Quick start

Requirements:

- Python 3.9+ with `Pillow` and `numpy`
- Poppler (`pdftoppm`) for PDF inputs
- macOS Apple Vision (default on macOS), or Tesseract on other platforms

```bash
python3 -m pip install -r requirements.txt
python3 pipeline.py take_home_documents.pdf --output output/run
python3 score.py output/run/structured_output.json answer_key.json
python3 score_tables.py output/run/tables.json evaluation/table3_expected.json
```

Useful live-session options:

```bash
# One new PDF page, with an explicit ID
python3 pipeline.py unseen.pdf --document-id 4 --output output/unseen

# Force the portable backend
python3 pipeline.py unseen.pdf --ocr-backend tesseract --output output/unseen

# Skip table reconstruction when only P&ID values are needed
python3 pipeline.py unseen.pdf --mode pid --output output/unseen

# Auto-select the tag grammar (default), or force one during diagnosis
python3 pipeline.py unseen.pdf --mode pid --profile auto --output output/unseen
python3 pipeline.py unseen.pdf --mode pid --profile generic_isa_pid --output output/unseen

# Build a worksheet containing every prediction, not only low-confidence items
python3 review.py output/unseen --output output/unseen/review_audit.csv

# With an answer key, label rows as TP/FP/FN or OUT_OF_SCOPE
python3 review.py output/final --answer-key answer_key.json --output output/final/review_audit.csv

# Override detector thresholds without editing business logic
python3 pipeline.py unseen.pdf --config config/pipeline_config.json --output output/unseen
```

The default 360 DPI run takes roughly 15-25 seconds per page on this machine. All recognition is local; no document or answer-key content is sent to an API.

## Outputs

- `structured_output.json`: one `{document_id, answers}` object per page
- `extraction_metadata.json`: field-level confidence, source bounding box, and contributing signals
- `review_queue.json`: low-confidence, conflicting, or template-only values requiring human review
- `raw_ocr.json`: every recognized string, confidence, alternative, and normalized page coordinate
- `tables.json`: table matrices plus bounding boxes and merged-cell metadata
- `tables/*.csv`: one directly inspectable CSV per substantive table

The raw detections make errors auditable. A bad value can be attributed to OCR, symbol classification, or normalization instead of disappearing inside a single opaque model call.

## Design

1. Render each page at 360 DPI.
2. OCR overlapping 3x3 tiles so small tags remain large enough to recognize.
3. Select an isolated document profile from OCR evidence. `legacy_nuclear_pid` preserves the supplied benchmark grammar; `generic_isa_pid` supports five/six-digit ISA-style equipment tags. If neither family has positive evidence, `unknown_pid` abstains from semantic extraction and requires a full-page review. `--profile` can override auto-selection without changing code.
4. Detect instrument bubbles and use local geometry (including diagonal leaders and U-bends) to classify symbols. Bubble text is OCR'd from original, autocontrast, and thresholded crops and resolved by vote.
5. Detect continuation hexagons from six-edge geometry. OCR reads are backed by a configurable short-code glyph matcher because general OCR engines often abstain on isolated one-character labels.
6. Apply conservative, profile-scoped normalization for tags, tubing sizes, named equipment, and boundary-interface legends. Generic pump tags require nearby pump-label evidence so off-page references such as `P-60001` are not treated as equipment.
7. Detect tables from repeated horizontal/vertical rules, recover merged cells, OCR each logical cell in a contact sheet, and apply within-table consistency checks.
8. Calibrate confidence from OCR, preprocessing agreement, geometry, and context; send weak or single-method results to review without changing the answer-key schema.

All operational thresholds live in `config/pipeline_config.json`. The extractor never reads the answer key.

## Answer-key scope

`evaluation/answer_key_scope.json` makes scoring assumptions explicit:

- Categories absent from a document's key are out of scope and are not false positives.
- Categories present in the key are scored as a strict reference set for reproducibility.
- Strict precision is not necessarily real-world precision when the supplied list is non-exhaustive. The manifest records the visually observed completeness concerns for documents 1 and 2.

`score.py` prints this policy before the metrics. In a live review, use `extraction_metadata.json` and the source bounding boxes to adjudicate apparent extras rather than silently deleting them.

## Validation

```bash
python3 -m unittest discover -s tests -v
```

The checked-in `output/final` directory is a reproducible example. The pipeline never reads either evaluation file; they are used only by the two scorers after extraction.
