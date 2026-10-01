# Ground-truth and review guide

## What you can label without process-engineering expertise

Separate observation from interpretation. First transcribe visible text, symbol shape, and drawing-grid location. Do not decide that `F2034` is a filter merely because it starts with `F`. Category assignment must be supported by the drawing legend, an explicit nearby equipment name, a known document profile, or an SME decision.

For each item record:

```text
visible_text | symbol_shape | grid/location | proposed_category | evidence | status
```

Use `confirmed`, `rejected`, `corrected`, or `needs_sme`. Keep `needs_sme` out of the scored ground truth until it is adjudicated.

## How to find missing values

Reviewing predictions can find false positives but cannot measure recall. Make an independent pass through the drawing before looking at the model output. Use the printed drawing grid and scan left-to-right, top-to-bottom, once per category. Record every in-scope object, then compare that list with `structured_output.json`. A visible in-scope object absent from the prediction is a false negative.

Use two passes for safety:

1. Reviewer A transcribes visible candidates without seeing model output.
2. Reviewer B compares candidates, model output, legend, and title block.
3. An SME resolves only the remaining category or scope questions.

## How to use the review worksheet

Without an answer key:

```bash
python3 review.py output/unseen_document_4 \
  --output output/unseen_document_4/review_audit.csv
```

With an answer key:

```bash
python3 review.py output/final \
  --answer-key answer_key.json \
  --output output/final/review_audit.csv
```

The worksheet includes every prediction, its confidence, source, bbox, queue status, and reasons. When a key is supplied it also adds missing expected values as `FN`. Fill in `review_decision`, `corrected_value`, and `review_notes`.

## Scoring policy

For each scored category:

```text
TP = prediction and ground truth agree
FP = predicted value is absent or assigned to the wrong category
FN = ground-truth value is missing from the prediction
precision = TP / (TP + FP)
recall = TP / (TP + FN)
```

A category omitted from an answer key is out of scope, not automatically empty. An explicitly empty list means the reviewer confirmed that no in-scope values exist. Report unresolved items separately; do not convert confidence into accuracy.

## Minimum evidence by field type

- Drawing number: title-block label plus adjacent identifier.
- Instrument bubble: symbol geometry, prefix text, and number must agree.
- Valve/equipment: visible tag plus a symbol or explicit equipment label.
- Filter: nearby `FILTER`/`STRAINER`, legend mapping, or profile-specific convention; an `F` prefix alone is insufficient.
- Continuation: closed connector geometry plus the complete interior label; exclude border coordinates and reference arrows.
- Table cell: exact row, column, text, and merged-cell position.

If the automatic profile is `unknown_pid`, treat the entire page as review-required. Do not score semantic categories until their conventions and scope have been confirmed.
