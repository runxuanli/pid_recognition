#!/usr/bin/env python3
"""Create a human-review worksheet from one pipeline output directory."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def values(raw) -> list[str]:
    if raw in (None, ""):
        return []
    if isinstance(raw, list):
        return [str(value) for value in raw]
    return [str(raw)]


def normalized(value: str) -> str:
    return value.strip().upper()


def load_documents(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(document["document_id"]): document for document in data.get("documents", [])}


def build_review_rows(result_dir: Path, answer_key: Path | None = None) -> list[dict]:
    predictions = load_documents(result_dir / "structured_output.json")
    metadata = load_documents(result_dir / "extraction_metadata.json")
    queue_path = result_dir / "review_queue.json"
    queued = json.loads(queue_path.read_text(encoding="utf-8")).get("items", []) if queue_path.exists() else []
    gold = load_documents(answer_key) if answer_key else {}
    queue_lookup = {
        (int(item["document_id"]), item["category"], normalized(str(item["value"]))): item
        for item in queued
    }
    rows = []
    for document_id, document in predictions.items():
        answers = document.get("answers", {})
        meta_document = metadata.get(document_id, {})
        page = meta_document.get("page", "")
        fields = meta_document.get("fields", {})
        gold_answers = gold.get(document_id, {}).get("answers", {})
        for category, raw_predictions in answers.items():
            expected = {normalized(value) for value in values(gold_answers.get(category, []))}
            category_is_scored = answer_key is not None and category in gold_answers
            field_lookup = {
                normalized(str(field.get("value", ""))): field
                for field in fields.get(category, [])
            }
            for prediction in values(raw_predictions):
                key = normalized(prediction)
                field = field_lookup.get(key, {})
                queue_item = queue_lookup.get((document_id, category, key), {})
                if answer_key is None:
                    comparison = "UNREVIEWED"
                elif not category_is_scored:
                    comparison = "OUT_OF_SCOPE"
                else:
                    comparison = "TP" if key in expected else "FP"
                rows.append({
                    "document_id": document_id,
                    "page": page,
                    "category": category,
                    "predicted_value": prediction,
                    "comparison_to_key": comparison,
                    "confidence": field.get("confidence", ""),
                    "source": field.get("source", ""),
                    "bbox": json.dumps(field.get("bbox"), separators=(",", ":")) if field.get("bbox") else "",
                    "in_review_queue": "yes" if queue_item else "no",
                    "queue_reasons": ";".join(queue_item.get("reasons", [])),
                    "review_decision": "",
                    "corrected_value": "",
                    "review_notes": "",
                })
        if answer_key is not None:
            for category, raw_expected in gold_answers.items():
                predicted = {normalized(value) for value in values(answers.get(category, []))}
                for expected_value in values(raw_expected):
                    if normalized(expected_value) in predicted:
                        continue
                    rows.append({
                        "document_id": document_id,
                        "page": page,
                        "category": category,
                        "predicted_value": expected_value,
                        "comparison_to_key": "FN",
                        "confidence": "",
                        "source": "answer_key",
                        "bbox": "",
                        "in_review_queue": "no",
                        "queue_reasons": "missing_from_prediction",
                        "review_decision": "",
                        "corrected_value": expected_value,
                        "review_notes": "Locate independently on the drawing before changing extraction logic.",
                    })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", help="Pipeline output directory")
    parser.add_argument("--answer-key", help="Optional answer-key JSON for TP/FP/FN labels")
    parser.add_argument("--output", default="review_audit.csv", help="CSV worksheet path")
    args = parser.parse_args()
    result_dir = Path(args.result_dir).resolve()
    rows = build_review_rows(result_dir, Path(args.answer_key).resolve() if args.answer_key else None)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else [
        "document_id", "page", "category", "predicted_value", "comparison_to_key",
        "confidence", "source", "bbox", "in_review_queue", "queue_reasons",
        "review_decision", "corrected_value", "review_notes",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
