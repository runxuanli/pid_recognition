#!/usr/bin/env python3
"""Score structured extraction output against answer-key JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_SCOPE = Path(__file__).resolve().parent / "evaluation/answer_key_scope.json"


def load_documents(path: str) -> dict[int, dict]:
    data = json.loads(Path(path).read_text())
    return {int(doc["document_id"]): doc.get("answers", {}) for doc in data["documents"]}


def as_set(value) -> set[str]:
    if isinstance(value, list):
        return {str(item).strip().upper() for item in value}
    if value in (None, ""):
        return set()
    return {str(value).strip().upper()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction")
    parser.add_argument("answer_key")
    parser.add_argument("--scope", default=str(DEFAULT_SCOPE), help="answer-key scope manifest")
    args = parser.parse_args()
    predictions = load_documents(args.prediction)
    gold = load_documents(args.answer_key)
    scope_path = Path(args.scope)
    scope = json.loads(scope_path.read_text(encoding="utf-8")) if scope_path.exists() else {}
    policy = scope.get("policy", {})
    print("SCOPE")
    print(f"  absent categories: {policy.get('categories_absent_from_key', 'out_of_scope')}")
    print(f"  present categories: {policy.get('categories_present_in_key', 'strict_reference_set')}")
    if policy.get("metric_interpretation"):
        print(f"  interpretation: {policy['metric_interpretation']}")
    for document_id in sorted(gold):
        document_scope = scope.get("documents", {}).get(str(document_id), {})
        if document_scope:
            print(f"  doc {document_id}: {document_scope.get('completeness', 'unknown')} - {document_scope.get('scope', '')}")
    print()
    micro_tp = micro_fp = micro_fn = 0
    print(f"{'doc':>3}  {'category':42} {'precision':>9} {'recall':>7} {'f1':>7}  counts")
    for document_id, answers in gold.items():
        predicted_answers = predictions.get(document_id, {})
        for category, expected_raw in answers.items():
            expected = as_set(expected_raw)
            predicted = as_set(predicted_answers.get(category, []))
            tp, fp, fn = len(expected & predicted), len(predicted - expected), len(expected - predicted)
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            micro_tp += tp
            micro_fp += fp
            micro_fn += fn
            print(f"{document_id:>3}  {category:42} {precision:9.3f} {recall:7.3f} {f1:7.3f}  {tp}/{fp}/{fn}")
    precision = micro_tp / (micro_tp + micro_fp) if micro_tp + micro_fp else 0.0
    recall = micro_tp / (micro_tp + micro_fn) if micro_tp + micro_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    print(f"MICRO {'':39} {precision:9.3f} {recall:7.3f} {f1:7.3f}  {micro_tp}/{micro_fp}/{micro_fn}")
    print("\nNote: predictions in categories absent from the key are intentionally not counted as false positives.")


if __name__ == "__main__":
    main()
