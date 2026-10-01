#!/usr/bin/env python3
"""Report exact cell accuracy for reconstructed tables."""

import argparse
import json
from pathlib import Path


def signature(table):
    rows = table["rows"]
    return len(rows), max((len(row) for row in rows), default=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prediction")
    parser.add_argument("expected")
    args = parser.parse_args()
    predicted = json.loads(Path(args.prediction).read_text())["tables"]
    expected = json.loads(Path(args.expected).read_text())["tables"]
    predicted_by_shape = {signature(table): table for table in predicted}
    correct = total = 0
    for table in expected:
        shape = signature(table)
        candidate = predicted_by_shape.get(shape, {"rows": []})
        table_correct = table_total = 0
        for row_index, row in enumerate(table["rows"]):
            for column_index, value in enumerate(row):
                actual = candidate["rows"][row_index][column_index] if row_index < len(candidate["rows"]) and column_index < len(candidate["rows"][row_index]) else None
                table_correct += actual == value
                table_total += 1
        correct += table_correct
        total += table_total
        print(f"{shape[0]}x{shape[1]}: {table_correct}/{table_total} ({table_correct / table_total:.1%})")
    print(f"TOTAL: {correct}/{total} ({correct / total:.1%})")


if __name__ == "__main__":
    main()
