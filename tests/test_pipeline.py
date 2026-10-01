import unittest
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from pipeline import (
    OCRItem,
    build_confidence_report,
    detect_continuation_hexagons,
    extract_pid,
    load_config,
    majority_vote,
    normalized_text,
    select_document_profile,
)
from review import build_review_rows


class PipelineTests(unittest.TestCase):
    def item(self, text, x=.5, y=.5):
        return OCRItem(text, 1.0, x, y, x + .01, y + .01, [])

    def test_pid_and_legend_normalization(self):
        answers = extract_pid([
            self.item("PID- 1-SW-LR20794"),
            self.item("SS - SAMPLE SYSTEM"),
            self.item("DM - DEMINERALIZED WATER SYSTEM"),
            self.item("CS - CHEMICAL & VOLUME CONTROL SYSTEM"),
        ])
        self.assertEqual(answers["pid"], "PID-1-SW_LR20794")
        self.assertEqual(answers["system_boundary_interfaces"], ["SS", "DM", "CS"])

    def test_cyrillic_lookalikes(self):
        self.assertEqual(normalized_text("3/8°-Т"), "3/8°-T")

    def test_named_equipment_from_split_lines(self):
        answers = extract_pid([
            self.item("SAMPLE SINK", .20, .20),
            self.item("PRIMARY", .20, .19),
            self.item("MM-13B", .20, .22),
        ])
        self.assertEqual(answers["sample_sinks"], ["PRIMARY SAMPLE SINK MM-13B"])

    def test_compound_gauge_prefers_aligned_bubble_below_label(self):
        answers = extract_pid([
            self.item("COMPOUND", .34, .11),
            self.item("PI 2888", .36, .11),
            self.item("PI 2793", .34, .135),
        ])
        self.assertEqual(answers["compound_gauge"], ["PI2793"])

    def test_continuation_requires_detected_hexagon(self):
        plain = self.item("6", .55, .55)
        detected = self.item("7", .55, .65)
        detected.alternatives = [{"source": "continuation_hexagon", "geometry_score": .97}]
        answers = extract_pid([plain, detected])
        self.assertEqual(answers["continuation_connections (hexagon)"], ["7"])

    def test_preprocessing_vote(self):
        value, agreement, confidence = majority_vote([("PI2851", .8), ("PI2851", 1.0), ("PI285I", .9)])
        self.assertEqual(value, "PI2851")
        self.assertAlmostEqual(agreement, 2 / 3)
        self.assertAlmostEqual(confidence, .9)

    def test_synthetic_continuation_hexagon_geometry(self):
        image = Image.new("RGB", (240, 220), "white")
        draw = ImageDraw.Draw(image)
        draw.polygon([(88, 90), (112, 90), (124, 109), (112, 128), (88, 128), (76, 109)], outline="black", width=2)
        detections = detect_continuation_hexagons(image, load_config())
        self.assertTrue(any(70 <= d["bbox_px"][0] <= 85 and 115 <= d["bbox_px"][2] <= 130 for d in detections))

    def test_profile_selection_keeps_legacy_and_detects_generic(self):
        config = load_config()
        legacy = [self.item("PID-1-SS-LR20519"), self.item("PI2851")]
        generic = [self.item("BV510541A"), self.item("DV510542H"), self.item("P510541")]
        unknown = [self.item("RADWASTE SYSTEM"), self.item("DRAWING NUMBER"), self.item("H-16176")]
        self.assertEqual(select_document_profile(legacy, config), "legacy_nuclear_pid")
        self.assertEqual(select_document_profile(generic, config), "generic_isa_pid")
        self.assertEqual(select_document_profile(unknown, config), "unknown_pid")
        self.assertEqual(extract_pid(unknown, config, "unknown_pid"), {"pid": "H-16176"})

    def test_generic_profile_supports_six_digit_tags_without_changing_legacy(self):
        config = load_config()
        bubble = self.item("PI 510542", .60, .35)
        bubble.alternatives = [{"source": "instrument_bubble", "geometry_score": .9, "vote_agreement": 1.0}]
        answers = extract_pid([
            self.item("WW510-", .80, .80),
            self.item("P-60002", .83, .80),
            bubble,
            self.item("(PI)", .50, .50),
            self.item("510544", .50, .51),
            self.item("ME510543", .70, .50),
            self.item("510543", .70, .51),
            self.item("BV510541A", .30, .40),
            self.item("DV510542H", .40, .40),
            self.item("METERING PUMP NO. 1", .35, .44),
            self.item("P510541", .35, .45),
            self.item("P60001", .10, .45),
            self.item("FI", .55, .53),
            self.item("610544", .55, .54),
            self.item("FI510544", .55, .55),
        ], config, "generic_isa_pid")
        self.assertEqual(answers["pid"], "WW510-P-60002")
        self.assertEqual(answers["pressure_indicators"], ["PI510542", "PI510544", "PI510543"])
        self.assertEqual(answers["valves"], ["BV510541A", "DV510542H"])
        self.assertEqual(answers["vacuum_pumps"], ["P510541"])
        self.assertEqual(answers["flow_indicators"], ["FI510544"])

    def test_confidence_evidence_ignores_short_numeric_substrings(self):
        config = load_config()
        short = self.item("510", .10, .10)
        valve = self.item("BV510541A", .30, .40)
        answers = extract_pid([short, valve], config, "generic_isa_pid")
        report, _ = build_confidence_report(4, 1, answers, [short, valve], config)
        evidence = report["fields"]["valves"][0]
        self.assertEqual(evidence["bbox"][:2], [0.3, 0.4])

    def test_review_worksheet_includes_tp_fp_fn_and_queue_state(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "structured_output.json").write_text(json.dumps({"documents": [{
                "document_id": 1,
                "answers": {"pid": "PID-1-SS_LR20519", "pressure_indicators": ["PI9999"]},
            }]}))
            (root / "extraction_metadata.json").write_text(json.dumps({"documents": [{
                "document_id": 1,
                "page": 1,
                "fields": {"pressure_indicators": [{"value": "PI9999", "confidence": .4, "source": "page_ocr"}]},
            }]}))
            (root / "review_queue.json").write_text(json.dumps({"items": [{
                "document_id": 1, "category": "pressure_indicators", "value": "PI9999", "reasons": ["low_confidence"],
            }]}))
            key = root / "answer_key.json"
            key.write_text(json.dumps({"documents": [{
                "document_id": 1,
                "answers": {"pid": "PID-1-SS_LR20519", "pressure_indicators": [], "filters": ["F-192"]},
            }]}))
            rows = build_review_rows(root, key)
            statuses = {(row["category"], row["predicted_value"]): row["comparison_to_key"] for row in rows}
            self.assertEqual(statuses[("pid", "PID-1-SS_LR20519")], "TP")
            self.assertEqual(statuses[("pressure_indicators", "PI9999")], "FP")
            self.assertEqual(statuses[("filters", "F-192")], "FN")
            pi_row = next(row for row in rows if row["predicted_value"] == "PI9999")
            self.assertEqual(pi_row["in_review_queue"], "yes")


if __name__ == "__main__":
    unittest.main()
